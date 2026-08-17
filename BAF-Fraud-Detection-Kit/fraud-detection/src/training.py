"""
training.py -- full pipeline orchestration.

raw data -> validation -> stratified train/val/test split
         -> preprocessing fit on TRAIN only
         -> imbalance handling on TRAIN only (val/test distribution untouched)
         -> train models x imbalance strategies
         -> validate + threshold tuning on validation
         -> (evaluate.py does the final untouched-test evaluation)
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import evaluation, imbalance, models, threshold_optimization
from src.config import Config, resolve_path
from src.data_loader import load_and_split
from src.data_validation import check_no_leakage, validate_quality, validate_schema
from src.preprocessing import Preprocessor

logger = logging.getLogger("fraud_detection.training")

MODEL_TYPES = ["logistic_regression", "random_forest", "lightgbm", "xgboost"]
DENSE_MODELS = {"logistic_regression", "random_forest"}
TREE_MODELS = {"lightgbm", "xgboost"}


def prepare_frames(cfg: Config):
    """
    Builds the tree-native and dense feature views for train/val only. The
    test split is deliberately NOT transformed here: it is never touched
    during training or model selection (evaluate.py loads and transforms it
    separately, in its own process, for the one-time final evaluation).
    Skipping it here removes ~15% of the memory this function would
    otherwise hold for the entire ablation loop, on an 8GB machine where
    that has mattered in practice (see README "Compute constraints").
    """
    train_df, val_df, test_df = load_and_split(cfg)
    validate_schema(train_df, cfg)
    validate_quality(train_df, cfg)
    check_no_leakage(train_df)
    del test_df

    pre = Preprocessor(cfg)
    pre.fit(train_df)

    frames = {"raw": {"train": train_df, "val": val_df}, "preprocessor": pre}
    for split_name, df in [("train", train_df), ("val", val_df)]:
        frames[f"X_tree_{split_name}"] = pre.transform_tree(df)
        frames[f"X_dense_{split_name}"] = pre.transform_dense(df)
        frames[f"y_{split_name}"] = pre.get_target(df)

    # The raw (pre-transform) train frame is not needed again once its two
    # feature views exist; val's raw frame is kept (customer_age is read
    # from it for the fairness report later). Free train's raw copy now.
    del frames["raw"]["train"]
    del train_df
    gc.collect()
    return frames


def _frame_for(model_type: str, strategy: str, frames: dict, split: str):
    """SMOTE-based strategies need a fully numeric, non-missing matrix, so
    they use the dense view even for LightGBM/XGBoost (see imbalance.py
    module docstring for the full justification)."""
    use_dense = (model_type in DENSE_MODELS) or (strategy in ("smote", "smote_undersample"))
    key = "X_dense" if use_dense else "X_tree"
    return frames[f"{key}_{split}"]


def _train_one(model_type: str, strategy: str, frames: dict, cfg: Config, seed: int = 42):
    X_tr_full = _frame_for(model_type, strategy, frames, "train")
    y_tr_full = frames["y_train"]
    X_va = _frame_for(model_type, strategy, frames, "val")
    y_va = frames["y_val"]

    X_tr, y_tr, kwargs = imbalance.apply_strategy(X_tr_full, y_tr_full, strategy, cfg, seed)

    resampling_strategies = ("random_undersample", "smote", "smote_undersample")
    if model_type == "random_forest" and strategy in resampling_strategies:
        max_rows = cfg.imbalance.rf_resampling_max_rows
        before = len(X_tr)
        X_tr, y_tr = imbalance.cap_rows(X_tr, y_tr, max_rows, seed)
        if len(X_tr) < before:
            logger.info("RF cap_rows: %d -> %d rows for strategy=%s (compute constraint, see README)",
                        before, len(X_tr), strategy)

    t0 = time.time()
    if model_type == "logistic_regression":
        model = models.train_logistic_regression(X_tr, y_tr, cfg, kwargs, seed)
    elif model_type == "random_forest":
        model = models.train_random_forest(X_tr, y_tr, cfg, kwargs, seed)
    elif model_type == "lightgbm":
        model = models.train_lightgbm(X_tr, y_tr, X_va, y_va, cfg, kwargs, seed)
    elif model_type == "xgboost":
        model = models.train_xgboost(X_tr, y_tr, X_va, y_va, cfg, kwargs, seed)
    else:
        raise ValueError(model_type)
    elapsed = time.time() - t0

    p_va = models.predict_proba(model, X_va, model_type)
    if np.std(p_va) < 1e-9:
        raise RuntimeError(f"{model_type}/{strategy}: degenerate (constant) validation predictions")

    metrics = evaluation.evaluate_scores(
        y_va, p_va, cfg.evaluation.target_fpr_for_tpr, label=f"{model_type} / {strategy}"
    )
    metrics["train_rows"] = int(len(y_tr))
    metrics["train_positive_rate"] = float(y_tr.mean())
    metrics["train_seconds"] = round(elapsed, 2)
    logger.info("  fit in %.1fs on %d rows", elapsed, len(y_tr))
    return model, metrics, p_va


def run_ablation(cfg: Config, frames: dict, seed: int = 42):
    """
    Trains all 20 model x strategy combinations. To avoid holding 20 fitted
    models (some with 300 full trees) in memory simultaneously on an 8GB
    machine, each model is immediately serialized to a scratch cache dir and
    dropped from memory; only its (small) validation predictions are kept.
    The winner is reloaded from the cache once selected (see run_training).
    """
    cache_dir = resolve_path("models/_ablation_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    trained = {}
    for model_type in MODEL_TYPES:
        for strategy in cfg.imbalance.strategies:
            key = f"{model_type}__{strategy}"
            logger.info("=" * 70)
            logger.info("TRAINING: %s", key)
            logger.info("=" * 70)
            model, metrics, p_va = _train_one(model_type, strategy, frames, cfg, seed)
            results[key] = metrics
            joblib.dump(model, cache_dir / f"{key}.joblib")
            trained[key] = p_va
            del model
            gc.collect()
    return results, trained


def select_best(results: dict, primary_metric: str) -> str:
    return max(results, key=lambda k: results[k][primary_metric])


def run_training(cfg: Config):
    frames = prepare_frames(cfg)
    results, trained = run_ablation(cfg, frames)

    comparison = pd.DataFrame(results).T
    comparison.index.name = "model__strategy"
    metrics_dir = resolve_path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_dir / "model_comparison.csv")
    logger.info("\n%s", comparison[["roc_auc", "pr_auc",
                f"tpr_at_{int(cfg.evaluation.target_fpr_for_tpr*100)}pct_fpr"]].round(4).to_string())

    primary = cfg.evaluation.primary_metric
    best_key = select_best(results, primary)
    best_model_type, best_strategy = best_key.split("__")
    logger.info("Selected best model by %s: %s (%.4f)", primary, best_key, results[best_key][primary])

    cache_dir = resolve_path("models/_ablation_cache")
    best_model = joblib.load(cache_dir / f"{best_key}.joblib")
    p_va_best = trained[best_key]
    y_va = frames["y_val"]

    thr_result = threshold_optimization.optimize(
        y_va, p_va_best,
        cfg.threshold_optimization.thresholds,
        cfg.evaluation.cost_fp, cfg.evaluation.cost_fn,
    )
    thr_result["f1_sweep"].to_csv(metrics_dir / "threshold_sweep_f1.csv", index=False)
    thr_result["cost_sweep"].to_csv(metrics_dir / "threshold_sweep_cost.csv", index=False)

    selected_threshold = thr_result["best_threshold_cost"]

    older = (frames["raw"]["val"][cfg.protected_attribute.column] > cfg.protected_attribute.threshold).to_numpy()
    fairness = evaluation.fairness_report(
        y_va, p_va_best, older, cfg.evaluation.target_fpr_for_tpr,
        label=f"{best_key} on validation, customer_age > {cfg.protected_attribute.threshold}",
    )

    models_dir = resolve_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "final_model.joblib")
    joblib.dump(frames["preprocessor"], models_dir / "preprocessor.joblib")

    feature_cols = list(_frame_for(best_model_type, best_strategy, frames, "val").columns)
    if best_model_type == "lightgbm":
        model_iteration = int(best_model.best_iteration)
    elif best_model_type == "xgboost":
        model_iteration = int(best_model.best_iteration)
    else:
        model_iteration = None
    artifact_meta = {
        "model_type": best_model_type,
        "strategy": best_strategy,
        "model_iteration": model_iteration,
        "threshold": selected_threshold,
        "threshold_source": "min expected cost "
                            f"(fp={cfg.evaluation.cost_fp}, fn={cfg.evaluation.cost_fn}) on validation",
        "feature_columns": feature_cols,
        "primary_metric": primary,
        "primary_metric_value": results[best_key][primary],
        "val_metrics": results[best_key],
        "fairness_val": dict(fairness),
        "seed": cfg.seed,
    }
    with open(models_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2, default=str)

    logger.info("Saved final model artifacts to %s", models_dir)
    logger.info("Selected: %s | threshold=%.3f", best_key, selected_threshold)

    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)
    logger.info("Cleaned up ablation model cache at %s", cache_dir)

    return {
        "frames": frames,
        "results": results,
        "trained": trained,
        "best_key": best_key,
        "best_model_type": best_model_type,
        "best_strategy": best_strategy,
        "selected_threshold": selected_threshold,
        "threshold_result": thr_result,
        "fairness_val": fairness,
        "comparison": comparison,
    }
