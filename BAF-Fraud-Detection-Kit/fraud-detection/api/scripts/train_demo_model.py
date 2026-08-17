"""
api/scripts/train_demo_model.py -- produce an INTERIM, real (not fake)
model artifact set in the exact format src/training.py itself produces, so
that /predict is genuinely runnable today.

Why this script exists: at the time this API layer was built, the other
agent's src/training.py existed and worked, but no training run had been
executed yet, so fraud-detection/models/ did not contain final_model.joblib
/ preprocessor.joblib / model_meta.json. Rather than leaving /predict
permanently un-exercisable while waiting, this script runs the SAME
src.preprocessing.Preprocessor and src.models training code the real
pipeline uses, on a bounded sample of the real Base.csv data (fast: a single
logistic regression on ~150k rows, not the full four-model x five-strategy
ablation), and writes artifacts in the identical format/location.

This means: once the ML pipeline's own `python train.py` run completes and
overwrites fraud-detection/models/*, the API layer picks it up with ZERO
code changes -- same file names, same model_meta.json schema. Re-run
api/scripts/record_model_checksum.py after any retrain (by either agent).

This is explicitly an INTERIM demo model, not a production-quality one: it
trains one logistic regression on a subsample, not the tuned
LightGBM/XGBoost ablation the real pipeline performs. It exists so the API
security controls (auth, RBAC, rate limiting, integrity checks, audit
logging) can be verified against a real, working prediction, not a mock.

Run from fraud-detection/:
    python -m api.scripts.train_demo_model
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import evaluation, models, threshold_optimization
from src.config import load_config, resolve_path
from src.preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("api.train_demo_model")

SAMPLE_TRAIN_ROWS = 150_000
SAMPLE_VAL_ROWS = 30_000


def _load_sample(cfg, processed_dir: Path):
    train_p = processed_dir / "train_raw.parquet"
    val_p = processed_dir / "val_raw.parquet"

    if train_p.exists() and val_p.exists():
        logger.info("Reusing cached split parquet files at %s (read-only)", processed_dir)
        train_df = pd.read_parquet(train_p)
        val_df = pd.read_parquet(val_p)
    else:
        logger.info("No cached split found -- reading raw CSV directly (not writing a cache, to avoid "
                     "colliding with the ML pipeline's own data_loader cache).")
        raw_path = resolve_path(cfg.data.raw_path)
        df = pd.read_csv(raw_path)
        junk = [c for c in df.columns if c.lower().startswith("unnamed")]
        if junk:
            df = df.drop(columns=junk)
        from sklearn.model_selection import train_test_split

        target = cfg.data.target_col
        train_df, rest_df = train_test_split(df, train_size=0.7, stratify=df[target], random_state=cfg.seed)
        val_df, _test_df = train_test_split(rest_df, train_size=0.5, stratify=rest_df[target], random_state=cfg.seed)

    target = cfg.data.target_col
    if len(train_df) > SAMPLE_TRAIN_ROWS:
        train_df, _ = _stratified_sample(train_df, target, SAMPLE_TRAIN_ROWS, cfg.seed)
    if len(val_df) > SAMPLE_VAL_ROWS:
        val_df, _ = _stratified_sample(val_df, target, SAMPLE_VAL_ROWS, cfg.seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, target: str, n: int, seed: int):
    from sklearn.model_selection import train_test_split

    frac_keep = n / len(df)
    keep, drop = train_test_split(df, train_size=frac_keep, stratify=df[target], random_state=seed)
    return keep, drop


def main() -> None:
    cfg = load_config()
    processed_dir = resolve_path(cfg.data.processed_dir)
    train_df, val_df = _load_sample(cfg, processed_dir)
    logger.info(
        "Training on %d rows (%.4f%% fraud), validating on %d rows (%.4f%% fraud)",
        len(train_df), 100 * train_df[cfg.data.target_col].mean(),
        len(val_df), 100 * val_df[cfg.data.target_col].mean(),
    )

    pre = Preprocessor(cfg)
    pre.fit(train_df)

    X_tr = pre.transform_dense(train_df)
    y_tr = pre.get_target(train_df)
    X_va = pre.transform_dense(val_df)
    y_va = pre.get_target(val_df)

    model = models.train_logistic_regression(X_tr, y_tr, cfg, {"class_weight": "balanced"}, cfg.seed)

    p_va = models.predict_proba(model, X_va, "logistic_regression")
    if np.std(p_va) < 1e-9:
        raise RuntimeError("Demo model produced constant predictions -- refusing to save it")

    metrics = evaluation.evaluate_scores(y_va, p_va, cfg.evaluation.target_fpr_for_tpr, label="demo logistic_regression")

    thr_result = threshold_optimization.optimize(
        y_va, p_va, cfg.threshold_optimization.thresholds,
        cfg.evaluation.cost_fp, cfg.evaluation.cost_fn,
    )
    selected_threshold = thr_result["best_threshold_cost"]

    older = (val_df[cfg.protected_attribute.column] > cfg.protected_attribute.threshold).to_numpy()
    fairness = evaluation.fairness_report(
        y_va, p_va, older, cfg.evaluation.target_fpr_for_tpr,
        label=f"demo model, customer_age > {cfg.protected_attribute.threshold}",
    )

    models_dir = resolve_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "final_model.joblib")
    joblib.dump(pre, models_dir / "preprocessor.joblib")

    artifact_meta = {
        "model_type": "logistic_regression",
        "strategy": "class_weight",
        "model_iteration": None,
        "threshold": selected_threshold,
        "threshold_source": (
            f"min expected cost (fp={cfg.evaluation.cost_fp}, fn={cfg.evaluation.cost_fn}) "
            "on a validation sample -- INTERIM DEMO MODEL, see api/scripts/train_demo_model.py"
        ),
        "feature_columns": list(X_va.columns),
        "primary_metric": cfg.evaluation.primary_metric,
        "primary_metric_value": metrics[cfg.evaluation.primary_metric],
        "val_metrics": metrics,
        "fairness_val": dict(fairness),
        "seed": cfg.seed,
        "interim_demo_model": True,
        "trained_by": "api/scripts/train_demo_model.py",
    }
    with open(models_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2, default=str)

    logger.info("Saved interim demo model artifacts to %s", models_dir)
    logger.info("roc_auc=%.4f pr_auc=%.4f threshold=%.3f", metrics["roc_auc"], metrics["pr_auc"], selected_threshold)
    logger.info("Now run: python -m api.scripts.record_model_checksum")


if __name__ == "__main__":
    main()
