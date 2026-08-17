"""
evaluate.py -- final, ONE-TIME evaluation on the untouched test split.

    python evaluate.py [--config config.yaml]

Loads the artifacts train.py saved (model, preprocessor, threshold,
feature columns), scores the held-out test split that no fitting step has
ever seen, and writes:
  - reports/metrics/test_evaluation.json   (headline numbers)
  - reports/metrics/test_confusion_matrix.csv
  - reports/figures/roc_curve.png, pr_curve.png, confusion_matrix.png,
    feature_importance.png, shap_summary.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src import evaluation, explainability, models
from src.config import load_config, resolve_path, setup_logging
from src.data_loader import load_and_split
from src.prediction import load_artifacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logger = setup_logging(name="fraud_detection")
    cfg = load_config(args.config)

    model, preprocessor, meta = load_artifacts(cfg)
    logger.info("Loaded final model: %s / %s (threshold=%.3f)",
                meta["model_type"], meta["strategy"], meta["threshold"])

    _, _, test_df = load_and_split(cfg)
    y_test = preprocessor.get_target(test_df)

    model_type = meta["model_type"]
    strategy = meta["strategy"]
    use_dense = (model_type in ("logistic_regression", "random_forest")) or (
        strategy in ("smote", "smote_undersample")
    )
    X_test = preprocessor.transform_dense(test_df) if use_dense else preprocessor.transform_tree(test_df)
    X_test = X_test[meta["feature_columns"]]

    p_test = models.predict_proba(model, X_test, model_type)
    if np.std(p_test) < 1e-9:
        raise RuntimeError("Final model produced constant predictions on the test set -- refusing to report.")

    target_fpr = cfg.evaluation.target_fpr_for_tpr
    metrics = evaluation.evaluate_scores(y_test, p_test, target_fpr, label="FINAL TEST EVALUATION")

    threshold = float(meta["threshold"])
    confusion = evaluation.confusion_at_threshold(y_test, p_test, threshold)
    cost = evaluation.cost_sensitive_eval(
        y_test, p_test, threshold, cfg.evaluation.cost_fp, cfg.evaluation.cost_fn
    )

    older = (test_df[cfg.protected_attribute.column] > cfg.protected_attribute.threshold).to_numpy()
    fairness = evaluation.fairness_report(
        y_test, p_test, older, target_fpr,
        label=f"customer_age > {cfg.protected_attribute.threshold} (test set)",
    )

    metrics_dir = resolve_path("reports/metrics")
    figures_dir = resolve_path("reports/figures")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model": meta["model_type"],
        "strategy": meta["strategy"],
        "threshold": threshold,
        "test_metrics": metrics,
        "confusion_at_threshold": confusion,
        "cost_sensitive": cost,
        "fairness_test": fairness,
    }
    with open(metrics_dir / "test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    pd.DataFrame([confusion]).to_csv(metrics_dir / "test_confusion_matrix.csv", index=False)
    logger.info("Wrote reports/metrics/test_evaluation.json")

    # --- Figures -----------------------------------------------------------
    RocCurveDisplay.from_predictions(y_test, p_test)
    plt.title(f"ROC curve -- {model_type}/{strategy} (test), AUC={metrics['roc_auc']:.4f}")
    plt.savefig(figures_dir / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, p_test)
    plt.title(f"PR curve -- {model_type}/{strategy} (test), PR-AUC={metrics['pr_auc']:.4f}")
    plt.savefig(figures_dir / "pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    cm = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix at threshold={threshold:.3f} (test)")
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- Feature importance (gain-based) for LightGBM/XGBoost ---------------
    if model_type == "lightgbm":
        imp = explainability.lightgbm_feature_importance(model, top_n=20)
        imp.to_csv(metrics_dir / "feature_importance.csv", index=False)
        plt.figure(figsize=(8, 8))
        plt.barh(imp["feature"][::-1], imp["gain"][::-1])
        plt.xlabel("Gain")
        plt.title("Top 20 features by gain (LightGBM, final model)")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_importance.png", dpi=120)
        plt.close()
        logger.info("Top 5 features by gain:\n%s", imp.head(5).to_string(index=False))
    elif model_type == "xgboost":
        imp = explainability.xgboost_feature_importance(model, top_n=20)
        imp.to_csv(metrics_dir / "feature_importance.csv", index=False)
        plt.figure(figsize=(8, 8))
        plt.barh(imp["feature"][::-1], imp["gain"][::-1])
        plt.xlabel("Gain")
        plt.title("Top 20 features by gain (XGBoost, final model)")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_importance.png", dpi=120)
        plt.close()
    else:
        logger.info("Final model is %s -- gain-based importance only applies to tree models; skipped.", model_type)

    # --- SHAP summary + one individual fraud-flagged prediction -------------
    if model_type in ("lightgbm", "xgboost"):
        rng = np.random.default_rng(cfg.seed)
        sample_idx = rng.choice(len(X_test), size=min(3000, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_idx]

        explainer, shap_values = explainability.shap_summary(model, X_sample, model_type)

        import shap
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.title("SHAP summary -- final model (test sample)")
        plt.savefig(figures_dir / "shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()

        flagged = np.flatnonzero((p_test[sample_idx] >= threshold) & (y_test.iloc[sample_idx].to_numpy() == 1))
        if len(flagged) > 0:
            pick = sample_idx[flagged[0]]
            row = X_test.iloc[[pick]]
            row_shap = explainability.explain_single_prediction(explainer, row)
            top_feats = explainability.top_shap_features_for_row(row_shap, list(row.columns))
            top_feats.to_csv(metrics_dir / "individual_explanation.csv", index=False)
            logger.info(
                "Individual explanation for one real TEST-set fraud-flagged row "
                "(index %d, predicted prob=%.4f):\n%s",
                pick, p_test[pick], top_feats.to_string(index=False),
            )

            plt.figure()
            shap.plots.waterfall(row_shap[0], show=False, max_display=15)
            plt.title(f"SHAP explanation -- one real fraud-flagged test row (idx={pick})")
            plt.savefig(figures_dir / "shap_individual_explanation.png", dpi=120, bbox_inches="tight")
            plt.close()
        else:
            logger.warning("No true-positive fraud rows found in the SHAP sample; "
                            "skipping individual explanation (sample too small/unlucky).")
    else:
        logger.info("Final model is %s -- SHAP TreeExplainer only applies to tree models; skipped.", model_type)

    logger.info("=" * 70)
    logger.info("FINAL TEST RESULTS: ROC-AUC=%.4f  PR-AUC=%.4f  TPR@%.0f%%FPR=%.4f",
                metrics["roc_auc"], metrics["pr_auc"], target_fpr * 100,
                metrics[f"tpr_at_{int(target_fpr*100)}pct_fpr"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
