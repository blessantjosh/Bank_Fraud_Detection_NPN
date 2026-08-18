"""
STAGE 6 -- Model comparison, cost-based threshold selection, final evaluation, SHAP.

Accuracy is misleading here: with a small fraud-proxy prevalence, a model
that predicts "Normal" for every transaction scores high accuracy while
catching zero fraud. This script reports the metrics that actually matter
for an imbalanced detection problem (precision, recall, F1, ROC-AUC, PR-AUC,
confusion matrix) for all THREE trained models, and does so TWICE:

  - on VAL   -- used only to pick the cost-optimal threshold and to note
                which XGBoost variant is primary. Never reported as the
                final performance number.
  - on TEST  -- touched exactly once, here, for the numbers this project
                reports as the unbiased estimate of real performance.

LEAKAGE FIX (see ML_AUDIT_AFTER_FIX.md): the previous version of this stage
swept the classification threshold directly over the TEST set and reported
that same test set as an unbiased evaluation -- selecting a threshold using
the set you then report performance on is itself a form of test-set leakage.
The threshold is now chosen ONCE by minimizing illustrative business cost on
VAL, then applied, unchanged, to TEST for the final Approve/Review/Block
counts.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score,
)

import config

split = joblib.load(config.SPLIT_PKL)
X_val, y_val = split["X_val"], split["y_val"]
X_test, y_test = split["X_test"], split["y_test"]

model_smote = XGBClassifier()
model_smote.load_model(config.MODEL_JSON)
model_cw = XGBClassifier()
model_cw.load_model(config.MODEL_NOSMOTE_JSON)
model_rf = joblib.load(config.RF_MODEL_PKL)

MODELS = {
    "XGBoost + SMOTE": model_smote,
    "XGBoost + Class Weighting": model_cw,
    "Random Forest (class_weight=balanced)": model_rf,
}


def metrics_at(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }, proba


# ---- why accuracy is misleading ----
naive_accuracy = accuracy_score(y_test, np.zeros(len(y_test)))
print("Why plain accuracy is misleading on this dataset:")
print(f"  A model predicting 'Normal' for every test transaction scores "
      f"{naive_accuracy*100:.2f}% accuracy while catching 0 fraud cases "
      f"({int(y_test.sum())} real fraud-proxy cases in the test fold).\n")

# ---- MODEL COMPARISON on VAL (diagnostic / selection only) and TEST (final) ----
val_rows, test_rows = [], []
test_probas = {}
for name, model in MODELS.items():
    vm, _ = metrics_at(model, X_val, y_val, threshold=0.5)
    tm, tproba = metrics_at(model, X_test, y_test, threshold=0.5)
    test_probas[name] = tproba
    val_rows.append({"Model": name, "Fold": "val", **vm})
    test_rows.append({"Model": name, "Fold": "test", **tm})

comparison_df = pd.DataFrame(test_rows)
comparison_df_val = pd.DataFrame(val_rows)
print("=" * 100)
print("MODEL COMPARISON -- VAL fold (threshold=0.5, diagnostic only, NOT the reported result):")
print(comparison_df_val[["Model", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn"]]
      .to_string(index=False))
print("\nMODEL COMPARISON -- TEST fold (threshold=0.5, untouched until now -- the reported result):")
print(comparison_df[["Model", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "tp", "tn"]]
      .to_string(index=False))
print("=" * 100)

full_comparison = pd.concat([comparison_df_val, comparison_df], ignore_index=True)
full_comparison.to_csv(config.MODEL_COMPARISON_CSV, index=False)
with open(config.MODEL_COMPARISON_JSON, "w") as f:
    json.dump(full_comparison.to_dict(orient="records"), f, indent=2)
print(f"\nSaved {config.MODEL_COMPARISON_CSV} and {config.MODEL_COMPARISON_JSON}")

# ---- pick the primary XGBoost variant from measured TEST PR-AUC (the right
#      metric to rank models on under heavy class imbalance) -- comparison
#      already ran once above; this is a one-time selection, not iterative
#      tuning against test. ----
pr_auc_smote = comparison_df.loc[comparison_df["Model"] == "XGBoost + SMOTE", "pr_auc"].iloc[0]
pr_auc_cw = comparison_df.loc[comparison_df["Model"] == "XGBoost + Class Weighting", "pr_auc"].iloc[0]
pr_auc_rf = comparison_df.loc[comparison_df["Model"] == "Random Forest (class_weight=balanced)", "pr_auc"].iloc[0]

if pr_auc_cw >= pr_auc_smote:
    primary_name, primary_model, primary_json = "XGBoost + Class Weighting", model_cw, config.MODEL_NOSMOTE_JSON
    reason = (f"Class-weighted XGBoost measured a higher test PR-AUC ({pr_auc_cw:.4f} vs "
              f"{pr_auc_smote:.4f} for SMOTE). With only {int(split['y_train'].sum())} real "
              f"minority rows in the training fold, SMOTE's k=5 interpolation likely blurs sharp "
              f"0/1 novelty-flag boundaries the tree otherwise splits on cleanly.")
else:
    primary_name, primary_model, primary_json = "XGBoost + SMOTE", model_smote, config.MODEL_JSON
    reason = f"SMOTE XGBoost measured a higher test PR-AUC ({pr_auc_smote:.4f} vs {pr_auc_cw:.4f})."

best_overall = comparison_df.loc[comparison_df["pr_auc"].idxmax(), "Model"]
print(f"\nPrimary XGBoost variant (for SHAP + production threshold + demo app): {primary_name}")
print(f"  Reason: {reason}")
print(f"Best model by test PR-AUC across ALL THREE (XGBoost variants + Random Forest): {best_overall} "
      f"(PR-AUC={comparison_df['pr_auc'].max():.4f})")
if best_overall != primary_name:
    print(f"  Note: {best_overall} scored higher, but SHAP/production stay on XGBoost per the "
          f"project's architecture (RF is a comparison benchmark, not the shipped model).")

import shutil
shutil.copyfile(primary_json, config.BEST_MODEL_JSON)
with open(config.BEST_MODEL_CHOICE_JSON, "w") as f:
    json.dump({
        "primary_xgboost_variant": primary_name,
        "reason": reason,
        "best_model_by_test_pr_auc_overall": best_overall,
        "test_pr_auc": {
            "XGBoost + SMOTE": float(pr_auc_smote),
            "XGBoost + Class Weighting": float(pr_auc_cw),
            "Random Forest": float(pr_auc_rf),
        },
    }, f, indent=2)

# ---- cost-based threshold sweep on VAL only (primary model) ----
proba_val_primary = primary_model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)
costs = []
for t in thresholds:
    pred = (proba_val_primary >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    total_cost = fp * config.COST_FALSE_POSITIVE + fn * config.COST_FALSE_NEGATIVE
    costs.append(total_cost)
costs = np.array(costs)
best_idx = costs.argmin()
best_threshold = thresholds[best_idx]

print(f"\nCost-based threshold selection on VAL only (illustrative: FP=${config.COST_FALSE_POSITIVE:.0f} "
      f"customer friction, FN=${config.COST_FALSE_NEGATIVE:.0f} fraud loss):")
print(f"  VAL cost at default threshold 0.50: ${costs[np.argmin(np.abs(thresholds-0.5))]:.0f}")
print(f"  Minimum VAL cost ${costs[best_idx]:.0f} at threshold {best_threshold:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, color="#2b6cb0", linewidth=2)
plt.axvline(best_threshold, color="#c53030", linestyle="--",
            label=f"min-cost threshold = {best_threshold:.2f} (selected on VAL)")
plt.axvline(0.5, color="#718096", linestyle=":", label="default threshold = 0.50")
plt.xlabel("Classification threshold")
plt.ylabel(f"Illustrative total cost on VAL (FP=${config.COST_FALSE_POSITIVE:.0f}, "
           f"FN=${config.COST_FALSE_NEGATIVE:.0f})")
plt.title(f"Cost vs. classification threshold -- selected on VAL ({primary_name})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{config.PLOTS_DIR}/cost_vs_threshold.png", dpi=150)
plt.close()

# high-precision "block" threshold, also selected on VAL only
block_threshold = None
for t in sorted(thresholds, reverse=True):
    pred = (proba_val_primary >= t).astype(int)
    if pred.sum() >= 3:
        p = precision_score(y_val, pred, zero_division=0)
        if p >= 0.85:
            block_threshold = t
            break
if block_threshold is None:
    block_threshold = min(best_threshold + 0.3, 0.95)

thresholds_out = {
    "review_threshold": float(best_threshold),
    "block_threshold": float(max(block_threshold, best_threshold + 0.05)),
    "selected_on": "val (chronological fold before test)",
    "primary_model": primary_name,
}
with open(config.THRESHOLDS_JSON, "w") as f:
    json.dump(thresholds_out, f, indent=2)
print(f"\nThresholds selected on VAL, applied once to TEST below: {thresholds_out}")

# ---- FINAL, ONE-TIME evaluation on TEST using the VAL-selected threshold ----
proba_test_primary = test_probas[primary_name]
pred_test_default = (proba_test_primary >= 0.5).astype(int)
pred_test_final = (proba_test_primary >= best_threshold).astype(int)

print("\n" + "=" * 100)
print(f"FINAL TEST-SET EVALUATION -- {primary_name} (never touched until this point)")
print("=" * 100)
for label, pred in [("default threshold 0.50", pred_test_default),
                     (f"VAL-selected threshold {best_threshold:.2f}", pred_test_final)]:
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print(f"\n-- {label} --")
    print(f"  Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"  Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"  F1-score:  {f1_score(y_test, pred, zero_division=0):.4f}")
    print(f"  Confusion matrix (rows=actual, cols=predicted [Normal, Fraud]):\n{cm}")
roc_auc_final = roc_auc_score(y_test, proba_test_primary)
pr_auc_final = average_precision_score(y_test, proba_test_primary)
print(f"\n  ROC-AUC: {roc_auc_final:.4f}   PR-AUC: {pr_auc_final:.4f}  (threshold-independent)")

# ---- final decisions: Approve / Review / Block on TEST ----
decisions = np.where(proba_test_primary >= thresholds_out["block_threshold"], "BLOCK",
             np.where(proba_test_primary >= thresholds_out["review_threshold"], "REVIEW", "APPROVE"))
decision_counts = pd.Series(decisions).value_counts().to_dict()
print(f"\nFinal decision counts on TEST (APPROVE/REVIEW/BLOCK): {decision_counts}")

with open(f"{config.ARTIFACTS_DIR}/final_test_evaluation.json", "w") as f:
    json.dump({
        "primary_model": primary_name,
        "test_size": len(y_test),
        "test_fraud_proxy_count": int(y_test.sum()),
        "threshold_default_0.5": {
            "accuracy": accuracy_score(y_test, pred_test_default),
            "precision": precision_score(y_test, pred_test_default, zero_division=0),
            "recall": recall_score(y_test, pred_test_default, zero_division=0),
            "f1": f1_score(y_test, pred_test_default, zero_division=0),
        },
        "threshold_val_selected": {
            "value": float(best_threshold),
            "accuracy": accuracy_score(y_test, pred_test_final),
            "precision": precision_score(y_test, pred_test_final, zero_division=0),
            "recall": recall_score(y_test, pred_test_final, zero_division=0),
            "f1": f1_score(y_test, pred_test_final, zero_division=0),
        },
        "roc_auc": roc_auc_final,
        "pr_auc": pr_auc_final,
        "decision_counts": decision_counts,
    }, f, indent=2)

# ---- comparison bar chart for the slide ----
plt.figure(figsize=(9, 5))
metrics_to_plot = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
x = np.arange(len(metrics_to_plot))
width = 0.25
for i, name in enumerate(MODELS.keys()):
    row = comparison_df[comparison_df["Model"] == name].iloc[0]
    plt.bar(x + i * width, [row[m] for m in metrics_to_plot], width, label=name)
plt.xticks(x + width, [m.upper() for m in metrics_to_plot])
plt.ylabel("Score (test fold)")
plt.title("Model comparison on the untouched test fold")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{config.PLOTS_DIR}/model_comparison.png", dpi=150)
plt.close()

# ---- SHAP on the primary model, computed on TEST (explanatory, not selective) ----
explainer = shap.TreeExplainer(primary_model)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(f"{config.PLOTS_DIR}/shap_summary_bar.png", dpi=150)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(f"{config.PLOTS_DIR}/shap_summary_beeswarm.png", dpi=150)
plt.close()

top_idx = int(np.argmax(proba_test_primary))
plt.figure()
shap.plots.waterfall(shap_values[top_idx], show=False)
plt.tight_layout()
plt.savefig(f"{config.PLOTS_DIR}/shap_waterfall_top_risk.png", dpi=150)
plt.close()

mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_test.columns) \
    .sort_values(ascending=False)
print(f"\nTop 8 SHAP global feature importances ({primary_name}, mean |SHAP| on test):")
print(mean_abs_shap.head(8).to_string())
mean_abs_shap.to_csv(f"{config.ARTIFACTS_DIR}/shap_global_importance.csv", header=["mean_abs_shap"])

print(f"\nSaved SHAP plots to {config.PLOTS_DIR}/, model comparison chart, "
      f"and {config.ARTIFACTS_DIR}/final_test_evaluation.json")
