"""
STAGE 6 — Robust evaluation.

Accuracy is misleading here: with ~5% fraud prevalence, a model that
predicts "Normal" for every single transaction scores >94% accuracy while
catching zero fraud. This script reports the metrics that actually matter
for an imbalanced detection problem (precision, recall, F1, ROC-AUC, PR-AUC,
confusion matrix), demonstrates the accuracy-is-misleading point with real
numbers from this dataset, and adds a cost-based framing: assign an
illustrative cost to false positives (customer friction from wrongly
blocking a legitimate transaction) vs false negatives (a fraud loss that
goes uncaught), then sweep the classification threshold to show how the
total illustrative cost moves as the threshold changes.
"""
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score,
)

import config

split = joblib.load(config.SPLIT_PKL)
X_test, y_test = split["X_test"], split["y_test"]

model = XGBClassifier()
model.load_model(config.MODEL_JSON)
proba = model.predict_proba(X_test)[:, 1]
pred_default = (proba >= 0.5).astype(int)

# ---- why accuracy is misleading ----
naive_accuracy = accuracy_score(y_test, np.zeros(len(y_test)))
model_accuracy = accuracy_score(y_test, pred_default)
print("Why plain accuracy is misleading on this dataset:")
print(f"  A model predicting 'Normal' for every transaction scores "
      f"{naive_accuracy*100:.2f}% accuracy while catching 0 fraud cases.")
print(f"  Our model's accuracy ({model_accuracy*100:.2f}%) looks similar, "
      f"but the difference that matters is recall/precision on the fraud class.\n")

# ---- proper metrics at default 0.5 threshold ----
prec = precision_score(y_test, pred_default)
rec = recall_score(y_test, pred_default)
f1 = f1_score(y_test, pred_default)
auc = roc_auc_score(y_test, proba)
ap = average_precision_score(y_test, proba)
cm = confusion_matrix(y_test, pred_default)

print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"PR-AUC:    {ap:.4f}")
print(f"Confusion matrix (rows=actual, cols=predicted [Normal, Fraud]):\n{cm}")

# ---- cost-based threshold sweep ----
thresholds = np.linspace(0.01, 0.99, 99)
costs = []
for t in thresholds:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    total_cost = fp * config.COST_FALSE_POSITIVE + fn * config.COST_FALSE_NEGATIVE
    costs.append(total_cost)
costs = np.array(costs)
best_idx = costs.argmin()
best_threshold = thresholds[best_idx]

print(f"\nCost-based framing (illustrative: FP=${config.COST_FALSE_POSITIVE:.0f} "
      f"customer friction, FN=${config.COST_FALSE_NEGATIVE:.0f} fraud loss):")
print(f"  Cost at default threshold 0.50: ${costs[np.argmin(np.abs(thresholds-0.5))]:.0f}")
print(f"  Minimum cost ${costs[best_idx]:.0f} at threshold {best_threshold:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, color="#2b6cb0", linewidth=2)
plt.axvline(best_threshold, color="#c53030", linestyle="--",
            label=f"min-cost threshold = {best_threshold:.2f}")
plt.axvline(0.5, color="#718096", linestyle=":", label="default threshold = 0.50")
plt.xlabel("Classification threshold")
plt.ylabel(f"Illustrative total cost (FP=${config.COST_FALSE_POSITIVE:.0f}, "
           f"FN=${config.COST_FALSE_NEGATIVE:.0f})")
plt.title("Cost vs. classification threshold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{config.PLOTS_DIR}/cost_vs_threshold.png", dpi=150)
plt.close()

# also pick a high-precision "block" threshold for the demo app: the lowest
# threshold at which precision among test-set positives is still very high
block_threshold = None
for t in sorted(thresholds, reverse=True):
    pred = (proba >= t).astype(int)
    if pred.sum() >= 3:
        p = precision_score(y_test, pred, zero_division=0)
        if p >= 0.85:
            block_threshold = t
            break
if block_threshold is None:
    block_threshold = min(best_threshold + 0.3, 0.95)

thresholds_out = {
    "review_threshold": float(best_threshold),
    "block_threshold": float(max(block_threshold, best_threshold + 0.05)),
}
with open(config.THRESHOLDS_JSON, "w") as f:
    json.dump(thresholds_out, f, indent=2)

print(f"\nDemo tiering thresholds saved: {thresholds_out}")
print(f"Saved {config.PLOTS_DIR}/cost_vs_threshold.png and {config.THRESHOLDS_JSON}")
