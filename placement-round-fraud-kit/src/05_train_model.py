"""
STAGE 5 — Supervised model + explainability.

Trains the primary XGBoost classifier on the SMOTE-balanced training set,
plus a second XGBoost trained on the original imbalanced fold with
scale_pos_weight instead (for the SMOTE-vs-class-weighting comparison
printed here and reported in Stage 6). Then:
  - SHAP TreeExplainer for global feature importance (bar plot) and a
    single-prediction explanation (waterfall plot).
  - A shallow (depth=3) Decision Tree trained on the same labels, purely to
    print human-readable if/then rules for a slide -- not used for scoring.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import roc_auc_score, average_precision_score

import config

split = joblib.load(config.SPLIT_PKL)
X_train, y_train = split["X_train"], split["y_train"]
X_train_smote, y_train_smote = split["X_train_smote"], split["y_train_smote"]
X_test, y_test = split["X_test"], split["y_test"]
scale_pos_weight = split["scale_pos_weight"]

# ---- primary model: XGBoost on SMOTE-balanced training data ----
model_smote = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", random_state=config.RANDOM_STATE,
)
model_smote.fit(X_train_smote, y_train_smote)
proba_smote = model_smote.predict_proba(X_test)[:, 1]

# ---- comparison model: XGBoost with class weighting, no SMOTE ----
model_cw = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", random_state=config.RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
)
model_cw.fit(X_train, y_train)
proba_cw = model_cw.predict_proba(X_test)[:, 1]

auc_smote, ap_smote = roc_auc_score(y_test, proba_smote), average_precision_score(y_test, proba_smote)
auc_cw, ap_cw = roc_auc_score(y_test, proba_cw), average_precision_score(y_test, proba_cw)
print("SMOTE vs class-weighting on held-out test set:")
print(f"  SMOTE          ROC-AUC={auc_smote:.4f}  PR-AUC={ap_smote:.4f}")
print(f"  class-weighted ROC-AUC={auc_cw:.4f}  PR-AUC={ap_cw:.4f}")
if ap_cw >= ap_smote:
    print("  -> class-weighting measures BETTER here (higher PR-AUC, the metric that "
          "matters most under 5% prevalence). With only ~107 real minority rows, SMOTE's "
          "k=5 interpolation likely blurs the sharp 0/1 novelty-flag boundaries the tree "
          "otherwise splits on cleanly. We still ship the SMOTE model as primary (the brief "
          "explicitly requires demonstrating SMOTE), but this is the honest, measured "
          "comparison, not an assumption -- for a real deployment at this dataset size, "
          "class-weighting would be the better default.")
else:
    print("  -> SMOTE measures better here; kept as the primary model.")

model_smote.save_model(config.MODEL_JSON)
model_cw.save_model(config.MODEL_NOSMOTE_JSON)

# ---- SHAP explainability (on the primary SMOTE model) ----
explainer = shap.TreeExplainer(model_smote)
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

# individual explanation: the single highest-risk test transaction
top_idx = proba_smote.argmax()
plt.figure()
shap.plots.waterfall(shap_values[top_idx], show=False)
plt.tight_layout()
plt.savefig(f"{config.PLOTS_DIR}/shap_waterfall_top_risk.png", dpi=150)
plt.close()
print(f"\nSaved SHAP plots to {config.PLOTS_DIR}/ "
      f"(individual example: test row {top_idx}, predicted P(fraud)={proba_smote[top_idx]:.3f})")

# ---- shallow decision tree for human-readable rules ----
dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=config.RANDOM_STATE)
dt.fit(X_train, y_train)
rules = export_text(dt, feature_names=list(X_train.columns))
with open(config.DT_RULES_TXT, "w") as f:
    f.write(rules)
joblib.dump(dt, config.DT_MODEL_PKL)

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=list(X_train.columns), class_names=["Normal", "Fraud"],
          filled=True, rounded=True, fontsize=8)
plt.tight_layout()
plt.savefig(f"{config.PLOTS_DIR}/decision_tree.png", dpi=150)
plt.close()

print(f"\nDecision tree rules (depth=3, for the slide):\n{rules}")
print(f"Saved {config.DT_RULES_TXT} and {config.PLOTS_DIR}/decision_tree.png")
