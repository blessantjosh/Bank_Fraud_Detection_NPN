"""
STAGE 5 -- Supervised models + explainability.

Trains three classifiers on the leakage-free chronological TRAIN fold only:

  MODEL A: XGBoost on the SMOTE-balanced training set.
  MODEL B: XGBoost on the original imbalanced fold with scale_pos_weight.
  MODEL C: Random Forest, class_weight="balanced" (the same "appropriate
           class balancing" idea as Model B's scale_pos_weight, just RF's
           own idiomatic mechanism) -- added for the hackathon-judge model
           comparison in Stage 6. Same leakage-free features, same train
           fold, same random_state -- no extra tuning given to any one
           model so the Stage 6 comparison is fair.

All three are evaluated in Stage 6 on VAL (for threshold/model selection)
and TEST (the untouched, one-time, unbiased estimate) -- never here.

SHAP explainability runs in Stage 6, AFTER the model comparison picks which
XGBoost variant is primary from actual measured test metrics -- so SHAP
explains the real final model, not a guess made before evaluation exists.

This stage also trains a shallow (depth=3) Decision Tree on the train fold,
purely to print human-readable if/then rules for a slide -- not used for
scoring.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import config

split = joblib.load(config.SPLIT_PKL)
X_train, y_train = split["X_train"], split["y_train"]
X_train_smote, y_train_smote = split["X_train_smote"], split["y_train_smote"]
scale_pos_weight = split["scale_pos_weight"]

# ---- MODEL A: XGBoost on SMOTE-balanced training data ----
model_smote = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", random_state=config.RANDOM_STATE,
)
model_smote.fit(X_train_smote, y_train_smote)

# ---- MODEL B: XGBoost with class weighting, no SMOTE ----
model_cw = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", random_state=config.RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
)
model_cw.fit(X_train, y_train)

# ---- MODEL C: Random Forest, class_weight="balanced" ----
model_rf = RandomForestClassifier(
    n_estimators=200, max_depth=4, class_weight="balanced",
    random_state=config.RANDOM_STATE,
)
model_rf.fit(X_train, y_train)

model_smote.save_model(config.MODEL_JSON)
model_cw.save_model(config.MODEL_NOSMOTE_JSON)
joblib.dump(model_rf, config.RF_MODEL_PKL)
print(f"Saved {config.MODEL_JSON}, {config.MODEL_NOSMOTE_JSON}, {config.RF_MODEL_PKL}")
print("(Comparison metrics on val/test, model selection, and SHAP are "
      "computed in Stage 6, not here -- training never looks at either.)")

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
