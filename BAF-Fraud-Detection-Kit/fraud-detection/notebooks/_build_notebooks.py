"""
_build_notebooks.py -- generates the four required notebooks as real,
executable .ipynb files (not just narrative text). Run once from the
notebooks/ directory:

    python _build_notebooks.py

Then execute them for real outputs with, e.g.:

    jupyter nbconvert --to notebook --execute --inplace 01_data_exploration.ipynb

This script is a build tool, not part of the pipeline itself -- it exists so
the four notebooks are generated consistently from one place rather than
hand-maintained in parallel with src/.
"""

import nbformat as nbf

ROOT_SETUP = """\
import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config

cfg = load_config(ROOT / "config.yaml")
pd.set_option("display.max_columns", 40)
"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = cells
    n["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ============================================================================
# 01_data_exploration.ipynb
# ============================================================================
nb1 = nb([
    md("""# 01 -- Data Exploration

Bank Account Fraud (BAF), Feedzai NeurIPS 2022, **Base** variant. This is
**account-opening** fraud (a fake/stolen identity opening a new account),
not card-transaction fraud: there is no transaction amount, no transaction
timestamp, and no account/customer id in the raw file. Every column is
available at *application time*.

This notebook runs the "first thirty minutes" checks that decide the rest of
the pipeline: split protocol, class balance, constant columns, sentinel
fractions, categorical cardinality, and the protected attribute for
fairness."""),
    code(ROOT_SETUP),
    code("""from src.data_loader import load_raw

df = load_raw(cfg)
print(df.shape)
df.head()"""),
    md("## Class balance -- the number every other metric has to be read against"),
    code("""fraud_rate = df[cfg.data.target_col].mean()
print(f"Fraud rate: {fraud_rate:.4%}")
print(f"Accuracy of an all-zero (never predict fraud) classifier: {1 - fraud_rate:.4%}")
print("-> accuracy is meaningless on this data; PR-AUC / TPR@5%FPR are used instead.")"""),
    md("## Is the split random or temporal? (decides whether `month` is a feature)\n\n"
       "VERIFIED (see README.md): this kit uses a stratified RANDOM 70/15/15 split, "
       "not the NeurIPS paper's temporal protocol, so `month` is kept as an ordinary "
       "feature. The check below is the same one that led to that decision."),
    code("""print(df["month"].value_counts().sort_index())
print()
print(df.groupby("month")[cfg.data.target_col].agg(["mean", "size"]))"""),
    md("## Constant columns"),
    code("""nun = df.nunique().sort_values()
print(nun.head(10))
print()
print("device_fraud_count unique values:", df['device_fraud_count'].unique())"""),
    md("## Sentinel (-1 = missing) columns vs. legitimate-negative columns\n\n"
       "Six columns use `-1` as a missing sentinel. Two other columns "
       "(`credit_risk_score`, `velocity_6h`) have real negative values and must "
       "NOT be treated as missing -- see 01-DATASET-BIBLE.md in the sibling kit."),
    code("""for c in cfg.sentinel_cols:
    print(f"{c:35s} frac negative: {(df[c] < 0).mean():.4f}")
print()
for c in cfg.legitimate_negative_cols:
    print(f"{c:35s} frac negative (LEGITIMATE, not missing): {(df[c] < 0).mean():.4f}")"""),
    md("## Categorical columns"),
    code("""for c in cfg.categorical_cols:
    print(c, df[c].nunique(), df[c].unique())"""),
    md("## Protected attribute for fairness: `customer_age > 50`\n\n"
       "Ages are rounded to the decade (9 distinct values), so the BAF paper's "
       "strictly-greater-than-50 cut matters: `>= 50` would silently move an "
       "entire bucket of applicants across the line."),
    code("""print(df["customer_age"].value_counts().sort_index())
older = df["customer_age"] > cfg.protected_attribute.threshold
print()
print("Fraud rate, age > 50:", df.loc[older, cfg.data.target_col].mean())
print("Fraud rate, age <= 50:", df.loc[~older, cfg.data.target_col].mean())"""),
    md("## No identifier and no leakage columns\n\n"
       "There is no `account_id`/`customer_id` in the raw file (no identifier-"
       "exclusion step was needed) and no post-decision outcome columns "
       "(chargeback status, investigation result, etc.) -- every feature is "
       "available at application time."),
    code("""from src.data_validation import check_no_leakage
import re
id_pattern = re.compile(r"(^id$|_id$|^id_|account_id|customer_id)", re.IGNORECASE)
print("identifier-like columns:", [c for c in df.columns if id_pattern.search(c)])
print("leakage-like columns:", check_no_leakage(df))"""),
    md("""## What was explicitly skipped, and why

This dataset has no transaction amount, no transaction timestamp (only a
coarse 0-7 `month` index), and no transaction history, so the following
spec items do not apply and were skipped rather than faked:

- `amount_log`, `transactions_per_hour` -- no transaction amount column exists.
- `hour`, `day_of_week`, `is_weekend` -- no real timestamp, only `month` (0-7).
- `current_amount / historical_average` deviation features -- no transaction
  history to compute a historical average from.

See `src/feature_engineering.py::SKIPPED_FEATURES` for the same list kept
next to the code."""),
])

# ============================================================================
# 02_feature_engineering.ipynb
# ============================================================================
nb2 = nb([
    md("""# 02 -- Feature Engineering

Sentinel handling (-1 -> NaN + `_is_missing` flag) and the engineered
features, each tied to one of three account-opening fraud archetypes:
(A) synthetic identity, (B) identity theft, (C) mule farming.
See `src/feature_engineering.py` and `src/preprocessing.py` for the
production code this notebook demonstrates."""),
    code(ROOT_SETUP),
    code("""from src.data_loader import load_raw
from src.preprocessing import to_nan_and_flag, drop_constant_columns
from src.feature_engineering import add_features, SKIPPED_FEATURES

df = load_raw(cfg)
const = drop_constant_columns(df, cfg.data.target_col, keep=[cfg.data.month_col])
df = df.drop(columns=const)
print("dropped constant columns:", const)"""),
    md("## Step 1 -- sentinel -1 -> NaN + `_is_missing` flag\n\n"
       "Missingness is itself predictive here: a synthetic identity has no "
       "previous address precisely because it was invented last week. This "
       "step is also a correctness prerequisite -- without it, -1 silently "
       "corrupts every ratio built from these columns below."),
    code("""before = df[cfg.sentinel_cols].describe()
df2 = to_nan_and_flag(df, cfg.sentinel_cols)
after = df2[cfg.sentinel_cols].describe()
print("BEFORE (raw, sentinel -1 mixed in):")
display(before)
print("\\nAFTER (sentinel converted to NaN):")
display(after)
[c for c in df2.columns if c.endswith("_is_missing")]"""),
    md("## Step 2 -- engineered features"),
    code("""df3 = add_features(df2)
new_cols = [c for c in df3.columns if c not in df2.columns]
print(f"{len(new_cols)} engineered features added:")
for c in new_cols:
    print(" -", c)"""),
    code("""df3[new_cols].describe().T"""),
    md("## Correlation of engineered features with the target (quick sanity check)"),
    code("""corrs = df3[new_cols + [cfg.data.target_col]].corr(numeric_only=True)[cfg.data.target_col].drop(cfg.data.target_col)
corrs.sort_values(key=abs, ascending=False)"""),
    md("## What was explicitly skipped, and why\n\n"
       "No transaction amount / timestamp / history exists in this dataset, "
       "so `amount_log`, `transactions_per_hour`, `hour`/`day_of_week`/`is_weekend`, "
       "and the `current_amount / historical_average` deviation feature were "
       "skipped rather than faked with substitute columns."),
    code("""for name, reason in SKIPPED_FEATURES.items():
    print(f"{name:40s} SKIPPED -- {reason}")"""),
])

# ============================================================================
# 03_model_training.ipynb
# ============================================================================
nb3 = nb([
    md("""# 03 -- Model Training

This notebook demonstrates the training methodology end to end: preprocessing
fit on TRAIN only, one imbalance strategy applied, and one model of each
family fit, using a fast subsample so the notebook runs in seconds.

**The authoritative, full-scale (700k-row train fold) result for every
model x imbalance-strategy combination is produced by `train.py` and saved
to `reports/metrics/model_comparison.csv` -- loaded and displayed at the
bottom of this notebook. Nothing in this notebook overrides those numbers;
this notebook exists to show the mechanism, not to re-derive the headline
results (that would just re-run train.py inside a notebook)."""),
    code(ROOT_SETUP),
    code("""from src.data_loader import load_and_split
from src.preprocessing import Preprocessor
from src import models, imbalance, evaluation

train_df, val_df, test_df = load_and_split(cfg)

# Demonstration subsample -- fast, illustrative only. The real comparison
# table below comes from the full 700k-row train fold via train.py.
demo_train = train_df.sample(n=60000, random_state=cfg.seed)
demo_val = val_df.sample(n=15000, random_state=cfg.seed)

pre = Preprocessor(cfg)
pre.fit(demo_train)
X_tr_tree, y_tr = pre.transform_tree(demo_train), pre.get_target(demo_train)
X_va_tree, y_va = pre.transform_tree(demo_val), pre.get_target(demo_val)
print(X_tr_tree.shape, X_va_tree.shape, y_tr.mean(), y_va.mean())"""),
    md("## Train one LightGBM model with the `class_weight` (scale_pos_weight) strategy"),
    code("""X_res, y_res, kwargs = imbalance.apply_strategy(X_tr_tree, y_tr, "class_weight", cfg, cfg.seed)
model = models.train_lightgbm(X_res, y_res, X_va_tree, y_va, cfg, kwargs, cfg.seed)
p_va = models.predict_proba(model, X_va_tree, "lightgbm")
evaluation.evaluate_scores(y_va, p_va, cfg.evaluation.target_fpr_for_tpr, label="demo LightGBM / class_weight (60k-row subsample)")
print("\\nNOTE: this is a small-subsample demo run for illustration; see the full-scale table below for the real reported numbers.")"""),
    md("## The full-scale ablation result (all 4 models x 5 imbalance strategies, real 700k-row train fold)\n\n"
       "Produced by running `python train.py` from the project root."),
    code("""import pandas as pd
comparison_path = ROOT / "reports" / "metrics" / "model_comparison.csv"
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path, index_col=0)
    display(comparison[["roc_auc", "pr_auc", "tpr_at_5pct_fpr", "train_rows", "train_seconds"]].round(4))
else:
    print("Run `python train.py` from the project root first to produce this table.")"""),
])

# ============================================================================
# 04_model_evaluation.ipynb
# ============================================================================
nb4 = nb([
    md("""# 04 -- Model Evaluation

Final, one-time evaluation of the SELECTED model on the untouched TEST split
(never used for fitting the preprocessor, choosing an imbalance strategy, or
tuning the threshold). Reproduces the same numbers `evaluate.py` writes to
`reports/metrics/test_evaluation.json` and `reports/figures/`."""),
    code(ROOT_SETUP),
    code("""from src.data_loader import load_and_split
from src.prediction import load_artifacts
from src import models, evaluation, explainability

model, preprocessor, meta = load_artifacts(cfg)
print("Selected model:", meta["model_type"], "/", meta["strategy"])
print("Threshold:", meta["threshold"], "  source:", meta["threshold_source"])

_, _, test_df = load_and_split(cfg)
y_test = preprocessor.get_target(test_df)

model_type = meta["model_type"]
strategy = meta["strategy"]
use_dense = (model_type in ("logistic_regression", "random_forest")) or (strategy in ("smote", "smote_undersample"))
X_test = preprocessor.transform_dense(test_df) if use_dense else preprocessor.transform_tree(test_df)
X_test = X_test[meta["feature_columns"]]

p_test = models.predict_proba(model, X_test, model_type)
metrics = evaluation.evaluate_scores(y_test, p_test, cfg.evaluation.target_fpr_for_tpr, label="FINAL TEST EVALUATION")"""),
    md("## Confusion matrix and cost-sensitive view at the selected threshold"),
    code("""threshold = float(meta["threshold"])
confusion = evaluation.confusion_at_threshold(y_test, p_test, threshold)
cost = evaluation.cost_sensitive_eval(y_test, p_test, threshold, cfg.evaluation.cost_fp, cfg.evaluation.cost_fn)
print(confusion)
print(cost)"""),
    md("## Threshold sweep -- finer resolution near the low end (this is a ~1% base rate)"),
    code("""from src.threshold_optimization import sweep_thresholds, sweep_cost_sensitive
f1_sweep = sweep_thresholds(y_test, p_test, cfg.threshold_optimization.thresholds)
display(f1_sweep)"""),
    md("## Fairness -- predictive equality, `customer_age > 50` (the BAF paper's protected group)"),
    code("""older = (test_df[cfg.protected_attribute.column] > cfg.protected_attribute.threshold).to_numpy()
fairness = evaluation.fairness_report(y_test, p_test, older, cfg.evaluation.target_fpr_for_tpr, label="customer_age > 50 (test)")"""),
    md("## Feature importance (gain-based) -- only meaningful for the tree models"),
    code("""if model_type == "lightgbm":
    imp = explainability.lightgbm_feature_importance(model, top_n=20)
elif model_type == "xgboost":
    imp = explainability.xgboost_feature_importance(model, top_n=20)
else:
    imp = None
    print(f"Selected model is {model_type} -- gain-based importance only applies to tree models.")
if imp is not None:
    display(imp)
    plt.figure(figsize=(8, 8))
    plt.barh(imp["feature"][::-1], imp["gain"][::-1])
    plt.xlabel("Gain")
    plt.title(f"Top 20 features by gain ({model_type})")
    plt.tight_layout()
    plt.show()"""),
    md("## SHAP summary + one real individual fraud-flagged prediction"),
    code("""import shap
if model_type in ("lightgbm", "xgboost"):
    rng = np.random.default_rng(cfg.seed)
    sample_idx = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    explainer, shap_values = explainability.shap_summary(model, X_sample, model_type)
    shap.summary_plot(shap_values, X_sample, max_display=20)
else:
    print(f"Selected model is {model_type} -- shap.TreeExplainer only applies to tree models.")"""),
    code("""if model_type in ("lightgbm", "xgboost"):
    flagged = np.flatnonzero((p_test[sample_idx] >= threshold) & (y_test.iloc[sample_idx].to_numpy() == 1))
    if len(flagged) > 0:
        pick = sample_idx[flagged[0]]
        row = X_test.iloc[[pick]]
        row_shap = explainability.explain_single_prediction(explainer, row)
        print(f"Real TEST-set fraud-flagged row, index={pick}, predicted prob={p_test[pick]:.4f}")
        shap.plots.waterfall(row_shap[0], max_display=15)
    else:
        print("No true-positive fraud rows landed in this sample; re-run with a larger sample size.")"""),
])

for name, notebook in [
    ("01_data_exploration.ipynb", nb1),
    ("02_feature_engineering.ipynb", nb2),
    ("03_model_training.ipynb", nb3),
    ("04_model_evaluation.ipynb", nb4),
]:
    with open(name, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print("wrote", name)
