"""Shared paths and knobs for the whole pipeline. Edit here, not per-script."""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

RAW_CSV = os.path.join(DATA_DIR, "bank_transactions_data_2.csv")

FEATURES_CSV = os.path.join(ARTIFACTS_DIR, "features.csv")
FEATURES_SCALED_CSV = os.path.join(ARTIFACTS_DIR, "features_scaled.csv")
REFERENCE_PKL = os.path.join(ARTIFACTS_DIR, "reference.pkl")
SCALER_PKL = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
ANOMALY_VOTES_CSV = os.path.join(ARTIFACTS_DIR, "anomaly_votes.csv")

LABELED_CSV = os.path.join(ARTIFACTS_DIR, "labeled.csv")
SPLIT_PKL = os.path.join(ARTIFACTS_DIR, "split.pkl")

MODEL_JSON = os.path.join(ARTIFACTS_DIR, "xgb_model.json")
MODEL_NOSMOTE_JSON = os.path.join(ARTIFACTS_DIR, "xgb_model_classweight.json")
BEST_MODEL_JSON = os.path.join(ARTIFACTS_DIR, "xgb_model_best.json")
RF_MODEL_PKL = os.path.join(ARTIFACTS_DIR, "random_forest_model.pkl")
BEST_MODEL_CHOICE_JSON = os.path.join(ARTIFACTS_DIR, "best_model_choice.json")
DT_RULES_TXT = os.path.join(ARTIFACTS_DIR, "decision_tree_rules.txt")
DT_MODEL_PKL = os.path.join(ARTIFACTS_DIR, "decision_tree.pkl")
THRESHOLDS_JSON = os.path.join(ARTIFACTS_DIR, "thresholds.json")
MODEL_COMPARISON_CSV = os.path.join(ARTIFACTS_DIR, "model_comparison.csv")
MODEL_COMPARISON_JSON = os.path.join(ARTIFACTS_DIR, "model_comparison.json")

CV_PER_FOLD_CSV = os.path.join(ARTIFACTS_DIR, "cv_per_fold.csv")
CV_SUMMARY_CSV = os.path.join(ARTIFACTS_DIR, "cv_summary.csv")
CV_SUMMARY_JSON = os.path.join(ARTIFACTS_DIR, "cv_summary.json")
CV_N_SPLITS = 5

RANDOM_STATE = 42

# Chronological split boundaries, expressed as quantiles of TransactionDate
# over the full raw dataset. TRAIN = earliest ~64%, VAL = next ~16% (used
# only for model comparison / cost-threshold selection, never for the final
# reported test metrics), TEST = latest ~20% (touched exactly once, at the
# very end, for the numbers reported as the unbiased estimate).
TRAIN_QUANTILE = 0.64
VAL_QUANTILE = 0.80

# Assumed fraud prevalence used to set contamination for the unsupervised
# detectors. No ground truth exists, so this is a documented assumption
# (typical card-not-present fraud rates sit well under 5%); it is applied
# identically to all four detectors so their votes stay comparable.
CONTAMINATION = 0.05

# Illustrative cost-based framing for Section 6 (not real bank figures):
# a false positive means one legitimate customer gets friction/blocked;
# a false negative means a fraudulent transaction goes through uncaught.
COST_FALSE_POSITIVE = 5.0
COST_FALSE_NEGATIVE = 250.0

for d in (DATA_DIR, ARTIFACTS_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)
