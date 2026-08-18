"""Shared paths and knobs for the research_v2 pipeline (teammate feature set).

Mirrors src_research/config_research.py's structure/API exactly, pointed at
the v2 (teammate-feature) artifact/report tree instead of the in-house one.
The original src_research/, research/, artifacts_research/ trees are not
touched by anything in this file or imported by it.
"""
import os

SRC_RESEARCH_V2_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_RESEARCH_V2_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_CSV = os.path.join(DATA_DIR, "bank_transactions_data_2.csv")

# canonical input for this pipeline: the teammate's 18 engineered features,
# with ID/display columns reattached (already verified aligned, per task brief)
ARTIFACTS_RESEARCH_DIR = os.path.join(ROOT_DIR, "artifacts_research")
FEATURES_TEAMMATE_MERGED_CSV = os.path.join(ARTIFACTS_RESEARCH_DIR, "features_teammate_merged.csv")

RESEARCH_V2_DIR = os.path.join(ROOT_DIR, "research_v2")
PLOTS_V2_DIR = os.path.join(RESEARCH_V2_DIR, "plots")

ARTIFACTS_V2_DIR = os.path.join(ROOT_DIR, "artifacts_research_v2")
MODELS_V2_DIR = os.path.join(ARTIFACTS_V2_DIR, "models")

RANDOM_STATE = 42

ID_COLS = [
    "TransactionID", "AccountID", "TransactionDate", "TransactionType",
    "Location", "DeviceID", "IP Address", "MerchantID", "Channel",
    "CustomerOccupation",
]

# the teammate's 18 engineered feature columns (everything in
# features_teammate_merged.csv that is not an ID/display column)
FEATURE_COLS_V2 = [
    "TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts",
    "AccountBalance", "account_frequency", "device_frequency", "ip_frequency",
    "merchant_frequency", "amount_to_balance_ratio", "high_amount_transaction",
    "TransactionType_Debit", "Channel_Branch", "Channel_Online",
    "CustomerOccupation_Engineer", "CustomerOccupation_Retired",
    "CustomerOccupation_Student", "Location_FE",
]

for d in (RESEARCH_V2_DIR, PLOTS_V2_DIR, ARTIFACTS_V2_DIR, MODELS_V2_DIR):
    os.makedirs(d, exist_ok=True)


def load_features_v2():
    import pandas as pd
    df = pd.read_csv(FEATURES_TEAMMATE_MERGED_CSV)
    assert list(df.columns) == ID_COLS + FEATURE_COLS_V2, "Unexpected column layout in features_teammate_merged.csv"
    return df
