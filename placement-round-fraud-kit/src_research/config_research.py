"""Shared paths and knobs for the Phase 2-4 research scripts. Edit here, not per-script."""
import os

SRC_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_RESEARCH_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_CSV = os.path.join(DATA_DIR, "bank_transactions_data_2.csv")

RESEARCH_DIR = os.path.join(ROOT_DIR, "research")
PLOTS_DIR = os.path.join(RESEARCH_DIR, "plots")

ARTIFACTS_RESEARCH_DIR = os.path.join(ROOT_DIR, "artifacts_research")

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
]

CATEGORICAL_FEATURES = [
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerOccupation",
    "AccountID",
]

DATETIME_FEATURES = ["TransactionDate", "PreviousTransactionDate"]

for d in (DATA_DIR, RESEARCH_DIR, PLOTS_DIR, ARTIFACTS_RESEARCH_DIR):
    os.makedirs(d, exist_ok=True)


def load_raw():
    import pandas as pd

    df = pd.read_csv(RAW_CSV)
    df["TransactionDate_parsed"] = pd.to_datetime(df["TransactionDate"], format="%d-%m-%Y %H:%M")
    df["PreviousTransactionDate_parsed"] = pd.to_datetime(
        df["PreviousTransactionDate"], format="%d-%m-%Y %H:%M"
    )
    return df
