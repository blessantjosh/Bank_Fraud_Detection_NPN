"""
Shared feature-engineering logic for the fraud-detection pipeline.

Used by both 01_feature_engineering.py (batch, at training time) and
app_streamlit.py (single transaction, at inference time) so the two
never drift apart. All account-history / global lookup stats needed to
score a brand-new transaction are captured in the `reference` dict
returned by fit_engineer() and persisted with joblib.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

EPS = 1e-6
CATEGORICAL_COLS = ["TransactionType", "Location", "Channel", "CustomerOccupation"]
ONEHOT_MAX_CARDINALITY = 10
RAW_ID_COLS = ["DeviceID", "IP Address", "MerchantID"]  # engineered into counts, then dropped
NUMERIC_BASE_COLS = ["TransactionAmount", "CustomerAge", "TransactionDuration",
                      "LoginAttempts", "AccountBalance"]


def load_raw(path):
    """Read the raw CSV, parse dates, drop the broken PreviousTransactionDate column."""
    df = pd.read_csv(path)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="%d-%m-%Y %H:%M")
    if "PreviousTransactionDate" in df.columns:
        # Only 7 unique values, all within minutes of one export timestamp on
        # 2024-11-04 -> this is a data-export artifact, not a real per-account
        # "previous transaction" time. Real recency is derived below from
        # TransactionDate itself (TimeSinceLastTxn).
        df = df.drop(columns=["PreviousTransactionDate"])
    return df


def _encode_categoricals(df, encoders=None):
    """
    One-hot encode low-cardinality categoricals, label-encode high-cardinality
    ones. `encoders` is None at fit time (encoders are learned and returned);
    pass the fitted dict back in at inference time to encode a single new row
    the same way training did.
    """
    df = df.copy()
    fitted = {} if encoders is None else encoders
    onehot_frames = []

    for col in CATEGORICAL_COLS:
        if encoders is None:
            nunique = df[col].nunique()
            if nunique <= ONEHOT_MAX_CARDINALITY:
                fitted[col] = ("onehot", sorted(df[col].astype(str).unique().tolist()))
            else:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                fitted[col] = ("label", le)
                continue
        kind, spec = fitted[col]
        if kind == "onehot":
            # drop the first category as the reference level: keeping all K
            # dummies makes them sum to 1 (perfect multicollinearity), which
            # singularizes the covariance matrix the MCD detector needs to invert
            kept_cats = spec[1:]
            dummies = pd.get_dummies(df[col].astype(str), prefix=col)
            for cat in kept_cats:
                dummy_col = f"{col}_{cat}"
                if dummy_col not in dummies.columns:
                    dummies[dummy_col] = 0
            dummies = dummies[[f"{col}_{cat}" for cat in kept_cats]]
            onehot_frames.append(dummies)
        else:
            le = spec
            known = set(le.classes_)
            safe_vals = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            df[col + "_enc"] = le.transform(safe_vals)

    df = df.drop(columns=CATEGORICAL_COLS)
    if onehot_frames:
        df = pd.concat([df] + onehot_frames, axis=1)
    return df, fitted


def fit_engineer(df):
    """
    Batch feature engineering at training time. Returns (engineered_df, reference).
    reference holds every lookup table needed to engineer a single brand-new
    transaction later (see transform_new below).
    """
    df = df.sort_values(["AccountID", "TransactionDate", "TransactionID"]).reset_index(drop=True)

    # ---- global lookups (used both as features and stored for inference) ----
    type_avg = df.groupby("TransactionType")["TransactionAmount"].mean().to_dict()
    device_counts = df["DeviceID"].value_counts().to_dict()
    ip_counts = df["IP Address"].value_counts().to_dict()
    merchant_counts = df["MerchantID"].value_counts().to_dict()

    df["_type_avg"] = df["TransactionType"].map(type_avg)
    df["Amount_vs_TypeAvg"] = (df["TransactionAmount"] - df["_type_avg"]) / (df["_type_avg"] + EPS)
    df["DeviceTxnCount"] = df["DeviceID"].map(device_counts)
    df["IPTxnCount"] = df["IP Address"].map(ip_counts)
    df["MerchantTxnCount"] = df["MerchantID"].map(merchant_counts)

    # ---- per-account chronological (leakage-safe: only prior rows) ----
    prior_avg = df.groupby("AccountID")["TransactionAmount"].transform(
        lambda s: s.shift().expanding().mean()
    )
    prior_avg_filled = prior_avg.fillna(df["_type_avg"])
    df["Amount_vs_AccountAvg"] = (df["TransactionAmount"] - prior_avg_filled) / (prior_avg_filled + EPS)

    df["DeviceNoveltyFlag"] = df.groupby("AccountID")["DeviceID"].transform(
        lambda s: (~s.duplicated()).astype(int)
    )
    df["LocationNoveltyFlag"] = df.groupby("AccountID")["Location"].transform(
        lambda s: (~s.duplicated()).astype(int)
    )

    gap_hours = df.groupby("AccountID")["TransactionDate"].diff().dt.total_seconds() / 3600.0
    median_gap = gap_hours.median()
    df["TimeSinceLastTxn"] = gap_hours.fillna(median_gap)

    # ---- per-account state needed to score a brand-new transaction later ----
    account_history = {}
    for acc, g in df.groupby("AccountID"):
        account_history[acc] = {
            "devices": set(g["DeviceID"]),
            "locations": set(g["Location"]),
            "last_time": g["TransactionDate"].max(),
            "running_mean_amount": g["TransactionAmount"].mean(),
            "n": len(g),
        }

    df = df.drop(columns=RAW_ID_COLS + ["_type_avg", "TransactionID", "TransactionDate", "AccountID"])
    df, encoders = _encode_categoricals(df, encoders=None)

    feature_cols = [c for c in df.columns]
    reference = {
        "type_avg": type_avg,
        "device_counts": device_counts,
        "ip_counts": ip_counts,
        "merchant_counts": merchant_counts,
        "account_history": account_history,
        "median_gap_hours": median_gap,
        "encoders": encoders,
        "feature_cols": feature_cols,
    }
    return df, reference


def transform_new(txn, reference):
    """
    Engineer features for ONE new transaction dict (as collected from the
    Streamlit form) using stats captured at training time. txn keys mirror the
    raw CSV columns (TransactionAmount, TransactionType, Location, DeviceID,
    'IP Address', MerchantID, Channel, CustomerAge, CustomerOccupation,
    TransactionDuration, LoginAttempts, AccountBalance, AccountID, TransactionDate).
    """
    type_avg = reference["type_avg"].get(txn["TransactionType"],
                                          np.mean(list(reference["type_avg"].values())))
    amt = txn["TransactionAmount"]
    row = {
        "TransactionAmount": amt,
        "CustomerAge": txn["CustomerAge"],
        "TransactionDuration": txn["TransactionDuration"],
        "LoginAttempts": txn["LoginAttempts"],
        "AccountBalance": txn["AccountBalance"],
        "Amount_vs_TypeAvg": (amt - type_avg) / (type_avg + EPS),
        "DeviceTxnCount": reference["device_counts"].get(txn["DeviceID"], 0) + 1,
        "IPTxnCount": reference["ip_counts"].get(txn["IP Address"], 0) + 1,
        "MerchantTxnCount": reference["merchant_counts"].get(txn["MerchantID"], 0) + 1,
        "TransactionType": txn["TransactionType"],
        "Location": txn["Location"],
        "Channel": txn["Channel"],
        "CustomerOccupation": txn["CustomerOccupation"],
    }

    hist = reference["account_history"].get(txn["AccountID"])
    if hist is None:
        # brand-new account: no prior history to compare against
        row["Amount_vs_AccountAvg"] = row["Amount_vs_TypeAvg"]
        row["DeviceNoveltyFlag"] = 1
        row["LocationNoveltyFlag"] = 1
        row["TimeSinceLastTxn"] = reference["median_gap_hours"]
    else:
        prior_avg = hist["running_mean_amount"]
        row["Amount_vs_AccountAvg"] = (amt - prior_avg) / (prior_avg + EPS)
        row["DeviceNoveltyFlag"] = int(txn["DeviceID"] not in hist["devices"])
        row["LocationNoveltyFlag"] = int(txn["Location"] not in hist["locations"])
        gap = (txn["TransactionDate"] - hist["last_time"]).total_seconds() / 3600.0
        row["TimeSinceLastTxn"] = max(gap, 0.0)

    single_df = pd.DataFrame([row])
    single_df, _ = _encode_categoricals(single_df, encoders=reference["encoders"])
    single_df = single_df.reindex(columns=reference["feature_cols"], fill_value=0)
    return single_df
