"""
Shared feature-engineering logic for the fraud-detection pipeline.

Used by 01_feature_engineering.py (batch, at training time) and the Argus
dashboard's "Upload & Predict" page (single/batch transactions, at inference
time, via dashboard/backend/api_server.py) so the two never drift apart.

LEAKAGE-SAFE DESIGN (see ML_AUDIT_AFTER_FIX.md for the full writeup):

  add_causal_features(df)
      Strictly causal, per-account behavioral features (prior-average amount,
      device/location novelty, time-since-last-txn). Each value only looks at
      STRICTLY EARLIER rows of the SAME account in chronological order, so it
      is safe to compute before any train/val/test split exists -- moving the
      split boundary can never change these values.

  fit_global_stats(df)
      Fits every cross-transaction lookup/aggregate (transaction-type means,
      device/IP/merchant popularity, the median history-gap used to fill a
      first transaction's "time since last") using ONLY the rows passed in.
      Call this with the TRAINING fold only.

  apply_global_stats(df, stats)
      Maps a fitted `stats` dict onto ANY dataframe (train, val, test, or a
      single new transaction). Categories never seen during fitting fall back
      to documented defaults rather than peeking at this df's own values.

  finalize_matrix(df, encoders=None)
      Drops identifier/date columns and one-hot/label-encodes the remaining
      categoricals. Pass encoders=None to FIT (training fold only); pass the
      fitted dict back in to transform any other fold/row identically.

reference.pkl (used by the Argus dashboard's "Upload & Predict" page for
brand-new transactions) is built from the FULL dataset via
fit_global_stats/build_account_history -- this is intentional and does not
leak into evaluation: by the time a genuinely new transaction arrives live,
all 2,512 historical rows really are past data. It is a separate artifact
from the train-only stats used to build the leakage-free train/val/test
feature matrices.
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


def sort_chronological(df):
    return df.sort_values(["AccountID", "TransactionDate", "TransactionID"]).reset_index(drop=True)


def add_causal_features(df):
    """Per-account features that only ever look backward -- safe pre-split."""
    df = df.copy()
    df["_prior_avg_amount"] = df.groupby("AccountID")["TransactionAmount"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["DeviceNoveltyFlag"] = df.groupby("AccountID")["DeviceID"].transform(
        lambda s: (~s.duplicated()).astype(int)
    )
    df["LocationNoveltyFlag"] = df.groupby("AccountID")["Location"].transform(
        lambda s: (~s.duplicated()).astype(int)
    )
    df["_gap_hours"] = (
        df.groupby("AccountID")["TransactionDate"].diff().dt.total_seconds() / 3600.0
    )
    return df


def fit_global_stats(df):
    """Fit cross-transaction lookups using ONLY the given (training) rows."""
    type_avg = df.groupby("TransactionType")["TransactionAmount"].mean().to_dict()
    device_counts = df["DeviceID"].value_counts().to_dict()
    ip_counts = df["IP Address"].value_counts().to_dict()
    merchant_counts = df["MerchantID"].value_counts().to_dict()
    median_gap = df["_gap_hours"].median()
    if pd.isna(median_gap):
        median_gap = 0.0
    global_amount_mean = df["TransactionAmount"].mean()
    return {
        "type_avg": type_avg,
        "device_counts": device_counts,
        "ip_counts": ip_counts,
        "merchant_counts": merchant_counts,
        "median_gap_hours": float(median_gap),
        "global_amount_mean": float(global_amount_mean),
    }


def apply_global_stats(df, stats):
    """Map a fitted stats dict onto any dataframe (train, val, test, or a new row)."""
    df = df.copy()
    type_avg_default = stats["global_amount_mean"]
    type_avg_map = df["TransactionType"].map(stats["type_avg"]).fillna(type_avg_default)
    df["Amount_vs_TypeAvg"] = (df["TransactionAmount"] - type_avg_map) / (type_avg_map + EPS)
    df["DeviceTxnCount"] = df["DeviceID"].map(stats["device_counts"]).fillna(0).astype(int)
    df["IPTxnCount"] = df["IP Address"].map(stats["ip_counts"]).fillna(0).astype(int)
    df["MerchantTxnCount"] = df["MerchantID"].map(stats["merchant_counts"]).fillna(0).astype(int)

    prior_avg_filled = df["_prior_avg_amount"].fillna(type_avg_map)
    df["Amount_vs_AccountAvg"] = (df["TransactionAmount"] - prior_avg_filled) / (prior_avg_filled + EPS)
    df["TimeSinceLastTxn"] = df["_gap_hours"].fillna(stats["median_gap_hours"])
    return df


def _encode_categoricals(df, encoders=None):
    """
    One-hot encode low-cardinality categoricals, label-encode high-cardinality
    ones. `encoders` is None at fit time (encoders are learned and returned);
    pass the fitted dict back in at transform time to encode any other
    dataframe (another split, or a single new row) the same way.
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


def finalize_matrix(df, encoders=None):
    """
    Drop raw identifier/date columns not used as model inputs (but KEEP
    TransactionID -- it isn't a model feature, but downstream stages need a
    stable key to re-join a labeled row back to its original raw record,
    since the train/val/test concatenation order no longer matches the
    single global chronological sort). Encode remaining categoricals (fit if
    encoders=None).
    """
    drop_cols = [c for c in RAW_ID_COLS + ["_prior_avg_amount", "_gap_hours",
                                            "TransactionDate", "AccountID"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)
    df, fitted_encoders = _encode_categoricals(df, encoders=encoders)
    return df, fitted_encoders


def build_account_history(df):
    """
    Full per-account snapshot (devices/locations seen, last txn time, running
    mean amount) as of the END of the given dataframe. Used only to engineer
    brand-new, genuinely-future transactions in the demo app -- never to score
    a transaction that is already inside this dataframe.
    """
    account_history = {}
    for acc, g in df.groupby("AccountID"):
        account_history[acc] = {
            "devices": set(g["DeviceID"]),
            "locations": set(g["Location"]),
            "last_time": g["TransactionDate"].max(),
            "running_mean_amount": g["TransactionAmount"].mean(),
            "n": len(g),
        }
    return account_history


def _engineer_new_row(txn, hist, stats):
    """
    Shared per-transaction feature logic for both transform_new (single row)
    and transform_batch_new (many rows, CSV upload). `hist`
    is either None (no known prior history for this account) or an
    account_history-shaped dict (devices/locations/last_time/running_mean_amount).
    """
    type_avg = stats["type_avg"].get(txn["TransactionType"], stats["global_amount_mean"])
    amt = txn["TransactionAmount"]
    row = {
        "TransactionAmount": amt,
        "CustomerAge": txn["CustomerAge"],
        "TransactionDuration": txn["TransactionDuration"],
        "LoginAttempts": txn["LoginAttempts"],
        "AccountBalance": txn["AccountBalance"],
        "Amount_vs_TypeAvg": (amt - type_avg) / (type_avg + EPS),
        "DeviceTxnCount": stats["device_counts"].get(txn["DeviceID"], 0) + 1,
        "IPTxnCount": stats["ip_counts"].get(txn["IP Address"], 0) + 1,
        "MerchantTxnCount": stats["merchant_counts"].get(txn["MerchantID"], 0) + 1,
        "TransactionType": txn["TransactionType"],
        "Location": txn["Location"],
        "Channel": txn["Channel"],
        "CustomerOccupation": txn["CustomerOccupation"],
    }

    if hist is None:
        # brand-new account: no prior history to compare against
        row["Amount_vs_AccountAvg"] = row["Amount_vs_TypeAvg"]
        row["DeviceNoveltyFlag"] = 1
        row["LocationNoveltyFlag"] = 1
        row["TimeSinceLastTxn"] = stats["median_gap_hours"]
    else:
        prior_avg = hist["running_mean_amount"]
        row["Amount_vs_AccountAvg"] = (amt - prior_avg) / (prior_avg + EPS)
        row["DeviceNoveltyFlag"] = int(txn["DeviceID"] not in hist["devices"])
        row["LocationNoveltyFlag"] = int(txn["Location"] not in hist["locations"])
        gap = (txn["TransactionDate"] - hist["last_time"]).total_seconds() / 3600.0
        row["TimeSinceLastTxn"] = max(gap, 0.0)

    return row


def transform_new(txn, reference):
    """
    Engineer features for ONE new transaction dict (as collected by a live
    scoring caller) using stats captured in `reference` (fit on the FULL
    historical dataset -- see module docstring). txn keys mirror the raw CSV
    columns (TransactionAmount, TransactionType, Location, DeviceID,
    'IP Address', MerchantID, Channel, CustomerAge, CustomerOccupation,
    TransactionDuration, LoginAttempts, AccountBalance, AccountID, TransactionDate).
    """
    stats = reference["stats"]
    hist = reference["account_history"].get(txn["AccountID"])
    row = _engineer_new_row(txn, hist, stats)

    single_df = pd.DataFrame([row])
    single_df, _ = _encode_categoricals(single_df, encoders=reference["encoders"])
    single_df = single_df.reindex(columns=reference["feature_cols"], fill_value=0)
    return single_df


def transform_batch_new(df, reference):
    """
    Engineer features for a CSV upload of possibly-many new transactions
    (the dashboard's "Upload & Detect" feature). Same leakage-safe contract
    as transform_new: every global lookup comes from `reference["stats"]`
    (fit at training time, never from this upload), and every per-account
    behavioral feature only ever looks at STRICTLY EARLIER rows of the same
    account -- prior rows already in `reference["account_history"]`, plus
    any earlier rows of that same account already seen earlier in this same
    upload (processed in chronological order), never a later row and never
    the row's own value.

    df columns mirror the raw CSV schema: TransactionID, AccountID,
    TransactionAmount, TransactionDate, TransactionType, Location, DeviceID,
    'IP Address', MerchantID, Channel, CustomerAge, CustomerOccupation,
    TransactionDuration, LoginAttempts, AccountBalance. Returns a feature
    dataframe with the same row order as the (chronologically re-sorted)
    input, plus the sort order actually used so callers can re-align results
    back to the original upload order if needed.
    """
    stats = reference["stats"]
    df = df.sort_values(["AccountID", "TransactionDate"], kind="stable").reset_index()
    orig_index = df["index"]  # position in the caller's original dataframe
    df = df.drop(columns=["index"])

    running_history = {}  # this upload's own running state, seeded lazily from reference
    rows = []
    for _, txn in df.iterrows():
        acc = txn["AccountID"]
        hist = running_history.get(acc, reference["account_history"].get(acc))
        rows.append(_engineer_new_row(txn, hist, stats))

        amt = txn["TransactionAmount"]
        prev_n = hist["n"] if hist else 0
        prev_mean = hist["running_mean_amount"] if hist else amt
        new_n = prev_n + 1
        new_mean = prev_mean + (amt - prev_mean) / new_n
        devices = set(hist["devices"]) if hist else set()
        devices.add(txn["DeviceID"])
        locations = set(hist["locations"]) if hist else set()
        locations.add(txn["Location"])
        running_history[acc] = {
            "devices": devices,
            "locations": locations,
            "last_time": txn["TransactionDate"],
            "running_mean_amount": new_mean,
            "n": new_n,
        }

    feat_df = pd.DataFrame(rows)
    feat_df, _ = _encode_categoricals(feat_df, encoders=reference["encoders"])
    feat_df = feat_df.reindex(columns=reference["feature_cols"], fill_value=0)
    feat_df.index = orig_index.values
    return feat_df.sort_index()
