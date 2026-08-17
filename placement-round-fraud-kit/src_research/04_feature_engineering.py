"""
Phase 5 -- Feature Engineering.

Builds velocity, rolling-statistic, ratio, deviation, cyclical-time,
behavioral, and network-proxy (shared-infrastructure) features on top of
the raw dataset, on the same chronological per-account sort used by the
existing v1 pipeline (src/fe_utils.py::fit_engineer). Every feature that
looks at "this account's history" or "this device/IP's other users" is
computed so only STRICTLY PRIOR rows (by TransactionDate, tie-broken by
TransactionID) are visible -- never the current row and never a future one.

Output: artifacts_research/features_v2.csv -- the 20 v1 features (reused,
not recomputed, loaded straight from artifacts/features.csv and aligned
onto this script's identical sort order) plus every new Phase-5 feature
below, with TransactionID/AccountID kept as ID columns for traceability
(excluded from modeling -- see feature dictionary in research/04_feature_engineering.md).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import ARTIFACTS_RESEARCH_DIR, ROOT_DIR, load_raw

EPS = 1e-6
V1_FEATURES_CSV = os.path.join(ROOT_DIR, "artifacts", "features.csv")
V1_VOTES_CSV = os.path.join(ROOT_DIR, "artifacts", "anomaly_votes.csv")
OUT_CSV = os.path.join(ARTIFACTS_RESEARCH_DIR, "features_v2.csv")
DICT_JSON = os.path.join(ARTIFACTS_RESEARCH_DIR, "feature_dictionary.json")


def _sorted_raw():
    """Load raw data and sort exactly as src/fe_utils.py::fit_engineer does,
    so row order matches artifacts/features.csv 1:1 and the two can be
    concatenated by position."""
    df = load_raw()
    df = df.drop(columns=["PreviousTransactionDate", "PreviousTransactionDate_parsed"])
    df = df.rename(columns={"TransactionDate_parsed": "dt"})
    df = df.sort_values(["AccountID", "dt", "TransactionID"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------- velocity
def velocity_features(df):
    """Trailing N-day transaction counts per account, strictly excluding the
    current transaction (closed='left'). Small per-account transaction
    counts (avg 5.08/account over a 364-day span) mean these will mostly be
    zero -- reported honestly in the write-up, not smoothed over."""
    s = df.set_index("dt").groupby("AccountID")["TransactionAmount"]
    v1d = s.rolling("1D", closed="left").count().fillna(0.0)
    v7d = s.rolling("7D", closed="left").count().fillna(0.0)
    return v1d.reset_index(drop=True).rename("Velocity_1D_Count"), \
        v7d.reset_index(drop=True).rename("Velocity_7D_Count")


# ------------------------------------------------------------- rolling/expanding
def rolling_and_expanding_features(df):
    g = df.groupby("AccountID")["TransactionAmount"]
    shifted = g.shift()  # this account's history strictly before the current row

    exp_mean = shifted.groupby(df["AccountID"]).expanding().mean().reset_index(level=0, drop=True)
    exp_median = shifted.groupby(df["AccountID"]).expanding().median().reset_index(level=0, drop=True)
    exp_std = shifted.groupby(df["AccountID"]).expanding().std().reset_index(level=0, drop=True)
    exp_min = shifted.groupby(df["AccountID"]).expanding().min().reset_index(level=0, drop=True)
    exp_max = shifted.groupby(df["AccountID"]).expanding().max().reset_index(level=0, drop=True)

    roll3_mean = shifted.groupby(df["AccountID"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    roll3_std = shifted.groupby(df["AccountID"]).rolling(3, min_periods=2).std().reset_index(level=0, drop=True)

    out = pd.DataFrame({
        "Expanding_MeanAmount": exp_mean,
        "Expanding_MedianAmount": exp_median,
        "Expanding_StdAmount": exp_std,
        "Expanding_MinAmount": exp_min,
        "Expanding_MaxAmount": exp_max,
        "Rolling3_MeanAmount": roll3_mean,
        "Rolling3_StdAmount": roll3_std,
    })
    # First transaction per account has no prior history at all: no prior
    # mean/median/min/max exists yet. Fall back to the population mean/median
    # (the same "no history" convention already used in v1's Amount_vs_AccountAvg,
    # which falls back to the transaction-type average) rather than 0, so a
    # first transaction is not scored as a giant artificial deviation.
    pop_mean = df["TransactionAmount"].mean()
    pop_median = df["TransactionAmount"].median()
    out["Expanding_MeanAmount"] = out["Expanding_MeanAmount"].fillna(pop_mean)
    out["Expanding_MedianAmount"] = out["Expanding_MedianAmount"].fillna(pop_median)
    out["Expanding_MinAmount"] = out["Expanding_MinAmount"].fillna(df["TransactionAmount"].min())
    out["Expanding_MaxAmount"] = out["Expanding_MaxAmount"].fillna(df["TransactionAmount"].max())
    out["Rolling3_MeanAmount"] = out["Rolling3_MeanAmount"].fillna(pop_mean)
    # Std is undefined with 0 or 1 prior observations -- fill with 0 (no
    # observed variability yet), a documented, deliberate choice, not silent.
    out["Expanding_StdAmount"] = out["Expanding_StdAmount"].fillna(0.0)
    out["Rolling3_StdAmount"] = out["Rolling3_StdAmount"].fillna(0.0)
    return out.reset_index(drop=True)


# ------------------------------------------------------------- ratio / deviation
def ratio_and_deviation_features(df, roll_exp):
    amt = df["TransactionAmount"]
    bal = df["AccountBalance"]
    exp_mean = roll_exp["Expanding_MeanAmount"]
    exp_median = roll_exp["Expanding_MedianAmount"]
    exp_std = roll_exp["Expanding_StdAmount"]

    # Amount_ZScore_Account divides by this account's own prior std, which
    # can be genuinely near-zero for an account with only 2-3 prior
    # transactions that happened to be nearly identical in size -- dividing
    # by (std + 1e-6) in that case sends the z-score into the billions for a
    # perfectly ordinary next transaction (confirmed empirically: with a
    # 1e-6 epsilon, 966/2,512 rows exceeded |z|>1,000, max 1.5e9). A
    # population-scale floor is used instead: no account's assumed
    # variability is allowed to imply a tighter spread than 5% of the
    # dataset-wide TransactionAmount std ($291.95 -> floor $14.60). This is
    # a deliberate, documented choice, not a silent divide-by-near-zero.
    std_floor = 0.05 * df["TransactionAmount"].std()
    exp_std_floored = exp_std.clip(lower=std_floor)

    out = pd.DataFrame({
        "Amount_to_Balance_Ratio": amt / (bal + EPS),
        "Amount_to_RollingMean_Ratio": amt / (exp_mean + EPS),
        "Amount_minus_ExpandingMean": amt - exp_mean,
        "Amount_minus_ExpandingMedian": amt - exp_median,
        "Amount_ZScore_Account": (amt - exp_mean) / exp_std_floored,
    })
    return out


# ------------------------------------------------------------------ cyclical
def cyclical_time_features(df):
    hour = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
    dow = df["dt"].dt.dayofweek  # Monday=0
    out = pd.DataFrame({
        "Hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "Hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "DOW_sin": np.sin(2 * np.pi * dow / 7.0),
        "DOW_cos": np.cos(2 * np.pi * dow / 7.0),
    })
    return out


# ---------------------------------------------------------------- behavioral
def behavioral_features(df, roll_exp):
    cust_count_so_far = df.groupby("AccountID").cumcount()  # prior-transaction count, leakage-safe by construction
    spend_cv = roll_exp["Expanding_StdAmount"] / (roll_exp["Expanding_MeanAmount"] + EPS)
    elevated_login = (df["LoginAttempts"] >= 2).astype(int)
    atm_credit_interaction = ((df["Channel"] == "ATM") & (df["TransactionType"] == "Credit")).astype(int)
    out = pd.DataFrame({
        "CustomerTxnCountSoFar": cust_count_so_far.values,
        "SpendCV_Account": spend_cv.values,
        "ElevatedLoginFlag": elevated_login.values,
        "ATM_Credit_InteractionFlag": atm_credit_interaction.values,
    })
    return out


# ------------------------------------------------------------- network proxy
def _prior_distinct_other_accounts(df, key_col):
    """For each row, count DISTINCT accounts (other than the row's own
    AccountID) that used the same key_col value (DeviceID / 'IP Address' /
    MerchantID) at any STRICTLY EARLIER TransactionDate. Leakage-safe by
    construction: only rows earlier in time within the same key group are
    ever counted, and a row never counts its own account."""
    tmp = df[[key_col, "AccountID", "dt"]].copy()
    tmp = tmp.sort_values([key_col, "dt"]).reset_index()  # keep original index to reassign later
    out = pd.Series(index=tmp["index"], dtype=int)
    for _, grp in tmp.groupby(key_col, sort=False):
        seen = set()
        for pos, (orig_idx, acc) in enumerate(zip(grp["index"], grp["AccountID"])):
            out.loc[orig_idx] = len(seen - {acc})
            seen.add(acc)
    out = out.sort_index()
    return out.values


def network_proxy_features(df):
    device_shared = _prior_distinct_other_accounts(df, "DeviceID")
    ip_shared = _prior_distinct_other_accounts(df, "IP Address")
    merchant_shared = _prior_distinct_other_accounts(df, "MerchantID")
    return pd.DataFrame({
        "DeviceSharedAccounts_Prior": device_shared,
        "IPSharedAccounts_Prior": ip_shared,
        "MerchantSharedAccounts_Prior": merchant_shared,
    })


# ------------------------------------------------------------------ encoding
def frequency_encoding(df, col, out_name):
    freq = df[col].value_counts(normalize=True)
    return df[col].map(freq).rename(out_name)


def main():
    df = _sorted_raw()
    n = len(df)

    v1d, v7d = velocity_features(df)
    roll_exp = rolling_and_expanding_features(df)
    ratio_dev = ratio_and_deviation_features(df, roll_exp)
    cyc = cyclical_time_features(df)
    behav = behavioral_features(df, roll_exp)
    net = network_proxy_features(df)
    loc_freq = frequency_encoding(df, "Location", "Location_Freq")

    new_features = pd.concat(
        [v1d, v7d, roll_exp, ratio_dev, cyc, behav, net, loc_freq], axis=1
    )
    assert len(new_features) == n

    # ---- load & align v1's 20 features (same sort order -> same row order) ----
    v1_features = pd.read_csv(V1_FEATURES_CSV)
    assert len(v1_features) == n, f"v1 features row count {len(v1_features)} != raw {n}"

    # Sanity check the alignment assumption is actually correct, not assumed:
    # recompute TimeSinceLastTxn independently here with the identical formula
    # src/fe_utils.py uses, and confirm it matches v1's column row-for-row.
    gap_hours = df.groupby("AccountID")["dt"].diff().dt.total_seconds() / 3600.0
    median_gap = gap_hours.median()
    time_since_last_check = gap_hours.fillna(median_gap).reset_index(drop=True)
    match = np.allclose(time_since_last_check.values, v1_features["TimeSinceLastTxn"].values, atol=1e-6)
    print(f"Row-order alignment check (recomputed TimeSinceLastTxn vs v1 features.csv): {'MATCH' if match else 'MISMATCH'}")
    if not match:
        raise RuntimeError("Row order between this script's sort and artifacts/features.csv does not match -- "
                            "cannot safely concatenate. Aborting rather than silently misaligning features.")

    features_v2 = pd.concat(
        [df[["TransactionID", "AccountID"]].reset_index(drop=True), v1_features.reset_index(drop=True),
         new_features.reset_index(drop=True)],
        axis=1
    )

    n_missing = int(features_v2.isna().sum().sum())
    print(f"Total missing cells in features_v2 after fills: {n_missing}")
    assert n_missing == 0, "Unexpected NaNs remain in engineered feature matrix"

    features_v2.to_csv(OUT_CSV, index=False)
    print(f"Saved {features_v2.shape[0]} rows x {features_v2.shape[1]} cols -> {OUT_CSV}")

    # ---- diagnostics used directly in the Phase 5 report ----
    diag = {
        "velocity_1d_value_counts": v1d.value_counts().to_dict(),
        "velocity_7d_value_counts": v7d.value_counts().to_dict(),
        "device_shared_gt0_rows": int((net["DeviceSharedAccounts_Prior"] > 0).sum()),
        "ip_shared_gt0_rows": int((net["IPSharedAccounts_Prior"] > 0).sum()),
        "merchant_shared_gt0_rows": int((net["MerchantSharedAccounts_Prior"] > 0).sum()),
        "device_shared_max": int(net["DeviceSharedAccounts_Prior"].max()),
        "ip_shared_max": int(net["IPSharedAccounts_Prior"].max()),
        "merchant_shared_max": int(net["MerchantSharedAccounts_Prior"].max()),
        "elevated_login_rows": int(behav["ElevatedLoginFlag"].sum()),
        "atm_credit_rows": int(behav["ATM_Credit_InteractionFlag"].sum()),
        "n_rows": n,
        "n_features_v2_cols": int(features_v2.shape[1]),
    }
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "phase5_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2, default=str)
    print(json.dumps(diag, indent=2, default=str))


if __name__ == "__main__":
    main()
