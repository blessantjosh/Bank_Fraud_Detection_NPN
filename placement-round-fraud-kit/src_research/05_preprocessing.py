"""
Phase 6 -- Preprocessing comparison.

1. Missing values: none exist (confirmed again here on features_v2.csv) --
   no imputation comparison is fabricated.
2. Scaling: StandardScaler vs MinMaxScaler vs RobustScaler vs
   QuantileTransformer, compared quantitatively on skew/kurtosis and on how
   much "coordinate space" each gives to the bulk of typical values vs the
   tail, on a representative set of continuous engineered features.
3. Encoding: label encoding (Location_enc, from v1) vs frequency encoding
   (Location_Freq, from Phase 5) vs one-hot, compared by (rough, weak-proxy)
   correlation / R^2 against artifacts/anomaly_votes.csv vote_count.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                    RobustScaler, StandardScaler)
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import ARTIFACTS_RESEARCH_DIR, ROOT_DIR, RANDOM_STATE

FEATURES_V2_CSV = os.path.join(ARTIFACTS_RESEARCH_DIR, "features_v2.csv")
VOTES_CSV = os.path.join(ROOT_DIR, "artifacts", "anomaly_votes.csv")

# Representative continuous features for the scaling comparison: the raw
# skewed money features plus the new Phase-5 engineered continuous features
# that inherit / amplify that skew.
SCALE_CANDIDATE_COLS = [
    "TransactionAmount", "AccountBalance", "TransactionDuration", "CustomerAge",
    "Expanding_MeanAmount", "Expanding_MedianAmount", "Expanding_StdAmount",
    "Expanding_MaxAmount", "Rolling3_MeanAmount", "Amount_to_Balance_Ratio",
    "Amount_to_RollingMean_Ratio", "Amount_minus_ExpandingMean",
    "Amount_ZScore_Account", "TimeSinceLastTxn", "SpendCV_Account",
]


def missing_value_check(df):
    total_missing = int(df.isna().sum().sum())
    print(f"Missing cells in features_v2.csv: {total_missing} / {df.size}")
    return total_missing


def scaling_comparison(df):
    X = df[SCALE_CANDIDATE_COLS].copy()

    scalers = {
        "raw": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "QuantileTransformer_normal": QuantileTransformer(
            output_distribution="normal", n_quantiles=min(1000, len(df)), random_state=RANDOM_STATE),
        "QuantileTransformer_uniform": QuantileTransformer(
            output_distribution="uniform", n_quantiles=min(1000, len(df)), random_state=RANDOM_STATE),
    }

    rows = []
    scaled_outputs = {}
    for name, scaler in scalers.items():
        if scaler is None:
            arr = X.values
        else:
            arr = scaler.fit_transform(X.values)
        scaled_outputs[name] = arr
        for j, col in enumerate(SCALE_CANDIDATE_COLS):
            col_vals = arr[:, j]
            skew = stats.skew(col_vals)
            kurt = stats.kurtosis(col_vals)  # excess (Fisher)
            rng = col_vals.max() - col_vals.min()
            q25, q75 = np.percentile(col_vals, [25, 75])
            iqr_coverage = (q75 - q25) / rng if rng > 0 else np.nan
            rows.append({
                "scaler": name, "feature": col, "skew": round(skew, 4),
                "excess_kurtosis": round(kurt, 4), "min": round(col_vals.min(), 4),
                "max": round(col_vals.max(), 4), "iqr_over_range": round(iqr_coverage, 5),
            })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "scaler_comparison.csv"), index=False)

    # Focused illustration: TransactionAmount specifically (the most skewed feature)
    amt_idx = SCALE_CANDIDATE_COLS.index("TransactionAmount")
    amt_summary = {}
    for name, arr in scaled_outputs.items():
        col_vals = arr[:, amt_idx]
        amt_summary[name] = {
            "skew": round(float(stats.skew(col_vals)), 4),
            "min": round(float(col_vals.min()), 4),
            "max": round(float(col_vals.max()), 4),
            "median": round(float(np.median(col_vals)), 4),
            "iqr_over_range": round(float((np.percentile(col_vals, 75) - np.percentile(col_vals, 25))
                                          / (col_vals.max() - col_vals.min())), 5),
        }
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "scaler_comparison_transactionamount.json"), "w") as f:
        json.dump(amt_summary, f, indent=2)

    print("\n=== Scaling comparison: TransactionAmount ===")
    print(json.dumps(amt_summary, indent=2))
    return comp_df, amt_summary


def scale_sensitivity_analysis(df):
    """
    Skew/kurtosis are mathematically invariant under any affine (location-
    scale) transform, so StandardScaler, MinMaxScaler, and RobustScaler will
    ALWAYS report identical skew/kurtosis to the raw feature and to each
    other -- confirmed empirically below, not just asserted. The real
    difference between these three scalers is which single-point-sensitive
    statistic each one uses as its scale denominator (std, full range, or
    IQR). This function measures that sensitivity directly: how much each
    scale statistic moves when the top 1% most extreme values are excluded.
    A scaler whose denominator swings wildly from a handful of points lets
    those same points dictate how compressed the *typical* 99% of the data
    becomes after scaling -- undesirable here, since Phase 3 deliberately
    kept outliers in the data as candidate fraud signal rather than
    removing them.
    """
    rows = []
    for col in SCALE_CANDIDATE_COLS:
        x = df[col].values
        p99 = np.percentile(x, 99)
        trimmed = x[x <= p99]
        std_full, std_trim = x.std(), trimmed.std()
        rng_full, rng_trim = x.max() - x.min(), trimmed.max() - trimmed.min()
        q25f, q75f = np.percentile(x, [25, 75])
        q25t, q75t = np.percentile(trimmed, [25, 75])
        iqr_full, iqr_trim = q75f - q25f, q75t - q25t
        rows.append({
            "feature": col,
            "std_pct_change_top1pct_trim": round(100 * (std_full - std_trim) / std_full, 2) if std_full else 0.0,
            "range_pct_change_top1pct_trim": round(100 * (rng_full - rng_trim) / rng_full, 2) if rng_full else 0.0,
            "iqr_pct_change_top1pct_trim": round(100 * (iqr_full - iqr_trim) / iqr_full, 2) if iqr_full else 0.0,
        })
    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "scaler_sensitivity_to_outliers.csv"), index=False)
    print("\n=== Scale-denominator sensitivity to top-1% trim (StandardScaler=std, MinMaxScaler=range, RobustScaler=IQR) ===")
    print(sens_df.to_string(index=False))
    return sens_df


def encoding_comparison(df):
    votes = pd.read_csv(VOTES_CSV)
    assert len(votes) == len(df), "anomaly_votes.csv row count does not match features_v2.csv"
    vote_count = votes["vote_count"].values.astype(float)

    label_enc = df["Location_enc"].values.astype(float)
    freq_enc = df["Location_Freq"].values.astype(float)

    corr_label = float(np.corrcoef(label_enc, vote_count)[0, 1])
    corr_freq = float(np.corrcoef(freq_enc, vote_count)[0, 1])

    # True one-hot: pull the raw Location string (not preserved in
    # features_v2.csv) under the identical sort used to build features_v2,
    # so row order lines up 1:1 -- this gives a real 43-category one-hot
    # design matrix rather than a lossy reconstruction from tied frequencies
    # (several of the 43 cities share the same transaction count, so
    # reconstructing categories from Location_Freq alone would incorrectly
    # merge distinct cities that happen to tie on frequency).
    raw = pd.read_csv(os.path.join(ROOT_DIR, "data", "bank_transactions_data_2.csv"))
    raw["dt"] = pd.to_datetime(raw["TransactionDate"], format="%d-%m-%Y %H:%M")
    raw = raw.sort_values(["AccountID", "dt", "TransactionID"]).reset_index(drop=True)
    assert len(raw) == len(df)
    location_raw = raw["Location"].values
    n_true_locations = len(np.unique(location_raw))

    onehot = pd.get_dummies(location_raw, drop_first=True).values.astype(float)
    reg = LinearRegression().fit(onehot, vote_count)
    r2_onehot = reg.score(onehot, vote_count)

    result = {
        "corr_label_enc_vs_votes": round(corr_label, 4),
        "corr_freq_enc_vs_votes": round(corr_freq, 4),
        "r2_label_enc_vs_votes": round(corr_label ** 2, 4),
        "r2_freq_enc_vs_votes": round(corr_freq ** 2, 4),
        "r2_onehot_vs_votes": round(r2_onehot, 4),
        "n_distinct_locations": int(n_true_locations),
        "vote_count_mean": round(float(vote_count.mean()), 4),
        "vote_count_std": round(float(vote_count.std()), 4),
    }
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "encoding_comparison.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Encoding comparison (Location vs anomaly-vote proxy) ===")
    print(json.dumps(result, indent=2))
    return result


def main():
    df = pd.read_csv(FEATURES_V2_CSV)
    missing_value_check(df)
    scaling_comparison(df)
    scale_sensitivity_analysis(df)
    encoding_comparison(df)


if __name__ == "__main__":
    main()
