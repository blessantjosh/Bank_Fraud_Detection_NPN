"""
Phase 3 -- Data Quality Assessment.

Covers: missing values (confirmation only), exact/near-duplicate detection,
and a five-method outlier comparison (IQR, Z-score, Modified Z-score,
percentile, Isolation Forest) on the five numeric features, with pairwise
overlap between methods. Results saved to artifacts_research/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import ARTIFACTS_RESEARCH_DIR, NUMERIC_FEATURES, RANDOM_STATE, load_raw


def missing_check(df: pd.DataFrame) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    return {"missing_cells": missing, "total_cells": total_cells, "pct_missing": round(100 * missing / total_cells, 4)}


def duplicate_check(df: pd.DataFrame) -> dict:
    exact_dupes = int(df.duplicated(keep=False).sum())
    id_dupes = int(df["TransactionID"].duplicated(keep=False).sum())

    df = df.copy()
    df["_txn_minute"] = df["TransactionDate_parsed"].dt.floor("min")
    near_dup_mask = df.duplicated(
        subset=["AccountID", "TransactionAmount", "_txn_minute"], keep=False
    )
    near_dupes = df.loc[near_dup_mask, ["TransactionID", "AccountID", "TransactionAmount", "_txn_minute"]]

    return {
        "exact_duplicate_rows": exact_dupes,
        "duplicate_transaction_ids": id_dupes,
        "near_duplicate_rows": int(near_dup_mask.sum()),
        "near_duplicate_examples": near_dupes.head(10).astype(str).to_dict(orient="records"),
    }


def iqr_flags(s: pd.Series) -> np.ndarray:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((s < lower) | (s > upper)).to_numpy()


def zscore_flags(s: pd.Series, thresh: float = 3.0) -> np.ndarray:
    z = (s - s.mean()) / s.std(ddof=0)
    return (z.abs() > thresh).to_numpy()


def modified_zscore_flags(s: pd.Series, thresh: float = 3.5) -> np.ndarray:
    median = s.median()
    mad = np.median(np.abs(s - median))
    if mad == 0:
        mad = 1e-9
    mod_z = 0.6745 * (s - median) / mad
    return (mod_z.abs() > thresh).to_numpy()


def percentile_flags(s: pd.Series, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    lo, hi = s.quantile(low), s.quantile(high)
    return ((s < lo) | (s > hi)).to_numpy()


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = set(np.where(a)[0]), set(np.where(b)[0])
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def outlier_analysis(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES].to_numpy()
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE, n_estimators=200)
    iso_pred = iso.fit_predict(X)  # -1 outlier, 1 inlier
    iso_flag_global = iso_pred == -1

    per_feature_rows = []
    method_flags = {}  # feature -> {method: bool array}

    for col in NUMERIC_FEATURES:
        s = df[col]
        flags = {
            "IQR": iqr_flags(s),
            "ZScore": zscore_flags(s),
            "ModifiedZScore": modified_zscore_flags(s),
            "Percentile_1_99": percentile_flags(s),
            "IsolationForest_joint": iso_flag_global,  # same joint flag repeated per feature row for comparison
        }
        method_flags[col] = flags
        per_feature_rows.append(
            {
                "feature": col,
                "IQR_count": int(flags["IQR"].sum()),
                "ZScore_count": int(flags["ZScore"].sum()),
                "ModifiedZScore_count": int(flags["ModifiedZScore"].sum()),
                "Percentile_1_99_count": int(flags["Percentile_1_99"].sum()),
                "IsolationForest_joint_count": int(flags["IsolationForest_joint"].sum()),
            }
        )

    summary_df = pd.DataFrame(per_feature_rows).set_index("feature")

    # pairwise overlap (Jaccard + raw intersection) per feature, univariate methods only
    overlap_rows = []
    methods = ["IQR", "ZScore", "ModifiedZScore", "Percentile_1_99"]
    for col in NUMERIC_FEATURES:
        flags = method_flags[col]
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                inter = int((flags[m1] & flags[m2]).sum())
                overlap_rows.append(
                    {
                        "feature": col,
                        "method_pair": f"{m1} vs {m2}",
                        "jaccard": round(jaccard(flags[m1], flags[m2]), 3),
                        "intersection_count": inter,
                    }
                )
        # also compare each univariate method against the joint IsolationForest flag
        for m in methods:
            inter = int((flags[m] & flags["IsolationForest_joint"]).sum())
            overlap_rows.append(
                {
                    "feature": col,
                    "method_pair": f"{m} vs IsolationForest_joint",
                    "jaccard": round(jaccard(flags[m], flags["IsolationForest_joint"]), 3),
                    "intersection_count": inter,
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)

    return summary_df, overlap_df, int(iso_flag_global.sum())


def main():
    df = load_raw()

    missing = missing_check(df)
    print("=== Missing values ===")
    print(json.dumps(missing, indent=2))

    dupes = duplicate_check(df)
    print("\n=== Duplicate analysis ===")
    print(json.dumps(dupes, indent=2, default=str))

    summary_df, overlap_df, iso_total = outlier_analysis(df)
    print("\n=== Outlier counts per method (per feature) ===")
    print(summary_df.to_string())
    print(f"\nIsolationForest total flagged rows (jointly, contamination=0.05): {iso_total} "
          f"({100*iso_total/len(df):.2f}% of {len(df)} rows)")

    print("\n=== Method overlap (Jaccard similarity + raw intersection) ===")
    print(overlap_df.to_string(index=False))

    quality_path = os.path.join(ARTIFACTS_RESEARCH_DIR, "data_quality_summary.json")
    with open(quality_path, "w") as f:
        json.dump({"missing": missing, "duplicates": dupes, "isolation_forest_total_flagged": iso_total}, f, indent=2, default=str)

    outlier_summary_path = os.path.join(ARTIFACTS_RESEARCH_DIR, "outlier_comparison.csv")
    summary_df.to_csv(outlier_summary_path)

    overlap_path = os.path.join(ARTIFACTS_RESEARCH_DIR, "outlier_method_overlap.csv")
    overlap_df.to_csv(overlap_path, index=False)

    print(f"\nSaved: {quality_path}\nSaved: {outlier_summary_path}\nSaved: {overlap_path}")


if __name__ == "__main__":
    main()
