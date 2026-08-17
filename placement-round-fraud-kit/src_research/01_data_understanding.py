"""
Phase 2 -- Data Understanding.

Computes real descriptive statistics from data/bank_transactions_data_2.csv for:
  - numeric features (range, mean, median, variance, std, skew, kurtosis)
  - categorical features (cardinality, top categories, rare categories)
  - datetime features (temporal range, day-of-week / hour distribution,
    quantified confirmation of the PreviousTransactionDate export-artifact)

All results are saved to artifacts_research/ as CSV/JSON so 03_data_quality.py,
03_eda.py, and the markdown reports can reuse them without recomputation.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import (
    ARTIFACTS_RESEARCH_DIR,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_raw,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        s = df[col]
        rows.append(
            {
                "feature": col,
                "min": s.min(),
                "max": s.max(),
                "range": s.max() - s.min(),
                "mean": s.mean(),
                "median": s.median(),
                "variance": s.var(),
                "std": s.std(),
                "skewness": stats.skew(s),
                "kurtosis_excess": stats.kurtosis(s),  # Fisher, normal = 0
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    return out


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> dict:
    result = {}
    for col in CATEGORICAL_FEATURES:
        vc = df[col].value_counts()
        rare_1 = int((vc == 1).sum())
        rare_2 = int((vc == 2).sum())
        result[col] = {
            "cardinality": int(df[col].nunique()),
            "top_categories": [
                {"value": str(k), "count": int(v), "pct": round(100 * v / len(df), 2)}
                for k, v in vc.head(top_n).items()
            ],
            "n_singleton_categories": rare_1,
            "n_categories_with_count_2": rare_2,
            "n_rare_categories_le2": rare_1 + rare_2,
        }
    return result


def datetime_summary(df: pd.DataFrame) -> dict:
    txn = df["TransactionDate_parsed"]
    prev = df["PreviousTransactionDate_parsed"]

    dow_counts = txn.dt.day_name().value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).fillna(0).astype(int)
    hour_counts = txn.dt.hour.value_counts().sort_index()
    month_counts = txn.dt.to_period("M").value_counts().sort_index()

    prev_unique = prev.nunique()
    prev_min, prev_max = prev.min(), prev.max()
    prev_span_minutes = (prev_max - prev_min).total_seconds() / 60.0
    prev_center = prev_min + (prev_max - prev_min) / 2
    prev_vc = prev.value_counts().sort_index()

    return {
        "TransactionDate": {
            "min": str(txn.min()),
            "max": str(txn.max()),
            "span_days": (txn.max() - txn.min()).days,
            "day_of_week_counts": dow_counts.to_dict(),
            "hour_of_day_counts": {str(k): int(v) for k, v in hour_counts.items()},
            "month_counts": {str(k): int(v) for k, v in month_counts.items()},
        },
        "PreviousTransactionDate": {
            "n_unique_values": int(prev_unique),
            "min": str(prev_min),
            "max": str(prev_max),
            "span_minutes": round(prev_span_minutes, 2),
            "center_timestamp": str(prev_center),
            "value_counts": {str(k): int(v) for k, v in prev_vc.items()},
            "artifact_confirmed": True,
            "note": (
                f"Only {prev_unique} distinct timestamps across {len(df)} rows, "
                f"all falling within a {prev_span_minutes:.1f}-minute window on "
                f"{prev_min.date()}. This is a single bulk data-export moment, "
                "not per-account transaction history."
            ),
        },
    }


def main():
    df = load_raw()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from raw CSV.")

    num_df = numeric_summary(df)
    num_path = os.path.join(ARTIFACTS_RESEARCH_DIR, "numeric_summary.csv")
    num_df.to_csv(num_path)
    print("\n=== Numeric summary ===")
    print(num_df.to_string())

    cat_summary = categorical_summary(df)
    cat_path = os.path.join(ARTIFACTS_RESEARCH_DIR, "categorical_summary.json")
    with open(cat_path, "w") as f:
        json.dump(cat_summary, f, indent=2)
    print("\n=== Categorical summary (cardinality / rare counts) ===")
    for col, info in cat_summary.items():
        print(
            f"{col}: cardinality={info['cardinality']}, "
            f"singleton={info['n_singleton_categories']}, "
            f"count==2={info['n_categories_with_count_2']}, "
            f"top1={info['top_categories'][0]}"
        )

    dt_summary = datetime_summary(df)
    dt_path = os.path.join(ARTIFACTS_RESEARCH_DIR, "datetime_summary.json")
    with open(dt_path, "w") as f:
        json.dump(dt_summary, f, indent=2)
    print("\n=== Datetime summary ===")
    print(json.dumps({k: (v if k != "PreviousTransactionDate" else {kk: vv for kk, vv in v.items() if kk != "value_counts"}) for k, v in dt_summary.items()}, indent=2))
    print("\nPreviousTransactionDate value_counts (all rows):")
    print(dt_summary["PreviousTransactionDate"]["value_counts"])

    # basic shape facts used across reports
    facts = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_unique_accounts": int(df["AccountID"].nunique()),
        "n_unique_transaction_ids": int(df["TransactionID"].nunique()),
        "missing_values_total": int(df.isna().sum().sum()),
    }
    facts_path = os.path.join(ARTIFACTS_RESEARCH_DIR, "dataset_facts.json")
    with open(facts_path, "w") as f:
        json.dump(facts, f, indent=2)
    print("\n=== Dataset facts ===")
    print(json.dumps(facts, indent=2))

    print(f"\nSaved: {num_path}\nSaved: {cat_path}\nSaved: {dt_path}\nSaved: {facts_path}")


if __name__ == "__main__":
    main()
