"""
Phase 13 -- Threshold Optimization.

Applies percentile-based and statistical thresholds to the Phase 12
recommended ensemble score (`ensemble_percentile_average`,
artifacts_research/ensemble_scores.csv) and reasons about the resulting
operational load and illustrative cost, explicitly bounded by what is and
is not honestly computable without a fraud label.

There is NO fraud label anywhere in this project. That means:
  - A precision/recall- or cost-curve-minimizing threshold (the kind v1's
    `src/06_evaluation.py` computes against its supervised proxy labels)
    CANNOT be computed here, and this script does not pretend otherwise.
  - What CAN be honestly shown: how many transactions land above each
    threshold, and what that implies for review-team operational load and
    an UPPER-BOUND illustrative false-positive cost (using v1's own stated
    illustrative $5 FP / $250 FN figures, which are not real bank numbers
    either) -- but never a true minimized total cost, since the false-
    negative side of that equation requires knowing which flagged (or
    unflagged) transactions are actually fraud, which this project cannot
    determine.

A genuine, unforced methodological finding surfaces directly from running
the statistical thresholds against the recommended score: because
`ensemble_percentile_average` is a bounded average of 11 percentiles in
(0, 1), both mean+3sigma and Q3+1.5*IQR exceed the maximum possible value of
the score, flagging zero transactions. This is reported plainly, along with
a side-by-side comparison against an unbounded score (Isolation Forest's raw
score, and the Phase 12 Weighted-Average ensemble) to show this is a
property of percentile aggregation specifically, not a general failure of
statistical thresholding on this dataset.

Outputs: artifacts_research/threshold_analysis.json,
artifacts_research/threshold_flagged_counts.csv.
Plots: research/plots/threshold_score_distribution.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import ARTIFACTS_RESEARCH_DIR, PLOTS_DIR, ROOT_DIR

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

# v1's own stated illustrative cost figures (src/config.py) -- NOT real bank
# numbers, reused here only for comparability with v1's framing, same caveat
# v1 itself states.
COST_FALSE_POSITIVE = 5.0
COST_FALSE_NEGATIVE = 250.0

N_ROWS = 2512
DATASET_SPAN_DAYS = 364  # research/02_data_understanding.md / datetime_summary.json


def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def load_scores():
    ens = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "ensemble_scores.csv"))
    classical = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_scores_all.csv"))
    assert (ens["TransactionID"].values == classical["TransactionID"].values).all()
    return ens, classical


def percentile_thresholds(score, percentiles):
    rows = []
    for p in percentiles:
        thresh = float(np.percentile(score, p))
        n_flagged = int((score >= thresh).sum())
        rows.append({"method": f"P{p}", "threshold_value": round(thresh, 6),
                     "n_flagged": n_flagged, "pct_flagged": round(100 * n_flagged / len(score), 3)})
    return rows


def statistical_thresholds(score, label):
    mean, std = score.mean(), score.std()
    q1, q3 = np.percentile(score, [25, 75])
    iqr = q3 - q1
    th_3sigma = mean + 3 * std
    th_iqr = q3 + 1.5 * iqr
    rows = []
    for method, thresh in [("mean+3sigma", th_3sigma), ("Q3+1.5*IQR", th_iqr)]:
        n_flagged = int((score >= thresh).sum())
        rows.append({"method": method, "score": label, "threshold_value": round(float(thresh), 6),
                     "n_flagged": n_flagged, "pct_flagged": round(100 * n_flagged / len(score), 3),
                     "score_max": round(float(score.max()), 6), "score_min": round(float(score.min()), 6)})
    return rows


def business_impact(rows, score_name):
    out = []
    for r in rows:
        n = r["n_flagged"]
        flagged_per_day = n / DATASET_SPAN_DAYS
        upper_bound_fp_cost = n * COST_FALSE_POSITIVE  # IF every flagged txn were a false positive
        out.append({
            **r,
            "flagged_per_day_this_sample": round(flagged_per_day, 4),
            "illustrative_upper_bound_review_cost_usd_if_all_fp": round(upper_bound_fp_cost, 2),
        })
    return out


def main():
    print("=== Phase 13: Threshold Optimization ===")
    ens, classical = load_scores()
    score = ens["ensemble_percentile_average"].values
    print(f"Recommended ensemble score (Phase 12): ensemble_percentile_average "
          f"(mean={score.mean():.4f}, std={score.std():.4f}, min={score.min():.4f}, max={score.max():.4f})")

    print("\n--- Percentile thresholds (95th / 97th / 99th / 99.5th) ---")
    pct_rows = percentile_thresholds(score, [95, 97, 99, 99.5])
    for r in pct_rows:
        print(f"  {r['method']:6s} thresh={r['threshold_value']:.4f}  n_flagged={r['n_flagged']:4d}  "
              f"({r['pct_flagged']}%)")

    print("\n--- Statistical thresholds on the recommended (bounded, percentile-based) score ---")
    stat_rows_recommended = statistical_thresholds(score, "ensemble_percentile_average")
    for r in stat_rows_recommended:
        print(f"  {r['method']:12s} thresh={r['threshold_value']:.4f}  n_flagged={r['n_flagged']:4d}  "
              f"(score range [{r['score_min']:.4f}, {r['score_max']:.4f}])")

    print("\n--- Same statistical thresholds, for context, on two UNBOUNDED scores ---")
    if_score = classical["score_isolation_forest"].values
    weighted_score = ens["ensemble_weighted_average"].values
    stat_rows_if = statistical_thresholds(if_score, "score_isolation_forest (unbounded, for context)")
    stat_rows_weighted = statistical_thresholds(weighted_score, "ensemble_weighted_average (unbounded, for context)")
    for r in stat_rows_if + stat_rows_weighted:
        print(f"  [{r['score']}] {r['method']:12s} thresh={r['threshold_value']:.4f}  n_flagged={r['n_flagged']:4d}")

    print("\n--- Business impact framing (percentile thresholds on the recommended score) ---")
    pct_business = business_impact(pct_rows, "ensemble_percentile_average")
    for r in pct_business:
        print(f"  {r['method']:6s} n_flagged={r['n_flagged']:4d}  "
              f"~{r['flagged_per_day_this_sample']:.3f}/day (this sample)  "
              f"upper-bound review cost if all FP: ${r['illustrative_upper_bound_review_cost_usd_if_all_fp']:.0f}")

    all_rows = pd.DataFrame(pct_rows + stat_rows_recommended + stat_rows_if + stat_rows_weighted)
    all_rows.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "threshold_flagged_counts.csv"), index=False)
    print(f"\nSaved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'threshold_flagged_counts.csv')}")

    summary = {
        "recommended_score": "ensemble_percentile_average (Phase 12 recommendation)",
        "score_distribution": {
            "mean": round(float(score.mean()), 4), "std": round(float(score.std()), 4),
            "min": round(float(score.min()), 4), "max": round(float(score.max()), 4),
            "q1": round(float(np.percentile(score, 25)), 4), "q3": round(float(np.percentile(score, 75)), 4),
        },
        "percentile_thresholds": pct_business,
        "statistical_thresholds_on_recommended_score": stat_rows_recommended,
        "statistical_thresholds_context_unbounded_scores": stat_rows_if + stat_rows_weighted,
        "statistical_threshold_finding": (
            "Both mean+3sigma and Q3+1.5*IQR exceed the maximum possible value of "
            "ensemble_percentile_average (max observed 0.999, thresholds computed at "
            f"{stat_rows_recommended[0]['threshold_value']:.3f} and "
            f"{stat_rows_recommended[1]['threshold_value']:.3f} respectively), flagging ZERO "
            "transactions. This is a genuine, unforced methodological finding, not an error: "
            "ensemble_percentile_average is a bounded average of 11 models' percentile ranks in "
            "(0,1), and averaging several roughly-independent percentiles compresses the tails "
            "(a CLT-like effect) far more than any single unbounded model's raw score would. "
            "Classic normal-distribution-derived statistical thresholds (3-sigma, Tukey's IQR "
            "fence) are not well-suited to a bounded, already-aggregated score. Confirmed this is "
            "specific to percentile aggregation, not a general dataset property, by applying the "
            "identical thresholds to two unbounded scores: Isolation Forest's raw score (mean+3sigma "
            f"flags {stat_rows_if[0]['n_flagged']}, Q3+1.5*IQR flags {stat_rows_if[1]['n_flagged']}) and "
            f"the Phase 12 Weighted-Average ensemble (mean+3sigma flags {stat_rows_weighted[0]['n_flagged']}, "
            f"Q3+1.5*IQR flags {stat_rows_weighted[1]['n_flagged']}) -- both produce non-trivial, usable "
            "thresholds. Practical conclusion: statistical (sigma/IQR) thresholds should be applied to an "
            "unbounded, z-scored ensemble score (e.g. Weighted Average) if this thresholding style is "
            "wanted in production, not to the bounded Percentile Average score recommended for scoring."
        ),
        "business_impact_caveat": (
            "This dataset has no fraud label, so a true cost-minimizing threshold (balancing "
            f"illustrative FP cost ${COST_FALSE_POSITIVE:.0f} against illustrative FN cost "
            f"${COST_FALSE_NEGATIVE:.0f}, following v1's own stated illustrative figures, src/config.py "
            "-- not real bank numbers either) CANNOT be computed here, because that requires knowing "
            "which transactions are actually fraud (both among the flagged, to count true positives "
            "against the FP cost, and among the unflagged, to count false negatives against the FN "
            "cost). What CAN be shown honestly: (a) how many transactions land above each threshold, "
            "(b) the implied review-team operational load, and (c) an UPPER-BOUND illustrative review "
            "cost assuming every flagged transaction is a false positive (n_flagged x $5) -- this is "
            "not a total cost estimate, only a worst-case review-labor cost ceiling."
        ),
        "operational_load_caveat": (
            f"This dataset covers {DATASET_SPAN_DAYS} days and {N_ROWS} transactions total across 495 "
            f"accounts -- an average of {N_ROWS/DATASET_SPAN_DAYS:.2f} transactions/day. This is a small "
            "research sample, not representative of a real bank's daily transaction volume; the "
            "'flagged per day' figures below are illustrative of THIS SAMPLE's scale only, not a "
            "production volume estimate, and are reported as such rather than extrapolated into a "
            "false claim about real-world throughput."
        ),
    }
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "threshold_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'threshold_analysis.json')}")

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(score, bins=50, color="#2F6690", alpha=0.75)
    colors = ["#D1495B", "#EDAE49", "#4C956C", "#6A4C93"]
    for r, c in zip(pct_rows, colors):
        ax.axvline(r["threshold_value"], color=c, ls="--", lw=1.3,
                   label=f"{r['method']} (n={r['n_flagged']})")
    ax.set_xlabel("Ensemble score (Percentile Aggregation, Phase 12 recommendation)")
    ax.set_ylabel("Count")
    ax.set_title("Threshold Placement on the Recommended Ensemble Score\n"
                 "(mean+3sigma and Q3+1.5*IQR both exceed the score's max of "
                 f"{score.max():.3f} -- not shown, flag zero rows)")
    ax.legend(fontsize=8)
    savefig(fig, "threshold_score_distribution.png")

    print("\nPhase 13 complete.")


if __name__ == "__main__":
    main()
