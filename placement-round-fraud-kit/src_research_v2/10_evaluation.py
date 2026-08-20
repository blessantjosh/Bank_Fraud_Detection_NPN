"""
Phase 10 (v2) -- Evaluation Framework, on the teammate's 18-feature matrix.

Mirrors src_research/10_evaluation.py's methodology 1:1. No fraud label
exists anywhere in this project, so "evaluation" here means the same
label-free things, plus one manual read:
  A. Internal cluster-validity metrics (Silhouette, Davies-Bouldin,
     Calinski-Harabasz) for each of the 9 models' implied binary partition
     (top-5%-by-score flagged vs. rest), computed on the RobustScaler-scaled
     18-feature space every model in Phase 8 (v2) shares.
  B. Stability/robustness: for Isolation Forest and LOF, refit on 5
     bootstrap resamples of the training split and measure how consistent
     the top-5%-flagged set is across runs (mean pairwise Jaccard). (No
     deep-learning model remains to include here -- the Autoencoder
     stability check was removed along with the model itself.)
  C. A manual business-evaluation pass over real top-1/2/5/10% transactions
     (with real TransactionID/AccountID and raw dollar values joined back
     from data/bank_transactions_data_2.csv), reasoned against Phase 1's
     scenario table, honest about which ones do and do not look like
     plausible fraud.

Outputs: artifacts_research_v2/{internal_validity_metrics_v2.csv,
stability_bootstrap_jaccard_v2.csv, business_evaluation_examples_v2.json}.
Plots: research_v2/plots/{internal_validity_comparison_v2.png,
stability_jaccard_comparison_v2.png}.
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research_v2 import (
    ARTIFACTS_V2_DIR, DATA_DIR, FEATURE_COLS_V2, MODELS_V2_DIR, PLOTS_V2_DIR, RANDOM_STATE, RAW_CSV,
    load_features_v2,
)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

TOP_PCT = 0.05
MODEL_NAMES = ["isolation_forest", "lof", "ocsvm", "elliptic_envelope", "dbscan",
               "hdbscan", "kmeans", "gmm", "hybrid_ensemble"]


def savefig(fig, name):
    path = os.path.join(PLOTS_V2_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def top_pct_partition(score, pct=TOP_PCT):
    thresh = np.percentile(score, 100 * (1 - pct))
    return (score >= thresh).astype(int)


def load_everything():
    df = load_features_v2()
    feature_cols = FEATURE_COLS_V2
    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = joblib.load(os.path.join(MODELS_V2_DIR, "shared_robust_scaler.pkl"))
    X_train = scaler.transform(X[idx_train])
    X_all = scaler.transform(X)

    scores = pd.read_csv(os.path.join(ARTIFACTS_V2_DIR, "model_scores_all.csv"))
    assert (scores["TransactionID"].values == df["TransactionID"].values).all()
    return df, feature_cols, X, X_train, X_all, idx_train, idx_val, scaler, scores


# ------------------------------------------------------- A. Internal validity metrics
def internal_validity_metrics(scores, X_all):
    score_col_for = {n: (f"score_{n}" if n != "hybrid_ensemble" else "hybrid_vote_count") for n in MODEL_NAMES}
    rows = []
    for name in MODEL_NAMES:
        col = score_col_for[name]
        s = scores[col].values.astype(float)
        mask = np.ones(len(scores), dtype=bool)
        s_m = s[mask]
        X_m = X_all[mask]
        flag = top_pct_partition(s_m)
        n_flagged = int(flag.sum())
        n_rest = int((1 - flag).sum())

        t0 = time.time()
        if n_flagged >= 2 and n_rest >= 2:
            sil = silhouette_score(X_m, flag)
            dbi = davies_bouldin_score(X_m, flag)
            ch = calinski_harabasz_score(X_m, flag)
        else:
            sil, dbi, ch = np.nan, np.nan, np.nan
        elapsed = time.time() - t0

        rows.append({
            "model": name, "n_rows_used": int(mask.sum()), "n_flagged_top5pct": n_flagged,
            "flagged_rate": round(n_flagged / mask.sum(), 4),
            "silhouette": round(float(sil), 4) if sil == sil else np.nan,
            "davies_bouldin": round(float(dbi), 4) if dbi == dbi else np.nan,
            "calinski_harabasz": round(float(ch), 2) if ch == ch else np.nan,
            "compute_sec": round(elapsed, 2),
        })
        print(f"  {name:20s}  n_flagged={n_flagged:4d}  silhouette={rows[-1]['silhouette']}"
              f"  DBI={rows[-1]['davies_bouldin']}  CH={rows[-1]['calinski_harabasz']}")

    out = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
    out.to_csv(os.path.join(ARTIFACTS_V2_DIR, "internal_validity_metrics_v2.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'internal_validity_metrics_v2.csv')}")

    plot_df = out.sort_values("silhouette", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    axes[0].barh(plot_df["model"], plot_df["silhouette"], color="#2F6690")
    axes[0].set_title("Silhouette (higher = better separated)")
    axes[1].barh(plot_df["model"], plot_df["davies_bouldin"], color="#D1495B")
    axes[1].set_title("Davies-Bouldin (lower = better separated)")
    axes[2].barh(plot_df["model"], plot_df["calinski_harabasz"], color="#4C956C")
    axes[2].set_title("Calinski-Harabasz (higher = better separated)")
    axes[2].set_xscale("log")
    fig.suptitle("Internal Validity of Each Model's Top-5%-Flagged vs. Rest Partition (v2)\n"
                  "(RobustScaler-scaled 18-feature space; no fraud label used or implied)", fontsize=11)
    savefig(fig, "internal_validity_comparison_v2.png")

    return out


# ------------------------------------------------------------ B. Stability / robustness
def bootstrap_jaccard_if(X_train, X_all, n_runs=5):
    flags = []
    for seed in range(n_runs):
        rng = np.random.RandomState(1000 + seed)
        boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        clf = IsolationForest(n_estimators=100, max_samples="auto", max_features=1.0,
                               contamination=0.05, random_state=seed, n_jobs=-1)
        clf.fit(X_train[boot_idx])
        score = -clf.decision_function(X_all)
        flags.append(top_pct_partition(score))
    return flags


def bootstrap_jaccard_lof(X_train, X_all, n_runs=5):
    flags = []
    for seed in range(n_runs):
        rng = np.random.RandomState(2000 + seed)
        boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        clf.fit(X_train[boot_idx])
        score = -clf.decision_function(X_all)
        flags.append(top_pct_partition(score))
    return flags


def mean_pairwise_jaccard(flags):
    n = len(flags)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = flags[i].astype(bool), flags[j].astype(bool)
            union = (f1 | f2).sum()
            inter = (f1 & f2).sum()
            vals.append(inter / union if union > 0 else np.nan)
    return float(np.mean(vals)), vals


def run_stability(X_train, X_all):
    print("\n=== Stability (v2): 5 bootstrap resamples each for IF, LOF ===")
    t0 = time.time()
    if_flags = bootstrap_jaccard_if(X_train, X_all)
    if_jaccard, if_pairs = mean_pairwise_jaccard(if_flags)
    print(f"  Isolation Forest: mean pairwise Jaccard = {if_jaccard:.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    lof_flags = bootstrap_jaccard_lof(X_train, X_all)
    lof_jaccard, lof_pairs = mean_pairwise_jaccard(lof_flags)
    print(f"  LOF: mean pairwise Jaccard = {lof_jaccard:.4f}  ({time.time()-t0:.1f}s)")

    rows = [
        {"model": "isolation_forest", "n_bootstrap_runs": 5,
         "mean_pairwise_jaccard_top5pct": round(if_jaccard, 4),
         "min_pairwise_jaccard": round(float(np.min(if_pairs)), 4),
         "max_pairwise_jaccard": round(float(np.max(if_pairs)), 4)},
        {"model": "lof", "n_bootstrap_runs": 5,
         "mean_pairwise_jaccard_top5pct": round(lof_jaccard, 4),
         "min_pairwise_jaccard": round(float(np.min(lof_pairs)), 4),
         "max_pairwise_jaccard": round(float(np.max(lof_pairs)), 4)},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ARTIFACTS_V2_DIR, "stability_bootstrap_jaccard_v2.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'stability_bootstrap_jaccard_v2.csv')}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    names = out["model"].tolist()
    means = out["mean_pairwise_jaccard_top5pct"].tolist()
    mins = out["min_pairwise_jaccard"].tolist()
    maxs = out["max_pairwise_jaccard"].tolist()
    yerr = [[m - lo for m, lo in zip(means, mins)], [hi - m for m, hi in zip(means, maxs)]]
    ax.bar(names, means, color=["#2F6690", "#D1495B"], yerr=yerr, capsize=6)
    ax.set_ylabel("Mean pairwise Jaccard overlap, top-5%-flagged set\n(5 bootstrap resamples of the training split)")
    ax.set_title("Flagging Stability Across Retrains (v2)\n(\"if we retrained tomorrow, would we flag the same transactions?\")")
    ax.set_ylim(0, 1)
    savefig(fig, "stability_jaccard_comparison_v2.png")

    return out


# ------------------------------------------------------- C. Business evaluation
def business_evaluation(df, scores, internal_metrics_df):
    ranked = internal_metrics_df.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False)
    top_by_silhouette = ranked.iloc[0]["model"]

    # Hybrid Ensemble's native score (hybrid_vote_count, 0-3) is too coarse
    # (only 4 distinct values) to carve out clean top-1%/2%/10% tiers.
    # Isolation Forest is used instead: continuous, native-contamination,
    # single production-ready model (Phase 8 v2 Section 1.1) -- consistent
    # with the in-house Phase 10's choice, reported explicitly here too.
    chosen_model = "isolation_forest"
    score_col = "score_isolation_forest"
    print(f"\nInternal-validity silhouette leader (v2): {top_by_silhouette} "
          f"(silhouette={ranked.iloc[0]['silhouette']})")
    print(f"Business-evaluation walkthrough uses: {chosen_model} "
          f"(continuous native score; Hybrid Ensemble's 0-3 vote_count is too coarse for "
          f"fine-grained 1%/2%/10% percentile tiers)")

    # Join raw (unscaled) values for readable narratives -- the 18-column
    # feature matrix is StandardScaler-scaled, not human-readable in dollars.
    raw = pd.read_csv(RAW_CSV)[["TransactionID", "TransactionAmount", "AccountBalance",
                                  "LoginAttempts", "TransactionDuration", "CustomerAge"]]
    raw = raw.rename(columns={"TransactionAmount": "raw_TransactionAmount",
                                "AccountBalance": "raw_AccountBalance",
                                "LoginAttempts": "raw_LoginAttempts",
                                "TransactionDuration": "raw_TransactionDuration",
                                "CustomerAge": "raw_CustomerAge"})
    merged = df.merge(raw, on="TransactionID", how="left")
    assert merged["raw_TransactionAmount"].notna().all()
    merged["raw_amount_to_balance_ratio"] = merged["raw_TransactionAmount"] / merged["raw_AccountBalance"]

    s = scores[score_col].values
    tiers = {}
    for pct, label in [(0.01, "top_1pct"), (0.02, "top_2pct"), (0.05, "top_5pct"), (0.10, "top_10pct")]:
        thresh = np.percentile(s, 100 * (1 - pct))
        idx = np.where(s >= thresh)[0]
        idx_sorted = idx[np.argsort(-s[idx])]
        tiers[label] = {
            "threshold_score": round(float(thresh), 4),
            "n_transactions": int(len(idx)),
            "transaction_ids": df.loc[idx_sorted, "TransactionID"].tolist(),
        }

    cols_of_interest = ["TransactionID", "AccountID", "raw_TransactionAmount", "raw_AccountBalance",
                         "raw_amount_to_balance_ratio", "high_amount_transaction", "raw_LoginAttempts",
                         "device_frequency", "ip_frequency", "account_frequency",
                         "CustomerOccupation_Student", "raw_TransactionDuration"]

    examples = {}
    for label in ["top_1pct", "top_2pct", "top_5pct", "top_10pct"]:
        n_take = 5 if label in ("top_1pct", "top_2pct") else 8
        ids = tiers[label]["transaction_ids"][:n_take]
        rows = merged[merged["TransactionID"].isin(ids)][cols_of_interest].copy()
        rows = rows.set_index("TransactionID").loc[ids].reset_index()
        examples[label] = rows.to_dict(orient="records")

    # also grab the weakest end of the top-10% tier, matching the in-house
    # Phase 10's "top-10% tail" spot-check
    tail_ids = tiers["top_10pct"]["transaction_ids"][-8:]
    tail_rows = merged[merged["TransactionID"].isin(tail_ids)][cols_of_interest].copy()
    tail_rows = tail_rows.set_index("TransactionID").loc[tail_ids].reset_index()
    examples["top_10pct_weakest_tail"] = tail_rows.to_dict(orient="records")

    with open(os.path.join(ARTIFACTS_V2_DIR, "business_evaluation_examples_v2.json"), "w") as f:
        json.dump({
            "chosen_model_for_walkthrough": chosen_model,
            "justification": ("Hybrid Ensemble's vote_count (0-3, only 4 distinct values) cannot "
                               "produce clean top-1%/2%/10% tiers; Isolation Forest is a continuous, "
                               "single, production-ready native-contamination model."),
            "tiers": tiers,
            "detailed_examples": examples,
        }, f, indent=2, default=str)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'business_evaluation_examples_v2.json')}")
    return tiers, examples


def main():
    print("=== Phase 10 (v2): Evaluation Framework ===")
    (df, feature_cols, X, X_train, X_all, idx_train, idx_val, scaler, scores) = load_everything()

    print("\n--- A. Internal validity metrics (Silhouette / Davies-Bouldin / Calinski-Harabasz) ---")
    internal_df = internal_validity_metrics(scores, X_all)

    stability_df = run_stability(X_train, X_all)

    print("\n--- C. Business evaluation ---")
    tiers, examples = business_evaluation(df, scores, internal_df)

    print("\nPhase 10 (v2) complete.")


if __name__ == "__main__":
    main()
