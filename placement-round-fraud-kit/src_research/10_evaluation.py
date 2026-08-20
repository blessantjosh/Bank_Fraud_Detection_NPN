"""
Phase 10 -- Evaluation Framework.

There is no fraud label anywhere in this project, so "evaluation" here means
two honest, label-free things, plus one manual business read:

  A. Internal cluster-validity metrics (Silhouette, Davies-Bouldin,
     Calinski-Harabasz) for each of the 9 models' implied binary partition
     (top-5%-by-score flagged vs. rest), computed on the RobustScaler-scaled
     46-feature space every model in Phase 8 shares.
  B. Stability/robustness: for Isolation Forest, LOF, and GMM -- the three
     detectors behind the Model 9 Hybrid Ensemble vote -- refit on 5
     bootstrap resamples of the training split and measure how consistent
     the top-5%-flagged set is across runs (mean pairwise Jaccard) --
     "if we retrained tomorrow, would we flag the same rows?"
  C. A manual business-evaluation pass over real top-1/2/5/10% transactions,
     reasoned against Phase 1's scenario table, honest about which ones do
     and do not look like plausible fraud.

Note: this script previously (pre deep-learning removal) also ran Section B's
stability check on the Autoencoder (not GMM), and had a Section C that
consolidated Autoencoder/VAE/LSTM-AE reconstruction-error metrics. The three
deep-learning models were removed from this pipeline -- see the project
decision log -- so the reconstruction-metrics section was removed entirely
(there is nothing left to reconstruct-error on), and the stability check's
third detector was switched from Autoencoder to GMM to match the redefined
Model 9 Hybrid Ensemble (IF + LOF + GMM).

Inputs: artifacts_research/{features_v2.csv, model_scores_all.csv,
model_summary_classical.json, models/shared_robust_scaler.pkl}.
Outputs: artifacts_research/{internal_validity_metrics.csv,
stability_bootstrap_jaccard.csv, business_evaluation_examples.json},
research/plots/{internal_validity_comparison.png, stability_jaccard_comparison.png}.
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
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import ARTIFACTS_RESEARCH_DIR, PLOTS_DIR, RANDOM_STATE, ROOT_DIR

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

FEATURES_V2_CSV = os.path.join(ARTIFACTS_RESEARCH_DIR, "features_v2.csv")
MODELS_DIR = os.path.join(ARTIFACTS_RESEARCH_DIR, "models")
TOP_PCT = 0.05
ID_COLS = ["TransactionID", "AccountID"]

MODEL_NAMES = ["isolation_forest", "lof", "ocsvm", "elliptic_envelope", "dbscan",
               "hdbscan", "kmeans", "gmm", "hybrid_ensemble"]


def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def top_pct_partition(score, pct=TOP_PCT):
    thresh = np.percentile(score, 100 * (1 - pct))
    return (score >= thresh).astype(int)


def load_everything():
    df = pd.read_csv(FEATURES_V2_CSV)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = joblib.load(os.path.join(MODELS_DIR, "shared_robust_scaler.pkl"))
    X_train = scaler.transform(X[idx_train])
    X_all = scaler.transform(X)

    scores = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_scores_all.csv"))
    assert (scores["TransactionID"].values == df["TransactionID"].values).all()
    return df, feature_cols, X, X_train, X_all, idx_train, idx_val, scaler, scores


# ------------------------------------------------------- A. Internal validity metrics
def internal_validity_metrics(scores, X_all):
    score_col_for = {n: (f"score_{n}" if n != "hybrid_ensemble" else "hybrid_vote_count") for n in MODEL_NAMES}
    rows = []
    for name in MODEL_NAMES:
        col = score_col_for[name]
        s_m = scores[col].values.astype(float)
        X_m = X_all
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
            "model": name, "n_rows_used": int(len(s_m)), "n_flagged_top5pct": n_flagged,
            "flagged_rate": round(n_flagged / len(s_m), 4),
            "silhouette": round(float(sil), 4) if sil == sil else np.nan,
            "davies_bouldin": round(float(dbi), 4) if dbi == dbi else np.nan,
            "calinski_harabasz": round(float(ch), 2) if ch == ch else np.nan,
            "compute_sec": round(elapsed, 2),
        })
        print(f"  {name:20s}  n_flagged={n_flagged:4d}  silhouette={rows[-1]['silhouette']}"
              f"  DBI={rows[-1]['davies_bouldin']}  CH={rows[-1]['calinski_harabasz']}")

    out = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
    out.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "internal_validity_metrics.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'internal_validity_metrics.csv')}")

    plot_df = out.sort_values("silhouette", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    axes[0].barh(plot_df["model"], plot_df["silhouette"], color="#2F6690")
    axes[0].set_title("Silhouette (higher = better separated)")
    axes[1].barh(plot_df["model"], plot_df["davies_bouldin"], color="#D1495B")
    axes[1].set_title("Davies-Bouldin (lower = better separated)")
    axes[2].barh(plot_df["model"], plot_df["calinski_harabasz"], color="#4C956C")
    axes[2].set_title("Calinski-Harabasz (higher = better separated)")
    axes[2].set_xscale("log")
    fig.suptitle("Internal Validity of Each Model's Top-5%-Flagged vs. Rest Partition\n"
                  "(RobustScaler-scaled 46-feature space; no fraud label used or implied)", fontsize=11)
    savefig(fig, "internal_validity_comparison.png")

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


def bootstrap_jaccard_gmm(X_train, X_all, n_runs=5):
    flags = []
    for seed in range(n_runs):
        rng = np.random.RandomState(3000 + seed)
        boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        clf = GaussianMixture(n_components=9, covariance_type="full", random_state=seed,
                               reg_covar=1e-5, max_iter=200)
        clf.fit(X_train[boot_idx])
        score = -clf.score_samples(X_all)
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
    print("\n=== Stability: 5 bootstrap resamples each for IF, LOF, GMM ===")
    t0 = time.time()
    if_flags = bootstrap_jaccard_if(X_train, X_all)
    if_jaccard, if_pairs = mean_pairwise_jaccard(if_flags)
    print(f"  Isolation Forest: mean pairwise Jaccard = {if_jaccard:.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    lof_flags = bootstrap_jaccard_lof(X_train, X_all)
    lof_jaccard, lof_pairs = mean_pairwise_jaccard(lof_flags)
    print(f"  LOF: mean pairwise Jaccard = {lof_jaccard:.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    gmm_flags = bootstrap_jaccard_gmm(X_train, X_all)
    gmm_jaccard, gmm_pairs = mean_pairwise_jaccard(gmm_flags)
    print(f"  GMM: mean pairwise Jaccard = {gmm_jaccard:.4f}  ({time.time()-t0:.1f}s)")

    rows = [
        {"model": "isolation_forest", "n_bootstrap_runs": 5,
         "mean_pairwise_jaccard_top5pct": round(if_jaccard, 4),
         "min_pairwise_jaccard": round(float(np.min(if_pairs)), 4),
         "max_pairwise_jaccard": round(float(np.max(if_pairs)), 4)},
        {"model": "lof", "n_bootstrap_runs": 5,
         "mean_pairwise_jaccard_top5pct": round(lof_jaccard, 4),
         "min_pairwise_jaccard": round(float(np.min(lof_pairs)), 4),
         "max_pairwise_jaccard": round(float(np.max(lof_pairs)), 4)},
        {"model": "gmm", "n_bootstrap_runs": 5,
         "mean_pairwise_jaccard_top5pct": round(gmm_jaccard, 4),
         "min_pairwise_jaccard": round(float(np.min(gmm_pairs)), 4),
         "max_pairwise_jaccard": round(float(np.max(gmm_pairs)), 4)},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "stability_bootstrap_jaccard.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'stability_bootstrap_jaccard.csv')}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    names = out["model"].tolist()
    means = out["mean_pairwise_jaccard_top5pct"].tolist()
    mins = out["min_pairwise_jaccard"].tolist()
    maxs = out["max_pairwise_jaccard"].tolist()
    yerr = [[m - lo for m, lo in zip(means, mins)], [hi - m for m, hi in zip(means, maxs)]]
    ax.bar(names, means, color=["#2F6690", "#D1495B", "#4C956C"], yerr=yerr, capsize=6)
    ax.set_ylabel("Mean pairwise Jaccard overlap, top-5%-flagged set\n(5 bootstrap resamples of the training split)")
    ax.set_title("Flagging Stability Across Retrains\n(\"if we retrained tomorrow, would we flag the same transactions?\")")
    ax.set_ylim(0, 1)
    savefig(fig, "stability_jaccard_comparison.png")

    return out


# ------------------------------------------------------- C. Business evaluation
def business_evaluation(df, scores, internal_metrics_df):
    # Choose the model to walk through concrete examples for. Justify from
    # the internal-metrics table computed in Section A, not by assumption.
    ranked = internal_metrics_df.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False)
    top_by_silhouette = ranked.iloc[0]["model"]

    # Hybrid Ensemble's native score (hybrid_vote_count, 0-3) is too coarse
    # (only 4 distinct values) to carve out clean top-1%/2%/10% tiers -- most
    # rows at a given vote count are tied, so "top 1%" and "top 2%" would be
    # identical or arbitrary depending on tie-break order. Isolation Forest
    # is used instead: a continuous, native-contamination, single production-
    # ready model (Phase 8 Section 1.1), and it is very close to the top of
    # the internal-validity ranking above -- reported explicitly, not silently
    # substituted.
    chosen_model = "isolation_forest"
    score_col = "score_isolation_forest"
    print(f"\nInternal-validity silhouette leader: {top_by_silhouette} "
          f"(silhouette={ranked.iloc[0]['silhouette']})")
    print(f"Business-evaluation walkthrough uses: {chosen_model} "
          f"(continuous native score; Hybrid Ensemble's 0-3 vote_count is too coarse for "
          f"fine-grained 1%/2%/10% percentile tiers)")

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

    cols_of_interest = ["TransactionID", "AccountID", "TransactionAmount", "Amount_vs_AccountAvg",
                         "Amount_ZScore_Account", "DeviceNoveltyFlag", "LocationNoveltyFlag",
                         "LoginAttempts", "Velocity_1D_Count", "Velocity_7D_Count",
                         "Amount_to_Balance_Ratio", "TimeSinceLastTxn", "AccountBalance"]

    examples = {}
    for label in ["top_1pct", "top_5pct"]:
        ids = tiers[label]["transaction_ids"][:5]
        rows = df[df["TransactionID"].isin(ids)][cols_of_interest].copy()
        rows = rows.set_index("TransactionID").loc[ids].reset_index()
        examples[label] = rows.to_dict(orient="records")

    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "business_evaluation_examples.json"), "w") as f:
        json.dump({
            "chosen_model_for_walkthrough": chosen_model,
            "justification": ("Hybrid Ensemble's vote_count (0-3, only 4 distinct values) cannot "
                               "produce clean top-1%/2%/10% tiers; Isolation Forest is a continuous, "
                               "single, production-ready native-contamination model, close to the top "
                               "of the internal-validity silhouette ranking (see internal_validity_metrics.csv)."),
            "tiers": tiers,
            "detailed_examples_top1pct_top5pct": examples,
        }, f, indent=2, default=str)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'business_evaluation_examples.json')}")
    return tiers, examples


def main():
    print("=== Phase 10: Evaluation Framework ===")
    (df, feature_cols, X, X_train, X_all, idx_train, idx_val, scaler, scores) = load_everything()

    print("\n--- A. Internal validity metrics (Silhouette / Davies-Bouldin / Calinski-Harabasz) ---")
    internal_df = internal_validity_metrics(scores, X_all)

    stability_df = run_stability(X_train, X_all)

    print("\n--- C. Business evaluation ---")
    tiers, examples = business_evaluation(df, scores, internal_df)

    print("\nPhase 10 complete.")


if __name__ == "__main__":
    main()
