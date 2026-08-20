"""
Phase 8 (v2), Part 2 -- Model 9 (Hybrid Ensemble) plus the cross-model
comparison across all 9 models, on the teammate's 18-feature matrix.

Model 9 -- Hybrid Ensemble: Isolation Forest + LOF + GMM majority vote
           (>=2 of 3). No deep-learning models are trained in this pipeline
           (Autoencoder/VAE/LSTM-AE were removed; see project history --
           this pipeline now runs 8 classical unsupervised detectors,
           trained in 06_models_classical.py, plus this Hybrid Ensemble).
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research_v2 import ARTIFACTS_V2_DIR, PLOTS_V2_DIR, load_features_v2

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})


def savefig(fig, name):
    path = os.path.join(PLOTS_V2_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


# --------------------------------------------------------- Model 9: Hybrid Ensemble
def model_hybrid_ensemble(flag_if, flag_lof, flag_gmm):
    vote_count = flag_if.astype(int) + flag_lof.astype(int) + flag_gmm.astype(int)
    flag_majority = (vote_count >= 2).astype(int)
    summary = {
        "components": "Isolation Forest (Model 1, native flag) + LOF (Model 2, native flag) + Gaussian Mixture Model (Model 8, top-5% negative-log-likelihood flag)",
        "rule": "flagged if >=2 of the 3 component flags fire (majority vote)",
        "vote_distribution": {int(k): int(v) for k, v in pd.Series(vote_count).value_counts().sort_index().items()},
        "majority_flagged_rate": round(float(flag_majority.mean()), 4),
        "pairwise_agreement_if_lof": round(float((flag_if == flag_lof).mean()), 4),
        "pairwise_agreement_if_gmm": round(float((flag_if == flag_gmm).mean()), 4),
        "pairwise_agreement_lof_gmm": round(float((flag_lof == flag_gmm).mean()), 4),
    }
    print("\n=== Model 9: Hybrid Ensemble (IF + LOF + GMM, majority vote) ===")
    print(json.dumps(summary, indent=2, default=float))
    return vote_count, flag_majority, summary


def main():
    df = load_features_v2()
    classical = pd.read_csv(os.path.join(ARTIFACTS_V2_DIR, "model_scores_classical.csv"))
    assert (classical["TransactionID"].values == df["TransactionID"].values).all()

    hybrid_votes, hybrid_flag, hybrid_summary = model_hybrid_ensemble(
        classical["flag_isolation_forest_native"].values,
        classical["flag_lof_native"].values,
        classical["flag_gmm_top5pct"].values,
    )

    all_scores = pd.DataFrame({
        "TransactionID": df["TransactionID"].values,
        "AccountID": df["AccountID"].values,
        "score_isolation_forest": classical["score_isolation_forest"].values,
        "score_lof": classical["score_lof"].values,
        "score_ocsvm": classical["score_ocsvm"].values,
        "score_elliptic_envelope": classical["score_elliptic_envelope"].values,
        "score_dbscan": classical["score_dbscan"].values,
        "score_hdbscan": classical["score_hdbscan"].values,
        "score_kmeans": classical["score_kmeans"].values,
        "score_gmm": classical["score_gmm"].values,
        "hybrid_vote_count": hybrid_votes,
        "flag_isolation_forest": classical["flag_isolation_forest_native"].values,
        "flag_lof": classical["flag_lof_native"].values,
        "flag_ocsvm": classical["flag_ocsvm_native"].values,
        "flag_elliptic_envelope": classical["flag_elliptic_envelope_native"].values,
        "flag_dbscan": classical["flag_dbscan_native"].values,
        "flag_hdbscan": classical["flag_hdbscan_native"].values,
        "flag_kmeans": classical["flag_kmeans_top5pct"].values,
        "flag_gmm": classical["flag_gmm_top5pct"].values,
        "flag_hybrid_ensemble": hybrid_flag,
    })
    all_scores.to_csv(os.path.join(ARTIFACTS_V2_DIR, "model_scores_all.csv"), index=False)
    print(f"\nSaved: {os.path.join(ARTIFACTS_V2_DIR, 'model_scores_all.csv')}")

    model_names = ["isolation_forest", "lof", "ocsvm", "elliptic_envelope", "dbscan",
                    "hdbscan", "kmeans", "gmm", "hybrid_ensemble"]
    score_cols = {n: (f"score_{n}" if n != "hybrid_ensemble" else "hybrid_vote_count") for n in model_names}
    flag_cols = {n: f"flag_{n}" for n in model_names}

    n_models = len(model_names)
    spearman_mat = np.full((n_models, n_models), np.nan)
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            s1 = all_scores[score_cols[m1]].values
            s2 = all_scores[score_cols[m2]].values
            rho, _ = spearmanr(s1, s2)
            spearman_mat[i, j] = rho

    spearman_df = pd.DataFrame(spearman_mat, index=model_names, columns=model_names)
    spearman_df.to_csv(os.path.join(ARTIFACTS_V2_DIR, "model_pairwise_spearman.csv"))

    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(spearman_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Spearman rho"}, vmin=-1, vmax=1)
    ax.set_title("Pairwise Spearman Rank Correlation Between All 9 Models' Anomaly Scores (v2)")
    savefig(fig, "model_pairwise_spearman_heatmap_v2.png")

    jaccard_mat = np.full((n_models, n_models), np.nan)
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            f1 = all_scores[flag_cols[m1]].values.astype(bool)
            f2 = all_scores[flag_cols[m2]].values.astype(bool)
            union = (f1 | f2).sum()
            inter = (f1 & f2).sum()
            jaccard_mat[i, j] = inter / union if union > 0 else np.nan

    jaccard_df = pd.DataFrame(jaccard_mat, index=model_names, columns=model_names)
    jaccard_df.to_csv(os.path.join(ARTIFACTS_V2_DIR, "model_pairwise_jaccard.csv"))

    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(jaccard_df, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": "Jaccard overlap"})
    ax.set_title("Pairwise Jaccard Overlap on Each Model's Flagged Set (v2)")
    savefig(fig, "model_pairwise_jaccard_heatmap_v2.png")

    rates = {n: all_scores[flag_cols[n]].mean() for n in model_names}
    fig, ax = plt.subplots(figsize=(9, 5))
    names_sorted = sorted(rates, key=lambda n: rates[n])
    ax.barh(names_sorted, [rates[n] * 100 for n in names_sorted], color="#2F6690")
    ax.set_xlabel("Flagged rate (%)")
    ax.set_title("Anomaly / Flagged Rate by Model, v2 (native contamination where defined, else top-5%)")
    ax.axvline(5.0, color="#D1495B", ls="--", lw=1, label="5% reference line")
    ax.legend()
    savefig(fig, "model_anomaly_rate_comparison_v2.png")

    with open(os.path.join(ARTIFACTS_V2_DIR, "model_comparison_summary.json"), "w") as f:
        json.dump({
            "anomaly_rates_pct": {n: round(float(rates[n] * 100), 2) for n in model_names},
        }, f, indent=2, default=float)

    print("\n=== Cross-model comparison summary (v2) ===")
    print("Anomaly rates (%):", json.dumps({n: round(rates[n]*100, 2) for n in model_names}, indent=2))
    print(f"\nSaved: model_scores_all.csv, model_pairwise_spearman.csv, model_pairwise_jaccard.csv, "
          f"model_comparison_summary.json (all in {ARTIFACTS_V2_DIR})")


if __name__ == "__main__":
    main()
