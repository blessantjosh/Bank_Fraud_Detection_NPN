"""
Phase 8 -- Model Development, Part 2 (Model 9): the Hybrid Ensemble
detector, plus the cross-model comparison across all 9 models built in this
phase (07_models_classical.py's 8 classical detectors + this file's Hybrid
Ensemble).

Model 9 -- Hybrid Ensemble: Isolation Forest + LOF + GMM, simple majority
           vote (>=2 of 3), following the same pattern as v1's 4-detector
           artifacts/anomaly_votes.csv but implemented fresh on this richer
           46-feature set with these 3 detectors, per the task spec (not a
           copy of v1's exact 4). GMM stands in as the third voter using its
           top-5%-by-anomaly-score flag: GMM has no native contamination-
           based .predict flag the way IsolationForest/LOF/OneClassSVM/
           EllipticEnvelope do, so the top-5%-of-score convention already
           used elsewhere in this phase is reused here too (this is exactly
           `flag_gmm_top5pct` from model_scores_classical.csv, Model 8).

Note: this file previously (pre deep-learning removal) also trained an
Autoencoder, VAE, and LSTM Autoencoder here as Models 9-11, with the Hybrid
Ensemble as Model 12 (IF + LOF + Autoencoder majority vote). Those three
deep-learning models were removed from this pipeline -- see the project
decision log -- and the Hybrid Ensemble's third voter was switched from the
Autoencoder to GMM as a direct consequence. All artifacts, plots, and
downstream scores in this phase were regenerated from scratch after the
removal, not hand-edited.
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
from config_research import ARTIFACTS_RESEARCH_DIR, PLOTS_DIR, ROOT_DIR

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

FEATURES_V2_CSV = os.path.join(ARTIFACTS_RESEARCH_DIR, "features_v2.csv")

MODEL_NAMES = ["isolation_forest", "lof", "ocsvm", "elliptic_envelope", "dbscan",
               "hdbscan", "kmeans", "gmm", "hybrid_ensemble"]


def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
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
        "components": "Isolation Forest (Model 1, native ~5% contamination flag) + "
                       "LOF (Model 2, native ~5% contamination flag) + GMM "
                       "(Model 8, top-5% negative-log-likelihood anomaly-score flag)",
        "rule": "flagged if >=2 of the 3 component flags fire (majority vote)",
        "vote_distribution": {int(k): int(v) for k, v in
                               pd.Series(vote_count).value_counts().sort_index().items()},
        "majority_flagged_rate": round(float(flag_majority.mean()), 4),
        "pairwise_agreement_if_lof": round(float((flag_if == flag_lof).mean()), 4),
        "pairwise_agreement_if_gmm": round(float((flag_if == flag_gmm).mean()), 4),
        "pairwise_agreement_lof_gmm": round(float((flag_lof == flag_gmm).mean()), 4),
        "cost_note": ("Cost = sum of its 3 components' costs at inference time (one tree "
                      "ensemble, one k-NN lookup, one Gaussian-mixture likelihood evaluation) -- "
                      "cheap in absolute terms at this data scale, and the majority-vote rule is "
                      "the cheapest possible combination step; the practical production cost is "
                      "maintaining and monitoring 3 models instead of 1, not raw compute."),
    }
    print("\n=== Model 9: Hybrid Ensemble (IF + LOF + GMM, majority vote) ===")
    print(json.dumps(summary, indent=2, default=float))
    return vote_count, flag_majority, summary


def main():
    df = pd.read_csv(FEATURES_V2_CSV)
    classical = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_scores_classical.csv"))
    assert (classical["TransactionID"].values == df["TransactionID"].values).all()

    # Hybrid ensemble uses IF/LOF's *native* contamination flags (already ~5%)
    # plus GMM's top-5% flag, all loaded from Model 1/2/8 above.
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
    all_scores.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_scores_all.csv"), index=False)
    print(f"\nSaved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'model_scores_all.csv')}")

    # ---------------- cross-model comparison: Spearman + Jaccard on top-5% ----------------
    score_cols = {n: (f"score_{n}" if n != "hybrid_ensemble" else "hybrid_vote_count") for n in MODEL_NAMES}
    flag_cols = {n: f"flag_{n}" for n in MODEL_NAMES}

    n_models = len(MODEL_NAMES)
    spearman_mat = np.full((n_models, n_models), np.nan)
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            s1 = all_scores[score_cols[m1]].values
            s2 = all_scores[score_cols[m2]].values
            rho, _ = spearmanr(s1, s2)
            spearman_mat[i, j] = rho

    spearman_df = pd.DataFrame(spearman_mat, index=MODEL_NAMES, columns=MODEL_NAMES)
    spearman_df.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_pairwise_spearman.csv"))

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(spearman_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Spearman rho"}, vmin=-1, vmax=1)
    ax.set_title("Pairwise Spearman Rank Correlation Between All 9 Models' Anomaly Scores")
    savefig(fig, "model_pairwise_spearman_heatmap.png")

    # Jaccard overlap on top-5%-flagged sets (native flags where available, else top-5% score-based)
    jaccard_mat = np.full((n_models, n_models), np.nan)
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            f1 = all_scores[flag_cols[m1]].values.astype(bool)
            f2 = all_scores[flag_cols[m2]].values.astype(bool)
            union = (f1 | f2).sum()
            inter = (f1 & f2).sum()
            jaccard_mat[i, j] = inter / union if union > 0 else np.nan

    jaccard_df = pd.DataFrame(jaccard_mat, index=MODEL_NAMES, columns=MODEL_NAMES)
    jaccard_df.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_pairwise_jaccard.csv"))

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(jaccard_df, annot=True, fmt=".2f", cmap="viridis", ax=ax,
                cbar_kws={"label": "Jaccard overlap"})
    ax.set_title("Pairwise Jaccard Overlap on Each Model's Flagged Set")
    savefig(fig, "model_pairwise_jaccard_heatmap.png")

    # anomaly rate bar chart
    rates = {n: all_scores[flag_cols[n]].mean() for n in MODEL_NAMES}
    fig, ax = plt.subplots(figsize=(9, 5))
    names_sorted = sorted(rates, key=lambda n: rates[n])
    ax.barh(names_sorted, [rates[n] * 100 for n in names_sorted], color="#2F6690")
    ax.set_xlabel("Flagged rate (%)")
    ax.set_title("Anomaly / Flagged Rate by Model (native contamination where defined, else top-5%)")
    ax.axvline(5.0, color="#D1495B", ls="--", lw=1, label="5% reference line")
    ax.legend()
    savefig(fig, "model_anomaly_rate_comparison.png")

    # rough, explicitly-caveated cross-check against v1's anomaly_votes.csv vote_count
    votes = pd.read_csv(os.path.join(ROOT_DIR, "artifacts", "anomaly_votes.csv"))
    assert len(votes) == len(df)
    vote_count_v1 = votes["vote_count"].values.astype(float)
    v1_crosscheck = {}
    for n in MODEL_NAMES:
        s = all_scores[score_cols[n]].values
        rho, _ = spearmanr(s, vote_count_v1)
        v1_crosscheck[n] = round(float(rho), 4)

    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_comparison_summary.json"), "w") as f:
        json.dump({
            "anomaly_rates_pct": {n: round(float(rates[n] * 100), 2) for n in MODEL_NAMES},
            "spearman_vs_v1_anomaly_votes_ROUGH_PROXY_NOT_GROUND_TRUTH": v1_crosscheck,
            "note": ("v1's anomaly_votes.csv vote_count (0-4) is a rough, weak proxy from a "
                      "different, smaller 4-detector v1 pipeline on a different (18-column, "
                      "pre-scaled) feature set -- used here only as a directionally-informative "
                      "cross-check, never as ground truth, since no fraud label exists anywhere "
                      "in this project."),
        }, f, indent=2, default=float)

    print("\n=== Cross-model comparison summary ===")
    print("Anomaly rates (%):", json.dumps({n: round(rates[n]*100, 2) for n in MODEL_NAMES}, indent=2))
    print("Spearman vs v1 vote_count (rough proxy):", json.dumps(v1_crosscheck, indent=2))
    print(f"\nSaved: model_scores_all.csv, model_pairwise_spearman.csv, model_pairwise_jaccard.csv, "
          f"model_comparison_summary.json (all in {ARTIFACTS_RESEARCH_DIR})")


if __name__ == "__main__":
    main()
