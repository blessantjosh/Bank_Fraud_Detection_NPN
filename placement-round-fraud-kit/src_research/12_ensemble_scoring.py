"""
Phase 12 -- Ensemble Anomaly Scoring.

Combines 11 of the 12 Phase 8 models' scores into 4 unsupervised ensemble
strategies. The Hybrid Ensemble (Model 12) is deliberately EXCLUDED from the
inputs to these new ensembles -- it is itself already a majority vote of
Isolation Forest + LOF + Autoencoder (Phase 8 Section 2.12), so folding it
back in as a 12th "model" would double-count those three detectors relative
to the other 8. This is the "well-justified subset" the task brief allows
for, stated explicitly rather than silently done. The Hybrid Ensemble's
vote_count is still carried along as a comparison point, not an input.

The 11 models combined: isolation_forest, lof, ocsvm, elliptic_envelope,
dbscan, hdbscan, kmeans, gmm, autoencoder, vae, lstm_ae.

LSTM-AE is NaN for 110/2,512 rows (accounts with <3 transactions, Phase 8
Section 2.11) -- every aggregation strategy below is NaN-aware: a row
missing the LSTM-AE score is combined using only the 10 other models for
that row, not silently imputed with a fabricated value (except where noted
explicitly for the PCA stacking proxy, which requires a complete matrix).

Four strategies:
  1. Weighted average -- weights = inverse mean disagreement with the other
     10 models (Spearman rho from Phase 8's model_pairwise_spearman.csv,
     mapped from [-1,1] to a [0,1] disagreement scale), applied to each
     model's own z-score-standardized score.
  2. Rank aggregation (Borda count) -- each model's raw ordinal rank
     (1 = least anomalous .. N = most anomalous), missing LSTM-AE ranks
     imputed with the neutral/median rank (documented, not silently done),
     summed across all 11 models.
  3. Percentile aggregation -- each model's score converted to its own
     empirical percentile (0-1), missing values SKIPPED (not imputed) and
     averaged only over the models actually available for that row.
  4. Stacking proxy (NOT supervised) -- PCA's first principal component
     across the 11 z-scored score columns (LSTM-AE's 110 missing rows
     zero-imputed, i.e. "no information," documented explicitly since PCA
     requires a complete matrix), oriented so higher = more anomalous.
     Explicitly labeled as an unsupervised consensus-axis proxy, not
     supervised stacking, since no label exists anywhere in this project to
     stack a meta-learner against.

Outputs: artifacts_research/{ensemble_scores.csv, ensemble_weights.json,
ensemble_pairwise_comparison.csv, ensemble_vs_v1_crosscheck.json}.
Plots: research/plots/{ensemble_weights_barplot.png,
ensemble_pairwise_spearman_heatmap.png, ensemble_pairwise_jaccard_heatmap.png,
ensemble_score_distributions.png}.
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
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import ARTIFACTS_RESEARCH_DIR, PLOTS_DIR, ROOT_DIR

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

TOP_PCT = 0.05
BASE_MODELS = ["isolation_forest", "lof", "ocsvm", "elliptic_envelope", "dbscan",
               "hdbscan", "kmeans", "gmm", "autoencoder", "vae", "lstm_ae"]


def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def top_pct_flag(score, pct=TOP_PCT):
    valid = ~np.isnan(score)
    flag = np.zeros(len(score), dtype=int)
    thresh = np.percentile(score[valid], 100 * (1 - pct))
    flag[valid] = (score[valid] >= thresh).astype(int)
    return flag


def load_data():
    scores = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_scores_all.csv"))
    lstm_applicable = (scores["lstm_ae_applicable"] == 1).values
    S = np.full((len(scores), len(BASE_MODELS)), np.nan)
    for j, m in enumerate(BASE_MODELS):
        col = scores[f"score_{m}"].values.astype(float)
        if m == "lstm_ae":
            col = col.copy()
            col[~lstm_applicable] = np.nan
        S[:, j] = col
    return scores, S, lstm_applicable


def zscore_columns(S):
    Z = np.full_like(S, np.nan)
    for j in range(S.shape[1]):
        col = S[:, j]
        valid = ~np.isnan(col)
        mu, sd = col[valid].mean(), col[valid].std()
        Z[valid, j] = (col[valid] - mu) / (sd if sd > 1e-12 else 1.0)
    return Z


# -------------------------------------------------------- Strategy 1: Weighted average
def compute_consensus_weights():
    spearman = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_pairwise_spearman.csv"), index_col=0)
    sub = spearman.loc[BASE_MODELS, BASE_MODELS]
    weights = {}
    disagreements = {}
    for m in BASE_MODELS:
        others = [o for o in BASE_MODELS if o != m]
        rho_vals = sub.loc[m, others].values
        # map rho in [-1, 1] to disagreement in [0, 1]: rho=1 -> 0 disagreement, rho=-1 -> 1
        disagreement = np.mean((1 - rho_vals) / 2)
        disagreements[m] = float(disagreement)
        weights[m] = 1.0 / (disagreement + 0.05)  # small epsilon floors the weight, avoids /~0 blowup
    total = sum(weights.values())
    weights = {m: w / total for m, w in weights.items()}
    return weights, disagreements


def weighted_average(Z, weights):
    w = np.array([weights[m] for m in BASE_MODELS])
    n = Z.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        valid = ~np.isnan(Z[i])
        if not valid.any():
            continue
        w_i = w[valid]
        out[i] = np.sum(Z[i, valid] * w_i) / np.sum(w_i)
    return out


# -------------------------------------------------------- Strategy 2: Rank aggregation (Borda)
def borda_rank_aggregation(S):
    n, k = S.shape
    borda_total = np.zeros(n)
    for j in range(k):
        col = S[:, j]
        valid = ~np.isnan(col)
        n_valid = valid.sum()
        ranks = np.full(n, np.nan)
        ranks[valid] = rankdata(col[valid], method="average")  # 1 = least anomalous
        # neutral/median rank imputation for missing rows -- documented, not silent:
        # a row missing this model's score contributes neither support for nor
        # against being anomalous from this model, so it gets that model's own
        # median rank rather than 0 (which would wrongly say "definitely not
        # anomalous by this model") or n_valid (which would wrongly say
        # "definitely anomalous by this model").
        median_rank = (n_valid + 1) / 2.0
        ranks[~valid] = median_rank
        borda_total += ranks
    return borda_total


# -------------------------------------------------------- Strategy 3: Percentile aggregation
def percentile_aggregation(S):
    n, k = S.shape
    P = np.full((n, k), np.nan)
    for j in range(k):
        col = S[:, j]
        valid = ~np.isnan(col)
        n_valid = valid.sum()
        ranks = np.full(n, np.nan)
        ranks[valid] = rankdata(col[valid], method="average")
        P[valid, j] = (ranks[valid] - 0.5) / n_valid  # empirical CDF, in (0, 1)
    out = np.full(n, np.nan)
    for i in range(n):
        valid = ~np.isnan(P[i])
        if not valid.any():
            continue
        out[i] = np.mean(P[i, valid])  # missing models SKIPPED, not imputed
    return out


# -------------------------------------------------------- Strategy 4: PCA stacking proxy
def pca_stacking(Z, consensus_ref):
    Z_imputed = np.where(np.isnan(Z), 0.0, Z)  # documented: missing LSTM-AE rows get
    # z=0 ("no information from this model"), required since PCA needs a complete matrix
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(Z_imputed).ravel()
    rho, _ = spearmanr(pc1, consensus_ref)
    if rho < 0:
        pc1 = -pc1
    explained_var = float(pca.explained_variance_ratio_[0])
    return pc1, explained_var


def main():
    print("=== Phase 12: Ensemble Anomaly Scoring ===")
    scores, S, lstm_applicable = load_data()
    Z = zscore_columns(S)

    print("\n--- Strategy 1: Weighted average (consensus-weighted) ---")
    weights, disagreements = compute_consensus_weights()
    for m in BASE_MODELS:
        print(f"  {m:20s} disagreement={disagreements[m]:.4f}  weight={weights[m]:.4f}")
    ens_weighted = weighted_average(Z, weights)

    print("\n--- Strategy 2: Rank aggregation (Borda count) ---")
    ens_borda = borda_rank_aggregation(S)

    print("\n--- Strategy 3: Percentile aggregation ---")
    ens_percentile = percentile_aggregation(S)

    print("\n--- Strategy 4: Stacking proxy (PCA, NOT supervised) ---")
    ens_pca, explained_var = pca_stacking(Z, ens_percentile)
    print(f"  PC1 explained variance ratio: {explained_var:.4f}")

    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "ensemble_weights.json"), "w") as f:
        json.dump({
            "weights": weights, "disagreements": disagreements,
            "pca_explained_variance_ratio_pc1": round(explained_var, 4),
            "note": ("Weights = 1 / (mean pairwise disagreement + 0.05), normalized to sum to 1; "
                     "disagreement_m = mean over the other 10 models of (1 - Spearman rho)/2, mapping "
                     "rho in [-1,1] to a [0,1] disagreement scale. Source: "
                     "artifacts_research/model_pairwise_spearman.csv (Phase 8, not recomputed)."),
        }, f, indent=2, default=float)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'ensemble_weights.json')}")

    out = pd.DataFrame({
        "TransactionID": scores["TransactionID"].values,
        "AccountID": scores["AccountID"].values,
        "ensemble_weighted_average": ens_weighted,
        "ensemble_rank_borda": ens_borda,
        "ensemble_percentile_average": ens_percentile,
        "ensemble_pca_stacking_proxy": ens_pca,
        "hybrid_vote_count_for_comparison": scores["hybrid_vote_count"].values,
    })
    out.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "ensemble_scores.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'ensemble_scores.csv')}")

    # ---------------- Compare the 4 strategies to each other ----------------
    strategy_cols = ["ensemble_weighted_average", "ensemble_rank_borda",
                      "ensemble_percentile_average", "ensemble_pca_stacking_proxy"]
    strategy_labels = ["Weighted Avg", "Rank (Borda)", "Percentile Avg", "PCA Stacking"]

    spearman_mat = np.full((4, 4), np.nan)
    for i, ci in enumerate(strategy_cols):
        for j, cj in enumerate(strategy_cols):
            rho, _ = spearmanr(out[ci], out[cj])
            spearman_mat[i, j] = rho
    spearman_df = pd.DataFrame(spearman_mat, index=strategy_labels, columns=strategy_labels)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(spearman_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"label": "Spearman rho"})
    ax.set_title("Pairwise Spearman Correlation -- 4 Ensemble Strategies")
    savefig(fig, "ensemble_pairwise_spearman_heatmap.png")

    flags = {}
    for c in strategy_cols:
        flags[c] = top_pct_flag(out[c].values)
    jaccard_mat = np.full((4, 4), np.nan)
    for i, ci in enumerate(strategy_cols):
        for j, cj in enumerate(strategy_cols):
            f1, f2 = flags[ci].astype(bool), flags[cj].astype(bool)
            union = (f1 | f2).sum()
            inter = (f1 & f2).sum()
            jaccard_mat[i, j] = inter / union if union > 0 else np.nan
    jaccard_df = pd.DataFrame(jaccard_mat, index=strategy_labels, columns=strategy_labels)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(jaccard_df, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={"label": "Jaccard overlap"})
    ax.set_title("Pairwise Jaccard Overlap on Top-5%-Flagged Set -- 4 Ensemble Strategies")
    savefig(fig, "ensemble_pairwise_jaccard_heatmap.png")

    # score distributions (rank-normalized to [0,1] for visual comparability across scales)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#2F6690", "#D1495B", "#4C956C", "#EDAE49"]
    for c, lbl, color in zip(strategy_cols, strategy_labels, colors):
        pct = (rankdata(out[c]) - 0.5) / len(out)
        ax.hist(pct, bins=40, alpha=0.5, label=lbl, color=color)
    ax.set_xlabel("Rank-normalized ensemble score (0-1, for visual comparability across scales)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of the 4 Ensemble Strategies (rank-normalized)")
    ax.legend()
    savefig(fig, "ensemble_score_distributions.png")

    # weights bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names_sorted = sorted(weights, key=lambda m: weights[m])
    ax.barh(names_sorted, [weights[n] for n in names_sorted], color="#2F6690")
    ax.set_xlabel("Consensus weight (normalized, sums to 1)")
    ax.set_title("Strategy 1 Weights: Inverse Mean Disagreement with the Other 10 Models")
    savefig(fig, "ensemble_weights_barplot.png")

    # ---------------- Cross-check against v1's vote_count and hybrid_vote_count ----------------
    votes_v1 = pd.read_csv(os.path.join(ROOT_DIR, "artifacts", "anomaly_votes.csv"))
    assert len(votes_v1) == len(out)
    v1_vote = votes_v1["vote_count"].values.astype(float)
    hybrid_vote = out["hybrid_vote_count_for_comparison"].values.astype(float)

    crosscheck = {}
    for c, lbl in zip(strategy_cols, strategy_labels):
        rho_v1, _ = spearmanr(out[c], v1_vote)
        rho_hybrid, _ = spearmanr(out[c], hybrid_vote)
        crosscheck[lbl] = {
            "spearman_vs_v1_vote_count_ROUGH_PROXY_NOT_GROUND_TRUTH": round(float(rho_v1), 4),
            "spearman_vs_hybrid_ensemble_vote_count": round(float(rho_hybrid), 4),
        }
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "ensemble_vs_v1_crosscheck.json"), "w") as f:
        json.dump(crosscheck, f, indent=2, default=float)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'ensemble_vs_v1_crosscheck.json')}")

    pairwise_summary = pd.DataFrame({
        "strategy_pair": [f"{strategy_labels[i]} vs {strategy_labels[j]}"
                          for i in range(4) for j in range(i + 1, 4)],
        "spearman": [spearman_mat[i, j] for i in range(4) for j in range(i + 1, 4)],
        "jaccard_top5pct": [jaccard_mat[i, j] for i in range(4) for j in range(i + 1, 4)],
    })
    pairwise_summary.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "ensemble_pairwise_comparison.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'ensemble_pairwise_comparison.csv')}")

    print("\n=== Cross-model summary ===")
    print(pairwise_summary.to_string())
    print(json.dumps(crosscheck, indent=2))
    print("\nPhase 12 complete.")


if __name__ == "__main__":
    main()
