"""
Phase 7 (v2) -- Dimensionality Reduction on the teammate's 18-feature matrix
(artifacts_research/features_teammate_merged.csv).

1. PCA: full 18-component explained-variance breakdown + scree plot
   (genuinely new computation here -- there is no Phase 4 PCA to reference,
   since Phase 2-4 were never re-run on this feature set).
2. UMAP: 2D projection of all 18 features.
3. t-SNE: 2D projection of all 18 features (perplexity=30, matching a
   standard default; the in-house Phase 4 t-SNE used the same).

All 18 columns are used for PCA/UMAP/t-SNE (not a 5-column subset) since,
unlike the in-house 46-column set, this is already the full, final feature
matrix -- there is no separate "raw numeric features" vs. "full engineered
set" distinction to draw here.

No autoencoder/deep-learning step here (removed): this pipeline's model
suite is now 8 classical unsupervised detectors + a Hybrid Ensemble
(IF + LOF + GMM, majority vote) -- see 06_models_classical.py and
07_models_deep.py. PCA/UMAP/t-SNE are independent of that change and are
kept as-is.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research_v2 import (
    ARTIFACTS_V2_DIR, FEATURE_COLS_V2, PLOTS_V2_DIR, RANDOM_STATE, load_features_v2,
)

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


# ------------------------------------------------------------------- PCA
def run_pca(df):
    X = df[FEATURE_COLS_V2].astype(float).values
    # features are already individually StandardScaler-scaled (verified in
    # 04_feature_verification.py), so PCA is run directly without an
    # additional scaling step -- re-scaling already-standardized columns
    # would be a no-op given each has mean~0/std~1 already.
    pca = PCA(n_components=len(FEATURE_COLS_V2), random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    var_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(var_ratio))],
        "explained_variance_ratio": var_ratio,
        "cumulative_variance_ratio": cum_var,
    })
    var_df.to_csv(os.path.join(ARTIFACTS_V2_DIR, "pca_explained_variance.csv"), index=False)

    loadings = pd.DataFrame(pca.components_.T, index=FEATURE_COLS_V2,
                             columns=[f"PC{i+1}" for i in range(len(var_ratio))])
    loadings.to_csv(os.path.join(ARTIFACTS_V2_DIR, "pca_loadings.csv"))

    n_for_80 = int(np.argmax(cum_var >= 0.80) + 1)
    n_for_90 = int(np.argmax(cum_var >= 0.90) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(range(1, len(var_ratio) + 1), var_ratio, marker="o", color="#2F6690")
    axes[0].set_xlabel("Principal component"); axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("PCA Scree Plot (18 Teammate Features)")
    axes[1].plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="#D1495B")
    axes[1].axhline(0.80, color="gray", ls="--", lw=0.8)
    axes[1].axhline(0.90, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("Number of components"); axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("PCA Cumulative Explained Variance")
    savefig(fig, "pca_scree_cumulative.png")

    print(f"PC1={var_ratio[0]*100:.2f}%  PC2={var_ratio[1]*100:.2f}%  PC3={var_ratio[2]*100:.2f}%")
    print(f"Components needed for 80% cumulative variance: {n_for_80}")
    print(f"Components needed for 90% cumulative variance: {n_for_90}")

    return {
        "var_ratio": var_ratio.tolist(),
        "cum_var": cum_var.tolist(),
        "n_components_for_80pct": n_for_80,
        "n_components_for_90pct": n_for_90,
    }


# ------------------------------------------------------------------ UMAP
def run_umap(df):
    import umap

    X = df[FEATURE_COLS_V2].astype(float).values
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(emb[:, 0], emb[:, 1], s=12, alpha=0.55, c="#4C956C", edgecolor="none")
    ax.set_xlabel("UMAP dimension 1"); ax.set_ylabel("UMAP dimension 2")
    ax.set_title("UMAP -- 2D Projection of the Teammate's 18 Features")
    savefig(fig, "umap_projection_v2.png")
    np.save(os.path.join(ARTIFACTS_V2_DIR, "umap_embedding_v2.npy"), emb)
    return emb


# ------------------------------------------------------------------ t-SNE
def run_tsne(df):
    X = df[FEATURE_COLS_V2].astype(float).values
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(emb[:, 0], emb[:, 1], s=12, alpha=0.55, c="#EDAE49", edgecolor="none")
    ax.set_xlabel("t-SNE dimension 1"); ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("t-SNE -- 2D Projection of the Teammate's 18 Features (perplexity=30)")
    savefig(fig, "tsne_projection_v2.png")
    np.save(os.path.join(ARTIFACTS_V2_DIR, "tsne_embedding_v2.npy"), emb)
    return emb


def main():
    df = load_features_v2()

    print("=== Phase 7.1: PCA ===")
    pca_summary = run_pca(df)

    print("\n=== Phase 7.2: UMAP ===")
    run_umap(df)

    print("\n=== Phase 7.3: t-SNE ===")
    run_tsne(df)

    with open(os.path.join(ARTIFACTS_V2_DIR, "phase7_dim_reduction_summary.json"), "w") as f:
        json.dump({"pca": pca_summary}, f, indent=2)
    print(f"\nSaved: {os.path.join(ARTIFACTS_V2_DIR, 'phase7_dim_reduction_summary.json')}")


if __name__ == "__main__":
    main()
