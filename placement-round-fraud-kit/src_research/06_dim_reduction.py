"""
Phase 7 -- Dimensionality Reduction.

1. PCA: explained variance was already computed in Phase 4
   (artifacts_research/pca_explained_variance.csv) -- referenced here, not
   recomputed.
2. UMAP: now that umap-learn is installed, run it on the same 5 standardized
   numeric features used for the Phase 4 PCA/t-SNE plots (apples-to-apples
   comparison), plus a second run on the full Phase 5/6 engineered feature
   set for the modeling handoff. This UMAP projection is computed directly on
   the raw/scaled feature matrix, independent of any learned latent space.

Note: this file previously also trained a PyTorch Autoencoder here (Dense(16)
->Dense(8)->bottleneck(4)->Dense(8)->Dense(16)) and saved it for reuse as
"Model 9" in the modeling phase. That Autoencoder (and the VAE / LSTM
Autoencoder trained downstream from it) has been removed from this pipeline
-- see the project decision log. UMAP above never depended on the
Autoencoder's latent space (it runs on the raw engineered feature matrix), so
it is unaffected and unchanged by that removal.
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import ARTIFACTS_RESEARCH_DIR, NUMERIC_FEATURES, PLOTS_DIR, RANDOM_STATE, ROOT_DIR

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

FEATURES_V2_CSV = os.path.join(ARTIFACTS_RESEARCH_DIR, "features_v2.csv")
ID_COLS = ["TransactionID", "AccountID"]


def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


# ------------------------------------------------------------------- PCA ref
def reference_pca():
    var_df = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "pca_explained_variance.csv"))
    print("Phase 4 PCA explained variance (referenced, not recomputed):")
    print(var_df.to_string(index=False))
    return var_df


# ------------------------------------------------------------------ UMAP
def umap_projection(df):
    import umap

    # (a) same 5 raw numeric features as Phase 4 PCA/t-SNE -- direct comparison
    X5 = StandardScaler().fit_transform(df[NUMERIC_FEATURES])
    reducer5 = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
    emb5 = reducer5.fit_transform(X5)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(emb5[:, 0], emb5[:, 1], s=12, alpha=0.55, c="#4C956C", edgecolor="none")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.set_title("UMAP -- 2D Projection of the Same 5 Numeric Features (Phase 4 PCA/t-SNE basis)")
    savefig(fig, "umap_projection.png")
    np.save(os.path.join(ARTIFACTS_RESEARCH_DIR, "umap_embedding_5feat.npy"), emb5)

    # (b) full Phase 5/6 engineered feature set -- for the modeling handoff
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    Xfull = RobustScaler().fit_transform(df[feature_cols].astype(float))
    reducer_full = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
    emb_full = reducer_full.fit_transform(Xfull)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(emb_full[:, 0], emb_full[:, 1], s=12, alpha=0.55, c="#2F6690", edgecolor="none")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.set_title("UMAP -- 2D Projection of the Full Engineered Feature Set (46 features)")
    savefig(fig, "umap_projection_full_features.png")
    np.save(os.path.join(ARTIFACTS_RESEARCH_DIR, "umap_embedding_full.npy"), emb_full)

    return emb5, emb_full


def main():
    df = pd.read_csv(FEATURES_V2_CSV)

    print("=== Phase 7.1: PCA (referenced from Phase 4) ===")
    reference_pca()

    print("\n=== Phase 7.2: UMAP ===")
    umap_projection(df)


if __name__ == "__main__":
    main()
