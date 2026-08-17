"""
Phase 7 -- Dimensionality Reduction.

1. PCA: explained variance was already computed in Phase 4
   (artifacts_research/pca_explained_variance.csv) -- referenced here, not
   recomputed.
2. UMAP: now that umap-learn is installed, run it on the same 5 standardized
   numeric features used for the Phase 4 PCA/t-SNE plots (apples-to-apples
   comparison), plus a second run on the full Phase 5/6 engineered feature
   set for the modeling handoff.
3. Autoencoder: PyTorch Dense(16)->Dense(8)->bottleneck(4)->Dense(8)->
   Dense(16), trained with a proper train/validation split on
   RobustScaler-scaled engineered features (see Phase 6 for why
   RobustScaler). Reusable architecture/training code lives in
   autoencoder_utils.py; weights saved to artifacts_research/autoencoder.pt
   for reuse as "Model 9" in the modeling phase.
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
from autoencoder_utils import reconstruction_errors, train_autoencoder

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


# ------------------------------------------------------------ autoencoder
def run_autoencoder(df):
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    X = df[feature_cols].astype(float).values

    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = RobustScaler().fit(X[idx_train])
    X_train = scaler.transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])
    X_all = scaler.transform(X)

    print(f"Training autoencoder: input_dim={X.shape[1]}, train n={len(idx_train)}, val n={len(idx_val)}")
    model, history = train_autoencoder(
        X_train, X_val, bottleneck_dim=4, epochs=200, lr=1e-3, batch_size=64, random_state=RANDOM_STATE
    )

    val_mse, val_mae, val_bottleneck, _ = reconstruction_errors(model, X_val)
    train_mse, train_mae, _, _ = reconstruction_errors(model, X_train)
    all_mse, all_mae, all_bottleneck, _ = reconstruction_errors(model, X_all)

    final_metrics = {
        "input_dim": X.shape[1],
        "bottleneck_dim": 4,
        "epochs": 200,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "train_mse_mean": round(float(train_mse.mean()), 6),
        "train_mae_mean": round(float(train_mae.mean()), 6),
        "val_mse_mean": round(float(val_mse.mean()), 6),
        "val_mae_mean": round(float(val_mae.mean()), 6),
        "val_mse_p95": round(float(np.percentile(val_mse, 95)), 6),
        "val_mse_p99": round(float(np.percentile(val_mse, 99)), 6),
        "val_mse_max": round(float(val_mse.max()), 6),
    }
    print(json.dumps(final_metrics, indent=2))

    # ---- save artifacts for reuse as "Model 9" ----
    import torch
    torch.save(model.state_dict(), os.path.join(ARTIFACTS_RESEARCH_DIR, "autoencoder.pt"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_RESEARCH_DIR, "autoencoder_scaler.pkl"))
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "autoencoder_config.json"), "w") as f:
        json.dump({
            "architecture": "input -> 16 -> 8 -> bottleneck(4) -> 8 -> 16 -> output",
            "feature_cols": feature_cols,
            **final_metrics,
        }, f, indent=2)

    errors_df = pd.DataFrame({
        "TransactionID": df["TransactionID"].values,
        "AccountID": df["AccountID"].values,
        "reconstruction_mse": all_mse,
        "reconstruction_mae": all_mae,
        "split": np.where(np.isin(np.arange(len(df)), idx_train), "train", "val"),
    })
    errors_df.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "autoencoder_reconstruction_errors.csv"), index=False)

    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "autoencoder_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---- bottleneck visualization: bottleneck_dim=4 -> project to 2D via PCA ----
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    bottleneck_2d = pca2.fit_transform(all_bottleneck)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sc = ax.scatter(bottleneck_2d[:, 0], bottleneck_2d[:, 1], s=12, alpha=0.6,
                     c=all_mse, cmap="magma")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Reconstruction MSE")
    ax.set_xlabel(f"Bottleneck PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% of bottleneck variance)")
    ax.set_ylabel(f"Bottleneck PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% of bottleneck variance)")
    ax.set_title("Autoencoder Bottleneck (4D, PCA-projected to 2D), Colored by Reconstruction Error")
    savefig(fig, "autoencoder_bottleneck_2d.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["train_mse"], label="Train MSE", color="#2F6690")
    ax.plot(history["val_mse"], label="Validation MSE", color="#D1495B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Autoencoder Training Curve")
    ax.legend()
    savefig(fig, "autoencoder_training_curve.png")

    return final_metrics


def main():
    df = pd.read_csv(FEATURES_V2_CSV)

    print("=== Phase 7.1: PCA (referenced from Phase 4) ===")
    reference_pca()

    print("\n=== Phase 7.2: UMAP ===")
    umap_projection(df)

    print("\n=== Phase 7.3: Autoencoder ===")
    run_autoencoder(df)


if __name__ == "__main__":
    main()
