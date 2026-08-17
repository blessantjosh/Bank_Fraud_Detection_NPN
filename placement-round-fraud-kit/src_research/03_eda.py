"""
Phase 4 -- Exploratory Data Analysis.

Univariate (hist+KDE+boxplot per numeric feature), bivariate (correlation
matrices, mutual information, numeric-vs-categorical boxplots, chi-square on
Channel x TransactionType), and multivariate (VIF, PCA, t-SNE, UMAP-if-available)
analysis. All plots saved to research/plots/, all numeric outputs saved to
artifacts_research/ as CSV/JSON.
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import (
    ARTIFACTS_RESEARCH_DIR,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    RANDOM_STATE,
    load_raw,
)

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})


def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


# ---------------------------------------------------------------- univariate
def univariate_plots(df: pd.DataFrame):
    for col in NUMERIC_FEATURES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2, 1]})
        sns.histplot(df[col], kde=True, ax=axes[0], color="#2F6690", edgecolor="white")
        axes[0].set_title(f"{col} -- Distribution (hist + KDE)")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")

        sns.boxplot(y=df[col], ax=axes[1], color="#81C3D7", fliersize=3)
        axes[1].set_title(f"{col} -- Boxplot")
        axes[1].set_ylabel(col)

        fig.suptitle(f"Univariate Analysis: {col}", fontsize=13, fontweight="bold", y=1.03)
        savefig(fig, f"univariate_{col}.png")


# ----------------------------------------------------------------- bivariate
def correlation_and_mi(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES]
    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")

    pearson.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "correlation_pearson.csv"))
    spearman.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "correlation_spearman.csv"))
    # keep the name the task spec calls out explicitly
    pearson.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "correlation_matrix.csv"))

    # mutual information matrix (symmetrized average of MI(x->y), MI(y->x))
    n = len(NUMERIC_FEATURES)
    mi_mat = np.zeros((n, n))
    Xs = StandardScaler().fit_transform(X)
    for i, col_i in enumerate(NUMERIC_FEATURES):
        target = Xs[:, i]
        others = np.delete(Xs, i, axis=1)
        mi_vals = mutual_info_regression(others, target, random_state=RANDOM_STATE)
        j_idx = [j for j in range(n) if j != i]
        for k, j in enumerate(j_idx):
            mi_mat[i, j] += mi_vals[k]
    mi_mat = (mi_mat + mi_mat.T) / 2
    np.fill_diagonal(mi_mat, np.nan)
    mi_df = pd.DataFrame(mi_mat, index=NUMERIC_FEATURES, columns=NUMERIC_FEATURES)
    mi_df.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "mutual_information_matrix.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.heatmap(pearson, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[0], square=True, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Pearson Correlation")
    sns.heatmap(spearman, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[1], square=True, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Spearman Correlation")
    fig.suptitle("Numeric Feature Correlation Matrices", fontsize=13, fontweight="bold", y=1.04)
    savefig(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(mi_df, annot=True, fmt=".3f", cmap="viridis", ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Mutual Information (standardized features)")
    savefig(fig, "mutual_information_heatmap.png")

    return pearson, spearman, mi_df


def numeric_vs_categorical(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Channel", y="TransactionAmount", ax=ax, palette="Blues")
    ax.set_title("TransactionAmount by Channel")
    savefig(fig, "boxplot_amount_by_channel.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x="TransactionType", y="TransactionAmount", ax=ax, palette="Blues")
    ax.set_title("TransactionAmount by TransactionType")
    savefig(fig, "boxplot_amount_by_transactiontype.png")

    # group stats used in the write-up
    by_channel = df.groupby("Channel")["TransactionAmount"].agg(["count", "mean", "median", "std"])
    by_type = df.groupby("TransactionType")["TransactionAmount"].agg(["count", "mean", "median", "std"])
    by_channel.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "amount_by_channel.csv"))
    by_type.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "amount_by_transactiontype.csv"))
    return by_channel, by_type


def categorical_vs_categorical(df: pd.DataFrame):
    ct = pd.crosstab(df["Channel"], df["TransactionType"])
    chi2, p, dof, expected = chi2_contingency(ct)
    ct.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "crosstab_channel_transactiontype.csv"))
    result = {"chi2_statistic": float(chi2), "p_value": float(p), "dof": int(dof)}
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "chisq_channel_transactiontype.json"), "w") as f:
        json.dump({"crosstab": ct.to_dict(), **result}, f, indent=2)
    return ct, result


# --------------------------------------------------------------- multivariate
def vif_analysis(df: pd.DataFrame):
    """Manual VIF via R^2 of each feature regressed on the rest (statsmodels
    not installed in this environment; sklearn LinearRegression gives an
    identical VIF = 1 / (1 - R^2) result)."""
    X = StandardScaler().fit_transform(df[NUMERIC_FEATURES])
    rows = []
    for i, col in enumerate(NUMERIC_FEATURES):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(others, y)
        r2 = reg.score(others, y)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        rows.append({"feature": col, "R2_vs_other_features": round(r2, 4), "VIF": round(vif, 3)})
    vif_df = pd.DataFrame(rows).set_index("feature")
    vif_df.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "vif.csv"))
    return vif_df


def pca_analysis(df: pd.DataFrame):
    X = StandardScaler().fit_transform(df[NUMERIC_FEATURES])
    pca = PCA(n_components=len(NUMERIC_FEATURES), random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X)

    var_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(NUMERIC_FEATURES))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    var_df.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "pca_explained_variance.csv"), index=False)

    loadings = pd.DataFrame(pca.components_.T, index=NUMERIC_FEATURES,
                             columns=[f"PC{i+1}" for i in range(len(NUMERIC_FEATURES))])
    loadings.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "pca_loadings.csv"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(var_df["component"], var_df["explained_variance_ratio"], color="#2F6690", label="Individual")
    ax.plot(var_df["component"], var_df["cumulative_variance_ratio"], color="#D1495B", marker="o", label="Cumulative")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Scree Plot -- 5 Standardized Numeric Features")
    ax.legend()
    savefig(fig, "pca_scree.png")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], s=12, alpha=0.55, c="#2F6690", edgecolor="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA -- 2D Projection of Numeric Features")
    savefig(fig, "pca_2d_scatter.png")

    return var_df, loadings, pcs


def tsne_analysis(df: pd.DataFrame):
    X = StandardScaler().fit_transform(df[NUMERIC_FEATURES])
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(emb[:, 0], emb[:, 1], s=12, alpha=0.55, c="#D1495B", edgecolor="none")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("t-SNE -- 2D Projection of Numeric Features")
    savefig(fig, "tsne_2d_scatter.png")

    np.save(os.path.join(ARTIFACTS_RESEARCH_DIR, "tsne_embedding.npy"), emb)
    return emb


def umap_analysis(df: pd.DataFrame):
    try:
        import umap
    except ImportError:
        print("umap-learn not installed in this environment -- skipping UMAP, PCA/t-SNE used instead.")
        return None

    X = StandardScaler().fit_transform(df[NUMERIC_FEATURES])
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(emb[:, 0], emb[:, 1], s=12, alpha=0.55, c="#4C956C", edgecolor="none")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.set_title("UMAP -- 2D Projection of Numeric Features")
    savefig(fig, "umap_2d_scatter.png")
    return emb


def main():
    df = load_raw()

    print("=== Univariate plots ===")
    univariate_plots(df)

    print("\n=== Bivariate: correlation + mutual information ===")
    pearson, spearman, mi_df = correlation_and_mi(df)
    print("Pearson:\n", pearson.round(3).to_string())
    print("Spearman:\n", spearman.round(3).to_string())
    print("Mutual information:\n", mi_df.round(3).to_string())

    print("\n=== Bivariate: numeric vs categorical ===")
    by_channel, by_type = numeric_vs_categorical(df)
    print("TransactionAmount by Channel:\n", by_channel.round(2).to_string())
    print("TransactionAmount by TransactionType:\n", by_type.round(2).to_string())

    print("\n=== Bivariate: categorical vs categorical (chi-square) ===")
    ct, chi_result = categorical_vs_categorical(df)
    print(ct.to_string())
    print(json.dumps(chi_result, indent=2))

    print("\n=== Multivariate: VIF ===")
    vif_df = vif_analysis(df)
    print(vif_df.to_string())

    print("\n=== Multivariate: PCA ===")
    var_df, loadings, pcs = pca_analysis(df)
    print(var_df.to_string(index=False))
    print("Loadings:\n", loadings.round(3).to_string())

    print("\n=== Multivariate: t-SNE ===")
    tsne_analysis(df)

    print("\n=== Multivariate: UMAP (optional) ===")
    umap_analysis(df)

    print("\nAll EDA artifacts written to:", ARTIFACTS_RESEARCH_DIR)
    print("All EDA plots written to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
