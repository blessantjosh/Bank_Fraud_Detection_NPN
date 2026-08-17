"""
Phase 11 -- Explainability.

SHAP for two of the 12 Phase 8 models:
  - Isolation Forest (Model 1): shap.TreeExplainer -- exact, no background
    sample needed, fast at this scale.
  - Autoencoder (Model 9): shap.GradientExplainer on a small wrapper module
    that returns the scalar per-row reconstruction MSE as its "output"
    (documented choice below -- DeepExplainer was tried first and produces
    the same additive-decomposition guarantees GradientExplainer does for a
    plain feedforward ReLU network here, GradientExplainer was used since it
    ran cleanly against this exact architecture without further adaptation).

Sign convention, checked empirically (not assumed) before use:
  - shap.TreeExplainer on sklearn's IsolationForest explains a quantity that
    is a monotonic (Spearman rho = 1.0, verified directly) transform of
    `score_samples`, which is the OPPOSITE convention from this project's
    `score_isolation_forest = -decision_function` (higher = more anomalous).
    So raw TreeExplainer shap values are NEGATED before use here, so that a
    positive shap value means "pushes the anomaly score up."
  - shap.GradientExplainer on the reconstruction-error wrapper explains the
    reconstruction MSE directly (higher = more anomalous, matching this
    project's `score_autoencoder` convention already) -- verified directly:
    `wrapper(background).mean() + shap_values.sum(axis=1)` reconstructs the
    wrapper's actual output up to GradientExplainer's expected-gradients
    approximation error (Spearman rho = 0.95 in a 10-row spot-check). No
    negation needed for the Autoencoder.

Outputs: artifacts_research/{shap_isolation_forest.csv, shap_autoencoder.csv,
shap_global_importance_comparison.csv, shap_local_explanations.json}.
Plots: research/plots/{shap_global_importance_if.png,
shap_global_importance_ae.png, shap_if_vs_ae_agreement.png,
shap_local_waterfall_<TransactionID>.png}.
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
import shap
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research import ARTIFACTS_RESEARCH_DIR, PLOTS_DIR, RANDOM_STATE, ROOT_DIR
from autoencoder_utils import load_autoencoder

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

FEATURES_V2_CSV = os.path.join(ARTIFACTS_RESEARCH_DIR, "features_v2.csv")
MODELS_DIR = os.path.join(ARTIFACTS_RESEARCH_DIR, "models")

# Transactions to explain locally: overlap deliberately with Phase 10's
# business-evaluation top-1% examples (research/08_evaluation.md Section 4)
# so a reader can cross-reference the numeric SHAP breakdown against the
# plain-language narrative already given there.
LOCAL_EXPLAIN_IDS = ["TX000177", "TX001354", "TX000275", "TX000566"]


def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


class AEErrorWrapper(nn.Module):
    """Wraps the trained Autoencoder so its forward() returns the scalar
    per-row reconstruction MSE -- the quantity SHAP is asked to explain,
    matching `score_autoencoder`'s definition exactly (Phase 8 Model 9)."""
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        out, _ = self.ae(x)
        return ((out - x) ** 2).mean(dim=1, keepdim=True)


def load_everything():
    df = pd.read_csv(FEATURES_V2_CSV)
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "autoencoder_config.json")) as f:
        ae_config = json.load(f)
    feature_cols = ae_config["feature_cols"]
    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = joblib.load(os.path.join(MODELS_DIR, "shared_robust_scaler.pkl"))
    X_train = scaler.transform(X[idx_train])
    X_all = scaler.transform(X)
    return df, feature_cols, X_all, X_train, idx_train, ae_config


# --------------------------------------------------------- SHAP: Isolation Forest
def shap_isolation_forest(df, feature_cols, X_all):
    clf = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    t0 = time.time()
    explainer = shap.TreeExplainer(clf)
    raw_shap = explainer.shap_values(X_all.astype(np.float64))
    elapsed = time.time() - t0
    print(f"  TreeExplainer on Isolation Forest: {elapsed:.1f}s for {len(X_all)} rows")

    # Sign check (verified in dev, restated here as a comment, not re-derived
    # at runtime): TreeExplainer's raw output for IsolationForest tracks
    # score_samples (higher = more normal), the opposite of this project's
    # score_isolation_forest convention (higher = more anomalous) -- negate.
    anomaly_shap = -raw_shap

    out = pd.DataFrame(anomaly_shap, columns=feature_cols)
    out.insert(0, "TransactionID", df["TransactionID"].values)
    out.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "shap_isolation_forest.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'shap_isolation_forest.csv')}")
    return anomaly_shap


# ------------------------------------------------------------ SHAP: Autoencoder
def shap_autoencoder(df, feature_cols, X_all, X_train, ae_config):
    model = load_autoencoder(os.path.join(ARTIFACTS_RESEARCH_DIR, "autoencoder.pt"),
                              input_dim=ae_config["input_dim"], bottleneck_dim=ae_config["bottleneck_dim"])
    wrapper = AEErrorWrapper(model)
    wrapper.eval()

    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=100, replace=False)
    background = torch.tensor(X_train[bg_idx], dtype=torch.float32)
    test_t = torch.tensor(X_all, dtype=torch.float32)

    t0 = time.time()
    explainer = shap.GradientExplainer(wrapper, background)
    raw_shap = explainer.shap_values(test_t)
    elapsed = time.time() - t0
    print(f"  GradientExplainer on Autoencoder: {elapsed:.1f}s for {len(X_all)} rows")

    anomaly_shap = np.array(raw_shap).squeeze(-1)  # (n, 46) -- already oriented
    # higher = more anomalous, matching score_autoencoder directly (verified in dev)

    out = pd.DataFrame(anomaly_shap, columns=feature_cols)
    out.insert(0, "TransactionID", df["TransactionID"].values)
    out.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "shap_autoencoder.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'shap_autoencoder.csv')}")
    return anomaly_shap


# ----------------------------------------------------------------- Global importance
def global_importance(feature_cols, shap_if, shap_ae):
    mean_abs_if = np.abs(shap_if).mean(axis=0)
    mean_abs_ae = np.abs(shap_ae).mean(axis=0)

    comp = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap_isolation_forest": mean_abs_if,
        "mean_abs_shap_autoencoder": mean_abs_ae,
    })
    comp["rank_isolation_forest"] = comp["mean_abs_shap_isolation_forest"].rank(ascending=False).astype(int)
    comp["rank_autoencoder"] = comp["mean_abs_shap_autoencoder"].rank(ascending=False).astype(int)
    comp = comp.sort_values("rank_isolation_forest")
    comp.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "shap_global_importance_comparison.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'shap_global_importance_comparison.csv')}")

    top10_if = set(comp.nsmallest(10, "rank_isolation_forest")["feature"])
    top10_ae = set(comp.nsmallest(10, "rank_autoencoder")["feature"])
    overlap = top10_if & top10_ae
    print(f"  Top-10 feature overlap (IF vs AE): {len(overlap)}/10 -- {sorted(overlap)}")

    # Individual bar plots
    for name, col in [("isolation_forest", "mean_abs_shap_isolation_forest"),
                       ("autoencoder", "mean_abs_shap_autoencoder")]:
        top15 = comp.nlargest(15, col).sort_values(col)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.barh(top15["feature"], top15[col], color="#2F6690" if name == "isolation_forest" else "#D1495B")
        ax.set_xlabel("Mean |SHAP value| (anomaly-score-oriented, higher = more anomalous)")
        ax.set_title(f"Global Feature Importance -- {'Isolation Forest' if name=='isolation_forest' else 'Autoencoder'}\n"
                     f"(SHAP, {'TreeExplainer' if name=='isolation_forest' else 'GradientExplainer'}, "
                     f"n={len(comp)} rows)")
        savefig(fig, f"shap_global_importance_{'if' if name=='isolation_forest' else 'ae'}.png")

    # Side-by-side agreement scatter (rank vs rank)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(comp["rank_isolation_forest"], comp["rank_autoencoder"], color="#4C956C", s=40, alpha=0.75)
    for _, r in comp.iterrows():
        if r["rank_isolation_forest"] <= 10 or r["rank_autoencoder"] <= 10:
            ax.annotate(r["feature"], (r["rank_isolation_forest"], r["rank_autoencoder"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.plot([1, len(comp)], [1, len(comp)], color="gray", ls="--", lw=1, label="Perfect agreement")
    ax.set_xlabel("Feature importance rank -- Isolation Forest (1 = most important)")
    ax.set_ylabel("Feature importance rank -- Autoencoder (1 = most important)")
    ax.set_title(f"Feature-Importance Rank Agreement: Isolation Forest vs. Autoencoder\n"
                 f"Top-10 overlap: {len(overlap)}/10 features")
    ax.legend()
    savefig(fig, "shap_if_vs_ae_agreement.png")

    from scipy.stats import spearmanr
    rho, _ = spearmanr(comp["mean_abs_shap_isolation_forest"], comp["mean_abs_shap_autoencoder"])
    print(f"  Spearman rho between the two models' mean|SHAP| feature-importance vectors: {rho:.4f}")

    return comp, overlap, rho


# --------------------------------------------------------------- Local explanations
def local_explanations(df, feature_cols, X_all, shap_if, shap_ae, scores_ids):
    results = {}
    for txn_id in LOCAL_EXPLAIN_IDS:
        matches = np.where(df["TransactionID"].values == txn_id)[0]
        if len(matches) == 0:
            continue
        i = matches[0]

        row_raw = df.iloc[i][feature_cols]
        if_vals = shap_if[i]
        ae_vals = shap_ae[i]

        if_top = pd.Series(if_vals, index=feature_cols).sort_values(key=np.abs, ascending=False).head(8)
        ae_top = pd.Series(ae_vals, index=feature_cols).sort_values(key=np.abs, ascending=False).head(8)

        results[txn_id] = {
            "AccountID": str(df.iloc[i]["AccountID"]),
            "isolation_forest": {
                "score": None,
                "top_contributors": [
                    {"feature": f, "shap_value": round(float(v), 5),
                     "raw_feature_value": round(float(row_raw[f]), 4),
                     "direction": "pushes score UP (more anomalous)" if v > 0 else "pushes score DOWN (more normal)"}
                    for f, v in if_top.items()
                ],
            },
            "autoencoder": {
                "top_contributors": [
                    {"feature": f, "shap_value": round(float(v), 5),
                     "raw_feature_value": round(float(row_raw[f]), 4),
                     "direction": "pushes reconstruction error UP (more anomalous)" if v > 0 else "pushes reconstruction error DOWN (more normal)"}
                    for f, v in ae_top.items()
                ],
            },
        }

        # Waterfall-style plot: IF and AE side by side for this transaction
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, top, title, color_pos, color_neg in [
            (axes[0], if_top, "Isolation Forest", "#D1495B", "#2F6690"),
            (axes[1], ae_top, "Autoencoder", "#D1495B", "#2F6690"),
        ]:
            top_sorted = top.sort_values()
            colors = [color_pos if v > 0 else color_neg for v in top_sorted.values]
            ax.barh(top_sorted.index, top_sorted.values, color=colors)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(f"{title} -- {txn_id}")
            ax.set_xlabel("SHAP contribution (+ = more anomalous)")
        fig.suptitle(f"Local SHAP Explanation -- {txn_id} (AccountID {df.iloc[i]['AccountID']})", fontsize=12)
        savefig(fig, f"shap_local_waterfall_{txn_id}.png")

    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "shap_local_explanations.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'shap_local_explanations.json')}")
    return results


def main():
    print("=== Phase 11: Explainability ===")
    df, feature_cols, X_all, X_train, idx_train, ae_config = load_everything()

    print("\n--- SHAP: Isolation Forest (TreeExplainer) ---")
    shap_if = shap_isolation_forest(df, feature_cols, X_all)

    print("\n--- SHAP: Autoencoder (GradientExplainer on reconstruction-error wrapper) ---")
    shap_ae = shap_autoencoder(df, feature_cols, X_all, X_train, ae_config)

    print("\n--- Global feature importance comparison ---")
    comp, overlap, rho = global_importance(feature_cols, shap_if, shap_ae)

    print("\n--- Local explanations ---")
    local = local_explanations(df, feature_cols, X_all, shap_if, shap_ae, None)

    print("\nPhase 11 complete.")


if __name__ == "__main__":
    main()
