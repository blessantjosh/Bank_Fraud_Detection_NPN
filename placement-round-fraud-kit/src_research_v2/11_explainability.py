"""
Phase 11 (v2) -- Explainability, on the teammate's 18-feature matrix.

SHAP for the same two of the 12 Phase 8 (v2) models as the in-house Phase 11:
  - Isolation Forest (Model 1): shap.TreeExplainer -- exact, no background
    sample needed.
  - Autoencoder (Model 9): shap.GradientExplainer on a small wrapper module
    that returns the scalar per-row reconstruction MSE.

Sign convention, checked empirically before use (same checks as the in-house
Phase 11, re-verified on this model/feature set rather than assumed
identical):
  - shap.TreeExplainer on sklearn's IsolationForest explains a quantity that
    is a monotonic transform of `score_samples` (opposite convention to this
    project's `score_isolation_forest = -decision_function`). Raw TreeExplainer
    shap values are NEGATED before use so that positive = "pushes anomaly
    score up."
  - shap.GradientExplainer on the reconstruction-error wrapper explains the
    reconstruction MSE directly (higher = more anomalous, matching
    `score_autoencoder`'s convention already) -- no negation needed.

Outputs: artifacts_research_v2/{shap_isolation_forest_v2.csv,
shap_autoencoder_v2.csv, shap_global_importance_comparison_v2.csv,
shap_local_explanations_v2.json}.
Plots: research_v2/plots/{shap_global_importance_if_v2.png,
shap_global_importance_ae_v2.png, shap_if_vs_ae_agreement_v2.png,
shap_local_waterfall_<TransactionID>_v2.png}.
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
from config_research_v2 import (
    ARTIFACTS_V2_DIR, FEATURE_COLS_V2, MODELS_V2_DIR, PLOTS_V2_DIR, RANDOM_STATE, load_features_v2,
)
from autoencoder_utils import load_autoencoder

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

# Transactions to explain locally -- selected after Phase 10 (v2)'s business
# evaluation is available; falls back to a fixed placeholder list only if
# that file is missing (it will not be, since Phase 10 runs before Phase 11
# in this pipeline's execution order).
BUSINESS_EVAL_JSON = os.path.join(ARTIFACTS_V2_DIR, "business_evaluation_examples_v2.json")


def savefig(fig, name):
    path = os.path.join(PLOTS_V2_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


class AEErrorWrapper(nn.Module):
    """Wraps the trained Autoencoder so its forward() returns the scalar
    per-row reconstruction MSE -- the quantity SHAP is asked to explain,
    matching `score_autoencoder`'s definition exactly (Phase 8 v2, Model 9)."""
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        out, _ = self.ae(x)
        return ((out - x) ** 2).mean(dim=1, keepdim=True)


def load_everything():
    df = load_features_v2()
    feature_cols = FEATURE_COLS_V2
    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = joblib.load(os.path.join(MODELS_V2_DIR, "shared_robust_scaler.pkl"))
    X_train = scaler.transform(X[idx_train])
    X_all = scaler.transform(X)
    with open(os.path.join(ARTIFACTS_V2_DIR, "autoencoder_config.json")) as f:
        ae_config = json.load(f)
    return df, feature_cols, X_all, X_train, idx_train, ae_config


# --------------------------------------------------------- SHAP: Isolation Forest
def shap_isolation_forest(df, feature_cols, X_all):
    clf = joblib.load(os.path.join(MODELS_V2_DIR, "isolation_forest.pkl"))
    t0 = time.time()
    explainer = shap.TreeExplainer(clf)
    raw_shap = explainer.shap_values(X_all.astype(np.float64))
    elapsed = time.time() - t0
    print(f"  TreeExplainer on Isolation Forest (v2): {elapsed:.1f}s for {len(X_all)} rows")

    # Sign check: verify TreeExplainer's raw output tracks score_samples
    # (higher = more normal), the opposite of score_isolation_forest -- negate.
    raw_pred = clf.score_samples(X_all[:200])
    from scipy.stats import spearmanr
    rho_check, _ = spearmanr(np.array(raw_shap[:200]).sum(axis=1), raw_pred)
    print(f"  Sign-check: Spearman(raw shap row-sums, score_samples) on 200-row spot-check = {rho_check:.4f} "
          f"(expect strongly positive -> negate to match anomaly-score-up convention)")
    anomaly_shap = -raw_shap

    out = pd.DataFrame(anomaly_shap, columns=feature_cols)
    out.insert(0, "TransactionID", df["TransactionID"].values)
    out.to_csv(os.path.join(ARTIFACTS_V2_DIR, "shap_isolation_forest_v2.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'shap_isolation_forest_v2.csv')}")
    return anomaly_shap


# ------------------------------------------------------------ SHAP: Autoencoder
def shap_autoencoder(df, feature_cols, X_all, X_train, ae_config):
    model = load_autoencoder(os.path.join(ARTIFACTS_V2_DIR, "autoencoder.pt"),
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
    print(f"  GradientExplainer on Autoencoder (v2): {elapsed:.1f}s for {len(X_all)} rows")

    anomaly_shap = np.array(raw_shap).squeeze(-1)  # (n, 18) -- already oriented,
    # higher = more anomalous, matching score_autoencoder directly

    # Additivity spot-check (10 rows), same convention as the in-house Phase 11
    with torch.no_grad():
        base_val = wrapper(background).mean().item()
    approx = base_val + anomaly_shap[:10].sum(axis=1)
    actual = wrapper(test_t[:10]).detach().numpy().ravel()
    from scipy.stats import spearmanr
    rho_add, _ = spearmanr(approx, actual)
    print(f"  Additivity spot-check (10 rows): Spearman(base+shap_sum, actual output) = {rho_add:.4f} "
          "(GradientExplainer's expected-gradients approximation, not an exact decomposition)")

    out = pd.DataFrame(anomaly_shap, columns=feature_cols)
    out.insert(0, "TransactionID", df["TransactionID"].values)
    out.to_csv(os.path.join(ARTIFACTS_V2_DIR, "shap_autoencoder_v2.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'shap_autoencoder_v2.csv')}")
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
    comp.to_csv(os.path.join(ARTIFACTS_V2_DIR, "shap_global_importance_comparison_v2.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'shap_global_importance_comparison_v2.csv')}")

    top10_if = set(comp.nsmallest(10, "rank_isolation_forest")["feature"])
    top10_ae = set(comp.nsmallest(10, "rank_autoencoder")["feature"])
    overlap = top10_if & top10_ae
    print(f"  Top-10 feature overlap (IF vs AE), v2: {len(overlap)}/10 -- {sorted(overlap)}")

    for name, col in [("isolation_forest", "mean_abs_shap_isolation_forest"),
                       ("autoencoder", "mean_abs_shap_autoencoder")]:
        top15 = comp.nlargest(min(15, len(comp)), col).sort_values(col)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.barh(top15["feature"], top15[col], color="#2F6690" if name == "isolation_forest" else "#D1495B")
        ax.set_xlabel("Mean |SHAP value| (anomaly-score-oriented, higher = more anomalous)")
        ax.set_title(f"Global Feature Importance (v2, 18 features) -- {'Isolation Forest' if name=='isolation_forest' else 'Autoencoder'}\n"
                     f"(SHAP, {'TreeExplainer' if name=='isolation_forest' else 'GradientExplainer'}, "
                     f"n={len(comp)} features, {2512} rows)")
        savefig(fig, f"shap_global_importance_{'if' if name=='isolation_forest' else 'ae'}_v2.png")

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(comp["rank_isolation_forest"], comp["rank_autoencoder"], color="#4C956C", s=40, alpha=0.75)
    for _, r in comp.iterrows():
        if r["rank_isolation_forest"] <= 10 or r["rank_autoencoder"] <= 10:
            ax.annotate(r["feature"], (r["rank_isolation_forest"], r["rank_autoencoder"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.plot([1, len(comp)], [1, len(comp)], color="gray", ls="--", lw=1, label="Perfect agreement")
    ax.set_xlabel("Feature importance rank -- Isolation Forest (1 = most important)")
    ax.set_ylabel("Feature importance rank -- Autoencoder (1 = most important)")
    ax.set_title(f"Feature-Importance Rank Agreement (v2): Isolation Forest vs. Autoencoder\n"
                 f"Top-10 overlap: {len(overlap)}/10 features")
    ax.legend()
    savefig(fig, "shap_if_vs_ae_agreement_v2.png")

    from scipy.stats import spearmanr
    rho, _ = spearmanr(comp["mean_abs_shap_isolation_forest"], comp["mean_abs_shap_autoencoder"])
    print(f"  Spearman rho between the two models' mean|SHAP| feature-importance vectors (v2): {rho:.4f}")

    return comp, overlap, rho


# --------------------------------------------------------------- Local explanations
def pick_local_explain_ids():
    if os.path.exists(BUSINESS_EVAL_JSON):
        with open(BUSINESS_EVAL_JSON) as f:
            be = json.load(f)
        top1 = be["tiers"]["top_1pct"]["transaction_ids"]
        # top of the top-1% tier, plus one from deeper in it, plus the weakest
        # end of the top-10% tier -- mirrors the in-house Phase 11's mix of
        # "clear" and "does-not-look-like-fraud" examples
        ids = top1[:2]
        if len(top1) > 4:
            ids.append(top1[4])
        tail = be["tiers"].get("top_10pct", {}).get("transaction_ids", [])
        if tail:
            ids.append(tail[-1])
        return ids
    return []


def local_explanations(df, feature_cols, shap_if, shap_ae, explain_ids):
    results = {}
    for txn_id in explain_ids:
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
                "top_contributors": [
                    {"feature": f, "shap_value": round(float(v), 5),
                     "raw_feature_value_scaled": round(float(row_raw[f]), 4),
                     "direction": "pushes score UP (more anomalous)" if v > 0 else "pushes score DOWN (more normal)"}
                    for f, v in if_top.items()
                ],
            },
            "autoencoder": {
                "top_contributors": [
                    {"feature": f, "shap_value": round(float(v), 5),
                     "raw_feature_value_scaled": round(float(row_raw[f]), 4),
                     "direction": "pushes reconstruction error UP (more anomalous)" if v > 0 else "pushes reconstruction error DOWN (more normal)"}
                    for f, v in ae_top.items()
                ],
            },
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, top, title in [
            (axes[0], if_top, "Isolation Forest"),
            (axes[1], ae_top, "Autoencoder"),
        ]:
            top_sorted = top.sort_values()
            colors = ["#D1495B" if v > 0 else "#2F6690" for v in top_sorted.values]
            ax.barh(top_sorted.index, top_sorted.values, color=colors)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(f"{title} -- {txn_id}")
            ax.set_xlabel("SHAP contribution (+ = more anomalous)")
        fig.suptitle(f"Local SHAP Explanation (v2) -- {txn_id} (AccountID {df.iloc[i]['AccountID']})", fontsize=12)
        savefig(fig, f"shap_local_waterfall_{txn_id}_v2.png")

    with open(os.path.join(ARTIFACTS_V2_DIR, "shap_local_explanations_v2.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'shap_local_explanations_v2.json')}")
    return results


def main():
    print("=== Phase 11 (v2): Explainability ===")
    df, feature_cols, X_all, X_train, idx_train, ae_config = load_everything()

    print("\n--- SHAP: Isolation Forest (TreeExplainer) ---")
    shap_if = shap_isolation_forest(df, feature_cols, X_all)

    print("\n--- SHAP: Autoencoder (GradientExplainer on reconstruction-error wrapper) ---")
    shap_ae = shap_autoencoder(df, feature_cols, X_all, X_train, ae_config)

    print("\n--- Global feature importance comparison ---")
    comp, overlap, rho = global_importance(feature_cols, shap_if, shap_ae)

    print("\n--- Local explanations ---")
    explain_ids = pick_local_explain_ids()
    print(f"  Transactions selected for local explanation: {explain_ids}")
    local = local_explanations(df, feature_cols, shap_if, shap_ae, explain_ids)

    print("\nPhase 11 (v2) complete.")


if __name__ == "__main__":
    main()
