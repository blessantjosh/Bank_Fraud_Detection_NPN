"""
Phase 11 (v2) -- Explainability, on the teammate's 18-feature matrix.

SHAP for the sole Phase 8 (v2) model this pipeline explains:
  - Isolation Forest (Model 1): shap.TreeExplainer -- exact, no background
    sample needed.

No other remaining classical model (LOF, OCSVM, Elliptic Envelope, DBSCAN,
HDBSCAN, K-Means, GMM) is naturally SHAP-compatible without an expensive
KernelExplainer, which is not used anywhere in this codebase -- so the
cross-model SHAP comparison that used to run Isolation Forest against the
Autoencoder (removed) has been dropped entirely. Isolation Forest SHAP is
the sole explainability output.

Sign convention, checked empirically before use (same check as the in-house
Phase 11, re-verified on this model/feature set rather than assumed
identical):
  - shap.TreeExplainer on sklearn's IsolationForest explains a quantity that
    is a monotonic transform of `score_samples` (opposite convention to this
    project's `score_isolation_forest = -decision_function`). Raw TreeExplainer
    shap values are NEGATED before use so that positive = "pushes anomaly
    score up."

Outputs: artifacts_research_v2/{shap_isolation_forest_v2.csv,
shap_local_explanations_v2.json}.
Plots: research_v2/plots/{shap_global_importance_if_v2.png,
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
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research_v2 import (
    ARTIFACTS_V2_DIR, FEATURE_COLS_V2, MODELS_V2_DIR, PLOTS_V2_DIR, RANDOM_STATE, load_features_v2,
)

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


def load_everything():
    df = load_features_v2()
    feature_cols = FEATURE_COLS_V2
    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = joblib.load(os.path.join(MODELS_V2_DIR, "shared_robust_scaler.pkl"))
    X_train = scaler.transform(X[idx_train])
    X_all = scaler.transform(X)
    return df, feature_cols, X_all, X_train, idx_train


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
    rho_check, _ = spearmanr(np.array(raw_shap[:200]).sum(axis=1), raw_pred)
    print(f"  Sign-check: Spearman(raw shap row-sums, score_samples) on 200-row spot-check = {rho_check:.4f} "
          f"(expect strongly positive -> negate to match anomaly-score-up convention)")
    anomaly_shap = -raw_shap

    out = pd.DataFrame(anomaly_shap, columns=feature_cols)
    out.insert(0, "TransactionID", df["TransactionID"].values)
    out.to_csv(os.path.join(ARTIFACTS_V2_DIR, "shap_isolation_forest_v2.csv"), index=False)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'shap_isolation_forest_v2.csv')}")
    return anomaly_shap


# ----------------------------------------------------------------- Global importance
def global_importance(feature_cols, shap_if):
    mean_abs_if = np.abs(shap_if).mean(axis=0)

    comp = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap_isolation_forest": mean_abs_if,
    })
    comp["rank_isolation_forest"] = comp["mean_abs_shap_isolation_forest"].rank(ascending=False).astype(int)
    comp = comp.sort_values("rank_isolation_forest")

    top15 = comp.nlargest(min(15, len(comp)), "mean_abs_shap_isolation_forest").sort_values(
        "mean_abs_shap_isolation_forest")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(top15["feature"], top15["mean_abs_shap_isolation_forest"], color="#2F6690")
    ax.set_xlabel("Mean |SHAP value| (anomaly-score-oriented, higher = more anomalous)")
    ax.set_title(f"Global Feature Importance (v2, 18 features) -- Isolation Forest\n"
                 f"(SHAP, TreeExplainer, n={len(comp)} features)")
    savefig(fig, "shap_global_importance_if_v2.png")

    print("  Top-5 features by mean|SHAP| (Isolation Forest, v2):")
    for _, r in comp.head(5).iterrows():
        print(f"    {r['feature']:35s} mean|SHAP|={r['mean_abs_shap_isolation_forest']:.5f}")

    return comp


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


def local_explanations(df, feature_cols, shap_if, explain_ids):
    results = {}
    for txn_id in explain_ids:
        matches = np.where(df["TransactionID"].values == txn_id)[0]
        if len(matches) == 0:
            continue
        i = matches[0]

        row_raw = df.iloc[i][feature_cols]
        if_vals = shap_if[i]

        if_top = pd.Series(if_vals, index=feature_cols).sort_values(key=np.abs, ascending=False).head(8)

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
        }

        fig, ax = plt.subplots(figsize=(7, 5.5))
        top_sorted = if_top.sort_values()
        colors = ["#D1495B" if v > 0 else "#2F6690" for v in top_sorted.values]
        ax.barh(top_sorted.index, top_sorted.values, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"Isolation Forest -- {txn_id}")
        ax.set_xlabel("SHAP contribution (+ = more anomalous)")
        fig.suptitle(f"Local SHAP Explanation (v2) -- {txn_id} (AccountID {df.iloc[i]['AccountID']})", fontsize=12)
        savefig(fig, f"shap_local_waterfall_{txn_id}_v2.png")

    with open(os.path.join(ARTIFACTS_V2_DIR, "shap_local_explanations_v2.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'shap_local_explanations_v2.json')}")
    return results


def main():
    print("=== Phase 11 (v2): Explainability ===")
    df, feature_cols, X_all, X_train, idx_train = load_everything()

    print("\n--- SHAP: Isolation Forest (TreeExplainer) ---")
    shap_if = shap_isolation_forest(df, feature_cols, X_all)

    print("\n--- Global feature importance ---")
    comp = global_importance(feature_cols, shap_if)

    print("\n--- Local explanations ---")
    explain_ids = pick_local_explain_ids()
    print(f"  Transactions selected for local explanation: {explain_ids}")
    local = local_explanations(df, feature_cols, shap_if, explain_ids)

    print("\nPhase 11 (v2) complete.")


if __name__ == "__main__":
    main()
