"""
blend_models.py -- does blending LightGBM + XGBoost + CatBoost actually beat
the single-model baseline? Test it, don't assume it (same philosophy as the
balancing ablation). Rank-average the three models' outputs, since AUC only
cares about relative order, not raw probability scale.
"""

from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

import baf
from run_pipeline import PARAMS, train_lgbm


def main(train_path: str, out_path: str):
    print(f"Loading {train_path} ...")
    df = baf.prepare(baf.load(train_path))
    tr, va = train_test_split(df, test_size=0.30, stratify=df[baf.TARGET], random_state=42)
    X_tr, y_tr = baf.xy(tr, drop_month=False)
    X_va, y_va = baf.xy(va, drop_month=False)

    results = {}
    preds = {}

    # ---- LightGBM (native categorical) --------------------------------
    print("\nTraining LightGBM...")
    m_lgb = train_lgbm(X_tr, y_tr, X_va, y_va)
    preds["lightgbm"] = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)
    results["lightgbm"] = baf.evaluate(y_va, preds["lightgbm"], 0.05, "LightGBM")

    # ---- XGBoost (native categorical via enable_categorical) -----------
    import xgboost as xgb
    print("\nTraining XGBoost...")
    # NOTE: XGBoost's min_child_weight is a SUM OF HESSIANS, not a row count
    # like LightGBM's min_data_in_leaf. At ~1.1% positive rate, per-row hessian
    # is tiny (~p*(1-p) ~= 0.01), so a value like 200 blocks every split and
    # silently leaves a stump (best_iteration=0, AUC exactly 0.5). Keep this
    # small; depth + subsample + reg do the regularisation work here instead.
    m_xgb = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=5.0, reg_alpha=0.1,
        tree_method="hist", enable_categorical=True,
        eval_metric="auc", early_stopping_rounds=100,
        n_jobs=-1, random_state=42,
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds["xgboost"] = m_xgb.predict_proba(X_va)[:, 1]
    results["xgboost"] = baf.evaluate(y_va, preds["xgboost"], 0.05, "XGBoost")

    # ---- CatBoost (native categorical via cat_features) -----------------
    from catboost import CatBoostClassifier, Pool
    print("\nTraining CatBoost...")
    cat_cols = [c for c in baf.CATEGORICAL_COLS if c in X_tr.columns]
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
    m_cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6,
        l2_leaf_reg=5.0, min_data_in_leaf=200,
        eval_metric="AUC", loss_function="Logloss",
        cat_features=cat_cols, early_stopping_rounds=100,
        random_seed=42, verbose=False,
    )
    m_cb.fit(Pool(X_tr_cb, y_tr, cat_features=cat_cols),
             eval_set=Pool(X_va_cb, y_va, cat_features=cat_cols), verbose=False)
    preds["catboost"] = m_cb.predict_proba(X_va_cb)[:, 1]
    results["catboost"] = baf.evaluate(y_va, preds["catboost"], 0.05, "CatBoost")

    # ---- Blend: average RANKS, not raw probabilities ---------------------
    print("\n" + "=" * 70)
    print("BLEND: rank-averaged LightGBM + XGBoost + CatBoost")
    print("=" * 70)
    ranks = np.column_stack([rankdata(preds[k]) for k in ["lightgbm", "xgboost", "catboost"]])
    blend_score = ranks.mean(axis=1)
    results["blend_equal"] = baf.evaluate(y_va, blend_score, 0.05, "Equal-weight rank blend")

    # Weighted blend, weighted toward the strongest individual model
    aucs = np.array([results[k]["roc_auc"] for k in ["lightgbm", "xgboost", "catboost"]])
    w = aucs - aucs.min() + 0.01
    w = w / w.sum()
    blend_w = (ranks * w).sum(axis=1)
    results["blend_weighted"] = baf.evaluate(y_va, blend_w, 0.05, "AUC-weighted rank blend")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    tbl = pd.DataFrame(results).T[["roc_auc", "pr_auc", "tpr_at_5pct_fpr"]]
    print(tbl.round(4).to_string())
    best = tbl["roc_auc"].idxmax()
    print(f"\nBest by ROC-AUC: {best}  ({tbl.loc[best, 'roc_auc']:.4f})")
    single_best = tbl.drop(["blend_equal", "blend_weighted"])["roc_auc"].max()
    blend_best = tbl.loc[["blend_equal", "blend_weighted"], "roc_auc"].max()
    gain = blend_best - single_best
    print(f"Blend vs best single model: {'+' if gain >= 0 else ''}{gain:.4f} AUC")

    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "sample_train.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "blend_results.json"
    main(train_path, out_path)
