"""
generate_results.py -- run baseline + ablation once, dump every number and
curve needed for charts into results.json. Reuses baf.py and the exact
LightGBM params/split from run_pipeline.py so the numbers match that run.
"""

from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split

import baf
from run_pipeline import PARAMS, train_lgbm, undersample, prior_correct


def downsample_curve(x, y, n=150):
    """Evenly sample n points along the curve for a light JSON payload."""
    if len(x) <= n:
        return list(map(float, x)), list(map(float, y))
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return [float(v) for v in np.asarray(x)[idx]], [float(v) for v in np.asarray(y)[idx]]


def main(train_path: str, out_path: str):
    print(f"Loading {train_path} ...")
    df = baf.load(train_path)
    df = baf.prepare(df)

    tr, va = train_test_split(df, test_size=0.30, stratify=df[baf.TARGET], random_state=42)
    X_tr, y_tr = baf.xy(tr, drop_month=False)
    X_va, y_va = baf.xy(va, drop_month=False)

    print("Training baseline model...")
    m_base = train_lgbm(X_tr, y_tr, X_va, y_va)
    p_base = m_base.predict(X_va, num_iteration=m_base.best_iteration)
    base_eval = baf.evaluate(y_va, p_base, 0.05, "baseline")

    print("Training scale_pos_weight model...")
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    m_spw = train_lgbm(X_tr, y_tr, X_va, y_va, scale_pos_weight=spw)
    p_spw = m_spw.predict(X_va, num_iteration=m_spw.best_iteration)
    spw_eval = baf.evaluate(y_va, p_spw, 0.05, "scale_pos_weight")

    print("Training undersample model...")
    Xu, yu = undersample(X_tr, y_tr, ratio=10)
    m_us = train_lgbm(Xu, yu, X_va, y_va)
    p_us_raw = m_us.predict(X_va, num_iteration=m_us.best_iteration)
    p_us = prior_correct(p_us_raw, yu.mean(), y_tr.mean())
    us_eval = baf.evaluate(y_va, p_us, 0.05, "undersample_10to1")

    # ---- ROC / PR curves (baseline model) --------------------------------
    fpr, tpr, _ = roc_curve(y_va, p_base)
    fpr_s, tpr_s = downsample_curve(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_va, p_base)
    rec_s, prec_s = downsample_curve(rec, prec)

    # ---- Fairness ----------------------------------------------------------
    older = (X_va["customer_age"] > 50).astype(int).to_numpy()
    fair = baf.fairness_report(y_va, p_base, older, 0.05, "customer_age > 50")

    # ---- Feature importance -------------------------------------------------
    imp = (pd.DataFrame({
        "feature": m_base.feature_name(),
        "gain": m_base.feature_importance("gain"),
    }).sort_values("gain", ascending=False).head(12))
    imp["share"] = imp["gain"] / imp["gain"].sum()

    out = {
        "n_rows": int(df.shape[0]),
        "fraud_rate": float(df[baf.TARGET].mean()),
        "split": {"train": int(len(tr)), "valid": int(len(va))},
        "ablation": {
            "none": base_eval,
            "scale_pos_weight": spw_eval,
            "undersample_10to1": us_eval,
        },
        "roc_curve": {"fpr": fpr_s, "tpr": tpr_s},
        "pr_curve": {"recall": rec_s, "precision": prec_s},
        "fairness": {
            "age_le_50": fair["group_0"],
            "age_gt_50": fair["group_1"],
            "fpr_ratio": fair["fpr_ratio"],
            "threshold": fair["threshold"],
        },
        "feature_importance": imp.to_dict(orient="records"),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "sample_train.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results.json"
    main(train_path, out_path)
