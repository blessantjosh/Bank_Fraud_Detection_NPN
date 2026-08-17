"""
export_demo_model.py -- train a browser-portable variant of the BAF model
and dump it to JSON for real, exact client-side inference (no server).

Why a separate model: the headline model (run_pipeline.py) uses LightGBM's
native pandas-category handling for housing_status/device_os/etc, which
splits on internal categorical bins that are non-trivial to reproduce
exactly in hand-written JS. This demo model one-hot encodes those same
columns instead, so every tree node is a plain numeric "<=" comparison --
exactly reproducible in ~30 lines of JS. It is trained on the same data,
same split, same core params, and its own real AUC is reported and stored
alongside the export so nobody mistakes it for the headline number.
"""

from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import baf
from run_pipeline import PARAMS


def main(train_path: str, out_path: str):
    print(f"Loading {train_path} ...")
    raw = baf.load(train_path)
    df = baf.prepare(raw)

    tr, va = train_test_split(df, test_size=0.30, stratify=df[baf.TARGET], random_state=42)
    X_tr_full, y_tr = baf.xy(tr, drop_month=False)
    X_va_full, y_va = baf.xy(va, drop_month=False)

    cat_cols = [c for c in baf.CATEGORICAL_COLS if c in X_tr_full.columns]
    print("One-hot encoding categorical columns:", cat_cols)

    # Union of categories across train+valid so both sides get identical dummy columns.
    cats = {c: sorted(pd.concat([X_tr_full[c], X_va_full[c]]).astype(str).unique()) for c in cat_cols}
    for c in cat_cols:
        X_tr_full[c] = pd.Categorical(X_tr_full[c].astype(str), categories=cats[c])
        X_va_full[c] = pd.Categorical(X_va_full[c].astype(str), categories=cats[c])

    X_tr = pd.get_dummies(X_tr_full, columns=cat_cols, prefix=cat_cols)
    X_va = pd.get_dummies(X_va_full, columns=cat_cols, prefix=cat_cols)
    X_tr, X_va = X_tr.align(X_va, join="outer", axis=1, fill_value=0)
    feature_names = list(X_tr.columns)
    print(f"Demo model feature count: {len(feature_names)}")

    demo_params = dict(PARAMS)
    # These only apply to native categorical splits -- irrelevant now, drop them.
    for k in ["max_cat_to_onehot", "cat_smooth", "cat_l2", "min_data_per_group"]:
        demo_params.pop(k, None)
    demo_params["num_leaves"] = 48  # keep the exported JSON smaller; small AUC cost

    print("Training demo (one-hot) model...")
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    model = lgb.train(
        demo_params, dtr, num_boost_round=2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    ev = baf.evaluate(y_va, p_va, 0.05, "demo (one-hot) model")

    # ---- Verify the raw-score reconstruction offset ------------------------
    # LightGBM's binary objective can add a global init score (boost_from_average).
    # Sum the dumped tree leaf values ourselves for a few rows and diff against
    # the library's own raw_score to get that constant exactly, empirically --
    # no guessing about internal defaults.
    dumped = model.dump_model()
    trees = dumped["tree_info"]

    def leaf_value_for_row(tree_struct, row):
        node = tree_struct
        while "leaf_value" not in node:
            fidx = node["split_feature"]
            thr = node["threshold"]
            v = row[fidx]
            miss = v is None or (isinstance(v, float) and np.isnan(v))
            if miss:
                go_left = node.get("default_left", True)
            else:
                go_left = v <= thr
            node = node["left_child"] if go_left else node["right_child"]
        return node["leaf_value"]

    sample_idx = X_va.index[:20]
    raw_scores = model.predict(X_va.loc[sample_idx], num_iteration=model.best_iteration, raw_score=True)
    offsets = []
    for pos, idx in enumerate(sample_idx):
        row = X_va.loc[idx].to_numpy(dtype=float)
        total = sum(leaf_value_for_row(t["tree_structure"], row) for t in trees)
        offsets.append(raw_scores[pos] - total)
    offsets = np.array(offsets)
    print(f"Offset check: mean={offsets.mean():.8f} std={offsets.std():.8f} (std should be ~0)")
    base_score = float(offsets.mean())

    # ---- Slim tree export (only fields the JS scorer needs) ----------------
    def slim(node):
        if "leaf_value" in node:
            return {"leaf_value": node["leaf_value"]}
        return {
            "split_feature": node["split_feature"],
            "threshold": node["threshold"],
            "default_left": node.get("default_left", True),
            "left_child": slim(node["left_child"]),
            "right_child": slim(node["right_child"]),
        }

    slim_trees = [slim(t["tree_structure"]) for t in trees]

    # ---- Defaults (median for numeric, mode for one-hot group) for every raw field
    raw_defaults = {}
    for c in raw.columns:
        if c in (baf.TARGET,):
            continue
        if c in cat_cols:
            raw_defaults[c] = raw[c].mode(dropna=True).iloc[0]
        elif raw[c].dropna().isin([0, 1]).all():
            raw_defaults[c] = int(raw[c].mode(dropna=True).iloc[0])
        else:
            raw_defaults[c] = float(raw[c].median())

    field_ranges = {}
    for c in raw.columns:
        if c == baf.TARGET:
            continue
        if c in cat_cols:
            continue
        field_ranges[c] = {"min": float(raw[c].min()), "max": float(raw[c].max())}

    out = {
        "base_score": base_score,
        "feature_names": feature_names,
        "sentinel_cols": baf.SENTINEL_COLS,
        "categorical_cols": cat_cols,
        "categories": cats,
        "raw_defaults": raw_defaults,
        "field_ranges": field_ranges,
        "eval": ev,
        "best_iteration": int(model.best_iteration),
        "n_leaves_setting": demo_params["num_leaves"],
        "trees": slim_trees,
    }
    with open(out_path, "w") as f:
        json.dump(out, f)
    import os
    print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB, {len(slim_trees)} trees)")


if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "sample_train.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "demo_model.json"
    main(train_path, out_path)
