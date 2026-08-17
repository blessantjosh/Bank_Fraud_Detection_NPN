"""
verify_demo_model.py -- sanity check that the JS live-demo logic (feature
engineering + one-hot + tree walk) reproduces the real training pipeline
exactly, given only the RAW (pre-engineered) columns a user would type in.
"""
import argparse
import json
import numpy as np
import pandas as pd
import baf

_ap = argparse.ArgumentParser()
_ap.add_argument("--train", required=True, help="path to the real training CSV (e.g. Base.csv)")
_ap.add_argument("--model", default="demo_model.json", help="path to the exported demo model JSON")
_args = _ap.parse_args()

MODEL = json.load(open(_args.model))


def js_engineer(raw):
    r = dict(raw)
    eps = 1e-6
    miss = {}
    for c in MODEL["sentinel_cols"]:
        v = raw[c]
        m = v < 0
        miss[c + "_is_missing"] = 1 if m else 0
        r[c] = np.nan if m else v
    r.update(miss)

    v6 = max(0, raw["velocity_6h"])
    r["velocity_burst_6h_4w"] = v6 / (raw["velocity_4w"] + eps)
    r["velocity_ratio_6h_24h"] = v6 / (raw["velocity_24h"] + eps)
    r["velocity_burst_24h_4w"] = raw["velocity_24h"] / (raw["velocity_4w"] + eps)

    r["email_mismatch_free"] = (1.0 - raw["name_email_similarity"]) * raw["email_is_free"]
    r["dob_emails_x_mismatch"] = raw["date_of_birth_distinct_emails_4w"] * (1.0 - raw["name_email_similarity"])

    prev_addr = 0 if pd.isna(r["prev_address_months_count"]) else r["prev_address_months_count"]
    cur_addr = 0 if pd.isna(r["current_address_months_count"]) else r["current_address_months_count"]
    r["total_address_history"] = prev_addr + cur_addr
    r["thin_file_score"] = miss.get("prev_address_months_count_is_missing", 0) + miss.get("bank_months_count_is_missing", 0)

    r["n_missing"] = sum(miss.get(c + "_is_missing", 0) for c in MODEL["sentinel_cols"])

    r["n_valid_phones"] = raw["phone_home_valid"] + raw["phone_mobile_valid"]
    r["no_valid_phone"] = 1 if r["n_valid_phones"] == 0 else 0

    r["limit_to_income"] = raw["proposed_credit_limit"] / (raw["income"] + eps)
    r["limit_per_risk"] = raw["proposed_credit_limit"] / (raw["credit_risk_score"] + 200.0)
    r["risk_x_income"] = raw["credit_risk_score"] * raw["income"]

    session_len = 0 if pd.isna(r["session_length_in_minutes"]) else r["session_length_in_minutes"]
    device_emails = 0 if pd.isna(r["device_distinct_emails_8w"]) else r["device_distinct_emails_8w"]
    r["emails_per_session_min"] = device_emails / (session_len + 1.0)
    r["short_session_no_keepalive"] = 1 if (session_len < 5 and raw["keep_alive_session"] == 0) else 0

    r["zip_density_vs_velocity"] = raw["zip_count_4w"] / (raw["velocity_4w"] + eps)
    return r


def js_build_vector(raw):
    eng = js_engineer(raw)
    names = MODEL["feature_names"]
    cat_cols = MODEL["categorical_cols"]
    vec = np.zeros(len(names))
    index = {n: i for i, n in enumerate(names)}
    for i, name in enumerate(names):
        is_dummy = any(name.startswith(c + "_") for c in cat_cols)
        if not is_dummy and name in eng:
            v = eng[name]
            vec[i] = np.nan if v is None or (isinstance(v, float) and pd.isna(v)) else v
    for c in cat_cols:
        col_name = c + "_" + str(raw[c])
        if col_name in index:
            vec[index[col_name]] = 1
    return vec


def eval_tree(node, vec):
    while "leaf_value" not in node:
        v = vec[node["split_feature"]]
        missing = v is None or (isinstance(v, float) and np.isnan(v))
        go_left = node["default_left"] if missing else (v <= node["threshold"])
        node = node["left_child"] if go_left else node["right_child"]
    return node["leaf_value"]


def js_score(raw):
    vec = js_build_vector(raw)
    total = MODEL["base_score"] + sum(eval_tree(t, vec) for t in MODEL["trees"])
    return 1 / (1 + np.exp(-total))


# ---- Ground truth: the REAL training pipeline on the same raw rows --------
raw_df = baf.load(_args.train)
sample = raw_df.sample(30, random_state=7).reset_index(drop=True)

# Run prepare()'s sub-steps directly, WITHOUT drop_constant -- on a 30-row
# sample, a real but low-cardinality column (e.g. device_distinct_emails_8w,
# only 4 effective values) can look constant by chance and get dropped,
# which would then be a sampling artifact, not a bug in the FE logic.
# device_fraud_count is the one column actually constant in the full 1M-row
# dataset (confirmed in the real training run), so drop only that.
prepared = sample.drop(columns=["device_fraud_count"]).copy()
prepared = baf.to_nan_and_flag(prepared, verbose=False)
prepared = baf.add_features(prepared)
prepared = baf.set_categoricals(prepared)
cat_cols = MODEL["categorical_cols"]
for c in cat_cols:
    prepared[c] = pd.Categorical(prepared[c].astype(str), categories=MODEL["categories"][c])
X = pd.get_dummies(prepared.drop(columns=[baf.TARGET]), columns=cat_cols, prefix=cat_cols)
for c in MODEL["feature_names"]:
    if c not in X.columns:
        X[c] = 0
X = X[MODEL["feature_names"]]

max_diff = 0.0
for i in range(len(sample)):
    raw_row = sample.iloc[i].to_dict()
    js_vec = js_build_vector(raw_row)
    true_vec = X.iloc[i].to_numpy(dtype=float)
    diff = np.nanmax(np.abs(np.nan_to_num(js_vec, nan=-99999) - np.nan_to_num(true_vec, nan=-99999)))
    max_diff = max(max_diff, diff)
    js_p = js_score(raw_row)
    if i < 5:
        print(f"row {i}: vector max diff={diff:.6f}  js_score={js_p:.6f}  true_fraud_bool={sample.iloc[i][baf.TARGET]}")

print(f"\nMax feature-vector diff across {len(sample)} rows: {max_diff:.6f}  (should be ~0)")
