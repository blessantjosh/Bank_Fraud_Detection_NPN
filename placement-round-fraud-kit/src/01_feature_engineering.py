"""
STAGE 1 -- Leakage-safe feature engineering.

Reads the raw transaction log, drops the broken PreviousTransactionDate
column and the raw TransactionID/AccountID identifiers, builds the
behavioral features (per-account deviation, novelty flags, recency,
device/IP/merchant popularity), encodes categoricals, and writes:

  features.csv         - unscaled, used by the tree-based supervised models
                          (XGBoost / Random Forest / Decision Tree don't need
                          scaling). Has a `split` column: train / val / test.
  features_scaled.csv   - StandardScaler-scaled, used by the distance-based
                          unsupervised detectors in Stage 2 (LOF, One-Class
                          SVM and MCD all assume comparable feature scales;
                          Isolation Forest doesn't strictly need it but is
                          unaffected by it). Same `split` column.

LEAKAGE FIX (see ML_AUDIT_AFTER_FIX.md): the split is chronological (train =
earliest transactions, val/test = later ones), and every statistic that
requires "fitting" -- transaction-type means, device/IP/merchant popularity,
the median history-gap fallback, the one-hot/label encoders, and the
StandardScaler -- is fit ONLY on the train fold, then applied unchanged to
val and test. Nothing about val or test rows contributes to any of these
fitted statistics.

Also persists reference.pkl (full-dataset account history + lookup stats) so
live scoring (the Argus dashboard's "Upload & Predict" page) can engineer a
brand-new transaction's features the same way training data was engineered.
reference.pkl intentionally uses the FULL dataset (train+val+test) because,
by the time a new transaction arrives live, all of history to date is
legitimately known -- this is a separate artifact from the train-only stats
used to build the features above, and does not affect evaluation.
"""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import fe_utils as fe

print(f"Loading raw data from {config.RAW_CSV}")
df = fe.load_raw(config.RAW_CSV)
print(f"  {len(df)} transactions across {df['AccountID'].nunique()} accounts")

df = fe.sort_chronological(df)
df = fe.add_causal_features(df)

# ---- chronological train / val / test split ----
cut_train = df["TransactionDate"].quantile(config.TRAIN_QUANTILE)
cut_val = df["TransactionDate"].quantile(config.VAL_QUANTILE)
split = pd.Series("test", index=df.index)
split[df["TransactionDate"] <= cut_val] = "val"
split[df["TransactionDate"] <= cut_train] = "train"
print(f"\nChronological split (train <= {cut_train}, val <= {cut_val}, test after):")
print(split.value_counts())

train_raw = df[split == "train"].copy()
val_raw = df[split == "val"].copy()
test_raw = df[split == "test"].copy()

# ---- fit every cross-transaction statistic on TRAIN ONLY ----
stats = fe.fit_global_stats(train_raw)

train_feat = fe.apply_global_stats(train_raw, stats)
val_feat = fe.apply_global_stats(val_raw, stats)
test_feat = fe.apply_global_stats(test_raw, stats)

train_feat, encoders = fe.finalize_matrix(train_feat, encoders=None)
val_feat, _ = fe.finalize_matrix(val_feat, encoders=encoders)
test_feat, _ = fe.finalize_matrix(test_feat, encoders=encoders)

# guard against a category appearing only in val/test producing a column
# train never saw (or vice versa) -- align everything to the train schema
val_feat = val_feat.reindex(columns=train_feat.columns, fill_value=0)
test_feat = test_feat.reindex(columns=train_feat.columns, fill_value=0)

# TransactionID is a row-identity key for downstream joins (e.g. the demo
# app's identifier-search tab), NOT a model feature -- exclude it here.
feature_cols = [c for c in train_feat.columns if c != "TransactionID"]
print(f"\nEngineered {len(feature_cols)} features:")
print(" ", ", ".join(feature_cols))

features = pd.concat([train_feat, val_feat, test_feat], ignore_index=True)
features["split"] = pd.Series(["train"] * len(train_feat) + ["val"] * len(val_feat) + ["test"] * len(test_feat))

# ---- StandardScaler for the unsupervised detectors: fit on TRAIN only ----
scaler = StandardScaler().fit(train_feat[feature_cols])
scaled_train = pd.DataFrame(scaler.transform(train_feat[feature_cols]), columns=feature_cols)
scaled_val = pd.DataFrame(scaler.transform(val_feat[feature_cols]), columns=feature_cols)
scaled_test = pd.DataFrame(scaler.transform(test_feat[feature_cols]), columns=feature_cols)
features_scaled = pd.concat([scaled_train, scaled_val, scaled_test], ignore_index=True)
features_scaled["split"] = features["split"].values

features.to_csv(config.FEATURES_CSV, index=False)
features_scaled.to_csv(config.FEATURES_SCALED_CSV, index=False)
joblib.dump(scaler, config.SCALER_PKL)

# ---- production reference ----
# IMPORTANT: `stats` (type_avg / device_counts / ip_counts / merchant_counts /
# median_gap_hours) must be the SAME train-fit dict used to build the model's
# training features, NOT a full-dataset refit. Refitting these on the full
# dataset silently breaks inference: full-dataset device/IP/merchant counts
# run ~1.5-2x higher than train-fold counts (more rows contribute to each
# count), so a genuinely new transaction would get count *features* on a
# different scale than the one the model's tree splits were calibrated
# against -- verified empirically to push predicted P(fraud) for almost
# every ordinary transaction above 0.6 (mean 0.96 replaying the full raw
# dataset through this path), a silent miscalibration bug, not a real
# fraud signal. `account_history` is fine to build from the FULL dataset
# (train+val+test): it is a per-account snapshot of real state, not a
# population-scale statistic, so it correctly reflects "everything known
# about this account so far" for a genuinely new incoming transaction.
full_account_history = fe.build_account_history(df)
reference = {
    "stats": stats,                # TRAIN-fit stats -- same scale the model was trained on
    "account_history": full_account_history,
    "encoders": encoders,          # keep the TRAIN-fit encoders so the model's input schema matches exactly what it was trained on
    "feature_cols": feature_cols,  # exact column order the shipped model expects
}
joblib.dump(reference, config.REFERENCE_PKL)

print(f"\nSaved:")
print(f"  {config.FEATURES_CSV}")
print(f"  {config.FEATURES_SCALED_CSV}")
print(f"  {config.REFERENCE_PKL}")
print(f"  {config.SCALER_PKL}")
