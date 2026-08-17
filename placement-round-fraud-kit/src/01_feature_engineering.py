"""
STAGE 1 — Feature engineering.

Reads the raw transaction log, drops the broken PreviousTransactionDate
column and the raw TransactionID/AccountID identifiers, builds the
behavioral features (per-account deviation, novelty flags, recency,
device/IP/merchant popularity), encodes categoricals, and writes two
feature matrices to artifacts/:

  features.csv         - unscaled, used by the tree-based supervised models
                          (XGBoost / Decision Tree don't need scaling)
  features_scaled.csv   - StandardScaler-scaled, used by the distance-based
                          unsupervised detectors in Stage 2 (LOF, One-Class
                          SVM and MCD all assume comparable feature scales;
                          Isolation Forest doesn't strictly need it but is
                          unaffected by it)

Also persists reference.pkl (account history + global lookup stats) and
scaler.pkl so the Streamlit demo can engineer a brand-new transaction's
features identically to how training data was engineered.
"""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import fe_utils as fe

print(f"Loading raw data from {config.RAW_CSV}")
df = fe.load_raw(config.RAW_CSV)
print(f"  {len(df)} transactions across {df['AccountID'].nunique()} accounts")

features, reference = fe.fit_engineer(df)
print(f"Engineered {features.shape[1]} features:")
print(" ", ", ".join(features.columns))

scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features), columns=features.columns, index=features.index
)

features.to_csv(config.FEATURES_CSV, index=False)
features_scaled.to_csv(config.FEATURES_SCALED_CSV, index=False)
joblib.dump(reference, config.REFERENCE_PKL)
joblib.dump(scaler, config.SCALER_PKL)

print(f"\nSaved:")
print(f"  {config.FEATURES_CSV}")
print(f"  {config.FEATURES_SCALED_CSV}")
print(f"  {config.REFERENCE_PKL}")
print(f"  {config.SCALER_PKL}")
