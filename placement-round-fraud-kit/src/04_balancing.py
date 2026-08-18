"""
STAGE 4 -- Chronological split (already applied in Stage 1) + data balancing.

LEAKAGE FIX (see ML_AUDIT_AFTER_FIX.md): the previous version of this stage
performed a RANDOM stratified train/test split here, after Stage 1/2/3 had
already computed global statistics and fit the anomaly detectors on the
complete dataset -- so by the time the split happened, test-set information
had already leaked into every upstream artifact.

The split now happens FIRST, chronologically, in Stage 1 (train = earliest
transactions, val = next slice, test = latest slice) -- see config.py's
TRAIN_QUANTILE/VAL_QUANTILE and ML_AUDIT_AFTER_FIX.md for why a chronological
split is the right choice for a transaction/fraud problem (train on the
past, evaluate on the future -- what a real deployment would face). This
stage only reads the split column Stage 1/2/3 already produced.

We resample only the TRAIN fold. Applying SMOTE before splitting -- or
resampling val/test -- would let synthetic points derived from later-fold
minority rows leak into training and inflate downstream scores.

SMOTE vs class-weighting:
  - SMOTE synthesizes new minority points by interpolating between a real
    minority point and its k nearest minority neighbors.
  - class_weight / scale_pos_weight instead just reweights the loss, using
    no synthetic data. It's the safer choice when the minority class is so
    tiny (dozens of rows) that SMOTE's k-NN neighborhoods become noisy.
  - We train both (Stage 5) and compare on the untouched test fold (Stage 6)
    rather than assuming one is better -- the actual counts are printed
    below so the choice is data-driven, not assumed.

VAL is used only for the cost-based threshold sweep (Stage 6) and for
picking which XGBoost variant is primary -- never for a metric reported as
the final unbiased estimate. TEST is never touched until the single, final
evaluation pass.
"""
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE

import config

df = pd.read_csv(config.LABELED_CSV)
# TransactionID is a row-identity key carried through for downstream joins
# (e.g. the demo app's identifier-search tab) -- not a model feature.
drop_cols = ["vote_count", "risk_tier", "is_fraud", "split", "TransactionID"]

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

X_train, y_train = train_df.drop(columns=drop_cols), train_df["is_fraud"]
X_val, y_val = val_df.drop(columns=drop_cols), val_df["is_fraud"]
X_test, y_test = test_df.drop(columns=drop_cols), test_df["is_fraud"]

print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud, {y_train.mean()*100:.2f}%)")
print(f"Val:   {len(X_val)} rows ({y_val.sum()} fraud, {y_val.mean()*100:.2f}%)")
print(f"Test:  {len(X_test)} rows ({y_test.sum()} fraud, {y_test.mean()*100:.2f}%)")

minority_count = y_train.sum()
k_neighbors = min(5, minority_count - 1)
print(f"\nMinority class in training fold: {minority_count} rows -> SMOTE k_neighbors={k_neighbors}")

smote = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_train_smote)} rows ({y_train_smote.sum()} fraud, "
      f"{y_train_smote.mean()*100:.2f}%)")

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"\nAlternative: class-weighting via scale_pos_weight={scale_pos_weight:.2f} "
      f"(no synthetic rows, trained directly on the {len(X_train)}-row imbalanced fold)")

joblib.dump({
    "X_train": X_train, "y_train": y_train,
    "X_train_smote": X_train_smote, "y_train_smote": y_train_smote,
    "X_val": X_val, "y_val": y_val,
    "X_test": X_test, "y_test": y_test,
    "scale_pos_weight": scale_pos_weight,
}, config.SPLIT_PKL)
print(f"\nSaved {config.SPLIT_PKL}")
