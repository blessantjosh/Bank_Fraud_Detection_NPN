"""
STAGE 4 — Stratified split + data balancing.

Fraud prevalence in the generated label is ~5% (minority class). We split
BEFORE resampling and only resample the training fold -- applying SMOTE
before splitting would let synthetic points derived from a test-fold
minority sample leak into training, inflating test scores.

SMOTE vs class-weighting:
  - SMOTE synthesizes new minority points by interpolating between a real
    minority point and its k nearest minority neighbors. With ~2000 rows and
    a ~5% minority class, the minority fold has enough points (~100+) for
    k=5 interpolation to stay meaningful without just re-drawing near-duplicates.
  - class_weight / scale_pos_weight instead just reweights the loss, using
    no synthetic data. It's the safer choice when the minority class is so
    tiny (dozens of rows) that SMOTE's k-NN neighborhoods become noisy.
  - Given this dataset's minority count, we use SMOTE as primary and also
    train a class-weighted model in Stage 5 for direct comparison -- the
    actual counts are printed below so the choice is data-driven, not assumed.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import config

df = pd.read_csv(config.LABELED_CSV)
y = df["is_fraud"]
X = df.drop(columns=["vote_count", "risk_tier", "is_fraud"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
)
print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud, {y_train.mean()*100:.2f}%)")
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
    "X_test": X_test, "y_test": y_test,
    "scale_pos_weight": scale_pos_weight,
}, config.SPLIT_PKL)
print(f"\nSaved {config.SPLIT_PKL}")
