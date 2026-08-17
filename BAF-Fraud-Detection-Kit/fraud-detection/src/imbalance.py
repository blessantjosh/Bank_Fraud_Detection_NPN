"""
imbalance.py -- the class-imbalance strategies compared in the ablation.

Strategies implemented (config.yaml -> imbalance.strategies):
  none               : raw class distribution, no correction
  class_weight       : class_weight="balanced" (LR/RF) / scale_pos_weight (LightGBM/XGBoost)
  random_undersample : keep all positives, sample N negatives per positive
  smote              : SMOTE oversampling of the minority class (train fold only)
  smote_undersample  : undersample majority first, then SMOTE the minority up

SMOTE needs a fully numeric, non-missing feature matrix, so SMOTE and
smote_undersample are run on the dense (imputed + one-hot + scaled) view of
the data for every model family, including LightGBM/XGBoost -- those two
otherwise use their native NaN/categorical handling for `none` and
`class_weight`. This is a deliberate, documented choice (see README), not an
oversight.

Random Forest is additionally capped to `rf_resampling_max_rows` for the
resampling-based strategies: fitting 300 trees on an oversampled >1M-row
frame is not feasible on an 8GB-RAM machine. See README "Compute constraints".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger("fraud_detection.imbalance")


def random_undersample(X: pd.DataFrame, y: pd.Series, ratio: int, seed: int = 42):
    """Keep all positives, keep `ratio` negatives per positive."""
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy()
    pos = np.flatnonzero(y_arr == 1)
    neg = np.flatnonzero(y_arr == 0)
    n_keep = min(len(pos) * ratio, len(neg))
    keep_neg = rng.choice(neg, size=n_keep, replace=False)
    idx = np.sort(np.concatenate([pos, keep_neg]))
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def smote_resample(X: pd.DataFrame, y: pd.Series, target_ratio: float, seed: int = 42):
    """
    SMOTE up to `target_ratio` = minority_count / majority_count.

    A controlled ratio (default 0.10, i.e. minority reaches 10% of majority)
    is used instead of the textbook 1:1 balance: on ~700k training rows a
    full 1:1 SMOTE would synthesize roughly 690k rows, which is unnecessary
    compute for a gain that ablation testing (see reports) does not support.
    """
    sm = SMOTE(sampling_strategy=target_ratio, random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res.reset_index(drop=True), y_res.reset_index(drop=True)


def smote_undersample_resample(
    X: pd.DataFrame, y: pd.Series, majority_ratio: float, target_ratio: float, seed: int = 42
):
    """
    Undersample the majority class down to `majority_ratio`:1 first (cheap),
    then SMOTE the minority class up to `target_ratio` of the (now smaller)
    majority. This keeps the expensive SMOTE k-NN step and the final
    training-set size small, which is exactly the "combined" strategy in
    the imbalance literature (e.g. SMOTE + Tomek/random undersampling).

    `target_ratio` must be strictly greater than 1/majority_ratio: SMOTE only
    ever adds synthetic minority rows, so the post-undersample minority ratio
    must already be below the ratio we ask SMOTE to reach.
    """
    rus = RandomUnderSampler(sampling_strategy=1.0 / majority_ratio, random_state=seed)
    X_u, y_u = rus.fit_resample(X, y)
    sm = SMOTE(sampling_strategy=target_ratio, random_state=seed)
    X_res, y_res = sm.fit_resample(X_u, y_u)
    return X_res.reset_index(drop=True), y_res.reset_index(drop=True)


def cap_rows(X: pd.DataFrame, y: pd.Series, max_rows: int, seed: int = 42):
    """Stratified subsample down to at most `max_rows`, used only for RF."""
    if len(X) <= max_rows:
        return X, y
    frac = max_rows / len(X)
    idx_pos = y[y == 1].index
    idx_neg = y[y == 0].index
    rng = np.random.default_rng(seed)
    keep_pos = rng.choice(idx_pos, size=max(1, int(len(idx_pos) * frac)), replace=False)
    keep_neg = rng.choice(idx_neg, size=max(1, int(len(idx_neg) * frac)), replace=False)
    idx = np.sort(np.concatenate([keep_pos, keep_neg]))
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def scale_pos_weight(y: pd.Series) -> float:
    pos = max(int((y == 1).sum()), 1)
    neg = int((y == 0).sum())
    return neg / pos


def apply_strategy(X: pd.DataFrame, y: pd.Series, strategy: str, cfg, seed: int = 42):
    """
    Returns (X_res, y_res, model_kwargs) where model_kwargs carries
    class_weight / scale_pos_weight to merge into the model constructor,
    or {} if the strategy is purely row-level resampling.
    """
    if strategy == "none":
        return X, y, {}
    if strategy == "class_weight":
        return X, y, {"class_weight": "balanced", "scale_pos_weight": scale_pos_weight(y)}
    if strategy == "random_undersample":
        ratio = cfg.imbalance.random_undersample_ratio
        X_res, y_res = random_undersample(X, y, ratio, seed)
        logger.info("random_undersample(%s:1) -> %d rows, positive rate %.4f",
                    ratio, len(y_res), y_res.mean())
        return X_res, y_res, {}
    if strategy == "smote":
        X_res, y_res = smote_resample(X, y, cfg.imbalance.smote_target_ratio, seed)
        logger.info("smote(target_ratio=%.2f) -> %d rows, positive rate %.4f",
                    cfg.imbalance.smote_target_ratio, len(y_res), y_res.mean())
        return X_res, y_res, {}
    if strategy == "smote_undersample":
        X_res, y_res = smote_undersample_resample(
            X, y,
            cfg.imbalance.smote_undersample_majority_ratio,
            cfg.imbalance.smote_undersample_final_ratio,
            seed,
        )
        logger.info("smote_undersample -> %d rows, positive rate %.4f", len(y_res), y_res.mean())
        return X_res, y_res, {}
    raise ValueError(f"Unknown imbalance strategy: {strategy}")
