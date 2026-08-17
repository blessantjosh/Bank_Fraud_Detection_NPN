"""
preprocessing.py -- sentinel handling, constant-column dropping, and the
tree-native / dense feature views used by the different model families.

Everything that "learns" from data (which columns are constant, imputation
medians, one-hot categories, scaler mean/std) is fit on the TRAIN split only
and then applied unchanged to val/test/new data, via the `Preprocessor` class.
Everything else (sentinel -> NaN, feature engineering) is a pure deterministic
transform with no fitted state.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config
from src.feature_engineering import add_features

logger = logging.getLogger("fraud_detection.preprocessing")


def drop_constant_columns(df: pd.DataFrame, target_col: str, keep: list[str] | None = None) -> list[str]:
    """Return the list of zero-variance columns (excluding target) to drop."""
    keep = keep or []
    const = [
        c for c in df.columns
        if c != target_col and c not in keep and df[c].nunique(dropna=False) <= 1
    ]
    return const


def to_nan_and_flag(df: pd.DataFrame, sentinel_cols: list[str]) -> pd.DataFrame:
    """
    Convert sentinel negatives to NaN and add an explicit `_is_missing` flag.

    Missingness is itself predictive here (a synthetic identity has no
    previous address because it was invented last week), so we keep both
    the flag and a true NaN rather than imputing the sentinel away. This is
    also a correctness prerequisite: without it, -1 silently corrupts every
    ratio built from these columns in feature_engineering.add_features.
    """
    df = df.copy()
    for col in sentinel_cols:
        if col not in df.columns:
            continue
        miss = df[col] < 0
        df[f"{col}_is_missing"] = miss.astype("int8")
        df.loc[miss, col] = np.nan
    return df


def set_categorical_dtype(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


class Preprocessor:
    """
    Fit on the training split only. Produces two views of the same data:

      - `transform_tree(df)`  : NaN kept, categoricals as pandas `category`
                                dtype. Used by LightGBM and XGBoost, both of
                                which route missing values and categorical
                                splits natively.
      - `transform_dense(df)` : median-imputed, one-hot encoded, standard
                                scaled numeric matrix. Used by Logistic
                                Regression and Random Forest, neither of
                                which accepts NaN or raw category dtype.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.target_col = cfg.data.target_col
        self.sentinel_cols = list(cfg.sentinel_cols)
        self.categorical_cols = list(cfg.categorical_cols)
        self.constant_cols_: list[str] = []
        self.feature_cols_tree_: list[str] = []
        self.feature_cols_dense_: list[str] = []
        self.numeric_cols_: list[str] = []
        self._imputer: SimpleImputer | None = None
        self._encoder: OneHotEncoder | None = None
        self._scaler: StandardScaler | None = None
        self.fitted_ = False

    # -- shared deterministic prep, no fitted state ------------------------
    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[c for c in self.constant_cols_ if c in df.columns], errors="ignore")
        df = to_nan_and_flag(df, self.sentinel_cols)
        df = add_features(df)
        return df

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        logger.info("Fitting Preprocessor on %d training rows", len(train_df))
        self.constant_cols_ = drop_constant_columns(
            train_df, self.target_col, keep=[self.cfg.data.month_col]
        )
        logger.info("Constant columns dropped: %s", self.constant_cols_)

        base = self._base_transform(train_df)
        feature_df = base.drop(columns=[self.target_col])
        self.feature_cols_tree_ = list(feature_df.columns)

        self.numeric_cols_ = [
            c for c in feature_df.columns if c not in self.categorical_cols
        ]
        cat_present = [c for c in self.categorical_cols if c in feature_df.columns]

        self._imputer = SimpleImputer(strategy="median")
        num_imputed = self._imputer.fit_transform(feature_df[self.numeric_cols_])

        self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        if cat_present:
            self._encoder.fit(feature_df[cat_present].astype(str))
            ohe_cols = list(self._encoder.get_feature_names_out(cat_present))
        else:
            ohe_cols = []

        self._scaler = StandardScaler()
        self._scaler.fit(num_imputed)

        self.feature_cols_dense_ = self.numeric_cols_ + ohe_cols
        self.fitted_ = True
        logger.info(
            "Preprocessor fitted: %d tree-native features, %d dense features",
            len(self.feature_cols_tree_), len(self.feature_cols_dense_),
        )
        return self

    def transform_tree(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted_, "call fit() first"
        base = self._base_transform(df)
        X = base.drop(columns=[self.target_col], errors="ignore")
        for c in self.feature_cols_tree_:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.feature_cols_tree_]
        X = set_categorical_dtype(X, self.categorical_cols)
        return X

    def transform_dense(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted_, "call fit() first"
        base = self._base_transform(df)
        X = base.drop(columns=[self.target_col], errors="ignore")
        for c in self.feature_cols_tree_:
            if c not in X.columns:
                X[c] = np.nan

        # float32 halves the memory footprint of the dense frame vs sklearn's
        # float64 default -- meaningful here since train/val/test dense frames
        # for every model family are held in memory for the whole ablation run.
        num = self._imputer.transform(X[self.numeric_cols_]).astype(np.float32)
        num = self._scaler.transform(num).astype(np.float32)
        num_df = pd.DataFrame(num, columns=self.numeric_cols_, index=X.index)

        cat_present = [c for c in self.categorical_cols if c in X.columns]
        if cat_present:
            ohe = self._encoder.transform(X[cat_present].astype(str))
            ohe_cols = list(self._encoder.get_feature_names_out(cat_present))
            ohe_df = pd.DataFrame(ohe, columns=ohe_cols, index=X.index)
            out = pd.concat([num_df, ohe_df], axis=1)
        else:
            out = num_df
        return out[self.feature_cols_dense_]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target_col].astype(int)
