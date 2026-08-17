import numpy as np
import pandas as pd

from src.preprocessing import Preprocessor, drop_constant_columns, to_nan_and_flag


def test_to_nan_and_flag_converts_sentinel_and_flags(synthetic_baf_df):
    out = to_nan_and_flag(synthetic_baf_df, ["prev_address_months_count"])
    assert "prev_address_months_count_is_missing" in out.columns
    was_neg = synthetic_baf_df["prev_address_months_count"] < 0
    assert out.loc[was_neg, "prev_address_months_count"].isna().all()
    assert (out["prev_address_months_count_is_missing"] == was_neg.astype(int)).all()


def test_to_nan_and_flag_does_not_touch_legitimate_negative_columns(synthetic_baf_df):
    before = synthetic_baf_df["credit_risk_score"].copy()
    out = to_nan_and_flag(synthetic_baf_df, ["prev_address_months_count"])
    assert (out["credit_risk_score"] == before).all()
    assert "credit_risk_score_is_missing" not in out.columns


def test_drop_constant_columns_detects_device_fraud_count(synthetic_baf_df):
    const = drop_constant_columns(synthetic_baf_df, "fraud_bool")
    assert "device_fraud_count" in const


def test_drop_constant_columns_can_protect_a_column(synthetic_baf_df):
    df = synthetic_baf_df.copy()
    df["month"] = 3  # force constant
    const = drop_constant_columns(df, "fraud_bool", keep=["month"])
    assert "month" not in const


def test_preprocessor_fit_transform_tree_has_no_sentinel_negatives(synthetic_baf_df, cfg):
    pre = Preprocessor(cfg)
    pre.fit(synthetic_baf_df)
    X = pre.transform_tree(synthetic_baf_df)
    for col in cfg.sentinel_cols:
        assert (X[col].dropna() >= 0).all()
    assert "device_fraud_count" not in X.columns


def test_preprocessor_dense_has_no_nan_and_is_numeric(synthetic_baf_df, cfg):
    pre = Preprocessor(cfg)
    pre.fit(synthetic_baf_df)
    X = pre.transform_dense(synthetic_baf_df)
    assert not X.isna().any().any()
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)


def test_preprocessor_transform_is_stable_on_unseen_rows(synthetic_baf_df, cfg):
    pre = Preprocessor(cfg)
    train = synthetic_baf_df.iloc[:1500]
    holdout = synthetic_baf_df.iloc[1500:]
    pre.fit(train)
    X_tree = pre.transform_tree(holdout)
    X_dense = pre.transform_dense(holdout)
    assert len(X_tree) == len(holdout)
    assert len(X_dense) == len(holdout)
    assert list(X_dense.columns) == pre.feature_cols_dense_
