import pytest

from src.data_validation import DataValidationError, check_no_leakage, validate_quality, validate_schema


def test_validate_schema_passes_on_synthetic_data(synthetic_baf_df, cfg):
    validate_schema(synthetic_baf_df, cfg)  # should not raise


def test_validate_schema_fails_on_missing_columns(synthetic_baf_df, cfg):
    bad = synthetic_baf_df.drop(columns=["credit_risk_score"])
    with pytest.raises(DataValidationError):
        validate_schema(bad, cfg)


def test_validate_schema_fails_on_non_binary_target(synthetic_baf_df, cfg):
    bad = synthetic_baf_df.copy()
    bad.loc[0, "fraud_bool"] = 2
    with pytest.raises(DataValidationError):
        validate_schema(bad, cfg)


def test_validate_quality_detects_constant_column(synthetic_baf_df, cfg):
    findings = validate_quality(synthetic_baf_df, cfg)
    assert "device_fraud_count" in findings["constant_columns"]


def test_validate_quality_detects_sentinel_fractions(synthetic_baf_df, cfg):
    findings = validate_quality(synthetic_baf_df, cfg)
    assert findings["sentinel_missing_fractions"]["prev_address_months_count"] > 0


def test_check_no_leakage_finds_nothing_on_clean_schema(synthetic_baf_df):
    assert check_no_leakage(synthetic_baf_df) == []


def test_check_no_leakage_flags_suspicious_column(synthetic_baf_df):
    bad = synthetic_baf_df.copy()
    bad["chargeback_flag"] = 0
    assert "chargeback_flag" in check_no_leakage(bad)
