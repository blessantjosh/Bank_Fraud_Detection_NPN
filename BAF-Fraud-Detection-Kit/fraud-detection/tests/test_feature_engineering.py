import numpy as np

from src.feature_engineering import SKIPPED_FEATURES, add_features
from src.preprocessing import to_nan_and_flag


def test_add_features_creates_expected_columns(synthetic_baf_df):
    df = to_nan_and_flag(synthetic_baf_df, [
        "prev_address_months_count", "current_address_months_count",
        "bank_months_count", "session_length_in_minutes",
        "device_distinct_emails_8w", "intended_balcon_amount",
    ])
    out = add_features(df)
    expected = [
        "velocity_burst_6h_4w", "velocity_ratio_6h_24h", "velocity_burst_24h_4w",
        "email_mismatch_free", "dob_emails_x_mismatch", "total_address_history",
        "thin_file_score", "n_missing", "n_valid_phones", "no_valid_phone",
        "limit_to_income", "limit_per_risk", "risk_x_income",
        "emails_per_session_min", "short_session_no_keepalive",
        "zip_density_vs_velocity",
    ]
    for col in expected:
        assert col in out.columns, f"missing engineered feature: {col}"


def test_velocity_ratio_uses_clipped_negative_velocity_6h(synthetic_baf_df):
    df = synthetic_baf_df.copy()
    df["velocity_6h"] = -50.0
    df["velocity_4w"] = 100.0
    out = add_features(df)
    # clip(lower=0) means a negative velocity_6h must never produce a negative ratio
    assert (out["velocity_burst_6h_4w"] >= 0).all()


def test_no_valid_phone_flag_is_consistent(synthetic_baf_df):
    out = add_features(synthetic_baf_df)
    both_invalid = (synthetic_baf_df["phone_home_valid"] == 0) & (synthetic_baf_df["phone_mobile_valid"] == 0)
    assert (out["no_valid_phone"] == both_invalid.astype(int)).all()


def test_skipped_features_are_documented():
    for name in ["amount_log", "transactions_per_hour", "hour", "day_of_week", "is_weekend"]:
        assert name in SKIPPED_FEATURES
        assert isinstance(SKIPPED_FEATURES[name], str) and len(SKIPPED_FEATURES[name]) > 0
