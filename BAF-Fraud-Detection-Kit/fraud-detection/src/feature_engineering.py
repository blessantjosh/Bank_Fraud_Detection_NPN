"""
feature_engineering.py -- engineered features tied to concrete fraud patterns.

This dataset is ACCOUNT-OPENING fraud: there is no transaction amount, no
transaction timestamp, and no account/customer id. Per the task spec we
therefore explicitly SKIP:
  - amount_log, transactions_per_hour            (no transaction amounts)
  - hour / day_of_week / is_weekend               (no real timestamp, only a
                                                    coarse 0-7 month index)
  - current_amount / historical_average deviation (no transaction history)
No substitute columns were invented for any of the above.

Every feature below is reused/adapted from the already-verified `baf.py`
toolkit and is tied to one of three fraud archetypes for THIS dataset:
  (A) synthetic identity  -- fabricated person, thin file, incoherent attributes
  (B) identity theft      -- real person, wrong human, contactability fails
  (C) mule farming        -- bulk applications, shared attributes, bursts
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("fraud_detection.feature_engineering")

EPS = 1e-6


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- (C) Velocity ratios --------------------------------------------
    # velocity_6h/24h/4w share units (applications/hour) over different
    # windows, so their ratios measure acceleration against baseline.
    # velocity_6h has legitimate negative values (generator artefact) --
    # clip at 0 before forming a ratio, but keep the raw column untouched.
    if "velocity_6h" in df.columns:
        v6 = df["velocity_6h"].clip(lower=0)
        if "velocity_4w" in df.columns:
            df["velocity_burst_6h_4w"] = v6 / (df["velocity_4w"] + EPS)
        if "velocity_24h" in df.columns:
            df["velocity_ratio_6h_24h"] = v6 / (df["velocity_24h"] + EPS)
    if {"velocity_24h", "velocity_4w"}.issubset(df.columns):
        df["velocity_burst_24h_4w"] = df["velocity_24h"] / (df["velocity_4w"] + EPS)

    # --- (A) Synthetic identity: email/name coherence --------------------
    if {"name_email_similarity", "email_is_free"}.issubset(df.columns):
        df["email_mismatch_free"] = (1.0 - df["name_email_similarity"]) * df["email_is_free"]
    if {"date_of_birth_distinct_emails_4w", "name_email_similarity"}.issubset(df.columns):
        df["dob_emails_x_mismatch"] = (
            df["date_of_birth_distinct_emails_4w"] * (1.0 - df["name_email_similarity"])
        )

    # --- (A) Thin file: no history because identity is new ---------------
    if {"prev_address_months_count", "current_address_months_count"}.issubset(df.columns):
        df["total_address_history"] = (
            df["prev_address_months_count"].fillna(0) + df["current_address_months_count"].fillna(0)
        )
    thin_parts = [c for c in ["prev_address_months_count_is_missing",
                              "bank_months_count_is_missing"] if c in df.columns]
    if thin_parts:
        df["thin_file_score"] = df[thin_parts].sum(axis=1).astype("int8")

    # Cross-column aggregate: how many independent history checks came back
    # empty. A tree can isolate any single -1 with one split, but this
    # sum is genuinely new information.
    sentinel_cols = [
        "prev_address_months_count", "current_address_months_count",
        "bank_months_count", "session_length_in_minutes",
        "device_distinct_emails_8w", "intended_balcon_amount",
    ]
    miss_cols = [f"{c}_is_missing" for c in sentinel_cols if f"{c}_is_missing" in df.columns]
    if len(miss_cols) >= 2:
        df["n_missing"] = df[miss_cols].sum(axis=1).astype("int8")

    # --- (B) Contactability -----------------------------------------------
    phones = [c for c in ["phone_home_valid", "phone_mobile_valid"] if c in df.columns]
    if len(phones) == 2:
        df["n_valid_phones"] = df["phone_home_valid"] + df["phone_mobile_valid"]
        df["no_valid_phone"] = (df["n_valid_phones"] == 0).astype("int8")

    # --- (A) Financial coherence -------------------------------------------
    if {"proposed_credit_limit", "income"}.issubset(df.columns):
        df["limit_to_income"] = df["proposed_credit_limit"] / (df["income"] + EPS)
    if {"proposed_credit_limit", "credit_risk_score"}.issubset(df.columns):
        df["limit_per_risk"] = df["proposed_credit_limit"] / (df["credit_risk_score"] + 200.0)
    if {"credit_risk_score", "income"}.issubset(df.columns):
        df["risk_x_income"] = df["credit_risk_score"] * df["income"]

    # --- (C) Device & session behaviour ------------------------------------
    if {"session_length_in_minutes", "device_distinct_emails_8w"}.issubset(df.columns):
        df["emails_per_session_min"] = (
            df["device_distinct_emails_8w"] / (df["session_length_in_minutes"] + 1.0)
        )
    if {"keep_alive_session", "session_length_in_minutes"}.issubset(df.columns):
        df["short_session_no_keepalive"] = (
            (df["session_length_in_minutes"] < 5) & (df["keep_alive_session"] == 0)
        ).astype("int8")

    # --- (C) Geographic / branch clustering --------------------------------
    if {"zip_count_4w", "velocity_4w"}.issubset(df.columns):
        df["zip_density_vs_velocity"] = df["zip_count_4w"] / (df["velocity_4w"] + EPS)

    return df


SKIPPED_FEATURES = {
    "amount_log": "no transaction amount column exists in account-opening data",
    "transactions_per_hour": "no transaction history exists; only application-time features",
    "hour": "no real timestamp, only a coarse 0-7 month index",
    "day_of_week": "no real timestamp, only a coarse 0-7 month index",
    "is_weekend": "no real timestamp, only a coarse 0-7 month index",
    "current_amount_vs_historical_average": "no transaction history to compute a historical average from",
}
