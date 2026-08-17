"""
data_validation.py -- schema and data-quality checks run before any modelling.

These checks encode facts already verified against the real Base.csv (see
01-DATASET-BIBLE.md) so a future re-run on a different variant/export fails
loudly instead of silently producing wrong features.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger("fraud_detection.data_validation")

EXPECTED_COLUMNS = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "payment_type", "zip_count_4w", "velocity_6h",
    "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "employment_status",
    "credit_risk_score", "email_is_free", "housing_status",
    "phone_home_valid", "phone_mobile_valid", "bank_months_count",
    "has_other_cards", "proposed_credit_limit", "foreign_request", "source",
    "session_length_in_minutes", "device_os", "keep_alive_session",
    "device_distinct_emails_8w", "device_fraud_count", "month", "fraud_bool",
]


class DataValidationError(ValueError):
    pass


def validate_schema(df: pd.DataFrame, cfg: Config) -> None:
    """Raise if the raw file doesn't look like BAF Base at all."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing expected BAF columns: {sorted(missing)}")

    target = cfg.data.target_col
    if target not in df.columns:
        raise DataValidationError(f"Target column '{target}' not found")
    if not set(df[target].unique()).issubset({0, 1}):
        raise DataValidationError(f"Target column '{target}' is not binary 0/1")

    logger.info("Schema OK: all %d expected columns present", len(EXPECTED_COLUMNS))


def validate_quality(df: pd.DataFrame, cfg: Config) -> dict:
    """
    Run and log the quality checks that decide the rest of the pipeline
    (see 01-DATASET-BIBLE.md section 4, "the checks that decide your
    architecture"). Returns a dict of findings for the report.
    """
    findings: dict = {}

    fraud_rate = df[cfg.data.target_col].mean()
    findings["fraud_rate"] = float(fraud_rate)
    logger.info("Fraud rate: %.4f%% (do-nothing accuracy would be %.4f%%)",
                100 * fraud_rate, 100 * (1 - fraud_rate))

    constant_cols = [
        c for c in df.columns
        if c != cfg.data.target_col and df[c].nunique(dropna=False) <= 1
    ]
    findings["constant_columns"] = constant_cols
    logger.info("Constant columns (to be dropped): %s", constant_cols)

    sentinel_fracs = {}
    for col in cfg.sentinel_cols:
        if col in df.columns:
            sentinel_fracs[col] = float((df[col] < 0).mean())
    findings["sentinel_missing_fractions"] = sentinel_fracs
    logger.info("Sentinel (-1) fractions: %s", sentinel_fracs)

    legit_neg = {}
    for col in cfg.legitimate_negative_cols:
        if col in df.columns:
            legit_neg[col] = float((df[col] < 0).mean())
    findings["legitimate_negative_fractions"] = legit_neg
    logger.info("Legitimate-negative columns, fraction negative: %s", legit_neg)

    dup_rows = int(df.duplicated().sum())
    findings["duplicate_rows"] = dup_rows
    if dup_rows:
        logger.warning("%d fully duplicated rows found", dup_rows)

    id_pattern = re.compile(r"(^id$|_id$|^id_|account_id|customer_id)", re.IGNORECASE)
    id_like = [c for c in df.columns if id_pattern.search(c) and c != cfg.data.target_col]
    findings["identifier_columns_found"] = id_like
    logger.info(
        "Identifier-style columns found: %s -- %s",
        id_like,
        "none, so no identifier-exclusion step is needed" if not id_like else
        "review before modelling",
    )

    cat_cardinality = {
        c: int(df[c].nunique()) for c in cfg.categorical_cols if c in df.columns
    }
    findings["categorical_cardinality"] = cat_cardinality
    logger.info("Categorical cardinalities: %s", cat_cardinality)

    if "month" in df.columns:
        by_month = df.groupby("month")[cfg.data.target_col].agg(["mean", "size"])
        findings["fraud_rate_by_month"] = by_month.to_dict()
        logger.info("Fraud rate by month:\n%s", by_month.to_string())

    return findings


def check_no_leakage(df: pd.DataFrame) -> list[str]:
    """
    Scan column names for terms that would indicate post-decision leakage
    (chargeback outcome, investigation result, etc.). On BAF Base, none
    exist -- every column is available at application time.
    """
    leakage_terms = ["chargeback", "investigation", "outcome", "resolved",
                      "settled", "dispute_result", "label_source"]
    hits = [c for c in df.columns if any(t in c.lower() for t in leakage_terms)]
    if hits:
        logger.warning("Potential leakage columns found: %s", hits)
    else:
        logger.info("No leakage-style columns found in schema.")
    return hits
