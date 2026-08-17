"""Shared fixtures: a small synthetic BAF-shaped frame, fast enough for unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config

N = 2000


@pytest.fixture(scope="session")
def cfg():
    return load_config(Path(__file__).resolve().parent.parent / "config.yaml")


@pytest.fixture()
def synthetic_baf_df():
    rng = np.random.default_rng(42)
    n = N
    df = pd.DataFrame({
        "income": rng.uniform(0.1, 0.9, n),
        "name_email_similarity": rng.uniform(0, 1, n),
        "prev_address_months_count": rng.choice(
            [-1] + list(range(0, 380)), n
        ).astype(float),
        "current_address_months_count": rng.choice(
            [-1] + list(range(0, 429)), n
        ).astype(float),
        "customer_age": rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], n),
        "days_since_request": rng.uniform(0, 79, n),
        "intended_balcon_amount": rng.choice(
            list(range(-16, 0)) + list(range(0, 114)), n
        ).astype(float),
        "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], n),
        "zip_count_4w": rng.integers(1, 6830, n),
        "velocity_6h": rng.uniform(-175, 16818, n),
        "velocity_24h": rng.uniform(1297, 9586, n),
        "velocity_4w": rng.uniform(2825, 7020, n),
        "bank_branch_count_8w": rng.integers(0, 2404, n),
        "date_of_birth_distinct_emails_4w": rng.integers(0, 39, n),
        "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE"], n),
        "credit_risk_score": rng.uniform(-191, 389, n),
        "email_is_free": rng.integers(0, 2, n),
        "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE"], n),
        "phone_home_valid": rng.integers(0, 2, n),
        "phone_mobile_valid": rng.integers(0, 2, n),
        "bank_months_count": rng.choice([-1] + list(range(0, 32)), n).astype(float),
        "has_other_cards": rng.integers(0, 2, n),
        "proposed_credit_limit": rng.uniform(200, 2000, n),
        "foreign_request": rng.integers(0, 2, n),
        "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[0.95, 0.05]),
        "session_length_in_minutes": rng.choice([-1] + list(range(0, 107)), n).astype(float),
        "device_os": rng.choice(["windows", "macintosh", "linux", "x11", "other"], n),
        "keep_alive_session": rng.integers(0, 2, n),
        "device_distinct_emails_8w": rng.choice([-1, 0, 1, 2], n).astype(float),
        "device_fraud_count": np.zeros(n),
        "month": rng.integers(0, 8, n),
    })
    fraud_prob = 0.011 + 0.05 * (df["credit_risk_score"] < -50).astype(float)
    df["fraud_bool"] = (rng.uniform(0, 1, n) < fraud_prob).astype(int)
    return df
