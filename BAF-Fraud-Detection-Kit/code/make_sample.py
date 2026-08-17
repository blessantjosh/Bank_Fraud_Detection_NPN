"""
make_sample.py -- generate a BAF-shaped synthetic sample.

Purpose: let you build and debug your ENTIRE pipeline before the real data
finishes downloading, and verify the toolkit works on your machine.

It reproduces the real schema exactly: same column names, same ranges, the
same -1 sentinel encodings, the same ~1.1% positive rate, the same 8 months
with drifting prevalence, and a deliberate age-linked false-positive bias so
the fairness code has something real to find.

    python make_sample.py --rows 60000 --out sample_train.csv

This is NOT the real data and scores on it mean nothing. It is a test harness.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def make(n: int = 60_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    month = rng.integers(0, 8, n)
    # Real BAF prevalence drifts between ~0.85% and ~1.5% across months.
    month_effect = np.array([0.85, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5])[month]

    age = rng.choice(np.arange(10, 100, 10), n, p=[.02, .14, .22, .2, .16, .12, .08, .04, .02])
    income = np.round(rng.choice(np.arange(0.1, 1.0, 0.1), n), 1)
    name_email_sim = rng.beta(2, 2, n)
    credit_risk = rng.normal(100, 90, n).clip(-191, 389)
    prop_limit = rng.choice([200, 500, 1000, 1500, 2000], n, p=[.3, .3, .2, .15, .05])

    vel_4w = rng.uniform(2825, 7020, n)
    vel_24h = rng.uniform(1297, 9586, n)
    vel_6h = rng.normal(5000, 3000, n).clip(-175, 16818)

    dob_emails = rng.poisson(1.2, n).clip(0, 39)
    zip_count = rng.gamma(2, 200, n).clip(1, 6830)

    # Sentinel-bearing columns: -1 marks missing, exactly as in the real data.
    prev_addr = np.where(rng.random(n) < 0.72, -1, rng.uniform(0, 380, n))
    curr_addr = np.where(rng.random(n) < 0.02, -1, rng.uniform(0, 429, n))
    bank_months = np.where(rng.random(n) < 0.25, -1, rng.uniform(0, 32, n))
    session_len = np.where(rng.random(n) < 0.03, -1, rng.gamma(2, 4, n).clip(0, 107))
    dev_emails = np.where(rng.random(n) < 0.02, -1, rng.choice([1, 2], n, p=[.95, .05]))
    balcon = np.where(rng.random(n) < 0.74, rng.uniform(-16, -1, n), rng.uniform(0, 114, n))

    phone_home = rng.binomial(1, 0.42, n)
    phone_mob = rng.binomial(1, 0.89, n)
    email_free = rng.binomial(1, 0.53, n)
    other_cards = rng.binomial(1, 0.22, n)
    foreign = rng.binomial(1, 0.025, n)
    keep_alive = rng.binomial(1, 0.58, n)

    # Latent fraud risk. Signals mirror the real domain logic so that the
    # engineered features in baf.py have something genuine to pick up.
    logit = (
        -6.6
        + 1.5 * (1 - name_email_sim)                    # synthetic identity
        - 0.011 * credit_risk                           # bank's own score
        + 0.9 * (prev_addr < 0)                         # thin file
        + 0.6 * (bank_months < 0)
        + 0.25 * dob_emails                             # DOB reused across emails
        + 1.1 * (1 - phone_mob) * (1 - phone_home)      # uncontactable
        + 0.0009 * prop_limit / (income + 0.05) * 0.35  # incoherent ask
        + 0.35 * (vel_6h / vel_4w)                      # burst
        + 0.5 * email_free * (1 - name_email_sim)
        + 0.4 * foreign
        - 0.5 * other_cards
        + np.log(month_effect)
    )
    p = 1 / (1 + np.exp(-logit))
    p *= 0.011 / p.mean()  # calibrate to a realistic ~1.1% base rate
    fraud = rng.binomial(1, p.clip(0, 1))

    # Deliberate bias: older applicants look riskier on the OBSERVED features
    # without actually being more fraudulent. This is what drives the
    # false-positive disparity the paper reports, and gives fairness code
    # a genuine effect to detect.
    older = age >= 50
    credit_risk = np.where(older, credit_risk - 30, credit_risk)
    prev_addr = np.where(older & (rng.random(n) < 0.15), -1, prev_addr)

    df = pd.DataFrame({
        "fraud_bool": fraud,
        "income": income,
        "name_email_similarity": name_email_sim,
        "prev_address_months_count": prev_addr,
        "current_address_months_count": curr_addr,
        "customer_age": age,
        "days_since_request": rng.exponential(1.5, n).clip(0, 79),
        "intended_balcon_amount": balcon,
        "payment_type": rng.choice([f"AA{c}" for c in "ABCDE"], n),
        "zip_count_4w": zip_count,
        "velocity_6h": vel_6h,
        "velocity_24h": vel_24h,
        "velocity_4w": vel_4w,
        "bank_branch_count_8w": rng.gamma(2, 100, n).clip(0, 2404),
        "date_of_birth_distinct_emails_4w": dob_emails,
        "employment_status": rng.choice([f"C{c}" for c in "ABCDEFG"], n,
                                        p=[.7, .1, .07, .05, .04, .03, .01]),
        "credit_risk_score": credit_risk,
        "email_is_free": email_free,
        "housing_status": rng.choice([f"B{c}" for c in "ABCDEFG"], n,
                                     p=[.37, .3, .15, .08, .05, .04, .01]),
        "phone_home_valid": phone_home,
        "phone_mobile_valid": phone_mob,
        "bank_months_count": bank_months,
        "has_other_cards": other_cards,
        "proposed_credit_limit": prop_limit,
        "foreign_request": foreign,
        "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[.993, .007]),
        "session_length_in_minutes": session_len,
        "device_os": rng.choice(["windows", "other", "linux", "macintosh", "x11"], n,
                                p=[.35, .3, .22, .08, .05]),
        "keep_alive_session": keep_alive,
        "device_distinct_emails_8w": dev_emails,
        "device_fraud_count": np.zeros(n, dtype=int),   # constant, as in BAF Base
        "month": month,
    })
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60_000)
    ap.add_argument("--out", default="sample_train.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = make(a.rows, a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  shape={df.shape}  fraud_rate={df.fraud_bool.mean():.4f}")
    print(df.groupby("month")["fraud_bool"].agg(["mean", "size"]))
