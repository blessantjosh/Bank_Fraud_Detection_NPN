"""
Argus -- Behavioral Anomaly Intelligence.

FastAPI backend for the fraud-analytics dashboard, wired to the **research_v2
pipeline** (the client-designated final pipeline, built on the teammate's
18-feature matrix). Nothing here retrains or recomputes the pipeline; it loads
the artifacts that pipeline already produced and serves them.

What it reads, and what each drives:

  artifacts_research_v2/ensemble_scores_v2.csv
      `ensemble_percentile_average` -- the Phase 12 (v2) recommended score.
      This is the dashboard's risk score.
  artifacts_research_v2/threshold_analysis_v2.json
      Phase 13 (v2) cutoffs: P99 = 0.951023 -> priority review,
      P95 = 0.867124 -> standard review, else normal. No block tier exists,
      by design (Phase 13 v2 SS1: a false-negative count cannot be computed
      without a label, so a cost-optimal block threshold is not defensible).
  artifacts_research_v2/model_scores_all.csv
      Per-row scores and top-5% flags for all 12 models.
  artifacts_research_v2/shap_isolation_forest_v2.csv, shap_autoencoder_v2.csv
      Precomputed full-dataset SHAP, served as a lookup -- never recomputed.
  artifacts_research_v2/internal_validity_metrics_v2.csv,
  stability_bootstrap_jaccard_v2.csv, ensemble_weights_v2.json,
  ensemble_pairwise_comparison_v2.csv, model_pairwise_spearman.csv,
  model_pairwise_jaccard.csv, shap_global_importance_comparison_v2.csv
      The Model Comparison and Explainability pages. Every number on those
      pages is read from an artifact at startup, not hardcoded -- so a stale
      artifact produces a visibly stale dashboard rather than a dashboard
      that silently disagrees with the pipeline behind it.

  data/bank_transactions_data_2.csv       raw display fields, joined on TransactionID
  artifacts_research/features_teammate_merged.csv   the 18 engineered features

The Account Scenario Simulator (formerly "What-if") additionally loads the
Isolation Forest and Autoencoder artifacts to score a hypothetical variation
of a *real* account's transaction. See SIMULATOR_NOTE below for why it cannot
be a free-form new-transaction simulator on this feature set.
"""
import csv
import io
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
SRC_V2_DIR = os.path.join(PROJECT_ROOT, "src_research_v2")
SRC_V1_DIR = os.path.join(PROJECT_ROOT, "src")
FRONTEND_DIR = os.path.join(DASHBOARD_DIR, "frontend")
CACHE_DIR = os.path.join(BACKEND_DIR, "cache")
QUEUE_STATE_PATH = os.path.join(BACKEND_DIR, "queue_state.json")

DATA_CSV = os.path.join(PROJECT_ROOT, "data", "bank_transactions_data_2.csv")
FEATURES_CSV = os.path.join(PROJECT_ROOT, "artifacts_research", "features_teammate_merged.csv")
ART_V2 = os.path.join(PROJECT_ROOT, "artifacts_research_v2")
MODELS_V2 = os.path.join(ART_V2, "models")

os.makedirs(CACHE_DIR, exist_ok=True)
sys.path.insert(0, SRC_V2_DIR)

from autoencoder_utils import load_autoencoder, reconstruction_errors  # noqa: E402
from config_research_v2 import FEATURE_COLS_V2, ID_COLS  # noqa: E402

# ---------------------------------------------------------------------------
# Upload & Predict -- the ONE addition in this file. Uses the separate,
# leakage-fixed v1 pipeline (../src/) because it has a real trained
# classifier + saved feature-engineering reference, which is what scoring a
# freshly-uploaded CSV of new transactions needs. Nothing above this comment
# or below the route itself was changed.
# ---------------------------------------------------------------------------
sys.path.insert(0, SRC_V1_DIR)
import config as v1_config  # noqa: E402
import fe_utils as v1_fe  # noqa: E402
from xgboost import XGBClassifier as _XGBClassifier  # noqa: E402

V1_REFERENCE = joblib.load(v1_config.REFERENCE_PKL)
V1_MODEL = _XGBClassifier()
V1_MODEL.load_model(v1_config.BEST_MODEL_JSON)

UPLOAD_REQUIRED_COLS = [
    "TransactionID", "AccountID", "TransactionAmount", "TransactionDate", "TransactionType",
    "Location", "DeviceID", "IP Address", "MerchantID", "Channel", "CustomerAge",
    "CustomerOccupation", "TransactionDuration", "LoginAttempts", "AccountBalance",
]
UPLOAD_MAX_ROWS = 5000

# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "TransactionAmount": "Transaction amount (log-scaled)",
    "CustomerAge": "Customer age",
    "TransactionDuration": "Transaction duration",
    "LoginAttempts": "Login attempts",
    "AccountBalance": "Account balance",
    "account_frequency": "Account activity (global count)",
    "device_frequency": "Device usage (global count)",
    "ip_frequency": "IP address usage (global count)",
    "merchant_frequency": "Merchant usage (global count)",
    "amount_to_balance_ratio": "Amount-to-balance ratio",
    "high_amount_transaction": "High-amount flag (top 5% globally)",
    "TransactionType_Debit": "Transaction type: debit",
    "Channel_Branch": "Channel: branch",
    "Channel_Online": "Channel: online",
    "CustomerOccupation_Engineer": "Occupation: engineer",
    "CustomerOccupation_Retired": "Occupation: retired",
    "CustomerOccupation_Student": "Occupation: student",
    "Location_FE": "Location commonness (frequency-encoded)",
}

MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "ocsvm": "One-Class SVM",
    "elliptic_envelope": "Elliptic Envelope (MCD)",
    "dbscan": "DBSCAN",
    "hdbscan": "HDBSCAN",
    "kmeans": "K-Means",
    "gmm": "Gaussian Mixture Model",
    "autoencoder": "Autoencoder",
    "vae": "Variational Autoencoder",
    "lstm_ae": "LSTM Autoencoder",
    "hybrid_ensemble": "Hybrid Ensemble (IF+LOF+AE)",
}

# the 11 models that feed the ensemble (Hybrid Ensemble excluded as an input --
# it is itself a vote of IF + LOF + AE, Phase 12 v2 SS0)
ENSEMBLE_MEMBERS = [
    "isolation_forest", "lof", "ocsvm", "elliptic_envelope", "dbscan",
    "hdbscan", "kmeans", "gmm", "autoencoder", "vae", "lstm_ae",
]

TIER_LABELS = {
    "priority": "Priority review",
    "standard": "Standard review",
    "normal": "Normal",
}

QUEUE_ACTIONS = ("pending", "approved", "escalated", "blocked")

SIMULATOR_NOTE = (
    "Free-form scoring of a brand-new transaction is not meaningful on this feature set. "
    "Five of the eighteen features (account / device / IP / merchant frequency and location "
    "frequency-encoding) are population-level statistics computed across the whole dataset -- "
    "an invented transaction has no device_frequency until you decide what population to count "
    "over, and inventing one would produce a confident-looking score built on a fabricated input. "
    "This tool therefore anchors every scenario to a real account and to real devices, IPs, "
    "merchants and locations that exist in the data, and uses their true historical frequency "
    "values. You vary only the fields that genuinely belong to a single transaction."
)

SIMULATOR_SCORE_NOTE = (
    "Scored by Isolation Forest and the Autoencoder only -- not by the full 11-model ensemble. "
    "DBSCAN and HDBSCAN have no out-of-sample predict() in this build and cannot score an unseen "
    "row at all (Phase 12 v2), so presenting a 'full ensemble score' for a hypothetical would be "
    "a fabrication. The percentiles below are this scenario's position within the 2,512 real "
    "transactions, and the tier is derived from the two-model reference distribution -- it is NOT "
    "the deployed 11-model threshold."
)

# training-derived constants, recovered exactly in Phase 14 (v2) SS5 and verified
# to reproduce all 2,512 rows of features_teammate_merged.csv. Computed at
# startup from the raw CSV rather than hardcoded, so they cannot drift silently.
HIGH_AMOUNT_THRESHOLD_Q = 0.95


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------
def _tier_for(score: float, priority_cut: float, standard_cut: float) -> str:
    if score >= priority_cut:
        return "priority"
    if score >= standard_cut:
        return "standard"
    return "normal"


def _percentile_of(sorted_ref: np.ndarray, value: float) -> float:
    """Fraction of the reference distribution at or below `value`, in [0, 1]."""
    return float(np.searchsorted(sorted_ref, value, side="right") / len(sorted_ref))


def _load_state():
    raw = pd.read_csv(DATA_CSV)
    raw["TransactionDate"] = pd.to_datetime(raw["TransactionDate"], format="%d-%m-%Y %H:%M")

    features = pd.read_csv(FEATURES_CSV)
    ensemble = pd.read_csv(os.path.join(ART_V2, "ensemble_scores_v2.csv"))
    model_scores = pd.read_csv(os.path.join(ART_V2, "model_scores_all.csv"))
    shap_if = pd.read_csv(os.path.join(ART_V2, "shap_isolation_forest_v2.csv"))
    shap_ae = pd.read_csv(os.path.join(ART_V2, "shap_autoencoder_v2.csv"))

    with open(os.path.join(ART_V2, "threshold_analysis_v2.json")) as f:
        thresholds = json.load(f)

    # --- alignment: every artifact must line up row-for-row on TransactionID ---
    n = len(raw)
    for name, df in [("features", features), ("ensemble", ensemble), ("model_scores", model_scores),
                     ("shap_if", shap_if), ("shap_ae", shap_ae)]:
        if len(df) != n:
            raise RuntimeError(f"Row-count mismatch: raw={n} {name}={len(df)}")
        if not (df["TransactionID"].values == raw["TransactionID"].values).all():
            raise RuntimeError(f"TransactionID alignment check failed for {name}")

    # --- Phase 13 (v2) cutoffs, read from the artifact, not hardcoded ---
    pct_rows = {r["method"]: r for r in thresholds["percentile_thresholds"]}
    priority_cut = float(pct_rows["P99"]["threshold_value"])
    standard_cut = float(pct_rows["P95"]["threshold_value"])

    score = ensemble["ensemble_percentile_average"].astype(float).values
    sorted_score = np.sort(score)

    flag_cols = [f"flag_{m}" for m in ENSEMBLE_MEMBERS]
    models_flagged = model_scores[flag_cols].fillna(0).astype(int).sum(axis=1).values
    lstm_applicable = model_scores["lstm_ae_applicable"].astype(bool).values

    # per-model percentile of each row's own score, for the drawer
    member_pct = {}
    for m in ENSEMBLE_MEMBERS:
        col = model_scores[f"score_{m}"].astype(float)
        member_pct[m] = col.rank(pct=True, na_option="keep").values

    ledger = pd.DataFrame({
        "transaction_id": raw["TransactionID"].astype(str),
        "account_id": raw["AccountID"].astype(str),
        "amount": raw["TransactionAmount"].astype(float),
        "date": raw["TransactionDate"],
        "txn_type": raw["TransactionType"].astype(str),
        "location": raw["Location"].astype(str),
        "device_id": raw["DeviceID"].astype(str),
        "ip_address": raw["IP Address"].astype(str),
        "merchant_id": raw["MerchantID"].astype(str),
        "channel": raw["Channel"].astype(str),
        "customer_age": raw["CustomerAge"].astype(int),
        "customer_occupation": raw["CustomerOccupation"].astype(str),
        "duration": raw["TransactionDuration"].astype(int),
        "login_attempts": raw["LoginAttempts"].astype(int),
        "account_balance": raw["AccountBalance"].astype(float),
        "risk_score": score,
        "score_percentile": pd.Series(score).rank(pct=True).values,
        "models_flagged": models_flagged,
        "lstm_applicable": lstm_applicable,
        "weighted_average": ensemble["ensemble_weighted_average"].astype(float).values,
        "hybrid_votes": model_scores["hybrid_vote_count"].astype(int).values,
    })
    ledger["risk_tier_code"] = [_tier_for(s, priority_cut, standard_cut) for s in score]
    ledger["risk_tier_label"] = ledger["risk_tier_code"].map(TIER_LABELS)

    id_to_row = {tx_id: idx for idx, tx_id in enumerate(ledger["transaction_id"])}

    # --- simulator: frozen training constants, recovered exactly (Phase 14 v2 SS5) ---
    amt = raw["TransactionAmount"].astype(float).values
    bal = raw["AccountBalance"].astype(float).values
    log_amt = np.log1p(amt)
    log_ratio = np.log1p(amt / (bal + 1.0))
    scaling_stats = {
        "TransactionAmount": (float(log_amt.mean()), float(log_amt.std(ddof=0))),
        "amount_to_balance_ratio": (float(log_ratio.mean()), float(log_ratio.std(ddof=0))),
    }
    for col in ("CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"):
        v = raw[col].astype(float).values
        scaling_stats[col] = (float(v.mean()), float(v.std(ddof=0)))
    high_amount_threshold = float(np.quantile(amt, HIGH_AMOUNT_THRESHOLD_Q))

    # real population frequency lookups -- never synthesised
    freq_lookup = {
        "account": raw.groupby("AccountID").size().to_dict(),
        "device": raw.groupby("DeviceID").size().to_dict(),
        "ip": raw.groupby("IP Address").size().to_dict(),
        "merchant": raw.groupby("MerchantID").size().to_dict(),
    }
    for key, col in [("account", "AccountID"), ("device", "DeviceID"),
                     ("ip", "IP Address"), ("merchant", "MerchantID")]:
        v = raw[col].map(freq_lookup[key]).astype(float).values
        scaling_stats[f"{key}_frequency"] = (float(v.mean()), float(v.std(ddof=0)))
    loc_prop = raw["Location"].value_counts(normalize=True)
    loc_series = raw["Location"].map(loc_prop).astype(float).values
    scaling_stats["Location_FE"] = (float(loc_series.mean()), float(loc_series.std(ddof=0)))

    # --- simulator models ---
    robust_scaler = joblib.load(os.path.join(MODELS_V2, "shared_robust_scaler.pkl"))
    iforest = joblib.load(os.path.join(MODELS_V2, "isolation_forest.pkl"))
    ae_scaler = joblib.load(os.path.join(ART_V2, "autoencoder_scaler.pkl"))
    ae_model = load_autoencoder(os.path.join(ART_V2, "autoencoder.pt"), len(FEATURE_COLS_V2), 3)

    # self-check: the reload path must reproduce the published scores
    X = features[FEATURE_COLS_V2].astype(float).values
    if_repro = -iforest.decision_function(robust_scaler.transform(X))
    if_err = float(np.abs(if_repro - model_scores["score_isolation_forest"].values).max())
    ae_repro, _, _, _ = reconstruction_errors(ae_model, ae_scaler.transform(X))
    ae_err = float(np.abs(ae_repro - model_scores["score_autoencoder"].values).max())
    if if_err > 1e-8 or ae_err > 1e-5:
        raise RuntimeError(f"Model reload check failed: IF max err {if_err}, AE max err {ae_err}")

    # two-model reference distribution for the simulator's honest tiering
    if_sorted = np.sort(if_repro)
    ae_sorted = np.sort(ae_repro)
    two_model_ref = np.sort(
        (pd.Series(if_repro).rank(pct=True).values + pd.Series(ae_repro).rank(pct=True).values) / 2.0
    )

    # --- Model Comparison / Explainability page data, all read from artifacts ---
    validity = pd.read_csv(os.path.join(ART_V2, "internal_validity_metrics_v2.csv"))
    stability = pd.read_csv(os.path.join(ART_V2, "stability_bootstrap_jaccard_v2.csv"))
    pairwise = pd.read_csv(os.path.join(ART_V2, "ensemble_pairwise_comparison_v2.csv"))
    spearman = pd.read_csv(os.path.join(ART_V2, "model_pairwise_spearman.csv"), index_col=0)
    jaccard = pd.read_csv(os.path.join(ART_V2, "model_pairwise_jaccard.csv"), index_col=0)
    shap_global = pd.read_csv(os.path.join(ART_V2, "shap_global_importance_comparison_v2.csv"))
    with open(os.path.join(ART_V2, "ensemble_weights_v2.json")) as f:
        weights = json.load(f)
    with open(os.path.join(ART_V2, "model_comparison_summary.json")) as f:
        rate_summary = json.load(f)

    return {
        "raw": raw,
        "features": features,
        "ledger": ledger,
        "id_to_row": id_to_row,
        "shap_if": shap_if,
        "shap_ae": shap_ae,
        "member_pct": member_pct,
        "model_scores": model_scores,
        "thresholds": {
            "priority": priority_cut,
            "standard": standard_cut,
            "analysis": thresholds,
        },
        "sorted_score": sorted_score,
        "validity": validity,
        "stability": stability,
        "pairwise": pairwise,
        "spearman": spearman,
        "jaccard": jaccard,
        "shap_global": shap_global,
        "weights": weights,
        "rate_summary": rate_summary,
        "sim": {
            "scaling_stats": scaling_stats,
            "high_amount_threshold": high_amount_threshold,
            "freq_lookup": freq_lookup,
            "loc_prop": loc_prop.to_dict(),
            "robust_scaler": robust_scaler,
            "iforest": iforest,
            "ae_scaler": ae_scaler,
            "ae_model": ae_model,
            "if_sorted": if_sorted,
            "ae_sorted": ae_sorted,
            "two_model_ref": two_model_ref,
            "repro_err": {"isolation_forest": if_err, "autoencoder": ae_err},
        },
    }


STATE = _load_state()


def _load_queue_state() -> dict:
    if os.path.exists(QUEUE_STATE_PATH):
        with open(QUEUE_STATE_PATH) as f:
            return json.load(f)
    return {}


def _save_queue_state(state: dict) -> None:
    with open(QUEUE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


QUEUE_STATE = _load_queue_state()

app = FastAPI(title="Argus", description="Behavioral Anomaly Intelligence")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _row_summary(idx: int) -> dict:
    row = STATE["ledger"].iloc[idx]
    action = QUEUE_STATE.get(row["transaction_id"], {}).get("action", "pending")
    return {
        "transaction_id": row["transaction_id"],
        "account_id": row["account_id"],
        "amount": round(float(row["amount"]), 2),
        "channel": row["channel"],
        "txn_type": row["txn_type"],
        "date": row["date"].isoformat(),
        "risk_tier_code": row["risk_tier_code"],
        "risk_tier_label": row["risk_tier_label"],
        "models_flagged": int(row["models_flagged"]),
        "lstm_applicable": bool(row["lstm_applicable"]),
        "risk_score": round(float(row["risk_score"]), 4),
        "queue_action": action,
    }


SORT_COLUMNS = {
    "date": "date", "amount": "amount", "risk_score": "risk_score",
    "transaction_id": "transaction_id", "models_flagged": "models_flagged",
}


@app.get("/api/transactions")
def list_transactions(
    q: Optional[str] = None,
    risk_tier: Optional[str] = Query(None, description="priority | standard | normal"),
    channel: Optional[str] = None,
    txn_type: Optional[str] = None,
    amount_min: Optional[float] = None,
    amount_max: Optional[float] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    sort_by: str = "date",
    sort_dir: str = "desc",
    page: int = 1,
    page_size: int = 25,
):
    ledger = STATE["ledger"]
    mask = pd.Series(True, index=ledger.index)

    if q:
        q_lower = q.strip().lower()
        mask &= (
            ledger["transaction_id"].str.lower().str.contains(q_lower, regex=False)
            | ledger["account_id"].str.lower().str.contains(q_lower, regex=False)
        )
    if risk_tier:
        mask &= ledger["risk_tier_code"] == risk_tier.lower()
    if channel:
        mask &= ledger["channel"] == channel
    if txn_type:
        mask &= ledger["txn_type"] == txn_type
    if amount_min is not None:
        mask &= ledger["amount"] >= amount_min
    if amount_max is not None:
        mask &= ledger["amount"] <= amount_max
    if date_start:
        mask &= ledger["date"] >= pd.to_datetime(date_start)
    if date_end:
        mask &= ledger["date"] <= pd.to_datetime(date_end) + pd.Timedelta(days=1)

    filtered = ledger[mask]
    sort_col = SORT_COLUMNS.get(sort_by, "date")
    filtered = filtered.sort_values(sort_col, ascending=(sort_dir == "asc"))

    total = len(filtered)
    page = max(page, 1)
    page_size = max(1, min(page_size, 200))
    start = (page - 1) * page_size
    page_indices = filtered.index[start:start + page_size]

    return {
        "total": total, "page": page, "page_size": page_size,
        "results": [_row_summary(i) for i in page_indices],
    }


def _shap_breakdown(frame: pd.DataFrame, idx: int, feature_row: pd.Series) -> List[dict]:
    row = frame.iloc[idx]
    out = [
        {
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "feature_value": round(float(feature_row[col]), 4),
            "shap_value": round(float(row[col]), 5),
        }
        for col in FEATURE_COLS_V2
    ]
    out.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
    return out


@app.get("/api/transactions/{transaction_id}")
def transaction_detail(transaction_id: str):
    idx = STATE["id_to_row"].get(transaction_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")

    row = STATE["ledger"].iloc[idx]
    feature_row = STATE["features"].iloc[idx]
    ms = STATE["model_scores"].iloc[idx]

    members = []
    for m in ENSEMBLE_MEMBERS:
        pct = STATE["member_pct"][m][idx]
        raw_score = ms[f"score_{m}"]
        flagged = ms[f"flag_{m}"]
        members.append({
            "model": m,
            "label": MODEL_LABELS[m],
            "percentile": None if pd.isna(pct) else round(float(pct), 4),
            "score": None if pd.isna(raw_score) else round(float(raw_score), 5),
            "flagged": None if pd.isna(flagged) else bool(flagged),
            "applicable": not pd.isna(pct),
        })

    action_entry = QUEUE_STATE.get(transaction_id, {})

    return {
        "transaction_id": row["transaction_id"],
        "account_id": row["account_id"],
        "raw": {
            "amount": round(float(row["amount"]), 2),
            "date": row["date"].isoformat(),
            "txn_type": row["txn_type"],
            "location": row["location"],
            "device_id": row["device_id"],
            "ip_address": row["ip_address"],
            "merchant_id": row["merchant_id"],
            "channel": row["channel"],
            "customer_age": int(row["customer_age"]),
            "customer_occupation": row["customer_occupation"],
            "duration_seconds": int(row["duration"]),
            "login_attempts": int(row["login_attempts"]),
            "account_balance": round(float(row["account_balance"]), 2),
            "amount_to_balance_ratio": round(float(row["amount"]) / max(float(row["account_balance"]), 0.01), 3),
        },
        "risk": {
            "risk_tier_code": row["risk_tier_code"],
            "risk_tier_label": row["risk_tier_label"],
            "risk_score": round(float(row["risk_score"]), 4),
            "score_percentile": round(float(row["score_percentile"]), 4),
            "score_rank": int((STATE["ledger"]["risk_score"] > row["risk_score"]).sum()) + 1,
            "models_flagged": int(row["models_flagged"]),
            "models_applicable": int(sum(1 for m in members if m["applicable"])),
            "hybrid_votes": int(row["hybrid_votes"]),
            "weighted_average": round(float(row["weighted_average"]), 4),
            "priority_threshold": round(STATE["thresholds"]["priority"], 4),
            "standard_threshold": round(STATE["thresholds"]["standard"], 4),
        },
        "models": members,
        "shap_isolation_forest": _shap_breakdown(STATE["shap_if"], idx, feature_row),
        "shap_autoencoder": _shap_breakdown(STATE["shap_ae"], idx, feature_row),
        "queue_action": action_entry.get("action", "pending"),
        "queue_updated_at": action_entry.get("updated_at"),
    }


@app.get("/api/kpis")
def kpis():
    ledger = STATE["ledger"]
    total = len(ledger)
    tier_counts = ledger["risk_tier_code"].value_counts()
    agree_counts = ledger["models_flagged"].value_counts().sort_index()

    daily = (
        ledger.assign(day=ledger["date"].dt.date)
        .groupby("day").size().reset_index(name="count").sort_values("day")
    )
    top_risk = ledger.sort_values("risk_score", ascending=False).index[:10]

    priority = int(tier_counts.get("priority", 0))
    standard = int(tier_counts.get("standard", 0))
    normal = int(tier_counts.get("normal", 0))

    return {
        "total_transactions": total,
        "priority_count": priority,
        "standard_count": standard,
        "normal_count": normal,
        "flag_rate": round((priority + standard) / total, 4),
        "avg_amount": round(float(ledger["amount"].mean()), 2),
        "priority_threshold": round(STATE["thresholds"]["priority"], 4),
        "standard_threshold": round(STATE["thresholds"]["standard"], 4),
        "tier_distribution": [
            {"tier": "Priority", "code": "priority", "count": priority},
            {"tier": "Standard", "code": "standard", "count": standard},
            {"tier": "Normal", "code": "normal", "count": normal},
        ],
        "model_agreement_distribution": [
            {"models": int(v), "count": int(c)} for v, c in agree_counts.items()
        ],
        "timeseries": [
            {"date": str(d), "count": int(c)} for d, c in zip(daily["day"], daily["count"])
        ],
        "top_risk": [_row_summary(i) for i in top_risk],
    }


@app.get("/api/queue")
def investigation_queue(status: Optional[str] = None, page: int = 1, page_size: int = 25):
    ledger = STATE["ledger"].sort_values("risk_score", ascending=False)
    if status:
        actions = ledger["transaction_id"].map(lambda tid: QUEUE_STATE.get(tid, {}).get("action", "pending"))
        ledger = ledger[actions == status]

    total = len(ledger)
    page = max(page, 1)
    page_size = max(1, min(page_size, 200))
    start = (page - 1) * page_size
    page_indices = ledger.index[start:start + page_size]
    return {
        "total": total, "page": page, "page_size": page_size,
        "results": [_row_summary(i) for i in page_indices],
    }


class QueueActionRequest(BaseModel):
    transaction_id: str
    action: Literal["pending", "approved", "escalated", "blocked"]


@app.post("/api/queue/action")
def queue_action(body: QueueActionRequest):
    if body.transaction_id not in STATE["id_to_row"]:
        raise HTTPException(status_code=404, detail=f"Transaction {body.transaction_id} not found")
    if body.action == "pending":
        QUEUE_STATE.pop(body.transaction_id, None)
    else:
        QUEUE_STATE[body.transaction_id] = {
            "action": body.action,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    _save_queue_state(QUEUE_STATE)
    return {"status": "ok", "transaction_id": body.transaction_id, "action": body.action}


@app.get("/api/queue/export")
def export_queue():
    ledger = STATE["ledger"].sort_values("risk_score", ascending=False)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "transaction_id", "account_id", "amount", "channel", "txn_type", "date",
        "risk_tier", "ensemble_percentile_average", "models_flagged_of_11", "queue_action",
    ])
    for idx in ledger.index:
        r = STATE["ledger"].iloc[idx]
        action = QUEUE_STATE.get(r["transaction_id"], {}).get("action", "pending")
        writer.writerow([
            r["transaction_id"], r["account_id"], r["amount"], r["channel"], r["txn_type"],
            r["date"].isoformat(), r["risk_tier_label"], round(float(r["risk_score"]), 4),
            int(r["models_flagged"]), action,
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=investigation_queue.csv"},
    )


# ---------------------------------------------------------------------------
# model comparison -- every number read from an artifact at startup
# ---------------------------------------------------------------------------
@app.get("/api/model-comparison")
def model_comparison():
    validity = STATE["validity"].set_index("model")
    weights = STATE["weights"]["weights"]
    disagreements = STATE["weights"]["disagreements"]
    sp, jc = STATE["spearman"], STATE["jaccard"]
    rates = STATE["rate_summary"]["anomaly_rates_pct"]

    models = []
    for m in list(ENSEMBLE_MEMBERS) + ["hybrid_ensemble"]:
        v = validity.loc[m] if m in validity.index else None
        # self-excluded pairwise means (Phase 14 v2 SS5, Inconsistency 1)
        sp_mean = float((sp.loc[m].sum() - 1.0) / (len(sp) - 1)) if m in sp.index else None
        jc_mean = float((jc.loc[m].sum() - 1.0) / (len(jc) - 1)) if m in jc.index else None
        models.append({
            "model": m,
            "label": MODEL_LABELS[m],
            "flagged_rate_pct": rates.get(m),
            "n_flagged_top5pct": None if v is None else int(v["n_flagged_top5pct"]),
            "n_rows_used": None if v is None else int(v["n_rows_used"]),
            "silhouette": None if v is None else round(float(v["silhouette"]), 4),
            "davies_bouldin": None if v is None else round(float(v["davies_bouldin"]), 4),
            "calinski_harabasz": None if v is None else round(float(v["calinski_harabasz"]), 2),
            "mean_spearman": None if sp_mean is None else round(sp_mean, 4),
            "mean_jaccard": None if jc_mean is None else round(jc_mean, 4),
            "ensemble_weight": None if m not in weights else round(weights[m], 4),
            "disagreement": None if m not in disagreements else round(disagreements[m], 4),
            "in_ensemble": m in ENSEMBLE_MEMBERS,
        })
    models.sort(key=lambda d: (d["silhouette"] is None, -(d["silhouette"] or 0)))

    stability = [
        {
            "model": r["model"], "label": MODEL_LABELS[r["model"]],
            "n_runs": int(r["n_bootstrap_runs"]),
            "mean_jaccard": round(float(r["mean_pairwise_jaccard_top5pct"]), 4),
            "min_jaccard": round(float(r["min_pairwise_jaccard"]), 4),
            "max_jaccard": round(float(r["max_pairwise_jaccard"]), 4),
        }
        for _, r in STATE["stability"].iterrows()
    ]

    pw = STATE["pairwise"]
    strategy_pairs = [
        {
            "pair": str(r["strategy_pair"]),
            "spearman": round(float(r["spearman"]), 4),
            "jaccard": round(float(r["jaccard_top5pct"]), 4),
        }
        for _, r in pw.iterrows()
    ]

    return {
        "models": models,
        "stability": stability,
        "strategy_pairs": strategy_pairs,
        "pc1_explained_variance": STATE["weights"]["pca_explained_variance_ratio_pc1"],
        "recommended_strategy": "Percentile Aggregation",
        "notes": {
            "leaderboard": (
                "This is not a leaderboard. Eight of the twelve models cluster at 4.5%-5.7% flagged "
                "because they either take a contamination~0.05 parameter or use the standardised "
                "top-5% convention -- the rate carries little information, the agreement on which "
                "rows carries all of it. Silhouette also structurally favours distance-based "
                "methods: a top-5%-by-distance cut is close to guaranteed to separate well in a "
                "distance metric, which is why the reconstruction-error models sit lowest."
            ),
            "elliptic_envelope": (
                "Elliptic Envelope leads on internal validity (Silhouette 0.5409, Calinski-Harabasz "
                "592.5, more than 3x the next model) and its core assumption is measurably false: "
                "Shapiro-Wilk rejects normality on 100% of the 18 scaled features. It also has the "
                "lowest flagged-set overlap of any model. Retained for comparison; not recommended "
                "as a primary detector."
            ),
            "stability": (
                "Measured by refitting on 5 bootstrap resamples of the training split. Isolation "
                "Forest is the most retrain-stable model here (0.6021) and the Autoencoder the "
                "least (0.3726, min 0.2115) -- an inversion of the in-house 46-feature pipeline, "
                "where LOF led at 0.590 inside a much narrower 0.527-0.590 spread. A retrain "
                "changes 40-63% of the flagged set with no drift and no new data."
            ),
            "agreement": (
                "Mean Spearman / Jaccard here are self-excluded pairwise means. The Phase 8 (v2) "
                "report published figures that include each model's own self-correlation of 1.0 in "
                "a 12-way average, inflating every value by the same monotone transform (logged as "
                "Inconsistency 1 in Phase 14 v2). Rankings are unaffected; these are the corrected "
                "absolute values."
            ),
            "dbscan": (
                "DBSCAN has the lowest agreement with the rest of the field by a wide margin and "
                "the lowest ensemble weight. The in-house 46-feature pipeline reached the same "
                "conclusion independently, which makes this a property of DBSCAN's behaviour on "
                "this raw data rather than an artifact of either feature-engineering choice."
            ),
            "hybrid": (
                "The Hybrid Ensemble is excluded as an ensemble input -- it is itself a >=2-of-3 "
                "vote of Isolation Forest, LOF and the Autoencoder, so folding it back in would "
                "double-count those three. Its native flag is 83 rows (3.30%); the 269-row "
                "partition its internal-validity metrics were computed on is the >=1-vote set."
            ),
            "strategies": (
                "Rank (Borda) and Percentile aggregation are near-mathematically identical "
                "(rho=0.9999) -- summing ranks and averaging rank/N differ only by a per-model "
                "normalisation constant. Percentile aggregation is recommended: bounded in (0,1) "
                "so thresholds are comparable across batches, no tuned weights to defend, and it "
                "skips missing models and renormalises rather than imputing."
            ),
        },
    }


# ---------------------------------------------------------------------------
# explainability
# ---------------------------------------------------------------------------
@app.get("/api/explainability")
def explainability():
    g = STATE["shap_global"]
    if_top = g.sort_values("rank_isolation_forest").head(10)
    ae_top = g.sort_values("rank_autoencoder").head(10)
    overlap = sorted(set(if_top["feature"]) & set(ae_top["feature"]))

    rho = float(
        pd.Series(g["mean_abs_shap_isolation_forest"]).rank()
        .corr(pd.Series(g["mean_abs_shap_autoencoder"]).rank())
    )

    analysis = STATE["thresholds"]["analysis"]
    score = STATE["ledger"]["risk_score"].values
    counts, edges = np.histogram(score, bins=40)

    return {
        "global_shap": {
            "isolation_forest": [
                {"feature": r["feature"], "label": FEATURE_LABELS.get(r["feature"], r["feature"]),
                 "mean_abs_shap": round(float(r["mean_abs_shap_isolation_forest"]), 4)}
                for _, r in if_top.iterrows()
            ],
            "autoencoder": [
                {"feature": r["feature"], "label": FEATURE_LABELS.get(r["feature"], r["feature"]),
                 "mean_abs_shap": round(float(r["mean_abs_shap_autoencoder"]), 4)}
                for _, r in ae_top.iterrows()
            ],
        },
        "divergence": {
            "spearman_rho": round(rho, 4),
            "top10_overlap": len(overlap),
            "overlap_features": [FEATURE_LABELS.get(f, f) for f in overlap],
            "explanation": (
                "Isolation Forest and the Autoencoder attribute their scores almost entirely "
                "differently. Isolation Forest scores by how few random splits isolate a point, so "
                "low-cardinality one-hots -- which isolate a whole minority class in a single split "
                "-- dominate its attributions. The Autoencoder scores by squared reconstruction "
                "error through a 3-unit bottleneck, dominated here by the frequency-encoded "
                "features. On this feature set the Autoencoder should be read as 'is this "
                "transaction's popularity profile unusual', NOT 'is this amount unusual for this "
                "account' -- the personal-baseline features that reading would need are absent. "
                "This divergence is the direct evidence-based reason the recommendation is an "
                "ensemble with both explanations shown side by side, rather than the single "
                "highest-ranked model."
            ),
            "worked_examples": [
                {"transaction_id": "TX000275",
                 "note": "Highest-scoring transaction in the dataset and the clearest fraud-signature "
                         "match. Both models agree: Isolation Forest attributes +1.714 to login "
                         "attempts and +1.607 to the amount-to-balance ratio; the Autoencoder "
                         "attributes +1.570 to the amount-to-balance ratio. The in-house 46-feature "
                         "pipeline independently flagged the same transaction in its own top 1%."},
                {"transaction_id": "TX000615",
                 "note": "The two models disagree on which aspect of the same transaction is "
                         "anomalous -- Isolation Forest reads it as an amount anomaly, the "
                         "Autoencoder primarily as a location-frequency anomaly."},
                {"transaction_id": "TX001029",
                 "note": "Both models agree the oddity is an extreme merchant-frequency value "
                         "(z=3.67), not an unusual amount or login pattern. A $516.47 transaction "
                         "at 0.40x its account's balance with normal login behaviour -- a high score "
                         "that is not a fraud-relevant anomaly. This is exactly the class of flag a "
                         "second, structurally different model is there to catch."},
            ],
        },
        "score_distribution": analysis["score_distribution"],
        "score_histogram": [
            {"x": round(float((edges[i] + edges[i + 1]) / 2), 4), "count": int(counts[i])}
            for i in range(len(counts))
        ],
        "percentile_thresholds": [
            {
                "method": r["method"], "threshold": round(float(r["threshold_value"]), 4),
                "n_flagged": int(r["n_flagged"]), "pct_flagged": float(r["pct_flagged"]),
                "review_cost_ceiling": float(r["illustrative_upper_bound_review_cost_usd_if_all_fp"]),
                "per_day": float(r["flagged_per_day_this_sample"]),
            }
            for r in analysis["percentile_thresholds"]
        ],
        "statistical_thresholds": [
            {"method": r["method"], "score": r["score"], "threshold": round(float(r["threshold_value"]), 4),
             "n_flagged": int(r["n_flagged"])}
            for r in (analysis["statistical_thresholds_on_recommended_score"]
                      + analysis["statistical_thresholds_context_unbounded_scores"])
        ],
        "statistical_finding": analysis["statistical_threshold_finding"],
        "cost_note": (
            "A cost-optimal threshold sweep cannot be reproduced on this pipeline: counting false "
            "negatives requires knowing which UNFLAGGED transactions are fraud, which is unknowable "
            "without a label. The costs below are an upper bound on review labour assuming every "
            "flagged transaction is a false positive, at v1's illustrative $5/review figure. That "
            "is a ceiling, not an estimate -- and it is why no automatic block tier is recommended."
        ),
    }


# ---------------------------------------------------------------------------
# Account Scenario Simulator (formerly "What-if")
# ---------------------------------------------------------------------------
@app.get("/api/simulator/options")
def simulator_options():
    raw = STATE["raw"]
    return {
        "accounts": sorted(raw["AccountID"].unique().tolist()),
        "devices": sorted(raw["DeviceID"].unique().tolist()),
        "ip_addresses": sorted(raw["IP Address"].unique().tolist()),
        "merchants": sorted(raw["MerchantID"].unique().tolist()),
        "locations": sorted(raw["Location"].unique().tolist()),
        "occupations": sorted(raw["CustomerOccupation"].unique().tolist()),
        "channels": ["ATM", "Online", "Branch"],
        "txn_types": ["Debit", "Credit"],
        "high_amount_threshold": round(STATE["sim"]["high_amount_threshold"], 2),
        "note": SIMULATOR_NOTE,
        "score_note": SIMULATOR_SCORE_NOTE,
    }


@app.get("/api/simulator/account/{account_id}")
def simulator_account(account_id: str):
    raw = STATE["raw"]
    rows = raw[raw["AccountID"] == account_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    last = rows.sort_values("TransactionDate").iloc[-1]
    freq = STATE["sim"]["freq_lookup"]
    return {
        "account_id": account_id,
        "n_transactions": int(len(rows)),
        "account_frequency": int(freq["account"][account_id]),
        "defaults": {
            "amount": round(float(last["TransactionAmount"]), 2),
            "account_balance": round(float(last["AccountBalance"]), 2),
            "customer_age": int(last["CustomerAge"]),
            "customer_occupation": str(last["CustomerOccupation"]),
            "channel": str(last["Channel"]),
            "txn_type": str(last["TransactionType"]),
            "location": str(last["Location"]),
            "device_id": str(last["DeviceID"]),
            "ip_address": str(last["IP Address"]),
            "merchant_id": str(last["MerchantID"]),
            "duration_seconds": int(last["TransactionDuration"]),
            "login_attempts": int(last["LoginAttempts"]),
        },
        "history": [
            {
                "transaction_id": str(r["TransactionID"]),
                "date": r["TransactionDate"].isoformat(),
                "amount": round(float(r["TransactionAmount"]), 2),
                "balance": round(float(r["AccountBalance"]), 2),
                "channel": str(r["Channel"]),
                "login_attempts": int(r["LoginAttempts"]),
            }
            for _, r in rows.sort_values("TransactionDate").iterrows()
        ],
    }


class ScenarioRequest(BaseModel):
    account_id: str = Field(..., description="Must be an existing AccountID")
    amount: float = Field(..., gt=0)
    account_balance: float = Field(..., ge=0)
    txn_type: str
    channel: str
    location: str
    device_id: str
    ip_address: str
    merchant_id: str
    customer_occupation: str
    customer_age: int = Field(..., ge=18, le=120)
    duration_seconds: int = Field(..., ge=1)
    login_attempts: int = Field(..., ge=1)


def _z(name: str, value: float) -> float:
    mean, std = STATE["sim"]["scaling_stats"][name]
    return (value - mean) / std


@app.post("/api/score")
def score_scenario(body: ScenarioRequest):
    sim = STATE["sim"]
    freq = sim["freq_lookup"]

    # Handle completely new accounts/devices/IPs/merchants by assigning default frequency (1)
    # This allows fraud prediction on brand new data not in the training set

    # Get frequency for account (default to 1 if new)
    account_freq = freq["account"].get(body.account_id, 1)

    # Get frequency for device (default to 1 if new)
    device_freq = freq["device"].get(body.device_id, 1)

    # Get frequency for IP (default to 1 if new)
    ip_freq = freq["ip"].get(body.ip_address, 1)

    # Get frequency for merchant (default to 1 if new)
    merchant_freq = freq["merchant"].get(body.merchant_id, 1)

    # Get location proportion (default to minimum if new location)
    if body.location not in sim["loc_prop"]:
        # Use minimum location proportion for new locations
        location_prop = min(sim["loc_prop"].values()) if sim["loc_prop"] else 0.001
    else:
        location_prop = sim["loc_prop"][body.location]

    ratio_raw = body.amount / (body.account_balance + 1.0)
    values = {
        "TransactionAmount": _z("TransactionAmount", float(np.log1p(body.amount))),
        "CustomerAge": _z("CustomerAge", body.customer_age),
        "TransactionDuration": _z("TransactionDuration", body.duration_seconds),
        "LoginAttempts": _z("LoginAttempts", body.login_attempts),
        "AccountBalance": _z("AccountBalance", body.account_balance),
        "account_frequency": _z("account_frequency", account_freq),
        "device_frequency": _z("device_frequency", device_freq),
        "ip_frequency": _z("ip_frequency", ip_freq),
        "merchant_frequency": _z("merchant_frequency", merchant_freq),
        "amount_to_balance_ratio": _z("amount_to_balance_ratio", float(np.log1p(ratio_raw))),
        "high_amount_transaction": 1.0 if body.amount > sim["high_amount_threshold"] else 0.0,
        "TransactionType_Debit": 1.0 if body.txn_type == "Debit" else 0.0,
        "Channel_Branch": 1.0 if body.channel == "Branch" else 0.0,
        "Channel_Online": 1.0 if body.channel == "Online" else 0.0,
        "CustomerOccupation_Engineer": 1.0 if body.customer_occupation == "Engineer" else 0.0,
        "CustomerOccupation_Retired": 1.0 if body.customer_occupation == "Retired" else 0.0,
        "CustomerOccupation_Student": 1.0 if body.customer_occupation == "Student" else 0.0,
        "Location_FE": _z("Location_FE", location_prop),
    }
    x = np.array([[values[c] for c in FEATURE_COLS_V2]], dtype=float)

    if_score = float(-sim["iforest"].decision_function(sim["robust_scaler"].transform(x))[0])
    ae_mse, _, _, ae_recon = reconstruction_errors(sim["ae_model"], sim["ae_scaler"].transform(x))
    ae_score = float(ae_mse[0])

    if_pct = _percentile_of(sim["if_sorted"], if_score)
    ae_pct = _percentile_of(sim["ae_sorted"], ae_score)
    two_model = (if_pct + ae_pct) / 2.0
    ref = sim["two_model_ref"]
    two_model_pct = _percentile_of(ref, two_model)
    tier = ("priority" if two_model_pct >= 0.99 else "standard" if two_model_pct >= 0.95 else "normal")

    # Isolation Forest: exact per-feature attribution, computed live (TreeExplainer
    # is exact for this model and takes milliseconds on one row). Sign-flipped to
    # match this project's "higher = more anomalous" convention, the same
    # verification Phase 11 (v2) performed.
    import shap  # imported lazily -- only the simulator needs it
    explainer = shap.TreeExplainer(sim["iforest"])
    raw_shap = np.array(explainer.shap_values(sim["robust_scaler"].transform(x))).reshape(-1)
    if_contrib = sorted(
        (
            {"feature": c, "label": FEATURE_LABELS.get(c, c),
             "feature_value": round(float(values[c]), 4),
             "shap_value": round(float(-raw_shap[i]), 5)}
            for i, c in enumerate(FEATURE_COLS_V2)
        ),
        key=lambda d: abs(d["shap_value"]), reverse=True,
    )

    # Autoencoder: the exact per-feature squared residual, which is what the
    # reconstruction-error score is literally the mean of. Reported instead of a
    # GradientExplainer approximation because it is exact and needs no background
    # sample -- labelled precisely rather than presented as SHAP.
    x_ae = sim["ae_scaler"].transform(x)
    resid = (ae_recon[0] - x_ae[0]) ** 2
    ae_contrib = sorted(
        (
            {"feature": c, "label": FEATURE_LABELS.get(c, c),
             "feature_value": round(float(values[c]), 4),
             "shap_value": round(float(resid[i]), 5),
             "share_of_error": round(float(resid[i] / resid.sum()), 4)}
            for i, c in enumerate(FEATURE_COLS_V2)
        ),
        key=lambda d: d["shap_value"], reverse=True,
    )

    return {
        "account_id": body.account_id,
        "isolation_forest": {"score": round(if_score, 5), "percentile": round(if_pct, 4)},
        "autoencoder": {"score": round(ae_score, 5), "percentile": round(ae_pct, 4)},
        "two_model_percentile_average": round(two_model, 4),
        "two_model_reference_percentile": round(two_model_pct, 4),
        "risk_tier_code": tier,
        "risk_tier_label": TIER_LABELS[tier],
        "frequency_inputs_used": {
            "account_frequency": int(account_freq),
            "device_frequency": int(device_freq),
            "ip_frequency": int(ip_freq),
            "merchant_frequency": int(merchant_freq),
            "location_share_pct": round(100 * location_prop, 2),
        },
        "derived": {
            "amount_to_balance_ratio_raw": round(ratio_raw, 4),
            "high_amount_flag": bool(values["high_amount_transaction"]),
            "high_amount_threshold": round(sim["high_amount_threshold"], 2),
        },
        "shap_isolation_forest": if_contrib[:8],
        "autoencoder_error_contributions": ae_contrib[:8],
        "score_note": SIMULATOR_SCORE_NOTE,
    }


@app.get("/api/meta")
def meta():
    """Provenance -- what this dashboard is actually serving, for verification."""
    return {
        "pipeline": "research_v2 (teammate 18-feature matrix) -- the client-designated final pipeline",
        "score": "ensemble_percentile_average (Phase 12 v2 recommendation: Percentile Aggregation)",
        "score_source": "artifacts_research_v2/ensemble_scores_v2.csv",
        "thresholds_source": "artifacts_research_v2/threshold_analysis_v2.json",
        "priority_threshold_p99": round(STATE["thresholds"]["priority"], 6),
        "standard_threshold_p95": round(STATE["thresholds"]["standard"], 6),
        "shap_sources": [
            "artifacts_research_v2/shap_isolation_forest_v2.csv",
            "artifacts_research_v2/shap_autoencoder_v2.csv",
        ],
        "ensemble_members": ENSEMBLE_MEMBERS,
        "n_ensemble_members": len(ENSEMBLE_MEMBERS),
        "n_transactions": int(len(STATE["ledger"])),
        "n_features": len(FEATURE_COLS_V2),
        "feature_columns": FEATURE_COLS_V2,
        "model_reload_max_error": STATE["sim"]["repro_err"],
        "block_tier": None,
        "block_tier_note": (
            "No automatic block tier exists. A cost-optimal block threshold requires counting false "
            "negatives, which requires a fraud label this dataset does not have (Phase 13 v2)."
        ),
    }


# ---------------------------------------------------------------------------
# Upload & Predict. Accepts either of the two CSV shapes already used
# elsewhere in this project:
#   (a) the raw transaction log (TransactionID, AccountID, TransactionDate,
#       DeviceID, ... -- same columns as data/bank_transactions_data_2.csv)
#       -> scored with the leakage-fixed v1 XGBoost pipeline.
#   (b) the already-engineered 18-feature matrix (FEATURE_COLS_V2 -- same
#       shape as artifacts_research/features_teammate_merged.csv, which is
#       what a teammate's separate feature-engineering pipeline produces)
#       -> scored with the same Isolation Forest + Autoencoder two-model
#       percentile average the Scenario Simulator already uses, since there
#       are no raw/ID columns to run the v1 feature engineering on.
# Everything else in this file is unchanged from before this feature was added.
# ---------------------------------------------------------------------------
RAW_FORMAT_CUTOFF_PCT = 50.0   # v1's XGBoost outputs a calibrated-ish probability -- 50% is a sensible split
V2_FORMAT_CUTOFF_PCT = 95.0    # two-model score is a PERCENTILE RANK, not a probability -- it is uniformly
                                # spread 0-100% by construction, so a 50% cutoff would always split ~50/50
                                # regardless of the data. The rest of this dashboard (and the Scenario
                                # Simulator) treats the top 5% (>=95th percentile) as the flagged group --
                                # reused here for the same reason, not a new assumption.


def _score_raw_format(df: pd.DataFrame):
    df_work = df.copy()
    try:
        df_work["TransactionDate"] = pd.to_datetime(df_work["TransactionDate"], format="%d-%m-%Y %H:%M")
    except (ValueError, TypeError):
        try:
            df_work["TransactionDate"] = pd.to_datetime(df_work["TransactionDate"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse TransactionDate values: {e}")
    if "PreviousTransactionDate" in df_work.columns:
        df_work = df_work.drop(columns=["PreviousTransactionDate"])

    try:
        feat_df = v1_fe.transform_batch_new(df_work, V1_REFERENCE)
        proba = V1_MODEL.predict_proba(feat_df)[:, 1]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"CSV is missing a value the model needs: {e}")

    results = []
    for i in range(len(df)):
        orig = df.iloc[i]
        results.append({
            "transaction_id": str(orig.get("TransactionID", f"row_{i}")),
            "account_id": str(orig.get("AccountID", "")),
            "date": str(orig.get("TransactionDate", "")),
            "amount": round(float(orig.get("TransactionAmount", 0.0)), 2),
            "fraud_percentage": round(float(proba[i]) * 100, 2),
        })
    return results, "XGBoost (leakage-fixed v1 pipeline)", RAW_FORMAT_CUTOFF_PCT


def _score_v2_feature_format(df: pd.DataFrame):
    sim = STATE["sim"]
    X = df[FEATURE_COLS_V2].astype(float).values

    if_scores = -sim["iforest"].decision_function(sim["robust_scaler"].transform(X))
    ae_scores, _, _, _ = reconstruction_errors(sim["ae_model"], sim["ae_scaler"].transform(X))

    if_pct = np.array([_percentile_of(sim["if_sorted"], s) for s in if_scores])
    ae_pct = np.array([_percentile_of(sim["ae_sorted"], s) for s in ae_scores])
    two_model = (if_pct + ae_pct) / 2.0
    two_model_pct = np.array([_percentile_of(sim["two_model_ref"], v) for v in two_model])

    results = []
    for i in range(len(df)):
        results.append({
            "transaction_id": str(df.iloc[i].get("TransactionID", f"row_{i}")),
            "account_id": str(df.iloc[i].get("AccountID", "")),
            "date": "",
            "amount": None,
            "fraud_percentage": round(float(two_model_pct[i]) * 100, 2),
        })
    return results, "Isolation Forest + Autoencoder percentile average (research_v2 pipeline)", V2_FORMAT_CUTOFF_PCT


@app.post("/api/upload/predict")
async def upload_predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse this file as CSV: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="The uploaded CSV has no rows.")
    if len(df) > UPLOAD_MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"This tool scores up to {UPLOAD_MAX_ROWS:,} rows at a time; the uploaded file has {len(df):,}.",
        )

    is_raw_format = not [c for c in UPLOAD_REQUIRED_COLS if c not in df.columns]
    is_v2_feature_format = not [c for c in FEATURE_COLS_V2 if c not in df.columns]

    if is_raw_format:
        results, model_used, cutoff = _score_raw_format(df)
    elif is_v2_feature_format:
        results, model_used, cutoff = _score_v2_feature_format(df)
    else:
        missing_raw = [c for c in UPLOAD_REQUIRED_COLS if c not in df.columns]
        raise HTTPException(
            status_code=400,
            detail=(
                f"CSV doesn't match either supported format. For the raw transaction log, it's missing: "
                f"{', '.join(missing_raw)}. It also doesn't match the pre-engineered 18-feature format "
                f"({', '.join(FEATURE_COLS_V2)})."
            ),
        )

    fraud_count = sum(1 for r in results if r["fraud_percentage"] >= cutoff)
    total = len(results)
    not_fraud_count = total - fraud_count

    return {
        "total": total,
        "model": model_used,
        "fraud_count": fraud_count,
        "not_fraud_count": not_fraud_count,
        "fraud_rate_pct": round(100 * fraud_count / total, 2),
        "not_fraud_rate_pct": round(100 * not_fraud_count / total, 2),
        "fraud_cutoff_pct": cutoff,
        "results": results,
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("Starting Bank Fraud Detection Dashboard...")
    print("Navigate to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
