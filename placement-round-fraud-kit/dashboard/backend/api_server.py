"""
Argus -- Behavioral Anomaly Intelligence.

FastAPI backend for the fraud-analytics dashboard. Reuses the v1 pipeline's
artifacts as-is (labeled.csv, anomaly_votes.csv, reference.pkl, xgb_model.json,
thresholds.json, decision_tree_rules.txt) -- nothing here retrains or
recomputes the underlying model. On startup it:

  1. Rebuilds the raw, sorted transaction table (fe.load_raw + the identical
     sort fit_engineer used) so raw fields can be joined back onto
     labeled.csv by row position.
  2. Scores every transaction with the SMOTE XGBoost model (the primary
     model per the brief) to get a continuous risk_score per row.
  3. Computes (and disk-caches) a SHAP TreeExplainer pass over all rows once,
     so per-transaction explanations are a dict lookup at request time, not
     a recomputation.
  4. Reproduces the cost-based threshold sweep from the held-out test split
     so the Explainability page can show a real curve, not two isolated
     numbers.

Everything else (the What-if Simulator, the investigation queue) reuses
these same in-memory structures.
"""
import csv
import io
import json
import os
import sys
from datetime import datetime
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
FRONTEND_DIR = os.path.join(DASHBOARD_DIR, "frontend")
CACHE_DIR = os.path.join(BACKEND_DIR, "cache")
QUEUE_STATE_PATH = os.path.join(BACKEND_DIR, "queue_state.json")

os.makedirs(CACHE_DIR, exist_ok=True)
sys.path.insert(0, SRC_DIR)

import config  # noqa: E402
import fe_utils as fe  # noqa: E402

import shap  # noqa: E402

# ---------------------------------------------------------------------------
# human-readable labels for the engineered feature names, used in the SHAP
# breakdown and the global-importance chart so the console doesn't just show
# raw column identifiers to an analyst.
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "TransactionAmount": "Transaction amount",
    "CustomerAge": "Customer age",
    "TransactionDuration": "Transaction duration",
    "LoginAttempts": "Login attempts",
    "AccountBalance": "Account balance",
    "Amount_vs_TypeAvg": "Amount vs. transaction-type average",
    "DeviceTxnCount": "Device transaction count",
    "IPTxnCount": "IP address transaction count",
    "MerchantTxnCount": "Merchant transaction count",
    "Amount_vs_AccountAvg": "Amount vs. account average",
    "DeviceNoveltyFlag": "New device for this account",
    "LocationNoveltyFlag": "New location for this account",
    "TimeSinceLastTxn": "Time since last transaction",
    "Location_enc": "Location (encoded)",
    "TransactionType_Debit": "Transaction type: debit",
    "Channel_Branch": "Channel: branch",
    "Channel_Online": "Channel: online",
    "CustomerOccupation_Engineer": "Occupation: engineer",
    "CustomerOccupation_Retired": "Occupation: retired",
    "CustomerOccupation_Student": "Occupation: student",
}

RISK_TIER_MAP = {
    "High confidence fraud": "high",
    "Medium confidence / needs review": "medium",
    "Normal": "normal",
}
RISK_TIER_LABELS = {v: k for k, v in RISK_TIER_MAP.items()}

QUEUE_ACTIONS = ("pending", "approved", "escalated", "blocked")

RAW_DISPLAY_COLS = [
    "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
    "TransactionType", "Location", "DeviceID", "IP Address", "MerchantID",
    "Channel", "CustomerAge", "CustomerOccupation", "TransactionDuration",
    "LoginAttempts", "AccountBalance",
]


# ---------------------------------------------------------------------------
# startup: load artifacts, score every row, cache SHAP, build the ledger
# ---------------------------------------------------------------------------
def _verdict_for(proba: float, thresholds: dict) -> str:
    if proba >= thresholds["block_threshold"]:
        return "block"
    if proba >= thresholds["review_threshold"]:
        return "review"
    return "approve"


VERDICT_LABELS = {"approve": "Auto-approve", "review": "Manual review", "block": "Block"}


def _load_state():
    reference = joblib.load(config.REFERENCE_PKL)
    feature_cols = reference["feature_cols"]

    model = XGBClassifier()
    model.load_model(config.MODEL_JSON)

    with open(config.THRESHOLDS_JSON) as f:
        thresholds = json.load(f)

    raw = fe.load_raw(config.RAW_CSV)
    raw_sorted = raw.sort_values(["AccountID", "TransactionDate", "TransactionID"]).reset_index(drop=True)

    labeled = pd.read_csv(config.LABELED_CSV)
    votes = pd.read_csv(config.ANOMALY_VOTES_CSV)

    if len(raw_sorted) != len(labeled) or len(labeled) != len(votes):
        raise RuntimeError(
            f"Row-count mismatch: raw={len(raw_sorted)} labeled={len(labeled)} votes={len(votes)}"
        )
    # spot-check the join lines up before trusting it for a single request
    if not np.isclose(raw_sorted["TransactionAmount"].iloc[0], labeled["TransactionAmount"].iloc[0]):
        raise RuntimeError("Row alignment check failed: raw_sorted vs labeled TransactionAmount mismatch at row 0")
    if not (votes["vote_count"] == labeled["vote_count"]).all():
        raise RuntimeError("Row alignment check failed: anomaly_votes.csv vote_count does not match labeled.csv")

    feature_df = labeled[feature_cols].reset_index(drop=True)
    proba = model.predict_proba(feature_df)[:, 1]

    shap_cache_path = os.path.join(CACHE_DIR, "shap_values.npy")
    if os.path.exists(shap_cache_path):
        shap_matrix = np.load(shap_cache_path)
        if shap_matrix.shape != feature_df.shape:
            shap_matrix = None
    else:
        shap_matrix = None
    explainer = shap.TreeExplainer(model)
    if shap_matrix is None:
        shap_matrix = explainer.shap_values(feature_df)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[1]
        np.save(shap_cache_path, shap_matrix)

    ledger = pd.DataFrame({
        "transaction_id": raw_sorted["TransactionID"].astype(str),
        "account_id": raw_sorted["AccountID"].astype(str),
        "amount": raw_sorted["TransactionAmount"].astype(float),
        "date": raw_sorted["TransactionDate"],
        "txn_type": raw_sorted["TransactionType"].astype(str),
        "location": raw_sorted["Location"].astype(str),
        "device_id": raw_sorted["DeviceID"].astype(str),
        "ip_address": raw_sorted["IP Address"].astype(str),
        "merchant_id": raw_sorted["MerchantID"].astype(str),
        "channel": raw_sorted["Channel"].astype(str),
        "customer_age": raw_sorted["CustomerAge"].astype(int),
        "customer_occupation": raw_sorted["CustomerOccupation"].astype(str),
        "duration": raw_sorted["TransactionDuration"].astype(int),
        "login_attempts": raw_sorted["LoginAttempts"].astype(int),
        "account_balance": raw_sorted["AccountBalance"].astype(float),
        "vote_count": labeled["vote_count"].astype(int),
        "risk_tier_label": labeled["risk_tier"].astype(str),
        "risk_tier_code": labeled["risk_tier"].map(RISK_TIER_MAP),
        "is_fraud": labeled["is_fraud"].astype(int),
        "flag_isoforest": votes["flag_isoforest"].astype(bool),
        "flag_lof": votes["flag_lof"].astype(bool),
        "flag_ocsvm": votes["flag_ocsvm"].astype(bool),
        "flag_mcd": votes["flag_mcd"].astype(bool),
        "risk_score": proba.astype(float),
    })
    ledger["verdict_code"] = ledger["risk_score"].apply(lambda p: _verdict_for(p, thresholds))
    ledger["verdict_label"] = ledger["verdict_code"].map(VERDICT_LABELS)

    id_to_row = {tx_id: idx for idx, tx_id in enumerate(ledger["transaction_id"])}

    # ---- cost-based threshold sweep, reproduced from the real held-out split ----
    split = joblib.load(config.SPLIT_PKL)
    X_test, y_test = split["X_test"], split["y_test"]
    test_proba = model.predict_proba(X_test)[:, 1]
    sweep_thresholds = np.linspace(0.01, 0.99, 99)
    sweep_costs = []
    for t in sweep_thresholds:
        pred = (test_proba >= t).astype(int)
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        sweep_costs.append(fp * config.COST_FALSE_POSITIVE + fn * config.COST_FALSE_NEGATIVE)
    sweep_costs = np.array(sweep_costs)
    best_idx = int(sweep_costs.argmin())

    return {
        "reference": reference,
        "feature_cols": feature_cols,
        "model": model,
        "explainer": explainer,
        "thresholds": thresholds,
        "raw": raw,
        "ledger": ledger,
        "feature_df": feature_df,
        "shap_matrix": shap_matrix,
        "id_to_row": id_to_row,
        "cost_sweep": {
            "thresholds": sweep_thresholds.tolist(),
            "costs": sweep_costs.tolist(),
            "min_threshold": float(sweep_thresholds[best_idx]),
            "min_cost": float(sweep_costs[best_idx]),
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


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
app = FastAPI(title="Argus", description="Behavioral Anomaly Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "vote_count": int(row["vote_count"]),
        "risk_score": round(float(row["risk_score"]), 4),
        "verdict_code": row["verdict_code"],
        "verdict_label": row["verdict_label"],
        "queue_action": action,
    }


SORT_COLUMNS = {
    "date": "date",
    "amount": "amount",
    "risk_score": "risk_score",
    "transaction_id": "transaction_id",
    "vote_count": "vote_count",
}


@app.get("/api/transactions")
def list_transactions(
    q: Optional[str] = None,
    risk_tier: Optional[str] = Query(None, description="high | medium | normal"),
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
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [_row_summary(i) for i in page_indices],
    }


@app.get("/api/transactions/{transaction_id}")
def transaction_detail(transaction_id: str):
    idx = STATE["id_to_row"].get(transaction_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")

    row = STATE["ledger"].iloc[idx]
    feature_row = STATE["feature_df"].iloc[idx]
    shap_row = STATE["shap_matrix"][idx]

    shap_breakdown = sorted(
        (
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "feature_value": round(float(feature_row[col]), 4),
                "shap_value": round(float(shap_row[i]), 5),
            }
            for i, col in enumerate(STATE["feature_cols"])
        ),
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )

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
        },
        "risk": {
            "risk_tier_code": row["risk_tier_code"],
            "risk_tier_label": row["risk_tier_label"],
            "vote_count": int(row["vote_count"]),
            "risk_score": round(float(row["risk_score"]), 4),
            "verdict_code": row["verdict_code"],
            "verdict_label": row["verdict_label"],
            "review_threshold": STATE["thresholds"]["review_threshold"],
            "block_threshold": STATE["thresholds"]["block_threshold"],
        },
        "detectors": {
            "isoforest": bool(row["flag_isoforest"]),
            "lof": bool(row["flag_lof"]),
            "ocsvm": bool(row["flag_ocsvm"]),
            "mcd": bool(row["flag_mcd"]),
        },
        "shap": shap_breakdown,
        "queue_action": action_entry.get("action", "pending"),
        "queue_updated_at": action_entry.get("updated_at"),
    }


@app.get("/api/kpis")
def kpis():
    ledger = STATE["ledger"]
    total = len(ledger)
    tier_counts = ledger["risk_tier_code"].value_counts()
    vote_counts = ledger["vote_count"].value_counts().sort_index()

    daily = (
        ledger.assign(day=ledger["date"].dt.date)
        .groupby("day")
        .size()
        .reset_index(name="count")
        .sort_values("day")
    )

    top_risk = ledger.sort_values("risk_score", ascending=False).index[:10]

    high = int(tier_counts.get("high", 0))
    medium = int(tier_counts.get("medium", 0))
    normal = int(tier_counts.get("normal", 0))

    return {
        "total_transactions": total,
        "high_risk_count": high,
        "review_count": medium,
        "normal_count": normal,
        "flag_rate": round((high + medium) / total, 4),
        "avg_amount": round(float(ledger["amount"].mean()), 2),
        "tier_distribution": [
            {"tier": "High", "code": "high", "count": high},
            {"tier": "Medium", "code": "medium", "count": medium},
            {"tier": "Normal", "code": "normal", "count": normal},
        ],
        "vote_distribution": [
            {"votes": int(v), "count": int(c)} for v, c in vote_counts.items()
        ],
        "timeseries": [
            {"date": str(d), "count": int(c)} for d, c in zip(daily["day"], daily["count"])
        ],
        "top_risk": [_row_summary(i) for i in top_risk],
    }


@app.get("/api/queue")
def investigation_queue(
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
):
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
        "total": total,
        "page": page,
        "page_size": page_size,
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
            "updated_at": datetime.utcnow().isoformat() + "Z",
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
        "risk_tier", "vote_count", "risk_score", "verdict", "queue_action",
    ])
    for idx in ledger.index:
        r = STATE["ledger"].iloc[idx]
        action = QUEUE_STATE.get(r["transaction_id"], {}).get("action", "pending")
        writer.writerow([
            r["transaction_id"], r["account_id"], r["amount"], r["channel"], r["txn_type"],
            r["date"].isoformat(), r["risk_tier_label"], r["vote_count"],
            round(float(r["risk_score"]), 4), r["verdict_label"], action,
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=investigation_queue.csv"},
    )


# ---------------------------------------------------------------------------
# model comparison -- real, measured numbers from the v1 pipeline, hardcoded
# here per the brief rather than recomputed (the training run that produced
# them is not reproduced by this dashboard).
# ---------------------------------------------------------------------------
MODEL_COMPARISON = {
    "detectors": [
        {"name": "Isolation Forest", "series": "series-1-blue", "flagged": 126, "rate": 0.0502},
        {"name": "Local Outlier Factor", "series": "series-2-orange", "flagged": 126, "rate": 0.0502},
        {"name": "One-Class SVM", "series": "series-3-aqua", "flagged": 120, "rate": 0.0478},
        {"name": "Elliptic Envelope (MCD)", "series": "series-4-yellow", "flagged": 126, "rate": 0.0502},
    ],
    "vote_distribution": [
        {"votes": 0, "count": 2230}, {"votes": 1, "count": 146}, {"votes": 2, "count": 79},
        {"votes": 3, "count": 34}, {"votes": 4, "count": 23},
    ],
    "confidence_tiers": [
        {"tier": "High (3-4 votes)", "code": "high", "count": 57},
        {"tier": "Medium (2 votes)", "code": "medium", "count": 79},
        {"tier": "Normal (0-1 votes)", "code": "normal", "count": 2376},
    ],
    "fraud_prevalence": {"count": 136, "total": 2512, "rate": 0.0541},
    "xgboost_variants": [
        {"name": "SMOTE-trained (primary)", "roc_auc": 0.9428, "pr_auc": 0.5934, "is_primary": True},
        {"name": "Class-weighted", "roc_auc": 0.9532, "pr_auc": 0.7398, "is_primary": False},
    ],
    "primary_model_note": (
        "Class-weighting measures marginally better on this held-out split, but SMOTE ships as the "
        "primary model per the brief's requirement. Reporting both honestly rather than only showing "
        "the stronger number."
    ),
    "confusion_matrix": {
        "threshold": 0.5, "n": 503,
        "tn": 457, "fp": 19, "fn": 12, "tp": 15,
        "precision": 0.441, "recall": 0.556, "f1": 0.492, "roc_auc": 0.943, "pr_auc": 0.593,
    },
    "accuracy_contrast": {
        "naive_accuracy": 0.9463,
        "model_accuracy": 0.9384,
        "explanation": (
            "A model that predicts \"normal\" for every transaction scores 94.63% accuracy while "
            "catching zero fraud. The real model's accuracy (93.84%) is lower -- not because it is "
            "worse, but because it spends correctness budget catching the fraud cases the naive "
            "baseline ignores entirely. Accuracy alone is the wrong scoreboard for a 5%-prevalence "
            "problem; precision, recall and PR-AUC are what matter here."
        ),
    },
}


@app.get("/api/model-comparison")
def model_comparison():
    return MODEL_COMPARISON


# ---------------------------------------------------------------------------
# explainability
# ---------------------------------------------------------------------------
GLOBAL_SHAP_IMPORTANCE = [
    {"feature": "Amount_vs_AccountAvg", "label": FEATURE_LABELS["Amount_vs_AccountAvg"], "mean_abs_shap": 0.944},
    {"feature": "Channel_Branch", "label": FEATURE_LABELS["Channel_Branch"], "mean_abs_shap": 0.696},
    {"feature": "LoginAttempts", "label": FEATURE_LABELS["LoginAttempts"], "mean_abs_shap": 0.679},
    {"feature": "LocationNoveltyFlag", "label": FEATURE_LABELS["LocationNoveltyFlag"], "mean_abs_shap": 0.626},
    {"feature": "Channel_Online", "label": FEATURE_LABELS["Channel_Online"], "mean_abs_shap": 0.603},
    {"feature": "CustomerOccupation_Retired", "label": FEATURE_LABELS["CustomerOccupation_Retired"], "mean_abs_shap": 0.408},
    {"feature": "DeviceNoveltyFlag", "label": FEATURE_LABELS["DeviceNoveltyFlag"], "mean_abs_shap": 0.354},
    {"feature": "TransactionAmount", "label": FEATURE_LABELS["TransactionAmount"], "mean_abs_shap": 0.322},
    {"feature": "TimeSinceLastTxn", "label": FEATURE_LABELS["TimeSinceLastTxn"], "mean_abs_shap": 0.260},
    {"feature": "MerchantTxnCount", "label": FEATURE_LABELS["MerchantTxnCount"], "mean_abs_shap": 0.253},
]


def _parse_decision_tree_rules(path: str) -> List[dict]:
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    stack = []
    rules = []
    for line in lines:
        marker_idx = line.index("|---")
        depth = len(line[:marker_idx]) // 4
        content = " ".join(line[marker_idx + 4:].strip().split())

        if content.startswith("class:"):
            outcome_code = content.split(":", 1)[1].strip()
            outcome = "Fraud (flagged)" if outcome_code == "1" else "Normal (clear)"
            conditions = [c for _, c in stack[:depth]]
            rules.append({"conditions": conditions, "outcome": outcome, "outcome_code": outcome_code})
        else:
            stack = stack[:depth]
            pretty = content.replace("<=", "≤").replace(">", ">")
            stack.append((depth, pretty))
    return rules


@app.get("/api/explainability")
def explainability():
    rules = _parse_decision_tree_rules(config.DT_RULES_TXT)
    sweep = STATE["cost_sweep"]
    cost_at_05_idx = int(np.argmin(np.abs(np.array(sweep["thresholds"]) - 0.5)))
    return {
        "global_shap_importance": GLOBAL_SHAP_IMPORTANCE,
        "decision_tree_rules": rules,
        "cost_sweep": {
            "points": [
                {"threshold": round(t, 2), "cost": c}
                for t, c in zip(sweep["thresholds"], sweep["costs"])
            ],
            "min_threshold": round(sweep["min_threshold"], 2),
            "min_cost": sweep["min_cost"],
            "default_threshold": 0.5,
            "default_cost": sweep["costs"][cost_at_05_idx],
            "cost_false_positive": config.COST_FALSE_POSITIVE,
            "cost_false_negative": config.COST_FALSE_NEGATIVE,
        },
    }


# ---------------------------------------------------------------------------
# what-if simulator
# ---------------------------------------------------------------------------
@app.get("/api/simulator/options")
def simulator_options():
    raw = STATE["raw"]
    return {
        "accounts": sorted(STATE["reference"]["account_history"].keys()),
        "devices": sorted(raw["DeviceID"].unique().tolist()),
        "locations": sorted(raw["Location"].unique().tolist()),
        "merchants": sorted(raw["MerchantID"].unique().tolist()),
        "occupations": sorted(raw["CustomerOccupation"].unique().tolist()),
        "channels": ["ATM", "Online", "Branch"],
        "txn_types": ["Debit", "Credit"],
    }


class ScoreRequest(BaseModel):
    account_id: Optional[str] = Field(None, description="Leave blank to simulate a brand-new account")
    amount: float = Field(..., gt=0)
    txn_type: str
    location: str
    device_id: str
    ip_address: str = "10.0.0.1"
    merchant_id: str
    channel: str
    customer_age: int = Field(..., ge=18, le=120)
    customer_occupation: str
    duration_seconds: int = Field(..., ge=1)
    login_attempts: int = Field(..., ge=1)
    account_balance: float = Field(..., ge=0)


@app.post("/api/score")
def score_transaction(body: ScoreRequest):
    txn = {
        "AccountID": body.account_id or None,
        "TransactionAmount": body.amount,
        "TransactionType": body.txn_type,
        "Location": body.location,
        "DeviceID": body.device_id,
        "IP Address": body.ip_address,
        "MerchantID": body.merchant_id,
        "Channel": body.channel,
        "CustomerAge": body.customer_age,
        "CustomerOccupation": body.customer_occupation,
        "TransactionDuration": body.duration_seconds,
        "LoginAttempts": body.login_attempts,
        "AccountBalance": body.account_balance,
        "TransactionDate": datetime.now(),
    }

    row = fe.transform_new(txn, STATE["reference"])
    proba = float(STATE["model"].predict_proba(row)[:, 1][0])
    verdict_code = _verdict_for(proba, STATE["thresholds"])

    shap_values = STATE["explainer"](row)
    contributions = sorted(
        (
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "feature_value": round(float(row[col].iloc[0]), 4),
                "shap_value": round(float(shap_values.values[0][i]), 5),
            }
            for i, col in enumerate(row.columns)
        ),
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )

    return {
        "risk_score": round(proba, 4),
        "verdict_code": verdict_code,
        "verdict_label": VERDICT_LABELS[verdict_code],
        "review_threshold": STATE["thresholds"]["review_threshold"],
        "block_threshold": STATE["thresholds"]["block_threshold"],
        "shap": contributions[:8],
    }


# ---------------------------------------------------------------------------
# serve the frontend (same-origin, avoids CORS entirely)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
