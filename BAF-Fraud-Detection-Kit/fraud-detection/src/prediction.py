"""
prediction.py -- score a new CSV with the saved model/preprocessor/threshold.

Output columns: fraud_probability, fraud_prediction (0/1), risk_level
(LOW/MEDIUM/HIGH/CRITICAL, boundaries configurable via config.yaml risk_levels).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import audit, models
from src.auth import admin_identity, require_admin
from src.config import Config, resolve_path

logger = logging.getLogger("fraud_detection.prediction")


def load_artifacts(cfg: Config, models_dir: str | Path | None = None):
    models_dir = resolve_path(models_dir or "models")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    model = joblib.load(models_dir / "final_model.joblib")
    with open(models_dir / "model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, preprocessor, meta


def risk_level(prob: np.ndarray, cfg: Config) -> np.ndarray:
    low_max = cfg.risk_levels.low_max
    medium_max = cfg.risk_levels.medium_max
    high_max = cfg.risk_levels.high_max
    levels = np.full(prob.shape, "CRITICAL", dtype=object)
    levels[prob < high_max] = "HIGH"
    levels[prob < medium_max] = "MEDIUM"
    levels[prob < low_max] = "LOW"
    return levels


def predict_dataframe(raw_df: pd.DataFrame, model, preprocessor, meta: dict, cfg: Config) -> pd.DataFrame:
    model_type = meta["model_type"]
    strategy = meta["strategy"]
    use_dense = (model_type in ("logistic_regression", "random_forest")) or (
        strategy in ("smote", "smote_undersample")
    )
    X = preprocessor.transform_dense(raw_df) if use_dense else preprocessor.transform_tree(raw_df)
    X = X[meta["feature_columns"]]

    prob = models.predict_proba(model, X, model_type)
    threshold = float(meta["threshold"])
    pred = (prob >= threshold).astype(int)
    level = risk_level(prob, cfg)

    out = raw_df.copy()
    out["fraud_probability"] = prob
    out["fraud_prediction"] = pred
    out["risk_level"] = level
    return out


def predict_csv(input_path: str | Path, output_path: str | Path, cfg: Config,
                 models_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Score a new CSV of applications. Gated behind require_admin(): this is
    the entry point that produces fraud decisions on real applicant data and
    writes to data/predictions/, so it is restricted to admins (see
    src/auth.py and README.md "Security note"). Every call that gets past
    the gate is also recorded to the append-only audit log (src/audit.py).
    """
    require_admin(cfg)
    identity = admin_identity()

    model, preprocessor, meta = load_artifacts(cfg, models_dir)

    input_path = Path(input_path)
    raw_bytes = input_path.read_bytes()
    input_hash = audit.sha256_bytes(raw_bytes)

    raw_df = pd.read_csv(input_path)
    junk = [c for c in raw_df.columns if c.lower().startswith("unnamed")]
    if junk:
        raw_df = raw_df.drop(columns=junk)

    out = predict_dataframe(raw_df, model, preprocessor, meta, cfg)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("Wrote %d predictions to %s", len(out), output_path)
    risk_counts = out["risk_level"].value_counts().to_dict()
    logger.info("Risk level distribution:\n%s", out["risk_level"].value_counts().to_string())

    audit.record_prediction_audit(
        log_path=resolve_path(audit.DEFAULT_AUDIT_LOG_PATH),
        admin_identity=identity,
        model_type=meta["model_type"],
        strategy=meta["strategy"],
        model_iteration=meta.get("model_iteration"),
        input_row_count=len(raw_df),
        input_content_hash=input_hash,
        risk_level_counts=risk_counts,
    )
    return out
