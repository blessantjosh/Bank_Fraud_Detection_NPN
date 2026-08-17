"""
api/model_service.py -- the one place this API layer touches the ML
pipeline in src/.

Interface assumption this module is built against (src/prediction.py as it
exists today, read before writing this file):

    model, preprocessor, meta = src.prediction.load_artifacts(cfg, models_dir)
    X = preprocessor.transform_tree(df) or preprocessor.transform_dense(df)
    prob = src.models.predict_proba(model, X, meta["model_type"])
    meta contains: model_type, strategy, threshold, feature_columns,
                   model_iteration, primary_metric, val_metrics, fairness_val

We deliberately do NOT call src.prediction.predict_csv() /
src.prediction.predict_dataframe() directly for the live /predict endpoint,
for one reason worth stating explicitly: predict_csv() is gated by
src.auth.require_admin(), which is a *different*, CLI-shaped credential
(FRAUD_ADMIN_TOKEN env var) designed for a trusted single-operator script,
not for a multi-user, multi-role HTTP service. This API layer has its own
JWT+RBAC authorization model (api/rbac.py) which is the real gate for HTTP
callers; re-running require_admin() underneath it would mean every API
caller of every role would need the *same* shared CLI token, which defeats
RBAC entirely. So we call the lower-level, ungated pieces
(load_artifacts + preprocessor.transform_* + models.predict_proba) directly
and enforce authorization at the HTTP boundary instead, in
api/routers/predict.py. This is documented, not hidden.

Model integrity: if fraud-detection/models/model_checksum.json exists (see
api/scripts/record_model_checksum.py), the sha256 of final_model.joblib and
preprocessor.joblib is verified against it on load. If it doesn't exist yet
-- e.g. because the ML pipeline hasn't finished a training run and recorded
one -- this is logged as a warning and, by default, does NOT block serving
(ENFORCE_MODEL_CHECKSUM=false locally). Production should set
ENFORCE_MODEL_CHECKSUM=true once a checksum has been recorded (see
api/SECURITY.md "Model integrity").
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from api.settings import settings

logger = logging.getLogger("fraud_api.model_service")

FRAUD_DETECTION_ROOT = Path(__file__).resolve().parent.parent
if str(FRAUD_DETECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAUD_DETECTION_ROOT))


class ModelNotAvailable(RuntimeError):
    """Raised when /predict is called but no usable, integrity-checked model
    is loaded (e.g. the ML pipeline hasn't produced models/final_model.joblib
    yet, or ENFORCE_MODEL_CHECKSUM=true and the checksum does not match)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelService:
    """Loads the trained model artifacts once, verifies their checksum, and
    serves predictions. Thread-safe lazy singleton via get_model_service()."""

    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.preprocessor = None
        self.meta: dict | None = None
        self.model_version: str = "unloaded"
        self.checksum_verified: bool | None = None
        self.load_error: str | None = None
        self._try_load()

    def _try_load(self) -> None:
        with self._lock:
            try:
                from src.config import load_config, resolve_path
                from src.prediction import load_artifacts

                cfg = load_config()
                models_dir = resolve_path(settings.models_dir)
                model, preprocessor, meta = load_artifacts(cfg, models_dir)

                self.checksum_verified = self._verify_checksum(models_dir)
                if settings.enforce_model_checksum and not self.checksum_verified:
                    raise ModelNotAvailable(
                        "Model checksum verification failed or no checksum is "
                        "recorded, and ENFORCE_MODEL_CHECKSUM=true."
                    )

                self.model = model
                self.preprocessor = preprocessor
                self.meta = meta
                self.model_version = (
                    f"{meta.get('model_type')}/{meta.get('strategy')}"
                    f"@iter={meta.get('model_iteration')}"
                )
                self._cfg = cfg
                logger.info(
                    "Model loaded: %s (checksum_verified=%s)",
                    self.model_version, self.checksum_verified,
                )
            except FileNotFoundError as exc:
                self.load_error = str(exc)
                logger.warning(
                    "No trained model artifacts found yet (%s). /predict will "
                    "return 503 until the ML pipeline (src/training.py) has "
                    "produced models/final_model.joblib, models/preprocessor.joblib "
                    "and models/model_meta.json, or until "
                    "api/scripts/train_demo_model.py has been run to produce an "
                    "interim demo model in the same artifact format.",
                    exc,
                )
            except Exception as exc:  # noqa: BLE001 -- log and degrade, never crash startup
                self.load_error = str(exc)
                logger.error("Failed to load model artifacts: %s", exc, exc_info=True)

    def _verify_checksum(self, models_dir: Path) -> bool | None:
        checksum_path = FRAUD_DETECTION_ROOT / settings.model_checksum_file
        if not checksum_path.exists():
            logger.warning(
                "No recorded model checksum at %s -- integrity of the loaded "
                "model artifacts cannot be verified. Run "
                "api/scripts/record_model_checksum.py after training.",
                checksum_path,
            )
            return None
        recorded = json.loads(checksum_path.read_text(encoding="utf-8"))
        ok = True
        for filename in ("final_model.joblib", "preprocessor.joblib"):
            expected = recorded.get(filename)
            actual = _sha256_file(models_dir / filename)
            if expected != actual:
                logger.error(
                    "Model integrity check FAILED for %s: expected sha256=%s, actual=%s",
                    filename, expected, actual,
                )
                ok = False
        return ok

    def reload(self) -> None:
        self._try_load()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def current_threshold(self, override: float | None) -> float:
        if override is not None:
            return override
        return float(self.meta["threshold"])

    def score(self, rows: list[dict], threshold_override: float | None = None):
        """Returns (probabilities, predictions, risk_levels, feature_df_tree)."""
        if not self.is_loaded:
            raise ModelNotAvailable(self.load_error or "Model not loaded")

        raw_df = pd.DataFrame(rows)
        model_type = self.meta["model_type"]
        strategy = self.meta["strategy"]
        use_dense = (model_type in ("logistic_regression", "random_forest")) or (
            strategy in ("smote", "smote_undersample")
        )
        X = (
            self.preprocessor.transform_dense(raw_df)
            if use_dense
            else self.preprocessor.transform_tree(raw_df)
        )
        X = X[self.meta["feature_columns"]]

        from src.models import predict_proba
        from src.prediction import risk_level

        prob = predict_proba(self.model, X, model_type)
        threshold = self.current_threshold(threshold_override)
        pred = (prob >= threshold).astype(int)
        level = risk_level(prob, self._cfg)
        return prob, pred, level, X

    def explain(self, X_row: pd.DataFrame, top_n: int = 8) -> list[dict]:
        """Best-effort local explanation for exactly one already-transformed
        row. Uses real SHAP (src.explainability, shap.TreeExplainer) for the
        tree model families (lightgbm/xgboost) the ML pipeline actually
        trains and selects from. For the dense linear models
        (logistic_regression), TreeExplainer does not apply, so we report a
        signed linear contribution (coefficient * scaled feature value)
        instead -- clearly labeled "linear_contribution", never mislabeled
        as SHAP. For random_forest we skip per-row explanation (no cheap,
        exact per-row attribution without a KernelExplainer, which is too
        slow for a live request) and return only global gain-based
        importance context.
        """
        model_type = self.meta["model_type"]
        feature_names = list(X_row.columns)

        if model_type in ("lightgbm", "xgboost"):
            try:
                from src.explainability import shap_summary

                _, shap_values = shap_summary(self.model, X_row, model_type)
                vals = np.asarray(shap_values.values).reshape(-1)
                order = np.argsort(-np.abs(vals))[:top_n]
                return [
                    {
                        "feature": feature_names[i],
                        "value": _safe_scalar(X_row.iloc[0, i]),
                        "contribution": float(vals[i]),
                        "method": "shap",
                    }
                    for i in order
                ]
            except Exception:
                logger.warning("SHAP explanation failed, falling back to no detail", exc_info=True)
                return []

        if model_type == "logistic_regression":
            coefs = np.asarray(self.model.coef_).reshape(-1)
            x = np.asarray(X_row.iloc[0]).reshape(-1)
            contrib = coefs * x
            order = np.argsort(-np.abs(contrib))[:top_n]
            return [
                {
                    "feature": feature_names[i],
                    "value": _safe_scalar(X_row.iloc[0, i]),
                    "contribution": float(contrib[i]),
                    "method": "linear_contribution",
                }
                for i in order
            ]

        return []


def _safe_scalar(v):
    if isinstance(v, (int, float, str)) or v is None:
        return v
    try:
        return float(v)
    except Exception:
        return str(v)


_service_lock = threading.Lock()
_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ModelService()
    return _service
