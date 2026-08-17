"""
api/routers/analytics.py -- GET /model/metrics and GET /model/figures/{name}.

Both routes are read-only views over report artifacts the training pipeline
already produces (evaluate.py -> reports/metrics/test_evaluation.json,
reports/metrics/feature_importance.csv, reports/figures/*.png). Nothing here
is computed or fabricated: if a report file is missing, the response says so
honestly (`available: false` / 404) rather than inventing a number.

Role gating mirrors the exact pattern api/routers/predict.py already uses for
PredictionItemBasic vs PredictionItemFull:
  - GET /model/metrics: require_view_predictions (everyone with predict-view
    access) for the headline metrics (ROC-AUC, PR-AUC, TPR@5%FPR, threshold).
    Fairness breakdown + feature importance are only included when
    full_explainability_allowed(role) is true (Admin, Risk Manager).
  - GET /model/figures/{name}: restricted to an explicit filename allow-list
    (never path-joined from caller input) and gated the same way as the full
    detail above (Admin, Risk Manager only), since these figures (SHAP
    summary, feature importance) are themselves explainability detail.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from api.rbac import CurrentUser, full_explainability_allowed, require_roles, require_view_predictions
from api.schemas import (
    ConfusionAtThreshold,
    FairnessGroup,
    FairnessSummary,
    FeatureImportanceItem,
    ModelMetricsResponse,
)
from api.models_db import Role

logger = logging.getLogger("fraud_api.analytics")

router = APIRouter(tags=["analytics"])

require_full_explainability = require_roles(Role.ADMIN, Role.RISK_MANAGER)

# fraud-detection/api/routers/analytics.py -> parents: routers, api, fraud-detection
_FRAUD_DETECTION_ROOT = Path(__file__).resolve().parent.parent.parent
_METRICS_DIR = _FRAUD_DETECTION_ROOT / "reports" / "metrics"
_FIGURES_DIR = _FRAUD_DETECTION_ROOT / "reports" / "figures"
_TEST_EVAL_PATH = _METRICS_DIR / "test_evaluation.json"
_FEATURE_IMPORTANCE_PATH = _METRICS_DIR / "feature_importance.csv"

# Explicit allow-list: never build a filesystem path from caller input.
_FIGURE_FILES: dict[str, Path] = {
    "roc_curve.png": _FIGURES_DIR / "roc_curve.png",
    "pr_curve.png": _FIGURES_DIR / "pr_curve.png",
    "shap_summary.png": _FIGURES_DIR / "shap_summary.png",
    "feature_importance.png": _FIGURES_DIR / "feature_importance.png",
    "confusion_matrix.png": _FIGURES_DIR / "confusion_matrix.png",
}


def _load_feature_importance() -> list[FeatureImportanceItem]:
    if not _FEATURE_IMPORTANCE_PATH.exists():
        return []
    items: list[FeatureImportanceItem] = []
    with open(_FEATURE_IMPORTANCE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                items.append(FeatureImportanceItem(
                    feature=row["feature"],
                    gain=float(row["gain"]),
                    share=float(row["share"]),
                ))
            except (KeyError, ValueError):
                continue
    return items


@router.get("/model/metrics", response_model=ModelMetricsResponse)
def model_metrics(current_user: CurrentUser = Depends(require_view_predictions)):
    if not _TEST_EVAL_PATH.exists():
        return ModelMetricsResponse(available=False, full_detail=False)

    try:
        with open(_TEST_EVAL_PATH, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.error("Failed to read/parse %s", _TEST_EVAL_PATH, exc_info=True)
        return ModelMetricsResponse(available=False, full_detail=False)

    test_metrics = report.get("test_metrics", {})
    full_detail = full_explainability_allowed(current_user.role)

    response = ModelMetricsResponse(
        available=True,
        full_detail=full_detail,
        model=report.get("model"),
        strategy=report.get("strategy"),
        threshold=report.get("threshold"),
        roc_auc=test_metrics.get("roc_auc"),
        pr_auc=test_metrics.get("pr_auc"),
        tpr_at_5pct_fpr=test_metrics.get("tpr_at_5pct_fpr"),
        n=test_metrics.get("n"),
        n_positive=test_metrics.get("n_positive"),
        positive_rate=test_metrics.get("positive_rate"),
    )

    if not full_detail:
        return response

    confusion = report.get("confusion_at_threshold")
    if confusion:
        try:
            response.confusion_at_threshold = ConfusionAtThreshold(**confusion)
        except (TypeError, ValueError):
            pass

    fairness = report.get("fairness_test")
    if fairness and "group_False" in fairness and "group_True" in fairness:
        try:
            response.fairness = FairnessSummary(
                protected_attribute="customer_age",
                protected_threshold=50,
                age_le_threshold=FairnessGroup(**fairness["group_False"]),
                age_gt_threshold=FairnessGroup(**fairness["group_True"]),
                fpr_ratio=fairness["fpr_ratio"],
                fairness_eval_threshold=fairness["threshold"],
            )
        except (TypeError, ValueError, KeyError):
            pass

    response.feature_importance = _load_feature_importance()
    return response


@router.get("/model/figures/{name}", include_in_schema=True)
def model_figure(name: str, current_user: CurrentUser = Depends(require_full_explainability)):
    path = _FIGURE_FILES.get(name)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown figure name")
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This figure has not been generated yet")
    return FileResponse(path, media_type="image/png")
