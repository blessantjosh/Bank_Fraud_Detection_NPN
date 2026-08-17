"""
api/routers/predict.py -- POST /predict (JSON rows) and POST /predict/file
(CSV upload), plus GET /predictions (view-only history for all roles,
including VIEWER).

Role gating (see api/rbac.py):
  - run predictions (/predict, /predict/file): FRAUD_ANALYST, RISK_MANAGER, ADMIN
  - view predictions (/predictions):           everyone, including VIEWER

Response detail gating (per the spec's explainability-security section):
  - VIEWER/FRAUD_ANALYST/AUDITOR-visible fields: fraud_probability,
    fraud_prediction, risk_level only.
  - ADMIN/RISK_MANAGER additionally get top_features (SHAP for tree models,
    a clearly-labeled linear-contribution approximation for the logistic
    regression demo model -- see api/model_service.py docstring).
"""

from __future__ import annotations

import csv
import io
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from api.audit import write_audit
from api.database import get_db
from api.model_service import ModelNotAvailable, get_model_service
from api.models_db import PredictionRecord, Role, SystemSetting
from api.rate_limit import limiter
from api.rbac import (
    CurrentUser,
    full_explainability_allowed,
    require_predict,
    require_threshold_change,
    require_view_predictions,
)
from api.schemas import (
    ApplicationRow,
    FeatureContribution,
    PredictionItemBasic,
    PredictionItemFull,
    PredictionRecordEntry,
    PredictRequest,
    PredictResponse,
    ThresholdUpdateRequest,
)
from api.settings import settings

router = APIRouter(tags=["predict"])

THRESHOLD_SETTING_KEY = "decision_threshold"
_MAX_CSV_ROWS = settings.max_upload_rows


def _threshold_override(db: Session) -> float | None:
    row = db.get(SystemSetting, THRESHOLD_SETTING_KEY)
    return float(row.value) if row else None


def _score_and_record(
    rows: list[dict], db: Session, current_user: CurrentUser, input_hash: str,
) -> PredictResponse:
    service = get_model_service()
    try:
        prob, pred, level, X = service.score(rows, threshold_override=_threshold_override(db))
    except ModelNotAvailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model is currently loaded. The ML pipeline may not "
                "have produced models/final_model.joblib yet, or model "
                "integrity verification failed. See /health."
            ),
        ) from exc

    full_detail = full_explainability_allowed(current_user.role)
    items = []
    for i in range(len(rows)):
        base = dict(
            fraud_probability=float(prob[i]),
            fraud_prediction=int(pred[i]),
            risk_level=str(level[i]),
        )
        if full_detail:
            contributions = service.explain(X.iloc[[i]])
            items.append(PredictionItemFull(
                **base,
                top_features=[FeatureContribution(**c) for c in contributions],
            ))
        else:
            items.append(PredictionItemBasic(**base))

    threshold_used = service.current_threshold(_threshold_override(db))
    risk_counts: dict[str, int] = {}
    for lv in level:
        risk_counts[str(lv)] = risk_counts.get(str(lv), 0) + 1

    record = PredictionRecord(
        user_id=current_user.id,
        model_version=service.model_version,
        input_row_count=len(rows),
        input_file_hash=input_hash,
        risk_level_counts=json.dumps(risk_counts),
        threshold_used=threshold_used,
    )
    db.add(record)
    db.commit()

    return PredictResponse(
        model_version=service.model_version,
        threshold_used=threshold_used,
        row_count=len(rows),
        predictions=items,
    )


@router.post("/predict", response_model=PredictResponse)
@limiter.limit(settings.predict_rate_limit)
def predict(
    request: Request,
    body: PredictRequest,
    current_user: CurrentUser = Depends(require_predict),
    db: Session = Depends(get_db),
):
    req_id = str(uuid.uuid4())
    rows = [r.model_dump() for r in body.rows]
    payload_bytes = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    import hashlib
    input_hash = hashlib.sha256(payload_bytes).hexdigest()

    try:
        result = _score_and_record(rows, db, current_user, input_hash)
    except HTTPException as exc:
        write_audit(
            db, request_id=req_id, user_id=current_user.id, role=current_user.role.value,
            action="PREDICT", resource="/predict", result="ERROR", detail=str(exc.detail),
        )
        raise

    write_audit(
        db, request_id=req_id, user_id=current_user.id, role=current_user.role.value,
        action="PREDICT", resource="/predict", result="SUCCESS",
        detail=f"rows={len(rows)}",
    )
    return result


_REQUIRED_CSV_COLUMNS = set(ApplicationRow.model_fields.keys())


@router.post("/predict/file", response_model=PredictResponse)
@limiter.limit(settings.predict_rate_limit)
def predict_file(
    request: Request,
    file: UploadFile,
    current_user: CurrentUser = Depends(require_predict),
    db: Session = Depends(get_db),
):
    """CSV upload scoring. Validated by: extension, size limit, and schema
    (required columns present, no unexpected extra columns) -- never trusted
    by extension alone, and the actual bytes are parsed and type-checked row
    by row through the same ApplicationRow Pydantic model as /predict."""
    req_id = str(uuid.uuid4())

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .csv files are accepted")

    raw = file.file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_bytes} byte limit",
        )

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is not valid UTF-8 text") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV has no header row")

    header = set(reader.fieldnames)
    missing = _REQUIRED_CSV_COLUMNS - header
    unexpected = header - _REQUIRED_CSV_COLUMNS
    if missing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"CSV is missing required columns: {sorted(missing)}")
    if unexpected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"CSV has unexpected columns: {sorted(unexpected)}")

    raw_rows = list(reader)
    if not raw_rows:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV has no data rows")
    if len(raw_rows) > _MAX_CSV_ROWS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"CSV exceeds the {_MAX_CSV_ROWS} row limit")

    try:
        validated = [ApplicationRow(**row) for row in raw_rows]
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"CSV failed schema validation: {exc}") from exc

    rows = [r.model_dump() for r in validated]
    import hashlib
    input_hash = hashlib.sha256(raw).hexdigest()

    try:
        result = _score_and_record(rows, db, current_user, input_hash)
    except HTTPException as exc:
        write_audit(
            db, request_id=req_id, user_id=current_user.id, role=current_user.role.value,
            action="PREDICT_FILE", resource="/predict/file", result="ERROR", detail=str(exc.detail),
        )
        raise

    write_audit(
        db, request_id=req_id, user_id=current_user.id, role=current_user.role.value,
        action="PREDICT_FILE", resource="/predict/file", result="SUCCESS",
        detail=f"rows={len(rows)} file={file.filename}",
    )
    return result


@router.get("/predictions", response_model=list[PredictionRecordEntry])
def list_predictions(
    limit: int = 50,
    current_user: CurrentUser = Depends(require_view_predictions),
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 200))
    records = (
        db.query(PredictionRecord)
        .order_by(desc(PredictionRecord.timestamp))
        .limit(limit)
        .all()
    )
    return [
        PredictionRecordEntry(
            id=r.id,
            timestamp=r.timestamp.isoformat(),
            user_id=r.user_id,
            model_version=r.model_version,
            input_row_count=r.input_row_count,
            risk_level_counts=json.loads(r.risk_level_counts),
            threshold_used=r.threshold_used,
        )
        for r in records
    ]


@router.patch("/settings/threshold")
def update_threshold(
    body: ThresholdUpdateRequest,
    current_user: CurrentUser = Depends(require_threshold_change),
    db: Session = Depends(get_db),
):
    """Risk Manager / Admin only, per the permission matrix."""
    threshold = body.threshold

    row = db.get(SystemSetting, THRESHOLD_SETTING_KEY)
    if row is None:
        row = SystemSetting(key=THRESHOLD_SETTING_KEY, value=str(threshold), updated_by=current_user.id)
        db.add(row)
    else:
        row.value = str(threshold)
        row.updated_by = current_user.id
    db.commit()

    write_audit(
        db, request_id=str(uuid.uuid4()), user_id=current_user.id, role=current_user.role.value,
        action="THRESHOLD_CHANGE", resource="/settings/threshold", result="SUCCESS",
        detail=f"new_threshold={threshold}",
    )
    return {"threshold": threshold}
