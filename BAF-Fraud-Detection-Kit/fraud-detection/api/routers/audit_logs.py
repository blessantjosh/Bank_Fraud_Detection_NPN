"""
api/routers/audit_logs.py -- GET /audit-logs.

Permission matrix detail (see api/rbac.py docstring):
  - AUDITOR, RISK_MANAGER, ADMIN: full audit log, all users' actions.
  - FRAUD_ANALYST: "limited audit view" -- implemented here as visibility
    restricted to that analyst's OWN actions only (their user_id), not the
    whole organization's activity.
  - VIEWER: no access at all (403).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import desc
from sqlalchemy.orm import Session

from api.audit import write_audit
from api.database import get_db
from api.models_db import AuditLog, Role
from api.rbac import CurrentUser, full_audit_view_allowed, require_audit_view
from api.schemas import AuditLogEntry

router = APIRouter(tags=["audit"])


@router.get("/audit-logs", response_model=list[AuditLogEntry])
def get_audit_logs(
    limit: int = 100,
    current_user: CurrentUser = Depends(require_audit_view),
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 500))
    query = db.query(AuditLog).order_by(desc(AuditLog.timestamp))

    if not full_audit_view_allowed(current_user.role):
        # FRAUD_ANALYST: limited to their own actions.
        query = query.filter(AuditLog.user_id == current_user.id)

    records = query.limit(limit).all()

    write_audit(
        db, request_id=str(uuid.uuid4()), user_id=current_user.id, role=current_user.role.value,
        action="VIEW_AUDIT_LOG", resource="/audit-logs", result="SUCCESS",
        detail="full" if full_audit_view_allowed(current_user.role) else "own-actions-only",
    )

    return [
        AuditLogEntry(
            id=r.id,
            timestamp=r.timestamp.isoformat(),
            user_id=r.user_id,
            role=r.role,
            action=r.action,
            resource=r.resource,
            result=r.result,
            request_id=r.request_id,
            detail=r.detail,
        )
        for r in records
    ]
