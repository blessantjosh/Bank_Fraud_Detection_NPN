"""
api/audit.py -- append-only audit-log writes to the `audit_log` table.

Distinct from src/audit.py (which is the ML pipeline's own JSONL log for CLI
predict.py runs). This one is the API layer's DB-backed log, covering every
authenticated action taken through the HTTP interface: logins (success and
failure), token refreshes, predictions, admin user changes, threshold
changes, and audit-log reads themselves.

Never call db.query(AuditLog).update(...) or .delete(...) anywhere in this
codebase -- there is deliberately no code path that mutates an existing row.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from api.models_db import AuditLog

logger = logging.getLogger("fraud_api.audit")


def write_audit(
    db: Session,
    *,
    request_id: str,
    user_id: str | None,
    role: str | None,
    action: str,
    resource: str,
    result: str,
    detail: str | None = None,
) -> None:
    entry = AuditLog(
        request_id=request_id,
        user_id=user_id,
        role=role,
        action=action,
        resource=resource,
        result=result,
        detail=detail,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.error("FAILED to write audit log entry (request_id=%s action=%s)", request_id, action, exc_info=True)
        raise
