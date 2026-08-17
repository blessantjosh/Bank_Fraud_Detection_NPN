"""
audit.py -- append-only prediction audit trail (JSON Lines).

Every prediction run this system makes writes one line to
reports/metrics/audit_log.jsonl: who ran it (an admin identity derived from
the auth token -- never the token itself), which model/checkpoint scored it,
how many rows and a content hash of the input file (NOT the raw applicant
data, so the log stays lightweight and never duplicates PII-adjacent data
into a second location), and a summary of the output risk levels (counts
per bucket, not individual rows).

This is an APPLICATION-LEVEL audit trail suitable for an internal tool. It
is NOT a replacement for a bank's real compliance/SIEM systems -- see
README.md "Governance & Audit" for the explicit caveat.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("fraud_detection.audit")

DEFAULT_AUDIT_LOG_PATH = "reports/metrics/audit_log.jsonl"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_prediction_audit(
    log_path,
    admin_identity: str,
    model_type: str,
    strategy: str,
    model_iteration,
    input_row_count: int,
    input_content_hash: str,
    risk_level_counts: dict,
) -> dict:
    """
    Append one audit entry. Every field here is something that actually
    happened in this run -- nothing is fabricated or backfilled. If the
    write itself fails, that failure is logged loudly and re-raised: a
    silently-dropped audit entry would defeat the entire point of an audit
    trail, so it must never be swallowed.
    """
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "admin_identity": admin_identity,
        "model_type": model_type,
        "strategy": strategy,
        "model_iteration": model_iteration,
        "input_row_count": int(input_row_count),
        "input_content_sha256": input_content_hash,
        "risk_level_counts": {str(k): int(v) for k, v in risk_level_counts.items()},
    }

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        logger.error(
            "FAILED to write audit log entry to %s -- this must be visible, "
            "not swallowed", log_path, exc_info=True,
        )
        raise

    logger.info(
        "Audit entry recorded: admin=%s model=%s/%s rows=%d risk_levels=%s",
        admin_identity, model_type, strategy, input_row_count, entry["risk_level_counts"],
    )
    return entry
