"""
api/models_db.py -- SQLAlchemy ORM models.

Three tables, matching the design in the task spec (and mirroring the
audit-log design principle the ML side already uses in src/audit.py: never
store passwords/tokens/full feature vectors/raw applicant PII in the audit
trail or prediction records -- metadata only).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    ADMIN = "ADMIN"
    FRAUD_ANALYST = "FRAUD_ANALYST"
    RISK_MANAGER = "RISK_MANAGER"
    AUDITOR = "AUDITOR"
    VIEWER = "VIEWER"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Role] = mapped_column(Enum(Role), nullable=False, default=Role.VIEWER)

    # TOTP secret, encrypted at rest with Fernet (api/security.py
    # encrypt_mfa_secret/decrypt_mfa_secret, key from MFA_ENCRYPTION_KEY) --
    # this column never holds the plaintext base32 secret, only a Fernet
    # token (~140 chars for a 32-char base32 secret). See api/SECURITY.md
    # "MFA secret at rest".
    mfa_secret: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    audit_entries = relationship("AuditLog", back_populates="user")
    predictions = relationship("PredictionRecord", back_populates="user")


class AuditLog(Base):
    """Append-only audit trail. No row is ever updated or deleted through
    the application -- there is deliberately no update/delete endpoint or
    repository method for this table. Never store: passwords, tokens,
    full feature vectors, or raw applicant PII here."""

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource: Mapped[str] = mapped_column(String(256), nullable=False)
    result: Mapped[str] = mapped_column(String(32), nullable=False)  # SUCCESS | DENIED | ERROR
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)  # short, non-sensitive context

    user = relationship("User", back_populates="audit_entries")


class PredictionRecord(Base):
    """Metadata about a prediction run only -- never raw applicant data."""

    __tablename__ = "prediction_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    input_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    input_file_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # sha256 hex
    risk_level_counts: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-encoded dict
    threshold_used: Mapped[float] = mapped_column(nullable=False)

    user = relationship("User", back_populates="predictions")


class SystemSetting(Base):
    """Small key/value store for the one thing Risk Manager can change at
    runtime: the decision threshold override used by /predict. Falls back to
    the trained model's own threshold (model_meta.json) when unset."""

    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class RevokedToken(Base):
    """JTI blocklist for logout / explicit revocation of otherwise-valid JWTs
    (JWTs are stateless by design, so revocation needs an explicit table)."""

    __tablename__ = "revoked_tokens"

    jti: Mapped[str] = mapped_column(String(36), primary_key=True)
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
