"""
api/security.py -- password hashing, JWT issuance/verification, TOTP MFA.

Password hashing: Argon2id via passlib + argon2-cffi. Verified available and
working in this environment (see api/README.md "Environment notes"); bcrypt
is kept as a documented fallback scheme in the CryptContext so existing
bcrypt hashes (e.g. from a future migration) still verify, but new hashes
are always Argon2id.

MFA secret at rest: encrypted with Fernet (symmetric, `cryptography`
library -- free, local, no KMS/paid service) before it is ever written to
the DB. The plaintext TOTP secret exists only transiently in memory: once
when generated at /auth/mfa/setup (encrypted immediately before the
User row is written), and once when decrypted to verify a submitted code
-- never logged, never returned by any endpoint after setup.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum

import jwt
from cryptography.fernet import Fernet, InvalidToken
from passlib.context import CryptContext

from api.settings import settings

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(password, hashed)
    except Exception:
        # Malformed/unknown hash format -- fail closed, never raise past auth.
        return False


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    MFA_PENDING = "mfa_pending"


def _create_token(subject: str, role: str, token_type: TokenType, expire_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "type": token_type.value,
        "iat": now,
        "exp": now + timedelta(minutes=expire_minutes),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.effective_jwt_secret(), algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str, role: str) -> str:
    return _create_token(user_id, role, TokenType.ACCESS, settings.access_token_expire_minutes)


def create_refresh_token(user_id: str, role: str) -> str:
    return _create_token(user_id, role, TokenType.REFRESH, settings.refresh_token_expire_minutes)


def create_mfa_pending_token(user_id: str, role: str) -> str:
    """Short-lived token proving password was correct, but MFA is still
    outstanding. Cannot be used at any protected endpoint (checked via its
    `type` claim), only at /auth/mfa/verify."""
    return _create_token(user_id, role, TokenType.MFA_PENDING, expire_minutes=5)


class TokenError(Exception):
    pass


def decode_token(token: str, expected_type: TokenType) -> dict:
    try:
        payload = jwt.decode(token, settings.effective_jwt_secret(), algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid token") from exc

    if payload.get("type") != expected_type.value:
        raise TokenError("Wrong token type for this operation")
    return payload


# --- MFA (TOTP) -----------------------------------------------------------

def generate_totp_secret() -> str:
    import pyotp

    return pyotp.random_base32()


def totp_provisioning_uri(secret: str, account_email: str) -> str:
    import pyotp

    return pyotp.totp.TOTP(secret).provisioning_uri(name=account_email, issuer_name=settings.mfa_issuer)


def verify_totp(secret: str, code: str) -> bool:
    import pyotp

    return pyotp.totp.TOTP(secret).verify(code, valid_window=1)


# --- MFA secret encryption at rest -----------------------------------------

def _fernet() -> Fernet:
    return Fernet(settings.effective_mfa_encryption_key().encode("utf-8"))


def encrypt_mfa_secret(plaintext_secret: str) -> str:
    """Encrypt a freshly-generated TOTP secret before it is written to the
    `users.mfa_secret` column. Returns a Fernet token (str, safe to store as
    text) -- never the plaintext secret."""
    return _fernet().encrypt(plaintext_secret.encode("utf-8")).decode("utf-8")


class MfaSecretError(Exception):
    """Raised when the stored MFA secret cannot be decrypted (wrong/rotated
    MFA_ENCRYPTION_KEY, or corrupted data) -- treated as an authentication
    failure by callers, never as a crash."""


def decrypt_mfa_secret(encrypted_secret: str) -> str:
    """Decrypt a stored MFA secret, in memory only, to verify a submitted
    TOTP code. Callers must never log the return value."""
    try:
        return _fernet().decrypt(encrypted_secret.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise MfaSecretError("Stored MFA secret could not be decrypted") from exc
