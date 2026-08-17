"""
api/routers/auth.py -- /auth/login, /auth/refresh, /auth/mfa/verify,
/auth/mfa/setup, /auth/mfa/enable.

Brute-force protection: /auth/login is rate-limited (see api/rate_limit.py)
and always returns the same generic "Invalid email or password" message on
any failure -- wrong password, unknown email, and inactive account all look
identical to the caller, so this endpoint never confirms whether an email
is registered.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.audit import write_audit
from api.database import get_db
from api.models_db import RevokedToken, Role, User
from api.rate_limit import limiter
from api.rbac import CurrentUser, get_current_user
from api.schemas import (
    LoginRequest,
    LoginResponse,
    MfaEnableRequest,
    MfaSetupResponse,
    MfaVerifyRequest,
    RefreshRequest,
    TokenPairResponse,
)
from api.security import (
    MfaSecretError,
    TokenError,
    TokenType,
    create_access_token,
    create_mfa_pending_token,
    create_refresh_token,
    decode_token,
    decrypt_mfa_secret,
    encrypt_mfa_secret,
    generate_totp_secret,
    totp_provisioning_uri,
    verify_password,
    verify_totp,
)
from api.settings import settings

router = APIRouter(prefix="/auth", tags=["auth"])

GENERIC_LOGIN_ERROR = "Invalid email or password"


@router.post("/login", response_model=LoginResponse)
@limiter.limit(settings.login_rate_limit)
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    req_id = str(uuid.uuid4())
    user = db.query(User).filter(User.email == body.email.lower()).first()

    if user is None or not user.is_active or not verify_password(body.password, user.hashed_password):
        write_audit(
            db, request_id=req_id, user_id=user.id if user else None,
            role=user.role.value if user else None, action="LOGIN", resource="/auth/login",
            result="DENIED", detail="invalid credentials",
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=GENERIC_LOGIN_ERROR)

    if settings.mfa_required and user.role == Role.ADMIN and user.mfa_enabled:
        mfa_token = create_mfa_pending_token(user.id, user.role.value)
        write_audit(
            db, request_id=req_id, user_id=user.id, role=user.role.value,
            action="LOGIN", resource="/auth/login", result="MFA_PENDING",
        )
        return LoginResponse(mfa_required=True, mfa_token=mfa_token)

    access = create_access_token(user.id, user.role.value)
    refresh = create_refresh_token(user.id, user.role.value)
    write_audit(
        db, request_id=req_id, user_id=user.id, role=user.role.value,
        action="LOGIN", resource="/auth/login", result="SUCCESS",
    )
    return LoginResponse(access_token=access, refresh_token=refresh, mfa_required=False)


@router.post("/mfa/verify", response_model=TokenPairResponse)
@limiter.limit(settings.login_rate_limit)
def mfa_verify(request: Request, body: MfaVerifyRequest, db: Session = Depends(get_db)):
    req_id = str(uuid.uuid4())
    try:
        payload = decode_token(body.mfa_token, TokenType.MFA_PENDING)
    except TokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    user = db.get(User, payload["sub"])
    if user is None or not user.is_active or not user.mfa_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    try:
        plaintext_secret = decrypt_mfa_secret(user.mfa_secret)
    except MfaSecretError as exc:
        write_audit(
            db, request_id=req_id, user_id=user.id, role=user.role.value,
            action="MFA_VERIFY", resource="/auth/mfa/verify", result="ERROR",
            detail="stored MFA secret could not be decrypted",
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session") from exc

    if not verify_totp(plaintext_secret, body.code):
        write_audit(
            db, request_id=req_id, user_id=user.id, role=user.role.value,
            action="MFA_VERIFY", resource="/auth/mfa/verify", result="DENIED",
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid MFA code")

    access = create_access_token(user.id, user.role.value)
    refresh = create_refresh_token(user.id, user.role.value)
    write_audit(
        db, request_id=req_id, user_id=user.id, role=user.role.value,
        action="MFA_VERIFY", resource="/auth/mfa/verify", result="SUCCESS",
    )
    return TokenPairResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenPairResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    req_id = str(uuid.uuid4())
    try:
        payload = decode_token(body.refresh_token, TokenType.REFRESH)
    except TokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    jti = payload.get("jti")
    if jti and db.get(RevokedToken, jti) is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

    user = db.get(User, payload["sub"])
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive account")

    # Rotate: revoke the used refresh token, issue a new pair.
    if jti:
        db.add(RevokedToken(jti=jti))
        db.commit()

    access = create_access_token(user.id, user.role.value)
    new_refresh = create_refresh_token(user.id, user.role.value)
    write_audit(
        db, request_id=req_id, user_id=user.id, role=user.role.value,
        action="TOKEN_REFRESH", resource="/auth/refresh", result="SUCCESS",
    )
    return TokenPairResponse(access_token=access, refresh_token=new_refresh)


@router.post("/mfa/setup", response_model=MfaSetupResponse)
def mfa_setup(current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)):
    """Admin-only (any authenticated user may enroll their own MFA; here we
    scope it to ADMIN because MFA_REQUIRED only applies to ADMIN accounts).
    Generates a secret and returns the otpauth:// URI for a real
    authenticator app -- mfa_enabled stays false until /auth/mfa/enable
    confirms the app is actually synced."""
    if current_user.role != Role.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="MFA enrollment is only required for admin accounts")

    user = db.get(User, current_user.id)
    secret = generate_totp_secret()
    # Encrypted immediately, before the row is ever written -- the plaintext
    # secret only exists in this function's local variables (returned once,
    # here, for the caller to enroll in their authenticator app) and is
    # never itself persisted or logged.
    user.mfa_secret = encrypt_mfa_secret(secret)
    user.mfa_enabled = False
    db.commit()
    return MfaSetupResponse(secret=secret, provisioning_uri=totp_provisioning_uri(secret, user.email))


@router.post("/mfa/enable")
def mfa_enable(
    body: MfaEnableRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.get(User, current_user.id)
    if not user.mfa_secret:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Call /auth/mfa/setup first")

    try:
        plaintext_secret = decrypt_mfa_secret(user.mfa_secret)
    except MfaSecretError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Stored MFA secret is invalid -- call /auth/mfa/setup again") from exc

    if not verify_totp(plaintext_secret, body.code):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid MFA code")
    user.mfa_enabled = True
    db.commit()
    return {"mfa_enabled": True}
