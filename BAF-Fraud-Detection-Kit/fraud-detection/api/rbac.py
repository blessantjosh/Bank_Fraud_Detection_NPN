"""
api/rbac.py -- role-based access control: the permission matrix and the
FastAPI dependencies that enforce it server-side on every protected route.

Permission matrix (exact, from the spec):

| Capability                         | VIEWER | FRAUD_ANALYST | RISK_MANAGER | AUDITOR | ADMIN |
|-------------------------------------|--------|----------------|---------------|---------|-------|
| view predictions (metadata/history) |   Y    |       Y        |       Y       |    Y    |   Y   |
| run predictions (/predict)          |        |       Y        |       Y       |         |   Y   |
| view audit log (own actions only)   |        |       Y        |               |         |       |
| view audit log (full)               |        |                |       Y       |    Y    |   Y   |
| change decision threshold           |        |                |       Y       |         |   Y   |
| full explainability (SHAP/features) |        |                |       Y       |         |   Y   |
| user management                     |        |                |               |         |   Y   |

Never trust a frontend-only check: every one of these is enforced here, as a
FastAPI dependency evaluated on the server for every request to a protected
route.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.database import get_db
from api.models_db import Role, RevokedToken, User
from api.security import TokenError, TokenType, decode_token

_BEARER_PREFIX = "Bearer "


class CurrentUser:
    __slots__ = ("id", "email", "role")

    def __init__(self, id: str, email: str, role: Role):
        self.id = id
        self.email = email
        self.role = role


def _extract_bearer_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_header[len(_BEARER_PREFIX):]


def get_current_user(request: Request, db: Session = Depends(get_db)) -> CurrentUser:
    token = _extract_bearer_token(request)
    try:
        payload = decode_token(token, TokenType.ACCESS)
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    jti = payload.get("jti")
    if jti and db.get(RevokedToken, jti) is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

    user_id = payload["sub"]
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive account")

    # Defense in depth: role claimed in the token must still match the DB
    # record (in case an admin changed the user's role after the token was
    # issued -- the token's own claim is not trusted as the source of truth).
    return CurrentUser(id=user.id, email=user.email, role=user.role)


def require_roles(*allowed_roles: Role):
    """FastAPI dependency factory: 403s any role not in allowed_roles."""

    def _dependency(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action",
            )
        return current_user

    return _dependency


# Convenience, named dependencies matching the permission matrix above.
require_admin = require_roles(Role.ADMIN)
require_predict = require_roles(Role.ADMIN, Role.RISK_MANAGER, Role.FRAUD_ANALYST)
require_view_predictions = require_roles(
    Role.ADMIN, Role.RISK_MANAGER, Role.FRAUD_ANALYST, Role.AUDITOR, Role.VIEWER
)
require_audit_view = require_roles(Role.ADMIN, Role.RISK_MANAGER, Role.AUDITOR, Role.FRAUD_ANALYST)
require_threshold_change = require_roles(Role.ADMIN, Role.RISK_MANAGER)


def full_explainability_allowed(role: Role) -> bool:
    return role in (Role.ADMIN, Role.RISK_MANAGER)


def full_audit_view_allowed(role: Role) -> bool:
    return role in (Role.ADMIN, Role.RISK_MANAGER, Role.AUDITOR)
