"""
api/routers/admin.py -- /admin/users (Admin only): list, create, update
role/active-status. There is deliberately no DELETE -- deactivate
(is_active=false) instead, so the audit trail and prediction history
FK references stay intact.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.audit import write_audit
from api.database import get_db
from api.models_db import User
from api.rbac import CurrentUser, require_admin
from api.schemas import UserCreateRequest, UserResponse, UserUpdateRequest
from api.security import hash_password

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[UserResponse])
def list_users(current_user: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at).all()
    return [_to_response(u) for u in users]


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    body: UserCreateRequest,
    current_user: CurrentUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    existing = db.query(User).filter(User.email == body.email.lower()).first()
    if existing is not None:
        # Deliberately vague to avoid confirming exact collision reason to a
        # caller who has already authenticated as admin -- still no need to
        # leak internals.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A user with this email already exists")

    user = User(email=body.email.lower(), hashed_password=hash_password(body.password), role=body.role)
    db.add(user)
    db.commit()
    db.refresh(user)

    write_audit(
        db, request_id=str(uuid.uuid4()), user_id=current_user.id, role=current_user.role.value,
        action="USER_CREATE", resource=f"/admin/users/{user.id}", result="SUCCESS",
        detail=f"created role={body.role.value}",
    )
    return _to_response(user)


@router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    body: UserUpdateRequest,
    current_user: CurrentUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active
    db.commit()
    db.refresh(user)

    write_audit(
        db, request_id=str(uuid.uuid4()), user_id=current_user.id, role=current_user.role.value,
        action="USER_UPDATE", resource=f"/admin/users/{user_id}", result="SUCCESS",
        detail=f"role={user.role.value} active={user.is_active}",
    )
    return _to_response(user)


def _to_response(u: User) -> UserResponse:
    return UserResponse(
        id=u.id, email=u.email, role=u.role, mfa_enabled=u.mfa_enabled,
        is_active=u.is_active, created_at=u.created_at.isoformat(),
    )
