"""
api/routers/health.py -- GET /health. Public, unauthenticated, deliberately
minimal: enough for a load balancer / demo script to know the service and
model are up, nothing that helps an attacker (no stack traces, no version
strings beyond what's already implied by model_version format, no secrets).
"""

from __future__ import annotations

from fastapi import APIRouter

from api.model_service import get_model_service
from api.schemas import HealthResponse
from api.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    service = get_model_service()
    return HealthResponse(
        status="ok",
        model_loaded=service.is_loaded,
        model_checksum_verified=service.checksum_verified,
        mfa_required=settings.mfa_required,
    )
