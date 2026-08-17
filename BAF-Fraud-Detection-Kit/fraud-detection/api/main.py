"""
api/main.py -- FastAPI app entrypoint for the security-hardened API layer.

Run locally:
    uvicorn api.main:app --reload --port 8000
(from the fraud-detection/ directory, so `src` and `api` both import cleanly)
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

_STATIC_DIR = Path(__file__).resolve().parent / "static"

from api.database import init_db
from api.model_service import get_model_service
from api.rate_limit import limiter
from api.routers import admin, analytics, audit_logs, auth, health, predict
from api.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    settings.effective_jwt_secret()  # triggers the loud warning if unset
    get_model_service()  # attempt to load the model now, not on first request
    logger.info(
        "Fraud Detection API started. mfa_required=%s enforce_model_checksum=%s database_url=%s",
        settings.mfa_required, settings.enforce_model_checksum, settings.database_url,
    )
    yield


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description=(
        "Security-hardened API layer over the BAF fraud-detection model. "
        "See api/SECURITY.md for exactly what is and is not covered by "
        "this codebase."
    ),
    lifespan=lifespan,
)

# --- rate limiting --------------------------------------------------------
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS: explicit allow-list, never "*" ----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)


# --- request id + security headers ----------------------------------------
@app.middleware("http")
async def add_request_id_and_security_headers(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # The real JSON API gets a fully locked-down CSP -- it never renders HTML,
    # so it needs nothing. /docs and /redoc are FastAPI's own HTML pages that
    # load Swagger UI's CSS/JS from a CDN; a blanket default-src 'none' here
    # silently blocks that (blank white page, no console-visible server
    # error), so those two routes get a CSP that allows exactly that CDN and
    # inline script (which is how swagger-ui-dist's bundle initializes) --
    # nothing else is relaxed.
    if request.url.path in ("/docs", "/redoc", "/docs/oauth2-redirect"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com; "
            "frame-ancestors 'none'"
        )
    elif request.url.path == "/dashboard":
        # The console is a single self-contained HTML file (inline <style>/
        # <script>, no external assets) that only ever calls this SAME API
        # (fetch to same-origin /auth, /predict, /predictions, /audit-logs,
        # /admin/users, /model/metrics, /model/figures/*) -- so unlike /docs,
        # this needs no third-party host at all. 'unsafe-inline' is required
        # because the script/style live in the page itself rather than a
        # separate file; connect-src is left to default-src 'self' since
        # every request this page makes is same-origin. img-src additionally
        # allows blob: (still not a third-party host) because report figures
        # are fetched with an Authorization header and rendered via
        # URL.createObjectURL -- a plain <img src="/model/figures/..."> could
        # not attach that header, and 'self' alone does not cover blob: URLs.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob:; "
            "frame-ancestors 'none'"
        )
    else:
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    # HSTS only makes sense once TLS actually terminates in front of this
    # process -- see api/SECURITY.md "TLS". Sending it unconditionally is
    # harmless (browsers ignore it over plain HTTP) and correct once a real
    # reverse proxy terminates TLS in production.
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    logger.info(
        "%s %s -> %s (%.1fms) [%s]",
        request.method, request.url.path, response.status_code,
        (time.time() - start) * 1000, request_id,
    )
    return response


# --- secure global error handling ------------------------------------------
# Never leak stack traces, file paths, or internal exception details to the
# client. Every unhandled exception is logged in full server-side (with
# traceback) and the client gets a generic message plus the request_id, so
# a real incident can still be correlated in server logs without exposing
# anything to the caller.

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    # Pydantic validation errors are safe to return (they describe the
    # caller's own malformed request, not server internals) but are
    # sanitized to field/type/msg only -- no raw input echoed back.
    errors = [
        {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Request validation failed", "errors": errors, "request_id": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled exception on %s %s [%s]", request.method, request.url.path, request_id, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please contact support with this request ID.", "request_id": request_id},
    )


# --- console (a real, live dashboard over this same API -- not a mockup) ---
@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(_STATIC_DIR / "dashboard.html")


# --- routers ----------------------------------------------------------------
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(audit_logs.router)
app.include_router(admin.router)
app.include_router(analytics.router)
