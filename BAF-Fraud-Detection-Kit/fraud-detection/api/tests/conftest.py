"""
api/tests/conftest.py -- test environment setup.

Uses a dedicated, disposable SQLite file (never the dev .env database) and a
real, actually-loaded model artifact (training an interim demo model via
api/scripts/train_demo_model.py if none exists yet -- see that script's
docstring for why an interim model exists at all). Every environment
variable is set here, BEFORE the first `import api...`, since
api/settings.py reads them once at import time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

API_TESTS_DIR = Path(__file__).resolve().parent
FRAUD_DETECTION_ROOT = API_TESTS_DIR.parent.parent
TEST_DB_PATH = API_TESTS_DIR / "test_fraud_api.db"

# Fresh DB for every test session.
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["JWT_SECRET"] = "test-only-secret-not-for-production-0123456789"
os.environ["MFA_REQUIRED"] = "false"
os.environ["ENFORCE_MODEL_CHECKSUM"] = "false"
os.environ["LOGIN_RATE_LIMIT"] = "5/minute"
os.environ["PREDICT_RATE_LIMIT"] = "30/minute"
os.environ["CORS_ALLOW_ORIGINS"] = "http://localhost:3000"
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = "test-admin@example.com"
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = "TestAdminPassword123"

if str(FRAUD_DETECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAUD_DETECTION_ROOT))

import pytest
from fastapi.testclient import TestClient

ADMIN_EMAIL = "test-admin@example.com"
ADMIN_PASSWORD = "TestAdminPassword123"


def _ensure_model_exists() -> None:
    models_dir = FRAUD_DETECTION_ROOT / "models"
    model_path = models_dir / "final_model.joblib"
    checksum_path = models_dir / "model_checksum.json"
    if not model_path.exists():
        from api.scripts import train_demo_model
        train_demo_model.main()
    if not checksum_path.exists():
        from api.scripts import record_model_checksum
        record_model_checksum.main()


_ensure_model_exists()

from api.main import app  # noqa: E402  (must come after env vars are set)
from api.rate_limit import limiter  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        # The app's own startup event only creates tables (api.database.init_db);
        # bootstrap-admin creation is a deliberately separate, explicit step
        # (see api/scripts/init_db.py docstring), so it's invoked here too.
        from api.scripts import init_db as init_db_script
        init_db_script.main()
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Every test starts with a clean rate-limit bucket, so tests are
    order-independent and the dedicated brute-force test gets a
    deterministic count."""
    limiter.reset()
    yield


@pytest.fixture()
def admin_token(client: TestClient) -> str:
    r = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def make_user(client: TestClient, admin_token: str, email: str, password: str, role: str) -> str:
    r = client.post(
        "/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": email, "password": password, "role": role},
    )
    assert r.status_code in (200, 201, 409), r.text
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]
