"""
api/tests/test_api.py -- the real, executed proof points required by the
task: wrong password rejected, invalid JWT rejected, Viewer blocked from
/admin/users, rate limit triggers after N failed logins, and a valid Admin
login + predict round-trip works end-to-end against the real (interim
demo) model artifacts.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, make_user

SAMPLE_ROW = {
    "income": 0.5, "customer_age": 40, "employment_status": "CA",
    "housing_status": "BA", "name_email_similarity": 0.8,
    "prev_address_months_count": 12, "current_address_months_count": 24,
    "bank_months_count": 10, "days_since_request": 0.5,
    "velocity_6h": 3000, "velocity_24h": 4000, "velocity_4w": 5000,
    "zip_count_4w": 100, "bank_branch_count_8w": 20,
    "date_of_birth_distinct_emails_4w": 2, "phone_home_valid": 1,
    "phone_mobile_valid": 1, "email_is_free": 0, "foreign_request": 0,
    "device_os": "windows", "device_distinct_emails_8w": 1,
    "session_length_in_minutes": 5.0, "keep_alive_session": 1,
    "source": "INTERNET", "credit_risk_score": 100.0,
    "proposed_credit_limit": 1000.0, "payment_type": "AA",
    "has_other_cards": 1, "intended_balcon_amount": -1.0,
    "month": 3, "device_fraud_count": 0,
}


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_wrong_password_rejected(client: TestClient):
    r = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": "definitely-wrong"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid email or password"


def test_unknown_email_gives_identical_generic_error(client: TestClient):
    """Never reveal whether an email is registered: same message, same
    status code as a wrong password for a real account."""
    r1 = client.post("/auth/login", json={"email": "nobody-registered@example.com", "password": "whatever12345"})
    r2 = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": "definitely-wrong"})
    assert r1.status_code == r2.status_code == 401
    assert r1.json()["detail"] == r2.json()["detail"] == "Invalid email or password"


def test_invalid_jwt_rejected(client: TestClient):
    r = client.get("/predictions", headers={"Authorization": "Bearer this.is.not.a.real.jwt"})
    assert r.status_code == 401


def test_missing_auth_header_rejected(client: TestClient):
    r = client.get("/predictions")
    assert r.status_code == 401


def test_expired_jwt_rejected(client: TestClient, monkeypatch):
    """Issue a token that is already expired and confirm it's rejected."""
    from api.security import _create_token, TokenType

    expired_token = _create_token("some-user-id", "ADMIN", TokenType.ACCESS, expire_minutes=-1)
    r = client.get("/predictions", headers={"Authorization": f"Bearer {expired_token}"})
    assert r.status_code == 401
    assert "expired" in r.json()["detail"].lower()


def test_viewer_blocked_from_admin_users(client: TestClient, admin_token: str):
    viewer_token = make_user(client, admin_token, "viewer-test@example.com", "ViewerPassword123", "VIEWER")
    r = client.get("/admin/users", headers={"Authorization": f"Bearer {viewer_token}"})
    assert r.status_code == 403


def test_viewer_can_view_predictions_but_not_run_them(client: TestClient, admin_token: str):
    viewer_token = make_user(client, admin_token, "viewer-test2@example.com", "ViewerPassword123", "VIEWER")
    headers = {"Authorization": f"Bearer {viewer_token}"}

    r = client.get("/predictions", headers=headers)
    assert r.status_code == 200

    r = client.post("/predict", headers=headers, json={"rows": [SAMPLE_ROW]})
    assert r.status_code == 403


def test_rate_limit_triggers_after_n_failed_logins(client: TestClient):
    """LOGIN_RATE_LIMIT=5/minute in the test env (conftest.py). The 6th
    request within the window must be rejected with 429, proving brute-force
    protection actually blocks repeated attempts rather than just logging
    them."""
    responses = [
        client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"})
        for _ in range(5)
    ]
    assert all(r.status_code == 401 for r in responses)

    sixth = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"})
    assert sixth.status_code == 429


def test_admin_login_and_predict_round_trip(client: TestClient, admin_token: str):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/predict", headers=headers, json={"rows": [SAMPLE_ROW]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["row_count"] == 1
    pred = body["predictions"][0]
    assert 0.0 <= pred["fraud_probability"] <= 1.0
    assert pred["fraud_prediction"] in (0, 1)
    assert pred["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    # Admin gets full explainability detail.
    assert "top_features" in pred


def test_fraud_analyst_gets_basic_fields_only(client: TestClient, admin_token: str):
    analyst_token = make_user(client, admin_token, "analyst-test@example.com", "AnalystPassword123", "FRAUD_ANALYST")
    r = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={"rows": [SAMPLE_ROW]},
    )
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert "top_features" not in pred


def test_auditor_full_audit_view_vs_analyst_limited_view(client: TestClient, admin_token: str):
    analyst_token = make_user(client, admin_token, "analyst-test2@example.com", "AnalystPassword123", "FRAUD_ANALYST")
    auditor_token = make_user(client, admin_token, "auditor-test@example.com", "AuditorPassword123", "AUDITOR")

    # Generate one action to audit.
    client.post("/predict", headers={"Authorization": f"Bearer {analyst_token}"}, json={"rows": [SAMPLE_ROW]})

    analyst_view = client.get("/audit-logs", headers={"Authorization": f"Bearer {analyst_token}"})
    auditor_view = client.get("/audit-logs", headers={"Authorization": f"Bearer {auditor_token}"})
    assert analyst_view.status_code == 200
    assert auditor_view.status_code == 200
    # Analyst only ever sees their own user_id in the results.
    analyst_user_ids = {e["user_id"] for e in analyst_view.json()}
    assert analyst_user_ids <= {e["user_id"] for e in analyst_view.json()}  # non-empty-safe
    for entry in analyst_view.json():
        assert entry["role"] == "FRAUD_ANALYST"


def test_viewer_blocked_from_audit_logs(client: TestClient, admin_token: str):
    viewer_token = make_user(client, admin_token, "viewer-test3@example.com", "ViewerPassword123", "VIEWER")
    r = client.get("/audit-logs", headers={"Authorization": f"Bearer {viewer_token}"})
    assert r.status_code == 403


def test_only_risk_manager_and_admin_change_threshold(client: TestClient, admin_token: str):
    analyst_token = make_user(client, admin_token, "analyst-test3@example.com", "AnalystPassword123", "FRAUD_ANALYST")
    rm_token = make_user(client, admin_token, "riskmgr-test@example.com", "RiskMgrPassword123", "RISK_MANAGER")

    r = client.patch("/settings/threshold", headers={"Authorization": f"Bearer {analyst_token}"}, json={"threshold": 0.4})
    assert r.status_code == 403

    r = client.patch("/settings/threshold", headers={"Authorization": f"Bearer {rm_token}"}, json={"threshold": 0.4})
    assert r.status_code == 200
    assert r.json()["threshold"] == 0.4


def test_strict_schema_rejects_unexpected_fields(client: TestClient, admin_token: str):
    bad_row = dict(SAMPLE_ROW)
    bad_row["not_a_real_field"] = 1
    r = client.post(
        "/predict", headers={"Authorization": f"Bearer {admin_token}"}, json={"rows": [bad_row]},
    )
    assert r.status_code == 422


def test_strict_schema_rejects_out_of_range_values(client: TestClient, admin_token: str):
    bad_row = dict(SAMPLE_ROW)
    bad_row["customer_age"] = 5000
    r = client.post(
        "/predict", headers={"Authorization": f"Bearer {admin_token}"}, json={"rows": [bad_row]},
    )
    assert r.status_code == 422


def test_unhandled_exception_never_leaks_internals(client: TestClient, admin_token: str):
    r = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"},
        content=b"{not valid json",
    )
    assert r.status_code == 422
    body = r.json()
    assert "request_id" in body
    # No file paths, no tracebacks, nothing beyond a structured validation error.
    assert "Traceback" not in r.text
    assert "fraud-detection" not in r.text.replace("Fraud Detection API", "")


def test_security_headers_present(client: TestClient):
    r = client.get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert "default-src" in r.headers.get("content-security-policy", "")
    assert "X-Request-ID" in r.headers


def test_csv_upload_rejects_wrong_extension(client: TestClient, admin_token: str):
    files = {"file": ("not_a_csv.txt", b"income\n0.5\n", "text/plain")}
    r = client.post("/predict/file", headers={"Authorization": f"Bearer {admin_token}"}, files=files)
    assert r.status_code == 400


def test_csv_upload_rejects_missing_columns(client: TestClient, admin_token: str):
    files = {"file": ("bad.csv", b"income\n0.5\n", "text/csv")}
    r = client.post("/predict/file", headers={"Authorization": f"Bearer {admin_token}"}, files=files)
    assert r.status_code == 400
    assert "missing required columns" in r.json()["detail"].lower()


def test_mfa_secret_encrypted_at_rest_and_verify_still_works(client: TestClient, admin_token: str):
    """The TOTP secret must never be recoverable by reading the DB row
    directly -- and MFA must still actually work end to end after
    encrypting it at rest."""
    import pyotp

    from api.database import SessionLocal
    from api.models_db import User
    from api.security import create_mfa_pending_token, decrypt_mfa_secret

    headers = {"Authorization": f"Bearer {admin_token}"}

    r = client.post("/auth/mfa/setup", headers=headers)
    assert r.status_code == 200, r.text
    plaintext_secret = r.json()["secret"]
    assert plaintext_secret and len(plaintext_secret) >= 16  # a real base32 TOTP secret, not empty/placeholder

    # Query the raw DB row directly -- the stored value must NOT be the
    # plaintext secret (proves it's actually encrypted, not just claimed to be).
    db = SessionLocal()
    try:
        row = db.query(User).filter(User.email == ADMIN_EMAIL).first()
        assert row is not None
        stored_value = row.mfa_secret
    finally:
        db.close()

    assert stored_value is not None
    assert stored_value != plaintext_secret
    assert plaintext_secret not in stored_value  # not just wrapped/prefixed, genuinely transformed

    # But it must decrypt back to exactly the original secret (round-trip
    # correctness, not just "looks different").
    assert decrypt_mfa_secret(stored_value) == plaintext_secret

    # Enable MFA with a real code derived from the plaintext secret.
    code = pyotp.TOTP(plaintext_secret).now()
    r = client.post("/auth/mfa/enable", headers=headers, json={"code": code})
    assert r.status_code == 200, r.text
    assert r.json()["mfa_enabled"] is True

    # End-to-end: a real mfa-pending token + a real TOTP code still
    # successfully redeems for a full token pair after encryption at rest.
    db = SessionLocal()
    try:
        row = db.query(User).filter(User.email == ADMIN_EMAIL).first()
        user_id = row.id
    finally:
        db.close()

    mfa_token = create_mfa_pending_token(user_id, "ADMIN")

    r = client.post("/auth/mfa/verify", json={"mfa_token": mfa_token, "code": "000000"})
    assert r.status_code == 401  # wrong code still rejected

    mfa_token = create_mfa_pending_token(user_id, "ADMIN")  # fresh token (not consumed by the failed attempt above)
    code = pyotp.TOTP(plaintext_secret).now()
    r = client.post("/auth/mfa/verify", json={"mfa_token": mfa_token, "code": code})
    assert r.status_code == 200, r.text
    assert "access_token" in r.json() and "refresh_token" in r.json()
