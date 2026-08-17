import pytest

from src.auth import ADMIN_TOKEN_ENV_VAR, AdminAuthError, generate_admin_credential, require_admin


@pytest.fixture()
def auth_cfg():
    salt_hex, hash_hex = generate_admin_credential("correct-horse-battery-staple")
    return {"auth": {"admin_salt_hex": salt_hex, "admin_token_hash_hex": hash_hex}}


def test_require_admin_rejected_when_no_token_set(auth_cfg, monkeypatch):
    monkeypatch.delenv(ADMIN_TOKEN_ENV_VAR, raising=False)
    with pytest.raises(AdminAuthError):
        require_admin(auth_cfg)


def test_require_admin_rejected_with_wrong_token(auth_cfg, monkeypatch):
    monkeypatch.setenv(ADMIN_TOKEN_ENV_VAR, "wrong-token")
    with pytest.raises(AdminAuthError):
        require_admin(auth_cfg)


def test_require_admin_accepted_with_correct_token(auth_cfg, monkeypatch):
    monkeypatch.setenv(ADMIN_TOKEN_ENV_VAR, "correct-horse-battery-staple")
    require_admin(auth_cfg)  # should not raise


def test_require_admin_rejected_when_no_credential_configured(monkeypatch):
    monkeypatch.setenv(ADMIN_TOKEN_ENV_VAR, "anything")
    with pytest.raises(AdminAuthError):
        require_admin({"auth": {"admin_salt_hex": None, "admin_token_hash_hex": None}})
