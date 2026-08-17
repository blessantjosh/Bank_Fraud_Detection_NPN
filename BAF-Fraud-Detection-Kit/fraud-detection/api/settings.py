"""
api/settings.py -- all environment-driven configuration for the API layer,
in one place, using pydantic-settings.

Every secret has a documented, hackathon-pragmatic default for local
dev/demo, and every default is called out as something that MUST be
overridden before any real deployment (see api/SECURITY.md).
"""

from __future__ import annotations

import logging
import secrets

from cryptography.fernet import Fernet
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("fraud_api.settings")

# Generated once per process if JWT_SECRET is not set. This means tokens
# issued before a restart become invalid after one -- an explicit, documented
# hackathon-timeline trade-off (see api/SECURITY.md "JWT_SECRET"). Production
# MUST set a persistent JWT_SECRET.
_EPHEMERAL_JWT_SECRET = secrets.token_hex(32)

# Generated once per process if MFA_ENCRYPTION_KEY is not set (same pattern
# as _EPHEMERAL_JWT_SECRET above). This means MFA secrets encrypted before a
# restart become undecryptable after one -- acceptable for a single-process
# demo, not for production. See api/SECURITY.md "MFA secret at rest".
_EPHEMERAL_MFA_ENCRYPTION_KEY = Fernet.generate_key().decode()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database -----------------------------------------------------
    # SQLite is the zero-setup local-dev/demo default. Swapping to Postgres
    # in production is a ONE-LINE change to this URL -- see api/README.md
    # "Database" section for the exact reasoning and the swap instructions.
    database_url: str = "sqlite:///./api/fraud_api.db"

    # --- auth / JWT -----------------------------------------------------
    jwt_secret: str = ""  # populated in model_post_init if unset
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # MFA is a real, working TOTP feature (pyotp) but OFF by default locally
    # so a hackathon demo isn't locked out without a real authenticator app
    # enrolled. Production should set MFA_REQUIRED=true for ADMIN accounts.
    # See api/SECURITY.md "MFA_REQUIRED".
    mfa_required: bool = False
    mfa_issuer: str = "BAF-Fraud-Detection-Kit"

    # Symmetric key (Fernet, cryptography library -- free, local, no KMS)
    # used to encrypt each user's TOTP secret before it is written to the
    # DB, and to decrypt it (in memory only, never logged) to verify a
    # code. Populated in effective_mfa_encryption_key() if unset.
    mfa_encryption_key: str = ""

    # --- rate limiting ---------------------------------------------------
    login_rate_limit: str = "5/minute"
    predict_rate_limit: str = "30/minute"

    # --- CORS -------------------------------------------------------------
    # Never "*". Comma-separated list of exact origins allowed to call this
    # API from a browser.
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- uploads -----------------------------------------------------------
    max_upload_bytes: int = 5 * 1024 * 1024  # 5 MB
    max_upload_rows: int = 50_000

    # --- model integrity --------------------------------------------------
    models_dir: str = "models"
    model_checksum_file: str = "models/model_checksum.json"
    # If true, refuse to serve /predict when no recorded checksum exists or
    # it doesn't match. Off by default in local dev (no trained model may
    # exist yet in a hackathon-timeline parallel build); ON is the intended
    # production posture. See api/SECURITY.md "Model integrity".
    enforce_model_checksum: bool = False

    # --- bootstrap admin (first-run convenience only) ----------------------
    # If set, api/scripts/init_db.py will create this admin account on first
    # run. Never hardcode a password in code -- this env var is the only
    # source, and it is required (no default) so a demo can't silently ship
    # with a guessable admin password.
    bootstrap_admin_email: str = ""
    bootstrap_admin_password: str = ""

    def effective_jwt_secret(self) -> str:
        if self.jwt_secret:
            return self.jwt_secret
        logger.warning(
            "JWT_SECRET is not set. Using a RANDOM, PROCESS-LOCAL secret "
            "generated at startup. All tokens will become invalid on "
            "restart, and multiple API instances will NOT be able to "
            "validate each other's tokens. This is acceptable for a "
            "single-process hackathon demo ONLY. Set a persistent JWT_SECRET "
            "before any shared or production deployment."
        )
        return _EPHEMERAL_JWT_SECRET

    def effective_mfa_encryption_key(self) -> str:
        if self.mfa_encryption_key:
            return self.mfa_encryption_key
        logger.warning(
            "MFA_ENCRYPTION_KEY is not set. Using a RANDOM, PROCESS-LOCAL "
            "Fernet key generated at startup to encrypt TOTP secrets at "
            "rest. Every enrolled MFA secret will become UNDECRYPTABLE "
            "(and those admin accounts will be unable to complete MFA "
            "login) after a restart. This is acceptable for a "
            "single-process hackathon demo ONLY. Set a persistent "
            "MFA_ENCRYPTION_KEY (generate one with "
            "`python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"`) before any shared "
            "or production deployment."
        )
        return _EPHEMERAL_MFA_ENCRYPTION_KEY

    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
