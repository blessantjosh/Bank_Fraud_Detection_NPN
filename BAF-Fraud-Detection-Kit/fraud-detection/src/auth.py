"""
auth.py -- a simple, CLI-appropriate admin gate.

This is an APPLICATION-LEVEL access control suitable for a trusted internal
CLI tool. It is NOT a substitute for a bank's real IAM / network security if
this pipeline is ever deployed as a service -- see README.md "Security note"
for the explicit caveat.

How it works:
  - The real credential is never stored anywhere in this repo. It is read
    once, at call time, from the FRAUD_ADMIN_TOKEN environment variable.
  - config.yaml stores only a SALTED HASH of the expected token
    (auth.admin_salt_hex / auth.admin_token_hash_hex), generated ahead of
    time with `python -m src.auth --generate <token>`. No default or
    backdoor credential is baked in: if config has no hash configured,
    every call is rejected.
  - require_admin() hashes the env var's value (PBKDF2-HMAC-SHA256, salted)
    and compares it to the stored hash with hmac.compare_digest, a
    constant-time comparison, so this is a real check, not `==`.
  - Every attempt (success or failure) is logged via the standard logging
    module. The credential itself is NEVER logged.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
import secrets

logger = logging.getLogger("fraud_detection.auth")

ADMIN_TOKEN_ENV_VAR = "FRAUD_ADMIN_TOKEN"
_PBKDF2_ITERATIONS = 200_000


class AdminAuthError(RuntimeError):
    """Raised when admin authentication is missing, misconfigured, or fails."""


def hash_token(token: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", token.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return digest.hex()


def generate_admin_credential(token: str) -> tuple[str, str]:
    """Return (salt_hex, hash_hex) to paste into config.yaml's `auth:` section."""
    salt_hex = secrets.token_hex(16)
    return salt_hex, hash_token(token, salt_hex)


def require_admin(cfg) -> None:
    """
    Raise AdminAuthError unless FRAUD_ADMIN_TOKEN is set in the environment
    and matches the salted hash configured in config.yaml. Call this at the
    top of any entry point that acts on real applicant data (predict.py) or
    that loads the saved model / writes to data/predictions/ on its behalf.
    """
    auth_cfg = cfg.get("auth") or {} if hasattr(cfg, "get") else {}
    salt_hex = auth_cfg.get("admin_salt_hex")
    expected_hash = auth_cfg.get("admin_token_hash_hex")

    if not salt_hex or not expected_hash:
        logger.error("Admin auth attempt REJECTED: no admin credential configured in config.yaml")
        raise AdminAuthError(
            "No admin credential configured. Run "
            "`python -m src.auth --generate <token>` and paste the printed "
            "salt/hash into config.yaml's `auth:` section, then set the "
            f"{ADMIN_TOKEN_ENV_VAR} environment variable to <token> before retrying."
        )

    token = os.environ.get(ADMIN_TOKEN_ENV_VAR)
    if not token:
        logger.error("Admin auth attempt REJECTED: %s is not set", ADMIN_TOKEN_ENV_VAR)
        raise AdminAuthError(
            f"This action is restricted to admins. Set the {ADMIN_TOKEN_ENV_VAR} "
            "environment variable and retry."
        )

    candidate_hash = hash_token(token, salt_hex)
    if not hmac.compare_digest(candidate_hash, expected_hash):
        logger.error("Admin auth attempt REJECTED: invalid %s", ADMIN_TOKEN_ENV_VAR)
        raise AdminAuthError("Invalid admin token.")

    logger.info("Admin auth attempt ACCEPTED")


def admin_identity() -> str:
    """
    A short, non-reversible identifier for the CURRENT admin token, safe to
    write into logs/audit trails. Never returns or logs the token itself.
    Call only after require_admin() has already succeeded.
    """
    token = os.environ.get(ADMIN_TOKEN_ENV_VAR, "")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _cli():
    ap = argparse.ArgumentParser(description="Generate a salted admin-token hash for config.yaml")
    ap.add_argument("--generate", metavar="TOKEN", required=True,
                    help="the admin token to hash (choose a long random secret; "
                         "this value is never stored, only its salted hash is)")
    args = ap.parse_args()
    salt_hex, hash_hex = generate_admin_credential(args.generate)
    print("Paste this into config.yaml under `auth:`:\n")
    print("auth:")
    print(f'  admin_salt_hex: "{salt_hex}"')
    print(f'  admin_token_hash_hex: "{hash_hex}"')
    print("\nThen, before running predict.py, set the environment variable "
          f"(NOT committed anywhere) to the token itself:")
    print(f"  bash/zsh:   export {ADMIN_TOKEN_ENV_VAR}=<token>")
    print(f'  PowerShell: $env:{ADMIN_TOKEN_ENV_VAR}="<token>"')


if __name__ == "__main__":
    _cli()
