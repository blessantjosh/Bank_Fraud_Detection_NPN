"""
api/scripts/init_db.py -- create tables, and (idempotently) create the
bootstrap ADMIN account from BOOTSTRAP_ADMIN_EMAIL / BOOTSTRAP_ADMIN_PASSWORD
if both are set and no user with that email exists yet.

Run from fraud-detection/:
    python -m api.scripts.init_db
"""

from __future__ import annotations

import logging
import sys

from api.database import SessionLocal, init_db
from api.models_db import Role, User
from api.security import hash_password
from api.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("fraud_api.init_db")


def main() -> None:
    init_db()
    logger.info("Database tables ensured at %s", settings.database_url)

    if not settings.bootstrap_admin_email or not settings.bootstrap_admin_password:
        logger.info(
            "BOOTSTRAP_ADMIN_EMAIL / BOOTSTRAP_ADMIN_PASSWORD not both set -- "
            "skipping bootstrap admin creation. Set both in your .env to "
            "create a first admin account, or use /admin/users once you "
            "have at least one admin."
        )
        return

    if len(settings.bootstrap_admin_password) < 10:
        logger.error("BOOTSTRAP_ADMIN_PASSWORD is too short (min 10 chars). Refusing to create the account.")
        sys.exit(1)

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == settings.bootstrap_admin_email.lower()).first()
        if existing is not None:
            logger.info("Bootstrap admin %s already exists -- no action taken.", settings.bootstrap_admin_email)
            return

        user = User(
            email=settings.bootstrap_admin_email.lower(),
            hashed_password=hash_password(settings.bootstrap_admin_password),
            role=Role.ADMIN,
        )
        db.add(user)
        db.commit()
        logger.info("Created bootstrap ADMIN account: %s", settings.bootstrap_admin_email)
    finally:
        db.close()


if __name__ == "__main__":
    main()
