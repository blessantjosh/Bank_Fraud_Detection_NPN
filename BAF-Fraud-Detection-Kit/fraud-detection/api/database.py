"""
api/database.py -- SQLAlchemy engine/session setup.

DATABASE_URL drives everything. Default is SQLite (zero-setup for a
hackathon demo). Swapping to PostgreSQL in production is a one-line env var
change (e.g. postgresql+psycopg://user:pass@host:5432/dbname) -- no code
change required, because every query in this codebase goes through the ORM
(see api/models_db.py, api/repositories.py) rather than raw/string-built SQL.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.settings import settings

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they don't exist. Called at API startup and by
    api/scripts/init_db.py. Never drops or migrates destructively -- for a
    real production Postgres deployment, replace this with Alembic
    migrations (documented in api/README.md)."""
    from api import models_db  # noqa: F401  (ensure models are registered)

    Base.metadata.create_all(bind=engine)
