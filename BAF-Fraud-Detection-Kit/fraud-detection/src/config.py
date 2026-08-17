"""
config.py -- load config.yaml into a plain, attribute-accessible object.

Every other module takes a `Config` (or a dict) rather than hardcoding
paths/params, so the whole pipeline is driven from one YAML file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """Dict that also allows attribute access, recursively, for convenience."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-file path relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def setup_logging(cfg: Config | None = None, name: str = "fraud_detection") -> logging.Logger:
    level_name = "INFO"
    log_dir = PROJECT_ROOT / "reports"
    if cfg is not None:
        level_name = cfg.get("logging", {}).get("level", "INFO")
        log_dir = resolve_path(cfg.get("logging", {}).get("log_dir", "reports"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g. re-imported in a notebook)

    logger.setLevel(getattr(logging, level_name, logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
