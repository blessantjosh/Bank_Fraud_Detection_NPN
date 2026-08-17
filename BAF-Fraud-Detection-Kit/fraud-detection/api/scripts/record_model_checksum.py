"""
api/scripts/record_model_checksum.py -- compute and record the SHA-256
checksums of the trained model artifacts, so api/model_service.py can
verify on load that the files being served are exactly the ones that were
approved/recorded, not a swapped or corrupted file.

Expected format written to fraud-detection/models/model_checksum.json:

    {
      "final_model.joblib": "<sha256 hex>",
      "preprocessor.joblib": "<sha256 hex>",
      "recorded_at_utc": "<iso8601>",
      "model_type": "...",       # copied from model_meta.json for convenience
      "strategy": "..."
    }

Run this once after every training run that produces a new
models/final_model.joblib + models/preprocessor.joblib (whether that
training run comes from src/training.py or from
api/scripts/train_demo_model.py). Neither of those training entry points
writes this file itself -- ownership of "is this the approved model" is
deliberately a separate, explicit step, not implicit in training.

Run from fraud-detection/:
    python -m api.scripts.record_model_checksum
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("api.record_model_checksum")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    models_dir = resolve_path("models")

    model_path = models_dir / "final_model.joblib"
    preprocessor_path = models_dir / "preprocessor.joblib"
    meta_path = models_dir / "model_meta.json"

    for p in (model_path, preprocessor_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} does not exist yet. Train a model first (src/training.py's "
                "run_training(), or api/scripts/train_demo_model.py for a fast interim model)."
            )

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    record = {
        "final_model.joblib": _sha256_file(model_path),
        "preprocessor.joblib": _sha256_file(preprocessor_path),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": meta.get("model_type"),
        "strategy": meta.get("strategy"),
    }

    out_path = resolve_path("models/model_checksum.json")
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Recorded model checksum to %s", out_path)
    logger.info("final_model.joblib   sha256=%s", record["final_model.joblib"])
    logger.info("preprocessor.joblib  sha256=%s", record["preprocessor.joblib"])


if __name__ == "__main__":
    main()
