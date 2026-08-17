"""
train.py -- end-to-end training entry point.

    python train.py [--config config.yaml]

Runs: load+split -> validate -> preprocess (fit on train) -> imbalance
ablation across 4 models x 5 strategies -> select best by PR-AUC -> threshold
tuning on validation -> save model/preprocessor/threshold artifacts to models/.

Final untouched-test evaluation is done separately by evaluate.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, setup_logging
from src.training import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to config.yaml")
    args = ap.parse_args()

    logger = setup_logging(name="fraud_detection")
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config or "config.yaml (default)")

    t0 = time.time()
    result = run_training(cfg)
    elapsed = time.time() - t0

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE in %.1f minutes", elapsed / 60)
    logger.info("=" * 70)
    logger.info("Best model: %s", result["best_key"])
    logger.info("Selected threshold: %.3f", result["selected_threshold"])
    logger.info("Validation %s: %.4f", cfg.evaluation.primary_metric,
                result["results"][result["best_key"]][cfg.evaluation.primary_metric])
    logger.info("Artifacts saved under models/. Run evaluate.py for the final test-set report.")


if __name__ == "__main__":
    main()
