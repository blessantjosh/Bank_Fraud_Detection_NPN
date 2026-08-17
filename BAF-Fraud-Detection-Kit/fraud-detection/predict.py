"""
predict.py -- score a new CSV with the saved model.

    python predict.py --input new_applications.csv --output data/predictions/scored.csv

Output columns added: fraud_probability, fraud_prediction, risk_level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, setup_logging
from src.prediction import predict_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to a CSV of new applications")
    ap.add_argument("--output", default="data/predictions/scored.csv")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logger = setup_logging(name="fraud_detection")
    cfg = load_config(args.config)

    out = predict_csv(args.input, args.output, cfg)
    print(f"\nScored {len(out):,} rows -> {args.output}")
    print(out[["fraud_probability", "fraud_prediction", "risk_level"]].head(10).to_string())


if __name__ == "__main__":
    main()
