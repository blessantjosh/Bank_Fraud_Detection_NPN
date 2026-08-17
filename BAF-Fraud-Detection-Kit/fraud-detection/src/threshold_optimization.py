"""
threshold_optimization.py -- sweep decision thresholds and pick one.

At a ~1.1% base rate the useful thresholds are almost all below 0.5, so the
sweep is deliberately finer at the low end (0.01/0.02/0.03/0.05...) instead
of uniform 0.1 steps, which would miss the entire operating region a real
fraud team lives in.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.evaluation import confusion_at_threshold, cost_sensitive_eval

logger = logging.getLogger("fraud_detection.threshold_optimization")


def sweep_thresholds(y_true, y_score, thresholds: list[float]) -> pd.DataFrame:
    rows = [confusion_at_threshold(y_true, y_score, t) for t in thresholds]
    return pd.DataFrame(rows)


def best_threshold_by_f1(sweep_df: pd.DataFrame) -> float:
    return float(sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"])


def sweep_cost_sensitive(y_true, y_score, thresholds: list[float], cost_fp: float, cost_fn: float) -> pd.DataFrame:
    rows = [cost_sensitive_eval(y_true, y_score, t, cost_fp, cost_fn) for t in thresholds]
    return pd.DataFrame(rows)


def best_threshold_by_cost(sweep_df: pd.DataFrame) -> float:
    return float(sweep_df.loc[sweep_df["total_cost"].idxmin(), "threshold"])


def optimize(y_true, y_score, thresholds: list[float], cost_fp: float, cost_fn: float) -> dict:
    """Run both sweeps and return the recommended thresholds + full tables."""
    f1_sweep = sweep_thresholds(y_true, y_score, thresholds)
    cost_sweep = sweep_cost_sensitive(y_true, y_score, thresholds, cost_fp, cost_fn)
    best_f1_thr = best_threshold_by_f1(f1_sweep)
    best_cost_thr = best_threshold_by_cost(cost_sweep)
    logger.info("Best threshold by F1: %.3f", best_f1_thr)
    logger.info("Best threshold by expected cost (fp=%s, fn=%s): %.3f", cost_fp, cost_fn, best_cost_thr)
    return {
        "f1_sweep": f1_sweep,
        "cost_sweep": cost_sweep,
        "best_threshold_f1": best_f1_thr,
        "best_threshold_cost": best_cost_thr,
    }
