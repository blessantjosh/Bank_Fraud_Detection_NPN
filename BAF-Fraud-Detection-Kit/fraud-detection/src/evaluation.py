"""
evaluation.py -- the metrics that actually matter at a ~1.1% base rate.

PR-AUC is the primary model-selection metric (config.evaluation.primary_metric),
not accuracy: a model predicting "never fraud" scores ~98.9% accuracy on this
data and is useless. ROC-AUC and TPR@5%FPR (the BAF paper's own domain metric,
chosen because every false positive is a real customer wrongly rejected) are
reported alongside it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("fraud_detection.evaluation")


def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(target_fpr, fpr, tpr))


def threshold_at_fpr(y_true, y_score, target_fpr: float = 0.05) -> float:
    fpr, _, thr = roc_curve(y_true, y_score)
    idx = int(np.searchsorted(fpr, target_fpr, side="right")) - 1
    idx = max(0, min(idx, len(thr) - 1))
    return float(thr[idx])


def evaluate_scores(y_true, y_score, target_fpr: float = 0.05, label: str = "") -> dict:
    y_true = np.asarray(y_true)
    res = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        f"tpr_at_{int(target_fpr * 100)}pct_fpr": tpr_at_fpr(y_true, y_score, target_fpr),
        "positive_rate": float(y_true.mean()),
        "accuracy_if_predict_all_0": float(1 - y_true.mean()),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    if label:
        logger.info("--- %s ---", label)
        for k, v in res.items():
            logger.info("  %-28s %s", k, f"{v:.4f}" if isinstance(v, float) else v)
    return res


def confusion_at_threshold(y_true, y_score, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(fpr), "fnr": float(fnr),
    }


def cost_sensitive_eval(y_true, y_score, threshold: float, cost_fp: float, cost_fn: float) -> dict:
    """Expected business cost at a given threshold: FP costs analyst/friction, FN costs unrecovered fraud."""
    cm = confusion_at_threshold(y_true, y_score, threshold)
    total_cost = cm["fp"] * cost_fp + cm["fn"] * cost_fn
    cm["cost_fp"] = cost_fp
    cm["cost_fn"] = cost_fn
    cm["total_cost"] = float(total_cost)
    cm["cost_per_application"] = float(total_cost / max(cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"], 1))
    return cm


def fairness_report(y_true, y_score, group, target_fpr: float = 0.05, label: str = "") -> dict:
    """
    Predictive equality across a protected group (customer_age > 50, per the
    BAF paper). A single global threshold is chosen to hit `target_fpr`
    overall, then each group's FPR is measured AT THAT SAME threshold --
    comparing groups at different thresholds is the most common way this
    metric gets misreported. FPR ratio (min/max) reported so 1.0 = parity.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    group = np.asarray(group)

    thr = threshold_at_fpr(y_true, y_score, target_fpr)
    pred = (y_score >= thr).astype(int)

    out = {}
    for g in np.unique(group):
        m = group == g
        neg, pos = m & (y_true == 0), m & (y_true == 1)
        out[f"group_{g}"] = {
            "n": int(m.sum()),
            "fpr": float(pred[neg].mean()) if neg.sum() else float("nan"),
            "tpr": float(pred[pos].mean()) if pos.sum() else float("nan"),
            "prevalence": float(y_true[m].mean()) if m.sum() else float("nan"),
        }

    fprs = [v["fpr"] for v in out.values() if not np.isnan(v["fpr"])]
    ratio = (min(fprs) / max(fprs)) if fprs and max(fprs) > 0 else float("nan")
    out["fpr_ratio"] = ratio
    out["threshold"] = thr

    if label:
        logger.info("--- Fairness: %s (global FPR target=%.0f%%) ---", label, target_fpr * 100)
        for k, v in out.items():
            if isinstance(v, dict):
                logger.info("  %s: n=%d FPR=%.4f TPR=%.4f prevalence=%.4f",
                            k, v["n"], v["fpr"], v["tpr"], v["prevalence"])
        logger.info("  predictive equality (FPR ratio, 1.0=parity): %.3f", ratio)
    return out
