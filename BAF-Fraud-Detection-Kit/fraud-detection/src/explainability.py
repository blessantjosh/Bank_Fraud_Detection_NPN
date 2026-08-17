"""
explainability.py -- gain-based feature importance and SHAP explanations.

SHAP is run against the selected final model (LightGBM, per training.py's
model-selection result) using shap.TreeExplainer, which is exact and fast
for tree ensembles -- no KernelExplainer approximation needed here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("fraud_detection.explainability")


def lightgbm_feature_importance(model, top_n: int = 20) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    imp["share"] = imp["gain"] / imp["gain"].sum()
    return imp.head(top_n)


def xgboost_feature_importance(model, top_n: int = 20) -> pd.DataFrame:
    score = model.get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": list(score.keys()), "gain": list(score.values())})
    imp = imp.sort_values("gain", ascending=False)
    imp["share"] = imp["gain"] / imp["gain"].sum()
    return imp.head(top_n)


def shap_summary(model, X_sample: pd.DataFrame, model_type: str):
    """
    Returns (explainer, shap_values) for a tree model. Caller is responsible
    for plotting (see notebooks/04_model_evaluation.ipynb and evaluate.py).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values


def explain_single_prediction(explainer, X_row: pd.DataFrame):
    """SHAP values for exactly one row -- used for the "one real fraud-flagged
    row" individual explanation required by the spec."""
    return explainer(X_row)


def top_shap_features_for_row(shap_values_row, feature_names, top_n: int = 10) -> pd.DataFrame:
    vals = np.asarray(shap_values_row.values).ravel()
    df = pd.DataFrame({"feature": feature_names, "shap_value": vals})
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")
