"""
models.py -- model factories/trainers for the four families compared here.

Logistic Regression : baseline, class_weight="balanced" (config-driven),
                       scaled + one-hot encoded dense features.
Random Forest        : n_estimators=300, class_weight="balanced", n_jobs=-1,
                       dense features (sklearn RF cannot consume NaN or
                       pandas `category` dtype).
LightGBM             : reuses the proven starting params from baf.py /
                       run_pipeline.py (min_data_in_leaf=200,
                       min_sum_hessian_in_leaf=1.0 matter a lot at 1.1% base
                       rate). Native NaN + categorical handling.
XGBoost              : enable_categorical=True, tree_method="hist".
                       min_child_weight is a SUM OF HESSIANS, not a row
                       count -- at ~1% base rate a value like 200 silently
                       produces a 0-split stump with AUC == 0.5. We use 5
                       (config-driven) and verify best_iteration > 0 and
                       non-constant predictions before trusting any result.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import Config

logger = logging.getLogger("fraud_detection.models")


def train_logistic_regression(X_tr, y_tr, cfg: Config, model_kwargs: dict, seed: int = 42):
    kwargs = dict(
        max_iter=cfg.models.logistic_regression.max_iter,
        solver=cfg.models.logistic_regression.solver,
        random_state=seed,
    )
    if model_kwargs.get("class_weight") == "balanced" and cfg.models.logistic_regression.class_weight_balanced:
        kwargs["class_weight"] = "balanced"
    model = LogisticRegression(**kwargs)
    model.fit(X_tr, y_tr)
    return model


def train_random_forest(X_tr, y_tr, cfg: Config, model_kwargs: dict, seed: int = 42):
    kwargs = dict(
        n_estimators=cfg.models.random_forest.n_estimators,
        max_depth=cfg.models.random_forest.max_depth,
        max_samples=cfg.models.random_forest.get("max_samples"),
        n_jobs=cfg.models.random_forest.n_jobs,
        random_state=seed,
    )
    if model_kwargs.get("class_weight") == "balanced" and cfg.models.random_forest.class_weight_balanced:
        kwargs["class_weight"] = "balanced"
    model = RandomForestClassifier(**kwargs)
    model.fit(X_tr, y_tr)
    return model


def _lgbm_params(cfg: Config, model_kwargs: dict, seed: int) -> dict:
    m = cfg.models.lightgbm
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": m.learning_rate,
        "num_leaves": m.num_leaves,
        "min_data_in_leaf": m.min_data_in_leaf,
        "min_sum_hessian_in_leaf": m.min_sum_hessian_in_leaf,
        "max_cat_to_onehot": m.max_cat_to_onehot,
        "feature_fraction": m.feature_fraction,
        "bagging_fraction": m.bagging_fraction,
        "bagging_freq": m.bagging_freq,
        "cat_smooth": m.cat_smooth,
        "cat_l2": m.cat_l2,
        "min_data_per_group": m.min_data_per_group,
        "lambda_l1": m.lambda_l1,
        "lambda_l2": m.lambda_l2,
        "n_jobs": -1,
        "verbose": -1,
        "seed": seed,
    }
    if "scale_pos_weight" in model_kwargs:
        params["scale_pos_weight"] = model_kwargs["scale_pos_weight"]
    return params


def train_lightgbm(X_tr, y_tr, X_va, y_va, cfg: Config, model_kwargs: dict, seed: int = 42):
    params = _lgbm_params(cfg, model_kwargs, seed)
    dtr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, y_va, reference=dtr, free_raw_data=False)
    model = lgb.train(
        params, dtr,
        num_boost_round=cfg.models.lightgbm.num_boost_round,
        valid_sets=[dva],
        callbacks=[
            lgb.early_stopping(cfg.models.lightgbm.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    if model.best_iteration <= 0:
        raise RuntimeError("LightGBM: best_iteration <= 0 -- degenerate model, refusing to trust it")
    return model


def _xgb_params(cfg: Config, model_kwargs: dict, seed: int) -> dict:
    m = cfg.models.xgboost
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": m.learning_rate,
        "max_depth": m.max_depth,
        "min_child_weight": m.min_child_weight,
        "subsample": m.subsample,
        "colsample_bytree": m.colsample_bytree,
        "reg_alpha": m.reg_alpha,
        "reg_lambda": m.reg_lambda,
        "tree_method": "hist",
        "seed": seed,
        "nthread": -1,
    }
    if "scale_pos_weight" in model_kwargs:
        params["scale_pos_weight"] = model_kwargs["scale_pos_weight"]
    return params


def train_xgboost(X_tr, y_tr, X_va, y_va, cfg: Config, model_kwargs: dict, seed: int = 42):
    params = _xgb_params(cfg, model_kwargs, seed)
    enable_cat = any(str(dt) == "category" for dt in X_tr.dtypes)
    dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=enable_cat)
    dva = xgb.DMatrix(X_va, label=y_va, enable_categorical=enable_cat)
    booster = xgb.train(
        params, dtr,
        num_boost_round=cfg.models.xgboost.num_boost_round,
        evals=[(dva, "valid")],
        early_stopping_rounds=cfg.models.xgboost.early_stopping_rounds,
        verbose_eval=False,
    )
    if booster.best_iteration is None or booster.best_iteration <= 0:
        raise RuntimeError("XGBoost: best_iteration <= 0 -- degenerate model (check min_child_weight)")

    preds = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    if np.std(preds) < 1e-9:
        raise RuntimeError(
            "XGBoost: validation predictions are constant -- degenerate model "
            "(classic min_child_weight-too-large footgun). Refusing to trust it."
        )
    return booster


def predict_proba(model, X, model_type: str) -> np.ndarray:
    """Uniform prediction interface across the four model families."""
    if model_type == "logistic_regression":
        return model.predict_proba(X)[:, 1]
    if model_type == "random_forest":
        return model.predict_proba(X)[:, 1]
    if model_type == "lightgbm":
        return model.predict(X, num_iteration=model.best_iteration)
    if model_type == "xgboost":
        enable_cat = any(str(dt) == "category" for dt in X.dtypes)
        d = xgb.DMatrix(X, enable_categorical=enable_cat)
        return model.predict(d, iteration_range=(0, model.best_iteration + 1))
    raise ValueError(f"Unknown model_type: {model_type}")
