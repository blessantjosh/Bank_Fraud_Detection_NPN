"""
STAGE 4b -- Robust cross-validated evaluation, BEFORE any fine-tuning.

Section 11 of the leakage-fix brief requires that IF hyperparameter tuning
is ever performed, it must be validated via cross-validation / a validation
fold -- never by looking at the test set:

    TRAIN -> cross-validation / validation -> select model -> FINAL TEST ONCE

This stage builds that robust evaluation harness FIRST, before any tuning
is attempted, using each model's current baseline hyperparameters (the same
ones trained in Stage 5). Two things this buys:

  1. An honest, variance-aware estimate (mean +/- std across 5 folds) of how
     stable each baseline configuration's precision/recall/F1/ROC-AUC/PR-AUC
     really is -- Stage 6's single train->val or train->test numbers are one
     draw; this shows the spread around them.
  2. A ready-made harness for future hyperparameter search: a grid/Optuna
     loop can be dropped in here, scored the same way, entirely inside the
     training fold -- so tuning never touches validation (used for model/
     threshold selection) or test (used exactly once, at the end).

LEAKAGE DISCIPLINE (same rules as every other stage): this script reads
ONLY the training fold from split.pkl. Validation and test are never
loaded here. Each fold's SMOTE resampling is fit on that fold's OWN
training portion and applied only to it -- never to the held-out CV fold --
for the same reason Stage 4 only ever fits SMOTE on the outer training set.
"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import config

split = joblib.load(config.SPLIT_PKL)
X_train, y_train = split["X_train"].reset_index(drop=True), split["y_train"].reset_index(drop=True)

METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def eval_fold(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_va)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_va, pred, zero_division=0),
        "recall": recall_score(y_va, pred, zero_division=0),
        "f1": f1_score(y_va, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_va, proba) if y_va.nunique() > 1 else float("nan"),
        "pr_auc": average_precision_score(y_va, proba),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def fresh_models():
    """Same baseline hyperparameters as Stage 5 -- no tuning applied yet."""
    return {
        "XGBoost + SMOTE": ("smote", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=config.RANDOM_STATE)),
        "XGBoost + Class Weighting": ("classweight", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=config.RANDOM_STATE)),
        "Random Forest (class_weight=balanced)": ("plain", RandomForestClassifier(
            n_estimators=200, max_depth=4, class_weight="balanced",
            random_state=config.RANDOM_STATE)),
    }


skf = StratifiedKFold(n_splits=config.CV_N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
fold_records = {name: [] for name in fresh_models()}

print(f"Running {config.CV_N_SPLITS}-fold stratified cross-validation on the "
      f"{len(X_train)}-row TRAINING FOLD ONLY ({int(y_train.sum())} fraud-proxy rows). "
      f"Validation and test are not read by this script.")

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    for name, (mode, model) in fresh_models().items():
        if mode == "smote":
            minority = y_tr.sum()
            k = min(5, max(int(minority) - 1, 1))
            X_tr_fit, y_tr_fit = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k).fit_resample(X_tr, y_tr)
        elif mode == "classweight":
            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            model.set_params(scale_pos_weight=spw)
            X_tr_fit, y_tr_fit = X_tr, y_tr
        else:
            X_tr_fit, y_tr_fit = X_tr, y_tr

        metrics = eval_fold(model, X_tr_fit, y_tr_fit, X_va, y_va)
        metrics["fold"] = fold_idx
        metrics["model"] = name
        fold_records[name].append(metrics)

    print(f"  fold {fold_idx}/{config.CV_N_SPLITS}: {len(tr_idx)} cv-train / {len(va_idx)} cv-val rows "
          f"({int(y_va.sum())} fraud-proxy in cv-val)")

per_fold_df = pd.concat([pd.DataFrame(v) for v in fold_records.values()], ignore_index=True)

summary_rows = []
for name, records in fold_records.items():
    df = pd.DataFrame(records)
    row = {"model": name}
    for m in METRIC_COLS:
        row[f"{m}_mean"] = df[m].mean()
        row[f"{m}_std"] = df[m].std()
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 100)
print(f"ROBUST {config.CV_N_SPLITS}-FOLD CV SUMMARY -- baseline hyperparameters, train fold only, BEFORE fine-tuning")
print("=" * 100)
print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("=" * 100)
print("\nUse this as the pre-tuning baseline: any future hyperparameter search must be scored inside this same "
      "train-fold-only CV harness, never against the validation fold (used for model/threshold selection in "
      "Stage 6) or the test fold (used exactly once, at the end).")

per_fold_df.to_csv(config.CV_PER_FOLD_CSV, index=False)
summary_df.to_csv(config.CV_SUMMARY_CSV, index=False)
with open(config.CV_SUMMARY_JSON, "w") as f:
    json.dump(summary_df.to_dict(orient="records"), f, indent=2)

print(f"\nSaved {config.CV_PER_FOLD_CSV}, {config.CV_SUMMARY_CSV}, {config.CV_SUMMARY_JSON}")
