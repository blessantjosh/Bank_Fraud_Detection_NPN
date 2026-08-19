# Fraud Detection Pipeline — Technical Documentation

Dataset: `bank_transactions_data_2.csv` — 2,512 transactions, 495 accounts, 16 raw columns, no fraud label.
All numbers below are from an actual run of the leakage-fixed pipeline in `src/` against this exact file, not
illustrative estimates. See `ML_AUDIT_AFTER_FIX.md` for the full leakage audit and exactly what changed.

---

## Stage 1 — Leakage-Safe Feature Engineering

**Dropped:** `PreviousTransactionDate` (7 unique values, all clustered within minutes of one export timestamp on
2024-11-04 — a data-export artifact, not a real per-account "previous transaction" time). `AccountID` (used only as
a grouping key during engineering, excluded from the model matrix). `DeviceID`, `IP Address`, `MerchantID`
(high-cardinality raw identifiers, converted into count features instead of encoded directly). `TransactionID` is
kept as a non-feature row-identity key (used to re-join a scored row back to its raw record; explicitly excluded
from the model matrix in every training/evaluation stage).

**20 engineered features** (after encoding):

| Feature | Definition |
|---|---|
| `TransactionAmount`, `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` | raw numeric columns |
| `Amount_vs_AccountAvg` | `(amount - prior_account_mean) / prior_account_mean`, where `prior_account_mean` is that account's **expanding mean of all prior transactions only** (computed via `groupby().shift().expanding()` — always causal, safe regardless of where a train/val/test split boundary falls). First-ever transaction for an account falls back to the (train-fit) type average. |
| `Amount_vs_TypeAvg` | deviation from the mean amount for that `TransactionType` (Debit/Credit), fit on the **training fold only** and mapped onto val/test |
| `DeviceNoveltyFlag` / `LocationNoveltyFlag` | 1 if this Device/Location has never appeared before in this account's *prior* history (first-ever transaction counts as novel = 1); always looks backward within the account, never at future rows |
| `TimeSinceLastTxn` | hours since this account's previous transaction, from `TransactionDate` itself; first transaction filled with the **training-fold** median gap |
| `DeviceTxnCount`, `IPTxnCount`, `MerchantTxnCount` | how many transactions that Device/IP/Merchant had **in the training fold**, mapped onto val/test (unseen identifiers → 0) |
| `Location_enc` | label-encoded (cardinality > 10), encoder fit on training fold only |
| `TransactionType_Debit`, `Channel_Branch`, `Channel_Online`, `CustomerOccupation_Engineer/Retired/Student` | one-hot, first category per group dropped as reference level; categories fixed from the training fold |

Two matrices are saved, both with a `split` column (`train`/`val`/`test`): unscaled (`features.csv`, used by the
tree-based models) and `StandardScaler`-scaled (`features_scaled.csv`, used by the distance-based detectors in
Stage 2). **The scaler and every lookup/encoder above are fit on the training fold only** — see the leakage audit.

### Chronological train / validation / test split

Split by `TransactionDate` (not randomly): train = earliest ~64% (1,608 rows, through 2023‑08‑28), val = next ~16%
(401 rows, through 2023‑10‑23), test = latest ~20% (503 rows). This mirrors the real deployment scenario — a model
trained on past transactions, evaluated on future ones — and is the methodologically correct choice for a
time-ordered transaction/fraud problem (see `ML_AUDIT_AFTER_FIX.md` §4 for the full justification).

---

## Stage 2 — Unsupervised Anomaly Ensemble (fit on TRAIN only)

No label exists, so four independent detectors vote on which transactions look statistically unusual.
Contamination assumption: **5%** (documented, unverified — applied identically to all four so votes are
comparable). **All four are fit exclusively on the 1,608 training-fold rows**, then used to *predict*
(out-of-sample) on val and test — none of them ever sees a val/test row during fitting. LOF uses scikit-learn's
`novelty=True` mode, the documented way to get an out-of-sample `.predict()` from LOF.

| Detector | Principle | Train anomaly rate | Val anomaly rate | Test anomaly rate |
|---|---|---|---|---|
| Isolation Forest | isolates points via random tree partitioning | 5.04% | 12.47% | 9.54% |
| Local Outlier Factor | local density much lower than neighbors | 4.42% | 9.73% | 12.33% |
| One-Class SVM | outside the learned soft boundary around normal data | 5.41% | 21.20% | 23.06% |
| MCD (EllipticEnvelope) | robust Mahalanobis distance | 5.04% | 4.24% | 3.38% |

**Notable, honest finding:** train anomaly rates land almost exactly on the 5% contamination target (expected,
since that's what each detector was fit to produce on its own fit data), but val/test rates — genuinely
out-of-sample — are 2–4x higher for three of the four detectors. This is a real signal that later transactions in
this dataset look more unusual relative to the *training period's* notion of "normal" (i.e. there is real
distribution drift over time), not a bug. Fitting on the whole dataset (the old, leaky approach) would have hidden
this: every detector would have "seen" the drifted data during fitting and calibrated its own boundary around it,
producing a deceptively uniform ~5% rate everywhere.

Note: `DeviceNoveltyFlag` is ~99.5% constant (almost every transaction is a "new" device for its account), which
makes MCD's robust covariance estimate ill-conditioned on some resamples — this produces benign convergence
warnings (suppressed in the script), not invalid output.

---

## Stage 3 — Confidence-Tiered Labeling

| Tier | Rule | Count (all folds) |
|---|---|---|
| **High confidence fraud** | 3–4 of 4 detectors agree | 79 |
| **Medium confidence / needs review** | exactly 2 agree | 137 |
| **Normal** | 0–1 agree | 2,296 |

Binary label for the supervised models: High + Medium → `is_fraud=1`; Normal → `is_fraud=0`. Prevalence differs by
fold for the same reason as Stage 2's drift finding — the detectors were fit on train, so train's own pseudo-fraud
rate sits near the 5% contamination target while val/test run hotter:

| Fold | Rows | Fraud-proxy | Prevalence |
|---|---|---|---|
| Train | 1,608 | 86 | 5.35% |
| Val | 401 | 59 | 14.71% |
| Test | 503 | 71 | 14.12% |

The 3-tier column is kept in `labeled.csv` for the demo, since "needs review" is a real operational category a
binary label would discard.

---

## Stage 4 — Balancing (TRAIN fold only)

SMOTE fit on the training fold only, `k_neighbors=5` (86 real minority rows is enough for 5-NN interpolation to
stay meaningful) → **3,044 rows, 1,522/1,522 (50/50)**.

Alternative: class-weighting via `scale_pos_weight = 17.70` (ratio of majority to minority in the training fold),
trained directly on the original 1,608-row imbalanced fold with no synthetic data. Val and test are **never**
resampled or used to fit anything in this stage.

---

## Stage 4b — Robust Cross-Validated Evaluation (before any fine-tuning)

Before fitting the final models in Stage 5, a robust 5-fold stratified cross-validation runs **on the training
fold only** (val/test are never read by this stage), for each model's baseline hyperparameters — the same
configuration used in Stage 5. This exists so that (a) the reported metrics have an honest variance estimate
(mean ± std across 5 folds) instead of a single point value, and (b) any future hyperparameter search has a
ready-made harness that stays entirely inside the training fold, per §11's requirement that tuning must never
touch validation (used for model/threshold selection) or test (used exactly once, at the end).

SMOTE is refit inside each CV split's own training portion only, never on the held-out CV fold — the same
resample-after-split discipline as Stage 4.

| Model | Precision (mean±std) | Recall (mean±std) | F1 (mean±std) | ROC-AUC (mean±std) | PR-AUC (mean±std) |
|---|---|---|---|---|---|
| XGBoost + SMOTE | 0.444 ± 0.083 | 0.546 ± 0.134 | 0.487 ± 0.099 | 0.915 ± 0.033 | 0.561 ± 0.077 |
| **XGBoost + Class Weighting** | **0.561 ± 0.085** | 0.546 ± 0.081 | **0.553 ± 0.080** | **0.949 ± 0.020** | **0.618 ± 0.058** |
| Random Forest (balanced) | 0.394 ± 0.063 | 0.732 ± 0.091 | 0.511 ± 0.072 | 0.926 ± 0.020 | 0.432 ± 0.145 |

Full per-fold numbers: `artifacts/cv_per_fold.csv`. Summary: `artifacts/cv_summary.csv` / `.json`.

**Read alongside Stage 6, not instead of it:** these CV numbers are all computed from *within* the training
period (folds drawn from the same 1,608 chronologically-earliest rows), so they run noticeably higher than the
val/test numbers in Stage 6 (e.g. class-weighted XGBoost's CV ROC-AUC of 0.949 vs. its test ROC-AUC of 0.831).
That gap is consistent with, and further evidence for, the distribution-drift finding in Stage 2/3 — later
transactions genuinely look different from the training period, so even a robust in-period CV estimate does not
fully anticipate out-of-period performance. This is exactly why the val/test split (not just CV) is kept as the
final word on generalization — CV alone, on this dataset, would have looked more optimistic than reality.

---

## Stage 5 — Models (trained on TRAIN fold only)

Three classifiers, same leakage-free features, same train fold, same `random_state=42` — no model gets extra
tuning or information the others don't:

- **Model A:** XGBoost (`n_estimators=200, max_depth=4, learning_rate=0.05`) on the SMOTE-balanced training set.
- **Model B:** identical XGBoost config, `scale_pos_weight=17.70`, on the original imbalanced training fold.
- **Model C (new):** Random Forest (`n_estimators=200, max_depth=4, class_weight="balanced"`) on the original
  imbalanced training fold — added for the hackathon model comparison.

## Stage 6 — Model Comparison, Cost-Based Threshold, Final Evaluation, SHAP

### Why accuracy is misleading

A model predicting "Normal" for every test transaction scores **85.88% accuracy** while catching 0 of the 71 real
fraud-proxy cases in the test fold. Accuracy alone would make that naive model look strong; it isn't.

### Model comparison — measured on the untouched TEST fold, threshold = 0.5

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN | TP | TN |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost + SMOTE | 0.500 | 0.394 | 0.441 | 0.801 | 0.468 | 28 | 43 | 28 | 404 |
| **XGBoost + Class Weighting** | **0.767** | 0.324 | 0.455 | **0.831** | **0.558** | **7** | 48 | 23 | 425 |
| Random Forest (balanced) | 0.453 | 0.338 | 0.387 | 0.797 | 0.431 | 29 | 47 | 24 | 403 |

Full table with the diagnostic VAL-fold numbers alongside: `artifacts/model_comparison.csv` /
`model_comparison.json`.

**Selected primary model: XGBoost + Class Weighting** — highest test PR-AUC (0.558, the metric that matters most
under this class imbalance), highest ROC-AUC (0.831), and fewest false positives (7) of the three. Random Forest,
added specifically for this comparison, does not beat either XGBoost variant here — a legitimate result of a fair,
same-data comparison, not a foregone conclusion. See `artifacts/best_model_choice.json` for the recorded reasoning.

*(For context: before the leakage fix, this same comparison reported ROC-AUC ≈ 0.94–0.95 and PR-AUC ≈ 0.59–0.74 —
those numbers were measuring how well XGBoost reproduced an anomaly ensemble that had already seen the test data
during fitting, not real generalization. The lower, leakage-free numbers above are the honest estimate.)*

### Global SHAP importance (mean |SHAP value| on test, XGBoost + Class Weighting)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `TransactionAmount` | 0.882 |
| 2 | `Amount_vs_AccountAvg` | 0.742 |
| 3 | `LoginAttempts` | 0.562 |
| 4 | `TransactionType_Debit` | 0.503 |
| 5 | `IPTxnCount` | 0.420 |
| 6 | `LocationNoveltyFlag` | 0.415 |
| 7 | `TimeSinceLastTxn` | 0.387 |
| 8 | `TransactionDuration` | 0.266 |

Full ranking: `artifacts/shap_global_importance.csv`. Plots: `artifacts/plots/shap_summary_bar.png`,
`shap_summary_beeswarm.png`, `shap_waterfall_top_risk.png`.

### Decision tree rules (depth=3, trained on train fold, for the slide)

```
|--- TransactionAmount <= 812.07
|   |--- LoginAttempts <= 2.50
|   |   |--- Amount_vs_AccountAvg <= 19.71  -> class: Normal
|   |   |--- Amount_vs_AccountAvg >  19.71  -> class: Fraud
|   |--- LoginAttempts >  2.50
|   |   |--- TransactionAmount <= 79.79     -> class: Normal
|   |   |--- TransactionAmount >  79.79     -> class: Fraud
|--- TransactionAmount >  812.07
|   |--- Amount_vs_AccountAvg <= 6.97
|   |   |--- Amount_vs_AccountAvg <= 5.18   -> class: Fraud
|   |   |--- Amount_vs_AccountAvg >  5.18   -> class: Normal
|   |--- Amount_vs_AccountAvg >  6.97
|   |   |--- TransactionDuration <= 43.50   -> class: Normal
|   |   |--- TransactionDuration >  43.50   -> class: Fraud
```

Plain-English summary: **transaction size and how far it deviates from the account's own prior average** are the
dominant split features, with login attempts and transaction duration refining the call at the margins — broadly
consistent with the SHAP ranking above.

### Cost-based decision threshold — selected on VALIDATION, applied once to TEST

**Leakage fix:** the threshold is swept on the VAL fold only (never test), then applied unchanged to test exactly
once. Illustrative costs (not real bank figures): false positive = $5 (customer friction), false negative = $250
(uncaught fraud loss) — a 50:1 ratio.

- VAL cost at default threshold 0.50: $7,545
- **Minimum VAL cost $1,450 at threshold 0.01** ("review" threshold)
- Block threshold (high-precision cutoff, also selected on VAL): 0.97

**Honest finding, not smoothed over:** a 0.01 review threshold is a genuinely degenerate result for daily
operations — it flags 90% of test transactions for review. It is nonetheless the *mathematically* cost-minimizing
threshold under the stated $5/$250 ratio: because the model's real, leakage-free probability separation is modest
(PR-AUC 0.56), the sweep finds that catching one more $250 loss is worth risking many more $5-friction false
positives, all the way down to a very low cutoff. Under the old, leakage-inflated pipeline, this same sweep landed
on a much more "reasonable-looking" 0.09 — that apparent reasonableness was itself an artifact of leakage
(sharper, over-confident probabilities), not evidence of a well-calibrated threshold. See `LIMITATIONS.md` and
`ML_AUDIT_AFTER_FIX.md` for what this implies for a real deployment (bank-supplied real costs, and/or a
review-capacity constraint on top of pure cost minimization, are both necessary next steps).

**Final TEST-set outcome, threshold=0.5 vs. VAL-selected threshold 0.01:**

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.50 (default) | 0.891 | 0.767 | 0.324 | 0.455 |
| 0.01 (VAL-selected, cost-optimal) | 0.239 | 0.156 | 1.000 | 0.270 |

ROC-AUC = 0.831, PR-AUC = 0.558 (threshold-independent).

Final Approve/Review/Block counts on the 503-row test fold, using the VAL-selected thresholds:
**APPROVE: 49, REVIEW: 451, BLOCK: 3.**

---

## Stage 7 — Live Scoring (Argus dashboard, "Upload & Predict")

There is no standalone demo app for Pipeline 1 anymore — live scoring is served from the Argus dashboard
(`dashboard/backend/api_server.py`, `POST /api/upload/predict`) instead. It engineers a batch of new transactions'
features with `fe_utils.transform_batch_new()` — built on the same per-row logic as `transform_new()`, so
training-time and live-scoring feature engineering can't drift apart — scores them with the Stage 6-selected
primary XGBoost model (`artifacts/xgb_model_best.json`), and returns a fraud probability per row. Verified directly
against both an existing account with history and a brand-new account with none — both score sensibly (a repeat
transaction on a known device/account scored far lower than the same amount on a brand-new account/device in a
spot check).

See `dashboard/README.md` for the full dashboard, including the Transaction Explorer and Investigation Queue pages
for browsing/searching individual transactions and accounts by identifier.

---

## Stage 8 — Limitations

See `LIMITATIONS.md` for the full, slide-ready caveats, and `ML_AUDIT_AFTER_FIX.md` for the complete leakage audit,
exactly what was fixed, and the reproducibility details.
