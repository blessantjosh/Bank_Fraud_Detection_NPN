# ML Audit: Data-Leakage Fix — `placement-round-fraud-kit` (v1 pipeline, `src/`)

Scope: the v1 pipeline in `src/` (raw data → 20 features → 4-detector unsupervised anomaly ensemble → voting →
pseudo fraud labels → SMOTE/class-weight XGBoost + Random Forest → SHAP → cost-based threshold → Approve/Review/
Block). The architecture is unchanged; every number in this document is from an actual run of the fixed pipeline
against `data/bank_transactions_data_2.csv`, not an estimate.

---

## 1. Original leakage problem

The pipeline computed every "global" statistic used as a feature — transaction-type means, device/IP/merchant
popularity counts, the median history-gap fallback, categorical encoders, and the `StandardScaler` — on the
**entire dataset**, then split into train/test *after* those statistics were already baked into every row. The
four unsupervised anomaly detectors (Isolation Forest, LOF, One-Class SVM, MCD) were also **fit on the complete
feature matrix**, so the pseudo-labels themselves were generated with test-set information already folded in. The
cost-based decision threshold was then swept directly over the test set and that same test set was reported as an
"unbiased" evaluation. All of this meant test performance numbers (previously ROC-AUC ≈ 0.94–0.95, PR-AUC ≈
0.59–0.74) were inflated by information the model should never have had access to at evaluation time.

## 2. Exactly where leakage existed

| # | Location (pre-fix) | What leaked |
|---|---|---|
| 1 | `fe_utils.fit_engineer()` — `type_avg`, `device_counts`, `ip_counts`, `merchant_counts` computed via `df.groupby(...).mean()` / `.value_counts()` on the full dataframe, before any split | Every row's `Amount_vs_TypeAvg`, `DeviceTxnCount`, `IPTxnCount`, `MerchantTxnCount` was a function of the entire dataset, including rows that would later become the test set |
| 2 | `fe_utils.fit_engineer()` — `median_gap` (fallback for `TimeSinceLastTxn` on a first transaction) computed from the full dataframe | Test-set transaction gaps influenced the fallback value used for train rows and vice versa |
| 3 | `01_feature_engineering.py` — a single `StandardScaler` fit on the complete feature matrix | Test-row values shifted the mean/variance the scaler used to transform train rows |
| 4 | `_encode_categoricals()` called once on the full dataframe (`encoders=None`) before any split existed | One-hot/label-encoding categories were determined by the complete dataset, not just training data |
| 5 | `02_anomaly_ensemble.py` — `iso.fit_predict(X)`, `lof.fit_predict(X)`, `ocsvm.fit_predict(X)`, `mcd.fit_predict(X)`, where `X` was the complete `features_scaled.csv` | All four unsupervised detectors — and therefore every pseudo-label — were fit with full knowledge of the eventual test rows |
| 6 | `04_balancing.py` — `train_test_split(..., stratify=y, random_state=...)` performed **after** stages 1–3 had already used the complete dataset | By the time any split existed, leakage from #1–#5 was already baked into every downstream artifact; the split itself was also random rather than chronological (see §4) |
| 7 | `06_evaluation.py` — the cost-based threshold sweep (`thresholds = np.linspace(...)`) ran directly against `X_test`/`y_test`, and that same test set was then reported as the "unbiased" evaluation | Threshold selection is itself a form of fitting; selecting it on the set you then report performance on lets the report see the test set twice |

## 3. Exactly how it was fixed

- **`fe_utils.py`** was split into explicit fit/apply stages: `add_causal_features()` (strictly backward-looking
  per-account features, safe pre-split by construction — see §3b), `fit_global_stats(train_df)` (fits
  `type_avg`/`device_counts`/`ip_counts`/`merchant_counts`/`median_gap_hours` from a given dataframe only),
  `apply_global_stats(df, stats)` (maps a fitted `stats` dict onto any other dataframe, with documented fallbacks
  for identifiers never seen during fitting), and `finalize_matrix(df, encoders=None)` (encodes categoricals,
  fitting the encoder only when `encoders=None` is passed).
- **`01_feature_engineering.py`** now splits the raw data chronologically *first* (§4), then calls
  `fit_global_stats()` on the training rows only, and applies the resulting `stats` dict — unchanged — to val and
  test. The `StandardScaler` and the categorical encoders are likewise fit on the training fold only and reused,
  unmodified, for val/test.
- **`02_anomaly_ensemble.py`** fits all four detectors on the training fold's scaled features only, then calls
  `.predict()` (out-of-sample) on val and test. LOF's default mode has no out-of-sample `predict`; the fix uses
  scikit-learn's documented `novelty=True` mode, which is built exactly for "fit on one set, score a different
  set."
- **`04_balancing.py`** no longer performs any split — it reads the `split` column Stage 1 already produced and
  resamples (SMOTE) only the training rows.
- **`06_evaluation.py`** sweeps the cost-based threshold on the validation fold only, then applies the resulting
  threshold, unchanged, to the test fold exactly once for the numbers reported as final.

### 3b. Why the per-account behavioral features didn't need this treatment

`Amount_vs_AccountAvg` (via `groupby("AccountID")["TransactionAmount"].shift().expanding().mean()`),
`DeviceNoveltyFlag`/`LocationNoveltyFlag` (via `.duplicated()` on a chronologically-sorted per-account group), and
the raw `TimeSinceLastTxn` gap were already strictly causal in the original code — each value only depends on
*strictly earlier* rows of the *same account*, which is invariant to wherever a train/val/test boundary later gets
drawn. These were kept as-is. The leakage in the original code was specifically in the four **global, cross-row**
statistics and the four **unsupervised detectors**, not in the per-account causal logic — see §3 above for exactly
what changed instead: the *fallback values* for these features (the training-fold median gap, the training-fold
type average used when an account has no prior transaction) are now fit on train only, same as everything else.

## 4. Train/test methodology: chronological split, and why

The split changed from a **random, stratified** 80/20 split (performed *after* the leaky stages) to a
**chronological** 64/16/20 split on `TransactionDate`:

- **Train**: earliest ~64% of transactions (1,608 rows, through 2023-08-28)
- **Validation**: next ~16% (401 rows, through 2023-10-23) — used only for model/threshold selection
- **Test**: latest ~20% (503 rows) — touched exactly once, for the final reported numbers

**Why chronological is the right choice here:** this is transaction data with a real time axis, and a production
fraud system is trained on the past and must generalize to the future — a random split lets the model implicitly
"see" transactions from after the cutoff during training (via shared distributional characteristics of a
time-drifting dataset) and evaluates it on a mix of past-and-future rows, which is not the deployment scenario. A
chronological split is the honest analogue of "train on everything known up to today, evaluate on tomorrow." A
three-way split (rather than two) was necessary specifically because §5's fix (fit anomaly detectors on train only)
and §13's fix (select the cost threshold without touching test) cannot both be satisfied with only a train/test
split — validation is the fold that absorbs model comparison and threshold selection so test stays genuinely
untouched until the one, final evaluation.

One limitation is stated plainly rather than hidden: cold-start accounts exist in val (14) and test (7) — accounts
with no prior transactions in the training fold. These are handled by the existing fallback logic (novelty flags
default to 1, `Amount_vs_AccountAvg` falls back to the type average) rather than silently dropped.

## 5. Pseudo-label generation methodology

Unchanged architecture: Isolation Forest, LOF, One-Class SVM, and MCD (via `EllipticEnvelope`) each vote
independently (contamination assumption: 5%, applied identically to all four, documented as unverified). 3–4 votes
→ "High confidence fraud", 2 votes → "Medium confidence / needs review", 0–1 → "Normal"; High+Medium collapse to
the binary `is_fraud` label used for supervised training.

**What changed:** all four detectors are now fit **exclusively on the 1,608 training rows** and used to *predict*
(not re-fit) on val and test. This surfaced a genuine, previously-invisible finding: train's own anomaly rate sits
almost exactly at the 5% contamination target (expected — that's what each detector was tuned to produce on its
fit data), but three of the four detectors flag val/test at 2–4x that rate (Isolation Forest 5.0%→9.5–12.5%,
One-Class SVM 5.4%→21–23%; MCD is the exception, staying low at 3–4%). This is evidence of real distribution drift
over time in this dataset, which the old whole-dataset fit had silently absorbed and hidden. Resulting fraud-proxy
prevalence: **train 5.35% (86/1,608), val 14.71% (59/401), test 14.12% (71/503)**.

## 6. Anomaly detector methodology (per detector)

| Detector | Fit on | Out-of-sample method |
|---|---|---|
| Isolation Forest | Train only | `.predict()` (natively supports new data) |
| Local Outlier Factor | Train only | `novelty=True`, then `.predict()` — LOF's default mode has no out-of-sample predict; `novelty=True` is scikit-learn's documented mechanism for exactly this case |
| One-Class SVM | Train only | `.predict()` (natively supports new data) |
| MCD (`EllipticEnvelope`) | Train only | `.predict()` (natively supports new data); `support_fraction=0.9` retained to keep the robust covariance estimate well-conditioned given `DeviceNoveltyFlag`'s near-constant distribution |

## 7. XGBoost training

Two variants, both trained on the training fold only, `random_state=42`, `n_estimators=200, max_depth=4,
learning_rate=0.05`:

- **Model A (SMOTE):** trained on `SMOTE(k_neighbors=5)`-resampled training data (86 real minority rows → 3,044
  rows, 50/50 after resampling). SMOTE is fit and applied to the training fold only; val/test are never resampled.
- **Model B (class-weighting):** trained directly on the imbalanced 1,608-row training fold with
  `scale_pos_weight=17.70` (the train fold's majority:minority ratio).

## 8. Random Forest training (new)

Added per the hackathon requirement for a third comparison model: `RandomForestClassifier(n_estimators=200,
max_depth=4, class_weight="balanced", random_state=42)`, trained on the **same** training fold and the **same**
leakage-free features as both XGBoost variants — no extra tuning, no different data. This makes the three-way
comparison in §11 fair by construction.

## 9. Evaluation results (measured, real, on the untouched TEST fold, threshold = 0.5)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN | TP | TN |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost + SMOTE | 0.500 | 0.394 | 0.441 | 0.801 | 0.468 | 28 | 43 | 28 | 404 |
| **XGBoost + Class Weighting** | **0.767** | 0.324 | 0.455 | **0.831** | **0.558** | **7** | 48 | 23 | 425 |
| Random Forest (balanced) | 0.453 | 0.338 | 0.387 | 0.797 | 0.431 | 29 | 47 | 24 | 403 |

A naive "always predict Normal" baseline scores 85.88% accuracy while catching 0 of the 71 real fraud-proxy cases
in test — accuracy alone would rank it above every real model above; it is reported but explicitly flagged as
insufficient, per the brief.

Full table (including the val-fold diagnostic numbers used for model/threshold selection):
`artifacts/model_comparison.csv` / `model_comparison.json`.

## 10. Model comparison and selection

**Primary model: XGBoost + Class Weighting** — highest test PR-AUC (0.558, the correct primary metric under this
level of class imbalance), highest ROC-AUC (0.831), and by far the fewest false positives (7 vs. 28–29 for the
other two). Random Forest, added specifically for this comparison, did not outperform either XGBoost variant here
— a genuine result of a fair, identical-data comparison (`artifacts/best_model_choice.json`), not a predetermined
conclusion. SHAP (§13) and the production threshold/demo app (§14) use this selected model.

*For context:* the pre-fix pipeline reported ROC-AUC ≈ 0.94–0.95 / PR-AUC ≈ 0.59–0.74 on the same architecture.
Those numbers measured how well XGBoost reproduced an anomaly ensemble that had already been fit with knowledge of
the test rows — not real generalization. The lower numbers above are the honest, leakage-free estimate. This is
exactly the expected, acceptable outcome the fix was asked to produce even if it made the headline numbers worse.

## 11. Cost-based threshold

**Leakage fix:** the threshold sweep now runs on the **validation** fold only; the resulting threshold is applied,
unchanged, to test exactly once. Illustrative costs (not real bank figures, unchanged from the original build):
false positive = $5 (customer friction), false negative = $250 (uncaught fraud), a 50:1 ratio.

- VAL cost at default threshold 0.50: $7,545
- **Minimum VAL cost $1,450 at threshold 0.01** ("review" threshold)
- High-precision "block" threshold (also selected on VAL): 0.97

**This is reported honestly even though it is not a practically usable answer.** A 0.01 review threshold flags 90%
of the test fold (451 of 503 rows) for manual review — mathematically cost-optimal under the stated 50:1 ratio
given the model's real (modest, leakage-free) probability separation, but operationally useless for a bank. Final
test-fold decision counts using this threshold: **APPROVE 49, REVIEW 451, BLOCK 3**. For comparison, at the
default 0.5 threshold: precision 0.767, recall 0.324, F1 0.455, confusion matrix `[[425 FP=7][FN=48 TP=23]]`. Both
sets of numbers are in `artifacts/final_test_evaluation.json`. See §14 (Limitations) for what this implies.

## 12. SHAP findings

Computed on the selected primary model (XGBoost + Class Weighting), on the test fold. Top drivers by mean |SHAP|:
`TransactionAmount` (0.882), `Amount_vs_AccountAvg` (0.742), `LoginAttempts` (0.562), `TransactionType_Debit`
(0.503), `IPTxnCount` (0.420), `LocationNoveltyFlag` (0.415), `TimeSinceLastTxn` (0.387). Full ranking:
`artifacts/shap_global_importance.csv`; plots in `artifacts/plots/`. As before the fix, SHAP explains what the
model learned from the anomaly ensemble's own pseudo-labels — internal consistency, not a verified causal fraud
mechanism (the label is circular by construction; see `LIMITATIONS.md`).

## 13. Remaining limitations

See `LIMITATIONS.md` for the full list. In summary: no genuine fraud label exists anywhere in this dataset (the
label is 4 detectors' own, train-fit consensus); the label is circular by construction; the dataset (2,512 rows,
~86–216 pseudo-fraud rows depending on fold) is far smaller than a real production deployment; real distribution
drift exists between train and val/test (§5/§6), meaning the 5% contamination assumption would need periodic
re-fitting in production; and the cost-optimal threshold as computed is not directly deployable without the bank's
real cost figures and a review-capacity constraint layered on top of pure cost minimization.

## 14. Reproducibility

`random_state=42` (or the seeded `config.RANDOM_STATE`) is used consistently for: the anomaly detectors
(Isolation Forest, MCD), SMOTE, both XGBoost variants, Random Forest, and the decision tree. LOF and One-Class SVM
are deterministic given fixed inputs and hyperparameters. The chronological split is deterministic (a fixed
quantile of `TransactionDate`, not a random draw). Re-running `01` through `06` in order against the same
`data/bank_transactions_data_2.csv` reproduces every number in this document exactly.

## 15. Verification checklist (all confirmed against actual runs, not assumed)

1. No feature statistic uses test data — `fit_global_stats()` is only ever called with `train_raw`/`df` (full-data
   call is for the production `reference.pkl`, a separate artifact that never touches evaluation; see
   `01_feature_engineering.py`'s docstring). ✅
2. No historical feature uses future transactions — per-account features are strictly causal (§3b). ✅
3. No anomaly detector is fitted on test data — all four are `.fit(X_train)` only (§6). ✅
4. No SMOTE is applied to test data — `04_balancing.py` resamples `X_train`/`y_train` only. ✅
5. No test data is used for hyperparameter tuning — no hyperparameter search has been added yet; the same fixed
   configuration is used for all three models per §10's fairness requirement. A robust, train-fold-only 5-fold CV
   harness (`04b_cross_validation.py`, §16) now exists specifically so that if/when tuning is added, it plugs into
   that harness rather than ever touching validation or test. ✅
6. No test data is used to choose the final threshold — threshold sweep runs on `X_val`/`y_val` only (§11). ✅
7. XGBoost trains successfully — both variants trained and saved (`artifacts/xgb_model.json`,
   `xgb_model_classweight.json`). ✅
8. Random Forest trains successfully — saved to `artifacts/random_forest_model.pkl`. ✅
9. Evaluation completes successfully — `artifacts/model_comparison.csv/json`,
   `artifacts/final_test_evaluation.json`. ✅
10. SHAP completes successfully — `artifacts/plots/shap_*.png`, `artifacts/shap_global_importance.csv`. ✅
11. New transaction inference still works — verified directly via `fe_utils.transform_new()` for both an existing
    account with history and a brand-new account/device with none; both scored sensibly (existing-account repeat
    transaction: P(fraud)=0.961 at an out-of-pattern amount; brand-new account/device, ordinary amount:
    P(fraud)=0.034). ✅
12. Existing UI/API continued to work at the time of this audit — the then-existing demo app booted and served
    HTTP 200 (verified headless); its identifier-search join was fixed to use `TransactionID` (§ below) since row
    order changed once the split became chronological-then-grouped rather than one global sort. That demo app has
    since been removed; live scoring is now served from the Bank Transaction Fraud & Anomaly Detection dashboard's "Upload & Predict" page instead,
    reusing the same `fe_utils.transform_new`/`transform_batch_new` functions this fix produced. ✅
13. Model comparison is generated — `artifacts/model_comparison.csv/json`, `artifacts/plots/model_comparison.png`.
    ✅
14. All metrics are generated from the actual leakage-free test set — every number in §9/§11 was printed by an
    actual run of `06_evaluation.py`, not hand-written. ✅

### One additional bug found and fixed during verification (not in the original leakage list, but load-bearing)

The then-existing demo app's "Search Identifier History" tab joined `labeled.csv` to the raw dataset by **row position**
(`pd.concat(..., axis=1)`), relying on both being sorted identically. Once Stage 1 started grouping rows by
`split` (train rows first, then val, then test) rather than one global chronological sort, that positional
assumption silently broke — every row would have been paired with the wrong transaction's tier. Fixed by carrying
`TransactionID` through `finalize_matrix()`/`features.csv`/`labeled.csv` as an explicit, un-modeled identity column
and joining on it (`raw.merge(labeled[...], on="TransactionID")`) instead of relying on row order.

## 16. Addendum: robust cross-validated evaluation, added before fine-tuning

A new stage, `04b_cross_validation.py`, was added between balancing (§4/§5) and final model training (§7/§8) to
satisfy §11's requirement that any hyperparameter tuning be validated via cross-validation, never against test —
even though no tuning has been added yet, the evaluation harness now exists first, per the requested order of
operations.

**Methodology:** 5-fold stratified cross-validation, run **exclusively on the 1,608-row training fold** (val and
test are never loaded by this script). Each model uses the exact same baseline hyperparameters as Stage 5/7 — no
tuning is performed, only a variance-aware re-measurement of the same configuration. SMOTE is refit inside each CV
split's own training portion only (never on the held-out CV fold), preserving the same resample-after-split
discipline as §4.

**Measured results** (mean ± std across 5 folds, train-fold-only, threshold=0.5):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost + SMOTE | 0.444 ± 0.083 | 0.546 ± 0.134 | 0.487 ± 0.099 | 0.915 ± 0.033 | 0.561 ± 0.077 |
| **XGBoost + Class Weighting** | **0.561 ± 0.085** | 0.546 ± 0.081 | **0.553 ± 0.080** | **0.949 ± 0.020** | **0.618 ± 0.058** |
| Random Forest (balanced) | 0.394 ± 0.063 | 0.732 ± 0.091 | 0.511 ± 0.072 | 0.926 ± 0.020 | 0.432 ± 0.145 |

Full detail: `artifacts/cv_per_fold.csv`, `artifacts/cv_summary.csv`/`.json`.

**This independently corroborates the drift finding in §5/§6.** All CV folds are drawn from *within* the training
period, so CV ROC-AUC/PR-AUC (e.g. 0.949 / 0.618 for the selected model) come out noticeably higher than that same
model's val/test ROC-AUC/PR-AUC (0.883 / 0.680 on val, 0.831 / 0.558 on test, §9). If the dataset had no real
temporal drift, a robust in-period CV estimate and true out-of-period performance would track much more closely.
The gap confirms the val/test split — not CV alone — has to be the final word on this pipeline's real
generalization, and that CV is best used here as a pre-tuning sanity check, not a substitute for the held-out
chronological folds.
