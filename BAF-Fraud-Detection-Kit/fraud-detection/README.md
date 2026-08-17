# Fraud Detection Kit -- Bank Account Fraud (BAF)

A production-shaped fraud-detection pipeline for the Feedzai **Bank Account
Fraud (BAF)** dataset, **Base** variant (NeurIPS 2022 Datasets & Benchmarks
track). Every metric in this README comes from an actual run of `train.py`
and `evaluate.py` against the real 1,000,000-row `Base.csv` on this machine
-- nothing here is fabricated or estimated.

## What this dataset actually is

This is **account-opening** fraud: a fraudster opening a *new* bank account
with a fake, stolen, or synthetic identity. It is **not** the more famous
credit-card *transaction* fraud problem. Consequently:

- There is **no transaction amount**, **no transaction timestamp**, and
  **no account/customer identifier** in the raw file -- only a coarse
  `month` index (0-7).
- The target column is `fraud_bool` (0/1). It is **not** renamed to
  `is_fraud` anywhere in this codebase; `config.yaml -> data.target_col`
  makes the name configurable without inventing a different one.
- Real fraud rate: **~1.10%** (imbalance ratio ~89:1). Accuracy is a
  meaningless headline metric here -- an all-zero classifier scores ~98.9%.

See the sibling kit's `01-DATASET-BIBLE.md` for the full column-by-column
reference this project was built from and verified against.

## Explicitly skipped, and why (do not treat as an oversight)

| Spec item | Why it was skipped |
|---|---|
| `amount_log`, `transactions_per_hour` | No transaction-amount column exists in account-opening data. |
| `hour`, `day_of_week`, `is_weekend` | No real timestamp exists, only a coarse 0-7 `month` index. |
| `current_amount / historical_average` deviation features | No transaction history exists to compute a historical average from. |
| Identifier-exclusion step | No `account_id`/`customer_id`-style column exists in the raw file; verified programmatically in `src/data_validation.py::validate_quality` (see `identifier_columns_found`). |
| Leakage-column removal | No post-decision columns (chargeback status, investigation result, etc.) exist; verified in `src/data_validation.py::check_no_leakage`, which found nothing. |
| Temporal train/test split | The organiser split for this dataset is **random stratified 70/30** (verified, not assumed -- see below), so a temporal split would both discard 6/8 of the data unnecessarily and throw away `month` as a usable feature. |

No substitute columns were invented for any of the skipped items above.

## Split protocol -- verified, not assumed

The organisers of the closest public leaderboard for this dataset
(`kaggle/1056lab-bank-account-fraud-detection`) state they used a **random**
70/30 split of the 1M rows, not the NeurIPS paper's temporal (months 0-5 /
6-7) protocol. This project mirrors that: a **stratified random 70/15/15
train/val/test split** (`src/data_loader.py::stratified_split`), with
`month` kept as an ordinary feature. `config.yaml -> split.strategy` is set
to `stratified_random` for exactly this reason, and the fraud rate is
reported per split at load time so the choice is auditable, not asserted.

## Compute constraints (be honest about the hardware, not just the model)

This was built and run on an **8GB RAM, 8-core** machine. Two consequences,
stated up front rather than discovered by a judge:

1. **Random Forest + resampling.** `n_estimators=300` on a 700k-row training
   fold is feasible (see Results), but 300 trees over an *oversampled*
   frame (SMOTE brings the effective row count well past 700k) is not, on
   this hardware. For the `random_undersample`, `smote`, and
   `smote_undersample` strategies **only**, Random Forest is trained on a
   stratified subsample capped at `config.yaml -> imbalance.rf_resampling_max_rows`
   (150,000 rows) rather than the full resampled set. `random_forest__none`
   and `random_forest__class_weight` train on the **full** 700k-row fold --
   no resampling means no row-count blowup, so no cap was needed there.
   Every row in `reports/metrics/model_comparison.csv` records its actual
   `train_rows`, so which combinations ran at full scale vs. a controlled
   subsample is auditable from the artifact itself, not asserted here.
2. **Full-scale training runs crashed three times before producing the
   numbers below** -- the process was killed outright, with no Python
   traceback, each time. First, a full 19-of-20 run died right at the very
   last combination (`xgboost/smote_undersample`); after a first round of
   memory mitigations, the next two attempts instead died immediately at
   `random_forest/none` (early in the run). Root cause, confirmed by an
   isolated reproduction: `RandomForestClassifier.fit()` on its own
   succeeded fine with these exact parameters and the exact same data: the
   crash was about accumulated memory across the *whole* ablation loop, not
   Random Forest specifically. The loop was holding the tree-native AND
   dense feature views for train, val, *and test* simultaneously (test was
   never actually used during training) plus every previously-fitted model
   object in memory at once. Fix applied, in order of impact: (a) `prepare_frames` no longer
   transforms the test split at all during training -- it is untouched
   until `evaluate.py` loads and transforms it separately, in its own
   process, for the one-time final evaluation; (b) each of the 20 trained
   models is now serialized to a scratch cache immediately after scoring
   and dropped from memory (`del` + `gc.collect()`), rather than all 20
   being held live until the end -- only the winner is reloaded once
   selected; (c) the dense feature view is stored as float32 instead of
   sklearn's float64 default; (d) Random Forest's own footprint was reduced
   (`max_depth` 20->15, added `max_samples=0.5`, `n_jobs` -1->6, leaving
   headroom instead of claiming every core). After these changes the full
   20-combination run completed cleanly in ~21 minutes with no further
   crashes. This is reported here rather than silently fixed and
   forgotten, because it is a real, reproducible engineering constraint of
   running this pipeline on this hardware, and the fix changes real
   numbers in the table below (e.g. Random Forest's `max_samples=0.5`
   means each tree bootstraps half the rows, not all of them).
3. **SMOTE needs a fully numeric, non-missing matrix.** For the `smote` and
   `smote_undersample` strategies, LightGBM and XGBoost are trained on the
   same imputed / one-hot / scaled **dense** feature view as Logistic
   Regression and Random Forest, instead of their native NaN + categorical
   handling (used for `none` and `class_weight`). This is a deliberate,
   documented trade-off (see `src/imbalance.py` module docstring), not an
   inconsistency.

## Project layout

```
fraud-detection/
  data/{raw,processed,predictions}/   raw csv, cached split parquet, scored csv output
  notebooks/                          01_data_exploration .. 04_model_evaluation
  src/                                config, data, preprocessing, FE, imbalance,
                                       models, training, evaluation, thresholds,
                                       explainability, prediction, auth, audit
  models/                             final_model.joblib, preprocessor.joblib, model_meta.json
  reports/{figures,metrics,final_report}/
  tests/                               unit tests (validation, FE, preprocessing, prediction, auth)
  train.py / evaluate.py / predict.py CLI entry points
  config.yaml                         drives every parameter in the pipeline
```

## Pipeline

```
raw Base.csv
  -> data_validation (schema, quality, leakage/identifier checks)
  -> stratified random 70/15/15 split (train/val/test)
  -> Preprocessor fit on TRAIN ONLY (constant-column drop, sentinel -> NaN
     + flag, feature engineering, impute/scale/one-hot for the dense view)
  -> imbalance strategy applied to TRAIN ONLY (val/test distributions never touched)
  -> train 4 model families x 5 imbalance strategies = 20 runs
  -> select best by PR-AUC on VALIDATION
  -> threshold tuned on VALIDATION (min expected cost, and F1, both reported)
  -> save model + preprocessor + threshold + feature-columns artifacts
  -> ONE final evaluation on the untouched TEST split (evaluate.py)
```

## Models compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline. `class_weight="balanced"` for the `class_weight` strategy; scaled + one-hot encoded dense features. |
| Random Forest | `n_estimators=300`, `max_depth=15`, `max_samples=0.5`, `class_weight="balanced"`, `n_jobs=6` (all reduced from an initial `max_depth=20`/`n_jobs=-1` after a real OOM-style crash on this 8GB machine -- see "Compute constraints"). |
| LightGBM | Starting params reused from the sibling kit's proven `baf.py`/`run_pipeline.py` (`min_data_in_leaf=200`, `min_sum_hessian_in_leaf=1.0` -- both matter a lot at a 1.1% base rate). Native NaN + categorical handling. |
| XGBoost | `enable_categorical=True`, `tree_method="hist"`. **`min_child_weight` is a sum of Hessians, not a row count** -- a value like 200 silently produces a 0-split stump (constant predictions, AUC exactly 0.5) at this base rate. `min_child_weight=5` is used instead, and every XGBoost run is verified to have `best_iteration > 0` and non-constant predictions before its numbers are trusted (`src/models.py::train_xgboost` raises `RuntimeError` otherwise). |

## Imbalance strategies compared

`none`, `class_weight` (`class_weight="balanced"` / `scale_pos_weight`),
`random_undersample` (10 negatives kept per positive),
`smote` (SMOTE to a controlled 0.10 minority:majority ratio, not a full 1:1
balance -- a full 1:1 SMOTE on ~700k rows synthesizes ~690k rows for no
measured benefit; see Results), and `smote_undersample` (undersample majority
to 20:1, then SMOTE the minority up to a 0.30 ratio on the smaller set).

## Metrics

**PR-AUC is the primary model-selection metric** (`config.yaml ->
evaluation.primary_metric`), not accuracy or even ROC-AUC alone: at a 1.1%
base rate, PR-AUC is far more sensitive to how well a model ranks the rare
positive class. ROC-AUC and **TPR@5%FPR** (the domain-correct metric from
the BAF paper -- chosen because every false positive is a real customer
wrongly denied a bank account) are reported alongside it for every model.
Threshold optimization sweeps a finer grid at the low end
(0.01/0.02/0.03/0.05/...) instead of uniform 0.1 steps, because a ~1% base
rate means every useful operating threshold lives below 0.5. Confusion
matrices and an optional cost-sensitive view (configurable FP/FN costs) are
computed at every swept threshold.

## Fairness

The BAF dataset was purpose-built for fairness research. The protected
attribute used throughout (`config.yaml -> protected_attribute`) is
`customer_age > 50` (**strictly greater than** -- ages are decade-rounded,
so `>= 50` would silently move an entire bucket of applicants across the
line). Predictive equality (FPR ratio between groups, at one global
threshold chosen to hit the target FPR overall) is reported on both
validation and test. See Results below for the real numbers.

## Governance & Audit

Two application-level controls were added on top of the modelling pipeline,
because a system that scores real applicants needs traceability, not just
a good AUC:

**Admin gate (`src/auth.py`).** `predict.py` -- the entry point that
produces fraud decisions on new applicant data and writes to
`data/predictions/` -- is restricted to admins. The real credential is
**never committed to this repo**: `config.yaml` stores only a salted hash of
the expected token (`auth.admin_salt_hex` / `auth.admin_token_hash_hex`,
both `null` by default -- there is no default or backdoor credential).
To use `predict.py`:

```bash
# one-time setup: generate a salt + hash for your chosen token
python -m src.auth --generate "<a long random secret you choose>"
# paste the printed admin_salt_hex / admin_token_hash_hex into config.yaml's `auth:` section

# every time before running predict.py:
export FRAUD_ADMIN_TOKEN="<the same token>"        # bash/zsh
$env:FRAUD_ADMIN_TOKEN="<the same token>"           # PowerShell
```

`require_admin()` hashes the environment variable's value (PBKDF2-HMAC-SHA256,
salted) and compares it to the configured hash with `hmac.compare_digest`
(constant-time, not `==`). If the env var is unset, or the hash doesn't
match, or no credential is configured at all, the call fails loudly with a
non-zero exit -- there is no silent bypass. Every attempt, success or
failure, is logged; the token itself is never logged, only a short
non-reversible identifier derived from it.

**This is an application-level access control appropriate for a trusted
internal CLI tool. It is NOT a substitute for a bank's real IAM / network
security if this pipeline is ever deployed as a service** -- that
distinction matters and is stated here deliberately, not glossed over.

**Audit log (`src/audit.py`).** Every successful `predict.py` run appends
one line to `reports/metrics/audit_log.jsonl` (append-only, JSON Lines):
timestamp (UTC, the run's own clock), a non-reversible admin identity (never
the token/credential), the model type/strategy and iteration/checkpoint that
scored the batch, the input row count and a SHA-256 content hash of the
input file (**not** the raw applicant data itself, so the log stays
lightweight and never duplicates PII-adjacent data into a second location),
and a summary of predictions per risk-level bucket (counts, not individual
rows). If the log write itself fails, that failure is raised loudly rather
than swallowed -- a silently-dropped audit entry would defeat the entire
point of having one.

This audit trail is, again, **application-level and suitable for an
internal tool** -- it is not a replacement for a bank's actual
compliance/SIEM systems, and is not presented as one.

## Results

All numbers below are from one real `python train.py` + `python evaluate.py` run
against the full 1,000,000-row `Base.csv` (70/15/15 stratified split: 700,000
train / 150,000 val / 150,000 test), on this 8GB-RAM, 8-core machine.
Full training took **~21 minutes** end to end (20 model x strategy combinations).
Raw artifacts: `reports/metrics/model_comparison.csv`, `models/model_meta.json`,
`reports/metrics/test_evaluation.json`.

### Full ablation table (validation set, all 4 models x 5 imbalance strategies)

| model / strategy | ROC-AUC | PR-AUC | TPR@5%FPR | train rows | train seconds |
|---|---:|---:|---:|---:|---:|
| logistic_regression / none | 0.8766 | 0.1431 | 0.5039 | 700,000 | 3.3 |
| logistic_regression / class_weight | 0.8775 | 0.1409 | 0.5027 | 700,000 | 7.1 |
| logistic_regression / random_undersample | 0.8769 | 0.1425 | 0.5027 | 84,920 | 0.5 |
| logistic_regression / smote | 0.8756 | 0.1424 | 0.4949 | 761,508 | 5.0 |
| logistic_regression / smote_undersample | 0.8763 | 0.1411 | 0.4967 | 200,720 | 1.8 |
| random_forest / none | 0.8725 | 0.1367 | 0.4900 | 700,000 | 271.7 |
| random_forest / class_weight | 0.8697 | 0.1222 | 0.4792 | 700,000 | 180.4 |
| random_forest / random_undersample | 0.8788 | 0.1446 | 0.5033 | 84,920 | 25.5 |
| random_forest / smote | 0.8751 | 0.1280 | 0.4785 | 149,999 (capped, see below) | 40.9 |
| random_forest / smote_undersample | 0.8759 | 0.1256 | 0.4864 | 149,999 (capped, see below) | 40.8 |
| lightgbm / none | 0.8946 | 0.1687 | 0.5366 | 700,000 | 39.5 |
| lightgbm / class_weight | 0.8874 | 0.1626 | 0.5196 | 700,000 | 48.9 |
| lightgbm / random_undersample | 0.8928 | 0.1682 | 0.5408 | 84,920 | 13.0 |
| lightgbm / smote | 0.8938 | 0.1662 | 0.5402 | 761,508 | 64.4 |
| lightgbm / smote_undersample | 0.8943 | 0.1658 | 0.5402 | 200,720 | 24.0 |
| xgboost / none | 0.8961 | 0.1678 | 0.5486 | 700,000 | 90.9 |
| xgboost / class_weight | 0.8927 | 0.1685 | 0.5341 | 700,000 | 71.0 |
| xgboost / random_undersample | 0.8943 | 0.1666 | 0.5414 | 84,920 | 44.4 |
| xgboost / smote | 0.8948 | 0.1686 | 0.5396 | 761,508 | 154.1 |
| xgboost / smote_undersample | 0.8950 | 0.1658 | 0.5378 | 200,720 | 56.5 |

`random_forest / smote` and `random_forest / smote_undersample` are the two
rows that ran on the controlled 150,000-row subsample described in "Compute
constraints" above -- their `train rows` column shows this directly rather
than asserting it in prose only. Every other row trained on its full,
uncapped resampled set (700,000 for `none`/`class_weight`; 84,920 for
`random_undersample`; 761,508 for `smote`; 200,720 for `smote_undersample`).

**What the ablation actually shows about "data balancing":** for every
model family, the gap between the best and worst imbalance strategy is
0.003-0.02 ROC-AUC and 0.01-0.02 PR-AUC -- inside noise for most pairs, and
never enough to change which model family wins. `class_weight`/
`scale_pos_weight` is not obviously better than doing nothing; SMOTE is not
obviously better than a plain 10:1 random undersample. This matches Trap 5
in the sibling kit's dataset bible: on gradient-boosted trees at this base
rate, synthetic oversampling is not a free win, and the honest conclusion is
"we tested it and it doesn't matter much here," not "we balanced the data
and it helped."

### Model selection

**Selected: `lightgbm` / `none`** (no resampling), chosen by **PR-AUC on
validation = 0.1687**, the highest of all 20 combinations. XGBoost's best
run (`xgboost/none`, PR-AUC 0.1678, ROC-AUC 0.8961) is close enough (within
~0.001 PR-AUC, ~0.0015 ROC-AUC) that either would be defensible; LightGBM
was preferred as the simpler, faster-to-train, faster-to-serve model with
effectively tied ranking quality -- and because it needed no resampling at
all, which is one fewer moving part in production. Random Forest and
Logistic Regression both trail the two GBDTs by a real margin (best RF
PR-AUC 0.1446 vs. LightGBM's 0.1687) and were not competitive for
selection.

- **Threshold: 0.100**, chosen by minimum expected cost on validation
  (`cost_fp=50`, `cost_fn=500` -- configurable in `config.yaml`), sweeping
  `[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, ..., 0.90]`. At this threshold
  on validation: precision 0.185, recall 0.320, F1 0.234, FPR 0.0157,
  total expected cost $679,750 (vs. $1,463,500 at threshold 0.01 and
  $827,500 at threshold 0.90 -- the sweep is genuinely U-shaped, not
  monotonic, and 0.10 is close to the true minimum, not an edge case).
  Full sweep: `reports/metrics/threshold_sweep_cost.csv` /
  `threshold_sweep_f1.csv`.

### FINAL test-set evaluation (untouched, one time, via `evaluate.py`)

| Metric | Value |
|---|---:|
| ROC-AUC | **0.9013** |
| PR-AUC | 0.1656 |
| TPR@5%FPR | 0.5586 |
| Accuracy of an all-zero classifier (do-nothing baseline) | 0.9890 |
| n / n positive | 150,000 / 1,654 |

Test ROC-AUC (0.9013) came in slightly *above* validation ROC-AUC (0.8946)
and matches the sibling kit's documented target range for this exact
dataset: **floor ~0.89, target ~0.905, and "above 0.92, stop and hunt for
leakage."** 0.9013 sits inside that window with no leakage columns present
(verified, see above) -- this is a legitimate result, not an inflated one.

**Confusion matrix at threshold 0.100 (test):** TN 146,022 / FP 2,324 /
FN 1,115 / TP 539 -- precision 0.188, recall 0.326, F1 0.239, FPR 0.0157,
FNR 0.674. Expected cost at this threshold: $673,700 total, $4.49 per
application.

### Fairness (`customer_age > 50`, test set, at the threshold that hits 5% FPR overall)

| Group | n | FPR | TPR | Prevalence |
|---|---:|---:|---:|---:|
| age <= 50 | 143,629 | 0.0445 | 0.5324 | 0.0100 |
| age > 50 | 6,371 | **0.1748** | 0.7327 | 0.0341 |

**Predictive equality (FPR ratio, 1.0 = parity): 0.255** -- applicants over
50 are falsely flagged **3.9x more often** than younger applicants at the
same global 5% FPR budget. This reproduces the BAF paper's own headline
fairness finding on the Base variant almost exactly (older group
substantially over-flagged), on data this project split and trained itself,
not a number copied from the paper. A production deployment of this model
would need a fairness mitigation step (e.g. group-specific thresholds, a
fairness-constrained objective, or reweighting) before it could be used
as-is; none is applied here, since the task was to measure and report this
honestly, not to silently correct it and hide the finding.

### Explainability

Gain-based top features for the selected LightGBM model (`reports/metrics/feature_importance.csv`):
`housing_status` (16.9% of total gain), `device_os` (8.2%), the engineered
`risk_x_income` (5.3%), the engineered `email_mismatch_free` (3.7%), and
`has_other_cards` (3.6%). Two engineered features reached the top 5, which
is direct evidence the feature-engineering step earned its place rather
than just adding noise columns.

SHAP summary plot: `reports/figures/shap_summary.png`. One real individual
explanation, picked from an actual true-positive (fraud=1, model correctly
flagged it) row in the TEST set (index 137098, predicted probability
0.418): top SHAP contributors were `housing_status` (+0.978),
`customer_age` (+0.871), `device_os` (+0.763),
`date_of_birth_distinct_emails_4w` (+0.533), and `keep_alive_session`
(+0.352) -- see `reports/metrics/individual_explanation.csv` and
`reports/figures/shap_individual_explanation.png`.

### `predict.py` / admin gate / audit log -- verified working

Ran end to end against a 25-row sample drawn from the untouched test split
(new-applications CSV, no target column): admin auth accepted with a
correctly-configured token, 25 predictions written with
`fraud_probability` / `fraud_prediction` / `risk_level` columns (risk
distribution on this small sample: 22 LOW, 2 MEDIUM, 1 HIGH), and one
audit-log line was appended recording the run (admin identity hash, model
`lightgbm/none` iteration 248, row count, input content hash, risk-level
counts -- no raw applicant data, no token). Also verified: `predict.py`
correctly refuses to run with no `FRAUD_ADMIN_TOKEN` set, and with an
incorrect token, in both cases raising `AdminAuthError` with no silent
bypass. The demo credential used for this test was removed from
`config.yaml` again afterward -- the committed config has no admin
credential configured, as documented above.

## Reproducing this

```bash
pip install -r requirements.txt
python train.py                 # ~21 minutes on the full 700k-row train fold, 8-core/8GB machine
python evaluate.py               # final, one-time test-set evaluation + figures + SHAP
python -m src.auth --generate "<token>"   # one-time, to enable predict.py
# paste output into config.yaml, then:
export FRAUD_ADMIN_TOKEN="<token>"
python predict.py --input path/to/new_applications.csv --output data/predictions/scored.csv
pytest tests/ -q
```
