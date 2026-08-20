# Phase 15 (v2) — Production Architecture (Teammate's 18-Feature Pipeline)

This phase describes how the pipeline recommended in Phase 14 (v2) (`research_v2/12_final_model_selection.md`) would actually run in a bank, and — equally important — which parts of that architecture **already exist as working code in this repository** versus which are designs that have not been built. Every claim of the first kind names the file; every claim of the second kind says so explicitly.

**Scope note.** This is the architecture for the *teammate's 18-feature pipeline*, which is the client's designated final dataset and pipeline. The in-house 46-feature pipeline (`research/`, `src_research/`, `artifacts_research/`) is retained as historical reference and is not deployed. Where the two architectures differ materially, the difference is called out — most of the differences come from one root cause: this feature set is built from **global population statistics** (frequency counts) rather than per-account running history, which changes the real-time-scoring problem substantially, in both directions.

---

## 1. Pipeline Overview

```
                          ┌──────────────────────────────────────┐
                          │  STAGE 1 — INGESTION                  │
                          │  data/bank_transactions_data_2.csv    │
                          │  16 raw columns · 2,512 rows · 495    │
                          │  accounts · no fraud label            │
                          └───────────────┬──────────────────────┘
                                          │
                          ┌───────────────▼──────────────────────┐
                          │  STAGE 2 — FEATURE ENGINEERING        │
                          │  18 features, StandardScaler-applied  │
                          │  → artifacts_research/                │
                          │      features_teammate_merged.csv     │
                          │  verified by src_research_v2/         │
                          │      04_feature_verification.py       │
                          │  + RobustScaler (train-fit) applied   │
                          │    on top, shared by every model:     │
                          │    models/shared_robust_scaler.pkl    │
                          └───────────────┬──────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                                           │
        ┌───────────▼────────┐                       ┌──────────▼──────────┐
        │ STAGE 3a           │                       │ STAGE 3b             │
        │ OUT-OF-SAMPLE      │                       │ BATCH-ONLY MODELS    │
        │ CAPABLE (6)        │                       │ DBSCAN, HDBSCAN      │
        │ IF, LOF, OCSVM,    │                       │ no .predict()        │
        │ EE, K-Means, GMM   │                       │ must refit on the    │
        │ + Hybrid Ensemble  │                       │ full history         │
        │ (IF+LOF+GMM vote)  │                       │ → Phase 14 §3        │
        └───────────┬────────┘                       └───────────┬──────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                          ┌───────────────▼──────────────────────┐
                          │  STAGE 4 — ENSEMBLE SCORE             │
                          │  percentile aggregation over the      │
                          │  available members (skip-and-         │
                          │  renormalise, Phase 12 v2 §1.3)       │
                          │  src_research_v2/12_ensemble_scoring  │
                          │  → ensemble_scores_v2.csv             │
                          │  secondary: weighted-average score,   │
                          │  unbounded, for sigma thresholds      │
                          └───────────────┬──────────────────────┘
                                          │
                          ┌───────────────▼──────────────────────┐
                          │  STAGE 5 — TIERING                    │
                          │  ≥ 0.9510 (P99) → priority review     │
                          │  ≥ 0.8671 (P95) → standard review     │
                          │  else            → normal             │
                          │  src_research_v2/13_threshold_optim.  │
                          │  → threshold_analysis_v2.json         │
                          │  NO AUTOMATIC BLOCK TIER              │
                          └───────────────┬──────────────────────┘
                                          │
                          ┌───────────────▼──────────────────────┐
                          │  STAGE 6 — EXPLANATION                │
                          │  IF SHAP (TreeExplainer, exact, ~8s)  │
                          │  sole explainability output --        │
                          │  no other classical model is          │
                          │  SHAP-compatible without an expensive │
                          │  KernelExplainer (not used here)      │
                          │  precomputed for all 2,512 rows:      │
                          │  shap_isolation_forest_v2.csv         │
                          └───────────────┬──────────────────────┘
                                          │
                          ┌───────────────▼──────────────────────┐
                          │  STAGE 7 — INVESTIGATION DASHBOARD    │
                          │  Bank Transaction Fraud & Anomaly Detection (dashboard/)                   │
                          │  FastAPI + static HTML/CSS/JS         │
                          │  serves tiers, queue, SHAP, model     │
                          │  comparison — now wired to THIS       │
                          │  pipeline's artifacts (§7)            │
                          └──────────────────────────────────────┘
```

**Read the diagram as three bands.** Stage 1–2 are batch data preparation. Stages 3–6 are the scoring pipeline, and their most consequential property is that **Stage 3b breaks real-time scoring for the full 8-model ensemble** (Phase 14 v2 §3). Stage 7 is the only component in the entire project that a human being actually looks at.

**No deep-learning model remains in this pipeline.** Autoencoder, VAE and LSTM-AE were removed; the pipeline now runs 8 classical unsupervised detectors (Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, DBSCAN, HDBSCAN, K-Means, GMM) plus Model 9, the Hybrid Ensemble, redefined as **Isolation Forest + LOF + GMM majority vote (≥2 of 3)** — GMM's top-5%-by-negative-log-likelihood flag now stands in for the removed Autoencoder's flag in that vote, with the same ≥2-of-3 logic and the same `hybrid_vote_count` / `flag_majority` output columns.

---

## 2. Stage 1 — Data Ingestion

**What exists.** `data/bank_transactions_data_2.csv` — 2,512 transactions, 495 accounts, 16 raw columns, no fraud label. Loaded once by `src_research_v2/04_feature_verification.py` for cross-checking against the engineered file, and by `src_research_v2/10_evaluation.py` for the business-evaluation walkthrough (which needs raw dollar amounts, not z-scores).

**What a production ingestion layer must add**, none of which is built here:

- **Schema contract.** The 18-feature matrix's column order is asserted at load time (`config_research_v2.py::load_features_v2()` asserts `list(df.columns) == ID_COLS + FEATURE_COLS_V2` and raises otherwise). That assertion is the right *idea* but the wrong *layer* — it catches a column reorder after the features have already been built. A production ingest needs the same contract on the 16 **raw** columns, before feature engineering runs.
- **`PreviousTransactionDate` must be dropped at ingest, permanently.** Phase 1 (`research/01_business_understanding.md`) established that every value in this column clusters within minutes of a single 2024-11-04 export moment — it is a snapshot artifact, not behavioural history. It is not used by any of the 18 features, which is correct, and the ingest layer should drop it explicitly rather than leaving a plausible-looking timestamp column available for someone to reach for later.
- **Late/out-of-order arrival policy.** Not applicable to a static research CSV; unavoidable in production, and it interacts badly with this feature set — see §3.2.

---

## 3. Stage 2 — Feature Engineering

### 3.1 Batch path — exists, and is verified rather than assumed

The 18 features arrive already engineered and StandardScaler-scaled in `artifacts_research/features_teammate_merged.csv`. `src_research_v2/04_feature_verification.py` re-derives and checks them against the raw CSV rather than trusting them, and its outputs are in `artifacts_research_v2/phase5_6_feature_verification.json`:

- 0 missing cells (0/70,336), 0 duplicate rows, 0 duplicate `TransactionID`s.
- Row alignment against the raw CSV confirmed exactly (`(raw["TransactionID"].values == df["TransactionID"].values).all()` → **True**).
- All 11 continuous columns verified mean≈0 / std≈1.0002 (the `ddof` artifact is explained in Phase 5 v2 §2.2); the 7 binary/dummy columns verified to remain 0/1.
- `amount_to_balance_ratio` verified to track the raw `TransactionAmount / AccountBalance` ratio at **r = 0.9467**.
- `high_amount_transaction` verified empirically to be a **global top-5%-by-raw-amount flag** (min flagged raw amount $878.63 vs. max unflagged $877.81, against a dataset 95th percentile of $878.18).
- One-hot baselines recovered: `TransactionType` drops Credit, `Channel` drops ATM, `CustomerOccupation` drops Doctor.

A second, model-facing scaling pass (`RobustScaler`, fit on the 2,009-row training split only) is applied on top and saved as `artifacts_research_v2/models/shared_robust_scaler.pkl`, shared identically by every model so that the cross-model Spearman/Jaccard comparisons are like-for-like.

**These verifications are the thing to port into production, not just the feature code.** Every one of them is a cheap assertion that would catch a real, silent failure — see Phase 16 (v2) §6.3.

### 3.2 Real-time path — this is the largest gap in the architecture, and it is a *different* gap than the in-house pipeline's

Four of the 18 features — `account_frequency`, `device_frequency`, `ip_frequency`, `merchant_frequency` — and a fifth, `Location_FE`, are **global counts or proportions computed over the whole dataset**. Phase 5 (v2) §1 flags the consequence directly and it is restated here because it is architecturally decisive:

> a global (whole-dataset) count includes each account's own **future** transactions, not just prior ones — fine for offline research/anomaly-scoring on a static dataset, but would need to become a prior-only running count before any live deployment.

This has three separate implications, which are often conflated:

1. **It is a leakage problem for any evaluation that pretends to be point-in-time.** No Phase 8–13 (v2) result claims to be point-in-time, so nothing already reported is invalidated. But a backtest that scored transactions "as of" their transaction date using these features would be measuring something it should not have known.
2. **It is a computability problem at inference.** A single fresh transaction has no `account_frequency` until you decide what population to count over. This is not a small implementation detail — it is why the Bank Transaction Fraud & Anomaly Detection What-if Simulator had to be redesigned rather than repointed (§7.3).
3. **It is a *drift* problem, and this is the one that differs most from the in-house pipeline.** A frequency encoding is a mapping from a category to a number derived from the population. When the population of devices/merchants/IPs changes — new devices onboard, a merchant closes, an IP range is reallocated — **every previously-seen category's encoded value shifts, even for categories whose own behaviour did not change at all.** The in-house pipeline's equivalent concern was a temporal-export artifact (`PreviousTransactionDate`); this pipeline's is population churn in the frequency encodings. Phase 16 (v2) §2 builds the monitoring around exactly this.

**What a production feature layer must do, and what it costs:**

| Feature | Production form | Cost |
|---|---|---|
| `TransactionAmount` | `StandardScaler(log1p(raw amount))` — **note the log transform**, recovered exactly in Phase 14 (v2) §5 Inconsistency 4 and not stated in Phase 5 (v2). Frozen mean/std | Trivial |
| `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` | Direct from the transaction record, scaled with a **frozen** training-fit `StandardScaler` (not refit per batch) | Trivial |
| `amount_to_balance_ratio` | `StandardScaler(log1p(amount / (balance + 1)))` — recovered **exactly** in Phase 14 (v2) §5 Inconsistency 5, correcting Phase 5 (v2) §2.3's conclusion that no exact formula was recoverable. Two fields on the same record, then frozen-scaled | Trivial |
| `high_amount_transaction` | A **frozen dollar threshold** (`raw amount > $878.179`, the training 95th percentile), not a recomputed percentile | Trivial, but see Phase 16 (v2) §2.4 — a percentile recomputed per batch is a silent-drift bug |
| `account_frequency`, `device_frequency`, `ip_frequency`, `merchant_frequency` | **Prior-only running counters** in a key-value store, incremented on write, read at score time — not a whole-dataset `groupby` | The real engineering work. Needs a feature store with per-key counters and a defined lookback window |
| `Location_FE` | A **frozen lookup table** from location → training-set proportion, with a documented default for unseen locations | Small, but the unseen-category default must be chosen deliberately, not left to `NaN` |
| `TransactionType_Debit`, `Channel_*`, `CustomerOccupation_*` | One-hot against **frozen** category lists (Credit / ATM / Doctor as dropped baselines) | Trivial, but an unseen category must fail loudly rather than silently encode as all-zeros |

**Honest assessment: none of this real-time feature layer is built.** What exists is the batch path and its verifications. The gap is smaller than it looks for thirteen of the eighteen features and larger than it looks for the five frequency-derived ones.

**One thing that improved while this phase was being written.** Phase 5 (v2) treated two of the features as only approximately recoverable from the raw columns. Both were subsequently recovered **exactly** (Phase 14 v2 §5, Inconsistencies 4 and 5): `TransactionAmount` is `StandardScaler(log1p(raw amount))` and `amount_to_balance_ratio` is `StandardScaler(log1p(amount / (balance + 1)))`. With those two settled, **all 18 features are exactly reproducible from the 16 raw columns**, which turns the table above from a partly-inferred specification into a complete one. It is also what makes the Bank Transaction Fraud & Anomaly Detection scenario simulator (§7.3) honest rather than approximate — it computes the real feature vector, not a stand-in.

**One genuine architectural advantage over the in-house pipeline, worth stating because it cuts the other way.** The in-house 46-feature set needs a per-account *history scan* at inference time (expanding means, rolling windows, time-since-last-transaction, novelty flags) — a read of every prior transaction for that account, or an equivalently complex incrementally-maintained state. This 18-feature set needs only **counter reads**: five integer lookups and a frozen scaler. Counters are dramatically cheaper to maintain, cheaper to keep consistent across replicas, and cheaper to backfill than per-account rolling statistics. The teammate's feature set is, in a real and concrete sense, the more *deployable* of the two — which is a genuine point in its favour that sits alongside, and does not cancel, Phase 5 (v2) §3's finding that it is the less *capable* of the two.

---

## 4. Stage 3 — Model Scoring

Nine models were built (Phase 8 v2): 8 classical unsupervised detectors plus the Hybrid Ensemble (Model 9). No deep-learning model is trained in this pipeline. They partition into two operationally distinct groups, and the partition — not the model quality — is what drives the architecture.

**3a. Out-of-sample capable (6 base models + the Hybrid Ensemble).** Isolation Forest, LOF (`novelty=True`), One-Class SVM, Elliptic Envelope, K-Means, GMM — all six fit on the 2,009-row training split and scored all 2,512 rows out-of-sample. The Hybrid Ensemble (IF + LOF + GMM majority vote) inherits out-of-sample capability from its three components. Artifacts: `artifacts_research_v2/models/*.pkl`, reloadable directly with `joblib.load`.

**3b. Batch-only (2 models).** DBSCAN and HDBSCAN have no out-of-sample `.predict()` in this build (Phase 12 v2 §0; HDBSCAN's `prediction_data=True` was not set at fit time). **This is the single fact that decides batch versus real-time for the full ensemble**, and Phase 14 (v2) §3 sets out the three options. Option B (drop both, aggregate over the remaining set) is the recommended path; Option C (re-enable HDBSCAN via `approximate_predict`) is a stronger follow-up here than it was in-house, because HDBSCAN is a materially more useful member on this feature set (8.88% flagged rate vs. the in-house pipeline's far higher noise rate, and a strong mean flagged-set agreement with the rest of the field).

**Measured scoring cost at n=2,512 × 18 features**, from `artifacts_research_v2/model_summary_classical.json` (fit + score, all configs tried):

| Model | Time (all configs) | Model | Time (all configs) |
|---|---:|---|---:|
| One-Class SVM (5 configs) | 0.78s | Elliptic Envelope (3 configs) | 2.76s |
| DBSCAN (9 configs) | 0.75s | LOF (5 configs) | 5.38s |
| HDBSCAN (4 configs) | 1.25s | Isolation Forest (5 configs) | 4.10s |

Every one of these is negligible at this scale. **Compute is not the constraint on this architecture; out-of-sample capability and artifact lifecycle are.**

---

## 5. Stage 4 — Ensemble Score

`src_research_v2/12_ensemble_scoring.py` combines the **8 classical models only** (Hybrid Ensemble excluded as an input, to avoid double-counting IF/LOF/GMM) into four scores for all 2,512 rows in `artifacts_research_v2/ensemble_scores_v2.csv`. The production score is **`ensemble_percentile_average`** (Phase 12 v2 §3, upheld by Phase 14 v2 §4). Distribution: mean 0.5000, std 0.2242, min 0.0576, max 0.9967.

Two properties make it the right choice for a production interface, both already verified:

- **It handles a missing member natively.** Percentile aggregation skips absent scores and renormalises over what is available (§1.3) — which is exactly the mechanism that makes Phase 14 (v2)'s Option B (drop DBSCAN/HDBSCAN) implementable without redesigning the score. This is not a lucky coincidence; it is why the strategy was preferred.
- **It is bounded in (0,1),** so a frozen reference distribution can be monitored directly and thresholds are comparable across batches of different sizes. Rank (Borda) aggregation, its near-mathematical twin (ρ=0.9999), does **not** have this property — a Borda sum's scale depends on how many rows were ranked together.

**Secondary score, computed in parallel at zero additional model cost: `ensemble_weighted_average`.** It is unbounded, which is the one thing the percentile score cannot do — Phase 13 (v2) §3 measured that mean+3σ and Q3+1.5×IQR both flag **zero** transactions on the percentile score (their thresholds, 1.1725 and 1.2120, exceed the score's maximum of 0.9967) while flagging 32 and 75 respectively on the weighted average. Keep both, and use the weighted average whenever a stakeholder wants a "three sigma" framing.

**A caution that must travel with any reduced-member deployment.** The published `ensemble_percentile_average` is an 8-model score. Dropping any further members would make it a **different score**, and Phase 13 (v2)'s thresholds would not transfer unrevalidated. The revalidation recipe is cheap (Spearman + top-5% Jaccard against the published score, the same two measures Phase 12 v2 §2 already uses) should the member set ever shrink further.

---

## 6. Stage 5 — Alert Generation

| Tier | Rule | Volume in this sample | Daily load at this sample's rate |
|---|---|---:|---:|
| **Priority review** | `ensemble_percentile_average` ≥ **0.9627** (P99) | 26 (1.04%) | ~0.07 transactions/day |
| **Standard review** | ≥ **0.8852** (P95) | 126 (5.02%) | ~0.35 transactions/day |
| Normal | below P95 | 2,386 (94.98%) | — |

Source: `artifacts_research_v2/threshold_analysis_v2.json`, `threshold_flagged_counts_v2.csv`.

**No automatic block tier**, and this is a deliberate, defensible refusal rather than an omission. Phase 13 (v2) §1 established that a cost-minimising threshold requires counting false negatives, which requires knowing which *unflagged* transactions are fraud — unknowable without a label. The v1 pipeline shipped a block tier derived from a cost sweep against supervised proxy labels; that sweep cannot be reproduced here, and blocking a customer's transaction on a score whose false-negative behaviour has never been measured is not something to hand a bank.

**The daily figures above are not a capacity plan.** This dataset covers 365 days at 6.88 transactions/day. The number that generalises is the *ratio* — P95 flags roughly 4.8× as many transactions as P99 — not the absolute counts.

**Alert content, per Phase 14 (v2) §4:** every alert carries the ensemble score, the tier, the per-model percentile breakdown, and the **Isolation Forest SHAP explanation** — the sole explainability output in this pipeline (§7.1). No other classical model here is naturally SHAP-compatible without an expensive KernelExplainer, which this codebase does not use.

---

## 7. Stage 6/7 — Explanation and the Bank Transaction Fraud & Anomaly Detection Investigation Dashboard

### 7.1 Explanation artifacts

The explainer was run over all 2,512 rows and saved, so serving an explanation is a lookup, never a recomputation:

- `artifacts_research_v2/shap_isolation_forest_v2.csv` — `shap.TreeExplainer`, **exact**, ~8s for the full dataset, sign-flipped so positive = pushes anomaly score up (verified against `score_samples`, ρ=1.0000 on a 200-row spot-check).
- `shap_local_explanations_v2.json` — the four worked local explanations.

This is the **sole** explainability output in this pipeline. There is no cross-model SHAP comparison and no `shap_global_importance_comparison_v2.csv` — the Autoencoder that the comparison used to run against was removed, and no other remaining classical model (LOF, OCSVM, Elliptic Envelope, DBSCAN, HDBSCAN, K-Means, GMM) is naturally SHAP-compatible without an expensive KernelExplainer, which this codebase does not use anywhere.

### 7.2 Bank Transaction Fraud & Anomaly Detection — what it is, and what changed

Bank Transaction Fraud & Anomaly Detection (`dashboard/`) is a browse-first fraud-analytics console: a FastAPI backend (`dashboard/backend/api_server.py`) that loads artifacts and serves both the API and a dependency-free static frontend (`dashboard/frontend/`, plain HTML/CSS/JS, no build step, no CDN, no external fonts — it runs fully offline). It retrains nothing.

**Bank Transaction Fraud & Anomaly Detection was originally wired to the v1 pipeline** (`artifacts/labeled.csv`, `xgb_model.json`, `reference.pkl`, `thresholds.json`), showing risk tiers derived from a supervised XGBoost model trained to reproduce a 4-detector unsupervised ensemble. **It has now been repointed at this pipeline.** The visual design, branding and page structure are unchanged — this was a data-source swap, not a redesign.

What Bank Transaction Fraud & Anomaly Detection now serves (as of this pipeline's artifact regeneration -- the dashboard's own backend/frontend code is repointed separately, outside the scope of this phase):

| Surface | Source (this pipeline) |
|---|---|
| Risk score | `ensemble_percentile_average` from `artifacts_research_v2/ensemble_scores_v2.csv` |
| Risk tier | Phase 13 (v2) cutoffs: ≥0.9627 → priority, ≥0.8852 → standard review, else normal |
| Per-transaction explanation | Precomputed **Isolation Forest** SHAP rows (sole explainability output) from `shap_isolation_forest_v2.csv` |
| Per-model detail | All 9 models' (8 classical + Hybrid Ensemble) per-row scores/flags from `model_scores_all.csv` |
| Model Comparison page | The 9-model comparison: flagged rates, internal validity (Phase 10 v2 §1), bootstrap stability (Phase 10 v2 §2), ensemble weights and strategy agreement over the 8 classical models (Phase 12 v2) |
| Explainability page | Global mean\|SHAP\| for Isolation Forest only, and the Phase 13 (v2) threshold sweep |
| Raw transaction fields | `data/bank_transactions_data_2.csv`, joined on `TransactionID` |

**Nothing on the dashboard is hand-typed from a report.** Every number is read from an artifact at startup, which is a deliberate property: it means a stale artifact produces a visibly stale dashboard rather than a dashboard that silently disagrees with the pipeline behind it.

### 7.3 The What-if Simulator — why it had to change, not just be repointed

The v1 simulator let a user type a brand-new hypothetical transaction and scored it via `fe_utils.transform_new`. **That is not honestly possible on this feature set**, for the reason in §3.2: `account_frequency`, `device_frequency`, `ip_frequency`, `merchant_frequency` and `Location_FE` are population statistics. A brand-new transaction has no `device_frequency` until you decide what population to count over, and inventing one would produce a confident-looking score built on a fabricated input.

Two options were considered (both were on the table; the choice is recorded here rather than left implicit):

- **(a)** Anchor the simulation to a **real existing account** and look up that account's true historical frequency values, letting the user vary only the fields that genuinely belong to a single transaction.
- **(b)** Disable the tab and explain why.

**Option (a) was implemented.** The rebuilt simulator requires the user to select an existing `AccountID`; it then loads that account's actual `account_frequency`, and the actual `device_frequency` / `ip_frequency` / `merchant_frequency` / `Location_FE` of the device, IP, merchant and location the user selects **from those that exist in the data** — real values, never synthesised. The user varies `TransactionAmount`, `TransactionType`, `Channel`, `Occupation`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` and `CustomerAge`.

The remaining features are then computed **exactly**, using the frozen training constants and the transformations recovered in Phase 14 (v2) §5: `TransactionAmount` = `StandardScaler(log1p(amount))`, `amount_to_balance_ratio` = `StandardScaler(log1p(amount / (balance + 1)))`, `high_amount_transaction` = `amount > $878.179`, and the four remaining continuous columns by their frozen means and standard deviations. This is not an approximation — the same transformation pipeline was verified to reproduce all 2,512 rows of `features_teammate_merged.csv` to within floating-point tolerance, and the resulting 18-vector is scored through the saved `shared_robust_scaler.pkl` + `isolation_forest.pkl` artifacts, which reproduce the published `score_isolation_forest` column to numerical precision.

The tab is relabelled **"Account Scenario Simulator"** and carries an explicit in-UI note stating that free-form new-transaction simulation is not meaningful on a feature set built from population-level frequency statistics, and that the frequency inputs are real historical values, not user-supplied. It is scored by Isolation Forest only — **not** by the full ensemble — because DBSCAN and HDBSCAN cannot score an unseen row at all (§4), and presenting a "full ensemble score" for a hypothetical would be a fabrication. The UI says this too.

This is the honest version of the feature. A simulator that silently invented frequency values would be the kind of thing that survives a demo and fails a model-risk review.

### 7.4 Running it

```
cd dashboard
python -m uvicorn backend.api_server:app --port 8000
# then open http://127.0.0.1:8000/
```

The backend serves the frontend itself via FastAPI `StaticFiles`, so there is no CORS configuration and nothing else to start. Requirements beyond the pipeline's own (`requirements.txt`): `fastapi`, `uvicorn`. No paid services, no external network calls.

### 7.5 The one thing Bank Transaction Fraud & Anomaly Detection produces that nothing else in this project does

Investigator decisions. `dashboard/backend/queue_state.json` accumulates Approve / Escalate / Block actions per transaction. **This is the only mechanism in the entire project capable of generating labelled data**, and every limitation in every report of this pipeline traces back to not having any. It should be treated as a first-class data product from day one: durably stored, versioned, and joined back to the score that produced the alert — because a year of it is what makes a supervised model, a real precision/recall number, and a genuine cost-optimised threshold possible.

---

## 8. Batch vs. Real-Time for This Specific Use Case

**Recommendation: nightly batch scoring, with a real-time path deferred until the frequency-counter feature store exists.**

The reasoning is specific to this pipeline, not generic:

1. **The full 8-model ensemble is batch-only by construction** (§4, Phase 14 v2 §3). Real-time requires committing to a reduced member set and revalidating the thresholds — work that has not been done.
2. **The volume does not demand it.** 6.88 transactions/day in this sample. Even scaled by three orders of magnitude, a nightly batch over the full history costs seconds of model time (§4).
3. **The feature layer is the blocker, not the models.** Five of eighteen features need prior-only counters backed by a feature store (§3.2). Until that exists, "real-time" would mean recomputing global frequencies per request, which is both wrong and slow.
4. **The output is a human review queue, not a transaction decision.** With no block tier (§6), nothing about this system needs to complete inside a payment authorisation window. A transaction flagged at 02:00 and reviewed at 09:00 loses nothing that a synchronous score would have gained.

**When real-time becomes necessary:** the day a block tier is introduced. That day should not come before the system has been validated against investigator-labelled outcomes (§7.5).

---

## 9. Model Versioning and Artifact Storage

**What exists today** — a flat, unversioned directory:

```
artifacts_research_v2/
  models/            isolation_forest.pkl, lof.pkl, ocsvm.pkl, elliptic_envelope.pkl,
                     dbscan.pkl, hdbscan.pkl, kmeans.pkl, gmm.pkl,
                     shared_robust_scaler.pkl
  model_scores_all.csv, model_summary_classical.json, model_comparison_summary.json
  ensemble_scores_v2.csv, ensemble_weights_v2.json
  shap_isolation_forest_v2.csv
  threshold_analysis_v2.json, threshold_flagged_counts_v2.csv
  internal_validity_metrics_v2.csv, stability_bootstrap_jaccard_v2.csv
```

No deep-learning artifacts remain -- `autoencoder.pt`, `autoencoder_scaler.pkl`, `autoencoder_config.json`, `vae.pt`/`vae_config.json`, `lstm_ae.pt`/`lstm_ae_config.json`, `shap_autoencoder_v2.csv` and `shap_global_importance_comparison_v2.csv` were all deleted along with the models that produced them.

This is adequate for research and inadequate for a bank. **What has to change:**

| Requirement | Why it matters here specifically |
|---|---|
| **Immutable, versioned artifact sets** | Phase 10 (v2) §2 measured 0.37–0.60 flagged-set Jaccard between *retrains of the same model on resampled data*. A retrain materially changes who gets reviewed. Reconstructing why a specific transaction was flagged six months ago requires the exact artifact set that flagged it. |
| **Scaler versioning alongside models** | `shared_robust_scaler.pkl` is as load-bearing as any model. A model paired with the wrong scaler produces plausible, wrong scores with no error. |
| **Frozen threshold constants stored as artifacts, not code** | The $878.18 `high_amount_transaction` boundary and the `Location_FE` lookup table are training-derived constants. If they are ever recomputed per batch instead of frozen, the pipeline drifts silently (Phase 16 v2 §2.4). |
| **The 8-member manifest** | The ensemble score is defined by *which* models went into it (the 8 classical detectors; the Hybrid Ensemble is deliberately excluded as an input). A reduced-member `ensemble_percentile_average` is a different score with the same column name. The member list must be recorded with the score. |
| **Score lineage** | Every alert should record the artifact-set version, the member manifest, and the threshold values that produced it. |

---

## 10. What Has to Change to Reach Production Scale

### 10.1 Feature engineering: from whole-dataset `groupby` to incremental counters

The five frequency-derived features are currently global aggregations over a 2,512-row DataFrame. At bank scale they become counter reads and writes against a key-value store, with a defined lookback window and a defined policy for unseen keys. **This is the single largest piece of unbuilt engineering in the architecture** (§3.2), and it is the prerequisite for everything real-time.

The good news, restated because it is genuinely favourable: counters are much simpler than the in-house pipeline's per-account expanding/rolling statistics. This is the easier of the two feature sets to productionise.

### 10.2 Models that hit a wall

| Model | Wall | Fix |
|---|---|---|
| One-Class SVM | QP solve, roughly O(n²)–O(n³) in support vectors; a concern past ~50k–100k rows | Subsample, or use a linear/Nyström approximation — either changes the score and requires revalidation |
| LOF | O(n²) neighbour search by default | An approximate-nearest-neighbour index; a real engineering task, not a config change |
| DBSCAN, HDBSCAN | Cannot score unseen rows at all | Phase 14 (v2) §3, Options B/C |
| Isolation Forest, K-Means, GMM, Elliptic Envelope, Hybrid Ensemble | None at realistic scale | — |

### 10.3 Everything that must be refit, not ported

- **Every scaler.** `StandardScaler` (the teammate's upstream pass) and `RobustScaler` (this pipeline's model-facing pass) both encode this dataset's means, standard deviations, medians and IQRs. On a different population they are wrong.
- **`Location_FE`'s lookup table** and **`high_amount_transaction`'s $878.18 threshold** — both training-derived constants of exactly the kind that fail silently.
- **Every model's `contamination` / `nu` / noise-rate setting.** The 5% figure used throughout is a documented *assumption*, not a measured fraud rate. It should be revisited against real base rates.
- **The ensemble weights**, if Weighted Average is retained — they are derived from a Spearman matrix that changes with the data.
- **Phase 13 (v2)'s thresholds**, if the member set changes (§5).

### 10.4 What does not change

The methodology. The verification pattern (§3.1), the two-explainer requirement (§6), the refusal to ship a block tier without a label (§6), the skip-and-renormalise ensemble rule (§5), and the honest reporting of what has and has not been measured — all of it transfers to any scale.

---

## 11. Handoff to Phase 16

- **Deployed score**: `ensemble_percentile_average` over the 8 classical models, Option B member set, nightly batch.
- **Deployed thresholds**: 0.9627 (priority) / 0.8852 (standard). No block tier.
- **Explanation layer**: Isolation Forest SHAP (sole explainability output), precomputed.
- **Investigator surface**: Bank Transaction Fraud & Anomaly Detection, now reading this pipeline's artifacts (§7.2), with an honestly-rebuilt Account Scenario Simulator (§7.3).
- **The three things Phase 16 must monitor hardest**, all identified above: (1) frequency-encoding drift as the device/merchant/IP population churns (§3.2), (2) the frozen training-derived constants that fail silently if recomputed (§9), and (3) silent member dropout in an ensemble whose score is defined by its member list (§9).
