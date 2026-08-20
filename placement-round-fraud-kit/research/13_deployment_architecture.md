# Phase 15 — Production Architecture

This is the deployment design for **this** system — every stage below names the file that implements it, or names precisely what does not exist yet. It is not a generic reference architecture with this project's labels pasted onto it.

**Scope boundary, stated first rather than buried in a closing caveat.** What follows is a research-prototype architecture proven at **2,512 transactions, 495 accounts, 364 days, 6.90 transactions/day** (Phase 13 §4). The original brief this work descends from describes roughly **1M rows** (`LIMITATIONS.md`). Nothing here has been run, load-tested, or profiled beyond the scale above. §7 sets out specifically what breaks at production volume and what has to be rebuilt rather than resized — that section is part of the design, not an apology attached to it.

---

## 1. Pipeline Overview

```
                            ┌──────────────────────────────────────────────┐
                            │  OFFLINE / TRAINING  (periodic, batch only)  │
                            └──────────────────────────────────────────────┘
   data/bank_transactions_data_2.csv
              │
              ▼
   ┌──────────────────────┐   src_research/04_feature_engineering.py
   │  1  DATA INGESTION   │──▶ 2  FEATURE ENGINEERING (batch, fit)
   │  config_research.py  │      → artifacts_research/features_v2.csv (2,512 × 48)
   │  ::load_raw()        │      → RobustScaler fit on the 2,009-row train split
   └──────────────────────┘        → models/shared_robust_scaler.pkl
              │                                    │
              │                                    ▼
              │                     ┌──────────────────────────────┐
              │                     │  MODEL TRAINING              │
              │                     │  07_models_classical.py      │
              │                     │  08_models_deep.py           │
              │                     │  → artifacts_research/models/│
              │                     └──────────────────────────────┘
              │                                    │
   ═══════════╪════════════════════════════════════╪══════════════════════════
              │        ONLINE / SCORING  (per batch or per transaction)
              ▼                                    ▼
   ┌──────────────────────┐          ┌───────────────────────────────┐
   │  2' FEATURE          │          │  3  MODEL SCORING             │
   │     ENGINEERING      │─────────▶│     6 out-of-sample classical │
   │     (transform)      │  46 cols │     models + Hybrid Ensemble  │
   │  batch: 04_fe.py     │  scaled  │     (IF+LOF+GMM), sign-       │
   │  live:  fe_utils     │          │     normalised so higher =    │
   │         ::transform_new (GAP §3)│     more anomalous → score_<model> × 6 + vote │
   └──────────────────────┘          └───────────────────────────────┘
                                                   │
                                                   ▼
                                     ┌───────────────────────────────┐
                                     │  4  ANOMALY SCORE             │
                                     │     percentile aggregation    │
                                     │     12_ensemble_scoring.py    │
                                     │  → ensemble_percentile_average│
                                     │     ∈ (0,1)                   │
                                     │  ⊦ Weighted Average (parallel,│
                                     │     unbounded, for σ rules)   │
                                     └───────────────────────────────┘
                                                   │
                                                   ▼
                                     ┌───────────────────────────────┐
                                     │  5  ALERT GENERATION          │
                                     │     13_threshold_optimization │
                                     │  ≥ 0.9145 → PRIORITY REVIEW   │
                                     │  ≥ 0.8406 → STANDARD REVIEW   │
                                     │  <  0.8406 → no alert         │
                                     │     (no automatic block tier) │
                                     └───────────────────────────────┘
                                                   │
                     ┌─────────────────────────────┴──────────────────┐
                     ▼                                                ▼
        ┌────────────────────────────┐                 ┌──────────────────────────┐
        │  EXPLANATION LAYER         │                 │  6  INVESTIGATION        │
        │  shap_isolation_forest.csv │────────────────▶│     DASHBOARD — "Bank Transaction Fraud & Anomaly Detection"  │
        │  (sole SHAP family; no     │  per-txn attrib │  dashboard/backend/      │
        │   remaining model here     │                 │    api_server.py         │
        │   is SHAP-compatible)      │                 │  FastAPI + static JS     │
        └────────────────────────────┘                 │                          │
                                                       │  6 pages, offline        │
                                                       └──────────────────────────┘
                                                                    │
                                                                    ▼
                                                       analyst verdict → queue_state.json
                                                       (the only human-labelled data
                                                        this system ever produces — §8)
```

---

## 2. Stage 1 — Data Ingestion

| | |
|---|---|
| **Implemented by** | `src_research/config_research.py::load_raw()` (research path) and `src/fe_utils.py::load_raw(path)` (v1 path) |
| **Input contract** | The 16 raw columns listed in Phase 2, with `TransactionDate` and `PreviousTransactionDate` parsed as `%d-%m-%Y %H:%M` |
| **Output** | A DataFrame with `TransactionDate_parsed` / `PreviousTransactionDate_parsed` added |

Two ingestion rules are already established by the analysis and must be enforced at the boundary, not rediscovered downstream:

- **`PreviousTransactionDate` is dropped, not used.** Phase 2 §4 found 7 unique values across all 2,512 rows spanning 6.0 minutes on 2024-11-04 — a single bulk-export timestamp stamped onto every row, ten months after the latest `TransactionDate`. Every recency and velocity feature is derived from `TransactionDate` sorted per `AccountID` instead. A production ingester should either receive a real per-account previous-transaction timestamp or omit the column entirely; silently accepting a constant here would let a dead feature into the model matrix.
- **Chronological sort is part of the contract, not a convenience.** Everything downstream assumes rows sorted `AccountID → TransactionDate → TransactionID` (Phase 5, and `src/fe_utils.py::fit_engineer`). Every expanding-window and prior-only feature is leakage-safe *only* under that ordering. Phase 5 verified this rather than assuming it, by recomputing `TimeSinceLastTxn` independently and checking it matched `artifacts/features.csv` row-for-row (**confirmed: MATCH**). That check should be a startup assertion in production, not a one-time validation — the Bank Transaction Fraud & Anomaly Detection backend already does the equivalent, raising a `RuntimeError` on a row-count or row-alignment mismatch between the raw CSV, `labeled.csv` and `anomaly_votes.csv` (`dashboard/backend/api_server.py:133–141`).

**Data-quality gates worth enforcing at ingestion**, all derived from Phase 3, which found the training data clean on every one of them (0 missing cells of 45,216; 0 duplicate rows; 0 duplicate `TransactionID`s; 0 near-duplicates under the same-account/same-amount/same-minute rule): reject on duplicate `TransactionID`, alert on any missing cell, and alert on near-duplicates. Phase 3's reasoning for the last one is the operationally important part — near-duplicate ledger entries would inflate the velocity/burst features and manufacture a fraud signal out of a data bug.

**No outlier removal or capping at ingestion.** Phase 3's decision is explicit and carries into production unchanged: in an unsupervised fraud system the outliers are the candidate signal, and removing them before modelling deletes the population the system exists to find.

---

## 3. Stage 2 — Feature Engineering

**46 features, one ordered schema.** The authoritative list is the non-ID columns of `artifacts_research/features_v2.csv`, in the order `07_models_classical.py::load_and_split()` reads them (`[c for c in df.columns if c not in ("TransactionID", "AccountID")]`) — 46 names in a fixed order. That ordering is a binding contract between this stage and every model artifact; the models were fit on columns in that order and will silently produce garbage if handed the same 46 columns permuted. It must be versioned with the models (§6), not treated as documentation.

### 3.1 Batch path — implemented and verified

`src_research/04_feature_engineering.py`, composed of six functions that each own one feature family:

| Function | Produces | Leakage safety |
|---|---|---|
| `velocity_features()` | `Velocity_1D_Count`, `Velocity_7D_Count` | `groupby('AccountID').rolling('1D'/'7D', closed='left')` — `closed='left'` excludes the current row |
| `rolling_and_expanding_features()` | `Expanding_{Mean,Median,Std,Min,Max}Amount`, `Rolling3_{Mean,Std}Amount` | `shift()` before the window opens, so only strictly prior rows are visible |
| `ratio_and_deviation_features()` | `Amount_to_Balance_Ratio`, `Amount_to_RollingMean_Ratio`, `Amount_minus_Expanding{Mean,Median}`, `Amount_ZScore_Account` | Inherits prior-only safety from the expanding stats |
| `cyclical_time_features()` | `Hour_sin/cos`, `DOW_sin/cos` | Same-row, no temporal dependency |
| `behavioral_features()` | `CustomerTxnCountSoFar`, `SpendCV_Account`, `ElevatedLoginFlag`, `ATM_Credit_InteractionFlag` | `cumcount()` counts only prior rows in sorted order |
| `network_proxy_features()` | `DeviceSharedAccounts_Prior`, `IPSharedAccounts_Prior`, `MerchantSharedAccounts_Prior` | `_prior_distinct_other_accounts()` — an explicit prior-only accumulation per key, own account excluded |

One implementation detail matters enough in production to name: `Amount_ZScore_Account` divides by `Expanding_StdAmount` floored at **5% of the dataset-wide `TransactionAmount` standard deviation ($291.95 → a $14.60 floor)**, not at an arbitrary epsilon (`04_feature_engineering.py:119`). Phase 6 §7.3 documents why — an earlier build used a near-zero epsilon and accounts with 2–3 near-identical prior transactions drove that one feature into the hundreds of millions, swamping every other feature's contribution to the autoencoder's loss. **The floor is a constant derived from the training data**, which makes it a versioned model parameter and a monitoring target (Phase 16 §2.4), not a hard-coded number.

Scaling: `RobustScaler`, fit on the training split only, persisted as `artifacts_research/models/shared_robust_scaler.pkl` and shared by all 9 models — there is no per-model scaler anymore now that the Autoencoder (which previously kept its own copy, `autoencoder_scaler.pkl`) has been removed from this pipeline. Phase 6 §6.2 chose it over `StandardScaler`/`MinMaxScaler`/`QuantileTransformer` on measured grounds: its IQR denominator moved 2.40% on average when the top 1% of values were trimmed, against 13.94% for the standard deviation and 30.42% for the range. For a system whose entire purpose is to retain and score outliers, the scaler's own baseline must not be dictated by them.

### 3.2 Real-time path — partially implemented, and this is the largest gap in the architecture

`src/fe_utils.py::transform_new(txn, reference)` engineers features for **one** transaction dict using statistics captured at training time in `artifacts/reference.pkl`, and reindexes to `reference["feature_cols"]`. It is real, working code — the Bank Transaction Fraud & Anomaly Detection What-if simulator calls it (`dashboard/backend/api_server.py:691`) and `DOCUMENTATION.md` Stage 7 records it being verified for both an existing account with history and a brand-new account with none.

**But it produces v1's 20 features, not the 46 this system scores on.** `reference.pkl`'s per-account state holds exactly four things — `running_mean_amount`, `devices`, `locations`, `last_time` — which is enough for `Amount_vs_AccountAvg`, `DeviceNoveltyFlag`, `LocationNoveltyFlag` and `TimeSinceLastTxn`, and nothing else. The 26 Phase 5 features are not reachable from it.

Closing the gap means extending the per-account state, and the required additions are enumerable exactly:

| Missing feature family | Additional per-account state required |
|---|---|
| `Expanding_{Mean,Median,Std,Min,Max}Amount` | running count, running sum, running sum-of-squares, running min/max, and a structure supporting a running median (a t-digest or a bounded reservoir — an exact running median needs unbounded history) |
| `Rolling3_{Mean,Std}Amount` | the last 3 amounts, as a fixed-size deque |
| `Velocity_1D_Count`, `Velocity_7D_Count` | timestamps within a trailing 7-day window, as a bounded deque |
| `CustomerTxnCountSoFar`, `SpendCV_Account` | derivable from the running count/sum/sum-of-squares above |
| `DeviceSharedAccounts_Prior`, `IPSharedAccounts_Prior`, `MerchantSharedAccounts_Prior` | **per-device / per-IP / per-merchant** distinct-account sets — the only state here that is *not* keyed by account, and therefore the only piece that cannot live in a per-account record |
| `Location_Freq` | a global location-frequency table, refreshed on the training cadence |

Every one of these is a bounded, incrementally-updatable statistic except the running median and the distinct-account sets. That is the design of the feature store described in §7 — and the reason the real-time path is a build item rather than a configuration change. **Until it is built, this system scores in batch.**

---

## 4. Stage 3 — Model Scoring

**Model set.** This pipeline now has 9 models total: 8 classical detectors (`07_models_classical.py`) plus the Model 9 Hybrid Ensemble (`08_models_deep.py`, majority vote of Isolation Forest + LOF + GMM — redefined from its earlier IF + LOF + Autoencoder form after the three deep-learning models, Autoencoder/VAE/LSTM-AE, were removed from this pipeline; see the project decision log). The deployed **online** scoring set is the models that can score a transaction they were not fit on: Isolation Forest, LOF (`novelty=True`), One-Class SVM, Elliptic Envelope, K-Means, GMM — plus the Hybrid Ensemble, computable online because all three of its inputs (IF, LOF, GMM) are themselves online-capable. **DBSCAN and HDBSCAN are excluded from the online path** because they have no out-of-sample `.predict` (Phase 8 §0) — they can only participate in a full-refit batch run, and by extension so can any ensemble score that includes them.

| Artifact | Loaded by |
|---|---|
| `artifacts_research/models/{isolation_forest,lof,ocsvm,elliptic_envelope,kmeans,gmm}.pkl` | `joblib.load` |
| `artifacts_research/models/{dbscan,hdbscan}.pkl` | `joblib.load` — batch-only, per above |
| `artifacts_research/models/shared_robust_scaler.pkl` | applied before every model |

**Two invariants that must survive into production code**, both established in Phase 8 §0:

1. **Sign convention.** Every `score_<model>` column is oriented so that **higher = more anomalous**. sklearn's `decision_function` for IsolationForest / LOF / OneClassSVM / EllipticEnvelope uses the opposite convention, so those four are negated at the point of scoring. Getting this wrong inverts the alert queue and would not be obvious from the score distribution alone.
2. **One scaler, fit once.** All models consume the same `RobustScaler` fit on the training split. Refitting the scaler at scoring time — an easy mistake, since the transform is one line — would silently change every score.

**One per-model behaviour the scoring service must handle explicitly:**

- **K-Means needs more than its pickle.** Its anomaly score is distance to the nearest centroid *among clusters holding ≥1% of training rows* (Phase 8 §1.7). Without that filter the score inverts: the most extreme transactions in the dataset formed their own micro-clusters and would have scored as the safest points in the book. The set of valid centroid indices is model state and must ship alongside `kmeans.pkl`.

(The removed LSTM-AE's permanent coverage hole — it only scored accounts with ≥3 transactions, 2,402 of 2,512 rows — no longer applies to this pipeline: all 8 classical models score every row.)

---

## 5. Stage 4 — Anomaly Score

**Implemented by** `src_research/12_ensemble_scoring.py`. **Primary output:** `ensemble_percentile_average` in `artifacts_research/ensemble_scores.csv`.

Each model's score is converted to its own empirical percentile via `(rank − 0.5) / n_valid`, and a transaction's score is the mean over the 8 classical models' percentiles (all 8 produce a score for every row now that the Autoencoder/VAE/LSTM-AE — the one family with a coverage hole — have been removed from this pipeline; see the project decision log). The aggregation code is still written NaN-aware (missing models skipped, remainder renormalised, not imputed — Phase 12 §1.3) for robustness, but in practice this branch is now a no-op.

The percentile-average design's real operational value is **model-outage behaviour**: if a model fails to load or times out, the score degrades gracefully rather than erroring — which is exactly why this strategy was preferred over PCA Stacking, which requires a complete matrix and would force a zero-imputation for any missing model's score.

**Reference distribution.** The published score has mean 0.5000, std 0.1970, min 0.1018, Q1 0.3413, Q3 0.6523, max 0.9994 (Phase 13, regenerated). By construction the mean of a percentile average sits near 0.5 whatever the data looks like — **which is exactly why the score's own distribution is not a drift signal**, and why Phase 16 monitors the input features and the cross-model agreement instead.

**Secondary score, computed in parallel:** the Phase 12 Weighted Average (weights in `artifacts_research/ensemble_weights.json`, ranging HDBSCAN 0.153 down to DBSCAN 0.085 across the 8 classical models). It costs nothing extra once the member models have run, and it is unbounded, which is the one thing the percentile score cannot do — see §6.

**Open item carried from Phase 14 §3, updated:** the published `ensemble_percentile_average` is the **8-model** (all-classical, batch) score, including DBSCAN and HDBSCAN. The 6-model online variant (excluding DBSCAN/HDBSCAN) recommended here has not been computed, so it must be validated against the published score (Spearman plus top-5% Jaccard, the measures Phase 12 §2 used) before any threshold below is applied to it.

---

## 6. Stage 5 — Alert Generation

**Implemented by** `src_research/13_threshold_optimization.py`; values in `artifacts_research/threshold_analysis.json` and `threshold_flagged_counts.csv`.

| Tier | Rule | Score cut | Volume in the 2,512-row sample |
|---|---|---:|---|
| **Priority review** | ≥ 99th percentile | 0.9167 | 26 (1.04%) |
| **Standard review** | ≥ 95th percentile | 0.8414 | 126 (5.02%) |
| No alert | below | — | 2,386 (94.98%) |

**No automatic block tier**, for the reason given in Phase 14 §4: v1's block threshold came from a cost sweep against supervised proxy labels, and Phase 13 §1 established that the sweep cannot be reproduced without a label, because a false-negative count requires knowing which *unflagged* transactions are fraud. Blocking on a score whose false-negative rate has never been measured is not defensible. Every output of this system lands in a human queue.

**Do not apply sigma or IQR rules to this score.** Phase 13 §3 found mean+3σ = 1.0911 and Q3+1.5×IQR = 1.1188, both above the score's observed maximum of 0.9994 — **both flag zero transactions**. Averaging eight bounded percentiles compresses the tails, so normal-theory thresholds have nothing to bite on. If a stakeholder wants a "more than three standard deviations from typical" framing, apply it to the parallel Weighted Average score instead, where mean+3σ = 2.2379 flags 16 transactions and Q3+1.5×IQR = 1.0468 flags 63. This is a property of the score's shape, not a defect in either strategy, and the two-score design in §5 exists so both framings are available without recomputation.

**Alert payload.** Each alert should carry: the transaction's raw fields, its `ensemble_percentile_average`, the **per-model percentile vector** (this is the score's explanation — see Phase 14 §1.15), and the Isolation Forest SHAP attribution from the explanation layer. The per-model vector is what lets a reviewer see whether a transaction is flagged by consensus or by one model's idiosyncrasy.

**Explanation layer.** `artifacts_research/shap_isolation_forest.csv` holds per-row, per-feature SHAP attributions for all 2,512 transactions, produced by `src_research/11_explainability.py` via `shap.TreeExplainer` (exact, no background sample needed). This is now the sole explainability output in this pipeline: the cross-model SHAP comparison this section previously described (Isolation Forest vs. the now-removed Autoencoder) was dropped along with the Autoencoder itself, and no other remaining classical model (LOF, OCSVM, Elliptic Envelope, DBSCAN, HDBSCAN, K-Means, GMM) is naturally SHAP-compatible without an expensive, unused `KernelExplainer`. Cost at this scale: TreeExplainer took 8.5s for all 2,512 rows, precomputable in batch.

---

## 7. Stage 6 — Investigation Dashboard (Bank Transaction Fraud & Anomaly Detection)

**This stage is built and verified.** `dashboard/` contains a FastAPI backend (`backend/api_server.py`) serving a plain HTML/CSS/JS frontend through `StaticFiles` — no build step, no CDN, no external fonts, no network calls at runtime.

```
cd dashboard
python -m uvicorn backend.api_server:app --reload
# then open http://127.0.0.1:8000/
```

Six pages: Overview (KPI tiles, risk-tier distribution, volume over time, top-10 riskiest), Transaction Explorer (search/filter/sort/paginate all 2,512, with a detail drawer showing raw fields, verdict, per-transaction SHAP and which detectors flagged it), Investigation Queue (risk-sorted, with Approve/Escalate/Block actions persisted to `backend/queue_state.json`, plus CSV export), Model Comparison, Explainability (global SHAP, decision-tree rules, a cost-threshold sweep recomputed at startup from `artifacts/split.pkl`), and a de-emphasised What-if Simulator. Verification is recorded in `dashboard/README.md`: every endpoint hit directly and confirmed against real data, all six pages plus both themes rendered end-to-end under headless Chromium with no console errors, and all frontend JS passing `node --check`.

### 7.1 What it currently scores against

Bank Transaction Fraud & Anomaly Detection sits on the **v1** pipeline, not on this research pipeline. Specifically: it loads `artifacts/reference.pkl`, `artifacts/labeled.csv`, `artifacts/anomaly_votes.csv` and `artifacts/thresholds.json`, scores every row with v1's SMOTE-trained XGBoost (`model.predict_proba(...)[:, 1]`, `api_server.py:144`), and precomputes a single `shap.TreeExplainer` pass over all rows into `backend/cache/shap_values.npy`. Verdicts come from `artifacts/thresholds.json`: `review_threshold = 0.09`, `block_threshold = 0.94`.

That means the risk score an analyst currently sees is **a supervised model's reproduction of v1's four-detector consensus** — which `LIMITATIONS.md` is blunt about: the 0.97-class ROC-AUC "measures how well XGBoost reproduces the anomaly ensemble's own judgment, not real-world fraud-catching accuracy." The dashboard surfaces this caveat in its own UI (Overview "About this system" panel and the sidebar footer).

### 7.2 Migrating Bank Transaction Fraud & Anomaly Detection onto the Phase 12 ensemble score

A realistic follow-up, scoped honestly. Five changes, in dependency order:

1. **Score source.** Replace `predict_proba` with a lookup of `ensemble_percentile_average` from `artifacts_research/ensemble_scores.csv`, joined on `TransactionID`. This is the cheapest change and it removes the supervised layer entirely — no XGBoost, no SMOTE, no proxy labels. It also removes the circularity that `LIMITATIONS.md` names as this project's central caveat: the displayed score becomes the unsupervised consensus directly, rather than a supervised model trained to imitate it.
2. **Thresholds.** `artifacts/thresholds.json`'s 0.09/0.94 probability cuts are meaningless against a percentile score and must be replaced by Phase 13's 0.8406/0.9145. The `_verdict_for()` helper (`api_server.py:106`) maps to three verdicts including Block — the Block branch should be **removed**, not re-pointed, per §6.
3. **Explanation layer.** The single `TreeExplainer`-over-XGBoost pass has no equivalent for an 8-model percentile average. It is replaced by the precomputed `shap_isolation_forest.csv` matrix plus the per-model percentile vector. The detail drawer then shows the Isolation Forest SHAP breakdown alongside the full per-model score vector — more UI work than today's single pass, and a better result: an analyst can see whether a transaction is flagged by consensus or by one model's idiosyncrasy, even without a second SHAP family (Phase 11's cross-model SHAP comparison was dropped when the Autoencoder was removed from this pipeline; no other remaining classical model is naturally SHAP-compatible).
4. **Model Comparison page.** Currently renders v1's four-detector comparison and the SMOTE-vs-class-weighted contrast from numbers hardcoded in `api_server.py`. Repoint at Phase 8's 9-model table (8 classical detectors + the Hybrid Ensemble, IF+LOF+GMM majority vote) and Phase 10's internal-validity metrics (`artifacts_research/internal_validity_metrics.csv`) — richer content, mechanically simple.
5. **What-if Simulator — blocked.** It calls `fe.transform_new(txn, STATE["reference"])` (`api_server.py:691`), which produces v1's 20 features. It cannot be migrated until the real-time feature-engineering gap in §3.2 is closed. The honest interim options are to leave it scoring against v1's model with a clear label saying so, or to hide it. **Do not** feed 20 features into models expecting 46.

Items 1, 2 and 4 are a day's work. Item 3 is a few days of UI work. Item 5 is the real-time feature store, which is §8-scale work.

### 7.3 The one thing Bank Transaction Fraud & Anomaly Detection produces that nothing else does

The Investigation Queue's Approve/Escalate/Block actions persist to `backend/queue_state.json`, written on the first queue action (the file does not exist until then, and `backend/cache/shap_values.npy` is likewise written on first startup). In a system whose defining limitation is the absence of any label, **that file is the seed of the first real one**. It is not a fraud label — an analyst's verdict is a judgement, not a confirmed outcome — but it is human, independent of the models, and it accumulates. Persisting it in a schema that can later be joined to case-management outcomes is the cheapest thing this project could do to escape the circularity described in `LIMITATIONS.md`, and it is worth doing before the volume that would make it useful arrives.

---

## 8. Batch vs. Real-Time for This Specific Use Case

**Recommendation: batch-first, and the reason is the intervention model, not the technology.**

The recommended tiers are both *review* tiers. Nothing in this system blocks a transaction or holds a customer's funds, so nothing in it needs to return an answer before the transaction completes. A transaction scored 30 minutes after it settles enters the same review queue it would have entered synchronously. Real-time scoring buys latency this use case has no use for.

Three further arguments, all specific to this system:

| Argument | Detail |
|---|---|
| **Two models can only run in batch** | DBSCAN and HDBSCAN have no out-of-sample `.predict` (Phase 8 §0). Keeping the full 11-model score Phase 12 published *requires* a batch refit. The 9-model online variant exists precisely to escape this — but it is unvalidated (§5). |
| **The real-time feature path does not exist yet** | §3.2: `transform_new()` reaches 20 of the 46 features. Batch feature engineering is implemented, tested, and leakage-verified. |
| **This sample's volume makes it moot** | 6.90 transactions/day over 364 days (Phase 13 §4). A nightly batch produces roughly 7 rows. Even the busiest observed month is ~226 transactions. |

**What would change the answer.** If the bank wants an inline *block* or *step-up authentication* decision, real-time becomes mandatory — and then §3.2's feature store, the 9-model set, and a measured false-negative rate all become prerequisites rather than improvements. That is a different system with a different validation bar, and this architecture should not be presented as being one config flag away from it.

**A defensible middle path** if latency is wanted before the feature store is ready: score in micro-batches (every 5–15 minutes) over recent transactions using the batch feature-engineering code against a warm history window. This keeps one feature-engineering implementation — which matters more than it sounds, because a second real-time implementation is the classic source of training/serving skew, and this pipeline's leakage safety (`closed='left'`, `shift()` before every window) is exactly the kind of subtlety a reimplementation gets wrong.

---

## 9. Model Versioning and Artifact Storage

**Current state, stated plainly: there is no versioning.** Artifacts are loose files in `artifacts/` and `artifacts_research/`, overwritten in place on each run. Nothing carries a version tag, a training timestamp, or a hash. For a research pipeline that is fine; for anything handed to a bank it is the first thing to fix.

**What a version must contain.** These artifacts are *jointly* valid and individually meaningless — versioning the models without the scaler and the feature schema is the failure mode to design against:

| Component | Artifact | Why it is inseparable |
|---|---|---|
| Feature schema | `feature_cols` (46 names, ordered) from `autoencoder_config.json` | Column order is a binding contract with every fitted model |
| Feature-engineering constants | the `Amount_ZScore_Account` denominator floor ($14.60 = 5% of the training `TransactionAmount` std), `Location_Freq` frequency table, `reference.pkl`-equivalent per-account state | Training-data-derived; a new training run changes them |
| Scaler | `models/shared_robust_scaler.pkl`, `autoencoder_scaler.pkl` | Fit on the 2,009-row train split; refitting at scoring time silently changes every score |
| Models | the 6 online-capable classical model artifacts + the Hybrid Ensemble's vote logic (+ DBSCAN/HDBSCAN for batch, 8 classical models total) | — |
| K-Means auxiliary state | the set of valid centroid indices (≥1% of training rows) | Without it the score inverts (§4) |
| Ensemble parameters | `ensemble_weights.json` for the parallel Weighted Average | Derived from `model_pairwise_spearman.csv`; shifts with the data |
| Thresholds | 0.9145 / 0.8406 from `threshold_analysis.json` | Percentiles of a specific score built by a specific model set |
| Reproducibility metadata | `random_state=42`, the 2,009/503 split, the LSTM-AE's separate account-level 342/86 split, library versions | Phase 8 §0 cross-checked the row-level split against the autoencoder's own `split` column and confirmed a match; that check is only meaningful if the split is recorded |

Isolation Forest (Model 1) is the model to copy here: `07_models_classical.py` already persists the fitted estimator, the shared scaler it was fit against, the ordered feature list it expects, and — via `11_explainability.py` — a full per-row SHAP attribution matrix. Every other model should be packaged to the same standard.

**Storage, free and local only.** Two options, both open-source and offline:

- **A dated, immutable artifact directory** — `artifacts_research/v2026-08-17T1744/...` — plus a `MANIFEST.json` recording the git commit, the training data's row count and SHA-256, library versions, and the table above. A symlink or a small pointer file marks the active version. Minimal machinery, no new dependency, and it makes rollback a one-line change.
- **A local MLflow file-store** (`mlflow.set_tracking_uri("file:./mlruns")`) if run-comparison and parameter tracking are wanted. Open-source, runs entirely offline, no service to pay for. Heavier, and only worth it if retraining becomes frequent.

The dated-directory option is recommended for this system's cadence. Either way, **the version identifier must be written onto every alert** — an alert with no record of which model version produced it cannot be audited, and a bank will ask.

**A gap to close before any handover: `requirements.txt` does not describe this pipeline.** It lists pandas, numpy, scikit-learn, xgboost, shap, imbalanced-learn, matplotlib and joblib — the v1 dependency set. The research pipeline additionally requires **optuna** (Phase 9, Isolation Forest and GMM tuning) and the standalone **hdbscan** package (Model 6, imported as `hdbscan` in `src_research/07_models_classical.py`, not `sklearn.cluster.HDBSCAN`) and **umap-learn** (Phase 7 dimensionality reduction) — **torch is no longer a dependency of this pipeline**, now that the Autoencoder, VAE and LSTM-AE (the only models that used it) have been removed — and the dashboard requires **fastapi** and **uvicorn** (documented in `dashboard/README.md`, not in `requirements.txt`). A clean-machine install from `requirements.txt` today cannot run Phase 9 or the dashboard. Splitting into `requirements.txt` / `requirements-research.txt` / `requirements-dashboard.txt`, with pinned versions, is a prerequisite for reproducible deployment.

---

## 10. What Has to Change to Reach Production Scale

The honest gap: **2,512 rows against the ~1M the brief describes** (`LIMITATIONS.md`) — a factor of roughly 400. Below is what genuinely breaks, separated from what merely gets slower.

### 10.1 Batch feature engineering: pandas → a distributed engine

The current implementation is single-process pandas over a fully-materialised DataFrame. What that means at 1M rows:

- **Maps cleanly across.** The expanding and rolling per-account statistics are window functions over a `PARTITION BY AccountID ORDER BY TransactionDate` frame — a direct translation to Spark SQL windows or Dask. `shift()` becomes `lag()`; `rolling('1D', closed='left')` becomes a `RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND 1 SECOND PRECEDING` frame. The leakage-safety semantics survive the translation, but they must be **re-verified after it**, not assumed — Phase 5's row-for-row `TimeSinceLastTxn` cross-check is the pattern to repeat.
- **Does not map cleanly.** `_prior_distinct_other_accounts()` — a per-device / per-IP / per-merchant running count of distinct *other* accounts seen strictly earlier. This is a stateful running distinct-count over an unbounded key space, not a partitioned window, and at scale it needs either an approximate structure (HyperLogLog per key) or a stateful streaming aggregation. It is the one feature family that requires a genuine redesign rather than a port.
- **Also needs attention.** The running median (`Expanding_MedianAmount`) is exact today because full history fits in memory. At scale it becomes a t-digest or a bounded reservoir, i.e. approximate — and the approximation error should be measured against the exact version on a sample before shipping.

### 10.2 Real-time scoring: a feature store, not a history scan

The naive real-time implementation recomputes an account's expanding statistics by scanning its full history on every transaction. At 1M rows with 5+ transactions per account that is a per-transaction database scan, and it gets slower as history grows.

The replacement is a keyed feature store holding incrementally-updatable state, enumerated in §3.2: per-account (count, sum, sum-of-squares, min, max, a running-median sketch, a 3-element amount deque, a 7-day timestamp deque, device set, location set, last timestamp) and per-device/IP/merchant (distinct-account sets or HLL sketches). `artifacts/reference.pkl` is already a small, in-memory version of exactly this shape — it holds 4 of roughly 12 required state elements — so the design is a generalisation of something that exists, not a new invention. Any free/local key-value store (Redis, SQLite, RocksDB) serves; no managed service is required.

The hard requirement is that the store's update and the score's read are **consistent and prior-only**: the state read for a transaction must reflect all strictly earlier transactions and none later. This is the same invariant `closed='left'` and `shift()` enforce in the batch path, and it is the invariant most likely to be quietly broken in a streaming reimplementation.

### 10.3 Models that hit a wall

| Model | Wall | Fix |
|---|---|---|
| One-Class SVM | Phase 8 §1.3: "a real scalability concern past ~50k–100k rows" — a QP roughly O(n²)–O(n³) in support vectors | Subsample the training set, or switch to `SGDOneClassSVM`; either changes the model and requires revalidation |
| LOF | O(n²) neighbour search; Phase 8 §1.2: needs an ANN index "well before six figures of rows" | An approximate-nearest-neighbour index (HNSW/Annoy — both free) |
| DBSCAN, HDBSCAN | Full refit over all history on every run, with no out-of-sample scoring | Drop from the online set (Phase 14 §3, Option B), or enable `prediction_data=True` for HDBSCAN and revalidate |
| GMM | 9-10 full-covariance components × 46 features = 1,000+ covariance parameters per component; Phases 8 and 9 both flagged overfitting at n=2,009 | More data genuinely helps here — but Phase 9's recommendation of `tied` covariance should be revisited on the larger sample rather than carried over unexamined |
| Isolation Forest, K-Means | No wall. Tree traversal and a centroid distance both scale fine | — |

### 10.4 Everything that must be refit, not ported

This is the part most likely to be underestimated. Several of this pipeline's central findings are **properties of this dataset**, and the models encode them:

- **Time features.** Every transaction in the training data falls Monday–Friday inside a 16:00–18:21 window, with Monday alone at 42.6% of volume (Phase 2 §4). `Hour_sin` has standard deviation **0.040**; `DOW_sin`/`DOW_cos` take only 5 distinct values (Phase 5 §2.3). A real 24/7 book turns four near-constant features into four informative ones. The scaler and every model must be refit — porting them would carry a scaler whose IQR for `Hour_sin` was estimated from a 2.3-hour window.
- **Network-proxy features.** 89.4% of devices and 93.2% of IPs in this dataset are shared across more than one account (Phase 5 §2.4), which Phase 5 correctly identifies as a data-generation artifact rather than a mule epidemic. A real bank's device-sharing base rate is far lower, so `DeviceSharedAccounts_Prior` and `IPSharedAccounts_Prior` will have a completely different distribution — and will, for the first time, actually mean something.
- **Velocity features.** 98.0% of `Velocity_1D_Count` and 91.1% of `Velocity_7D_Count` values are zero here, because 495 accounts averaging 5.08 transactions across 364 days rarely transact twice in a window (Phase 5 §2.1). A genuine transaction stream produces far more nonzero mass, turning a rare high-precision flag into a graded signal.
- **The novelty flags.** `DeviceNoveltyFlag` is 1 for 99.52% of rows and `LocationNoveltyFlag` for 94.27% (Phase 10 §4) — near-constant, because with ~5 transactions per account almost nothing repeats. With real account histories these become informative. This also removes the specific mechanism behind Isolation Forest's `TX000566` false signal (Phase 11 §2), which was driven by the *rarity* of `LocationNoveltyFlag = 0`.
- **The 5% contamination assumption.** `src/config.py::CONTAMINATION = 0.05` is documented as unverified, and `LIMITATIONS.md` lists re-tuning it against the bank's actual historical fraud rate as a pre-deployment requirement. At 1M rows a 5% flag rate is **50,000 alerts** — the assumption stops being a modelling detail and becomes a staffing decision.
- **The `$14.60` z-score floor.** 5% of *this* dataset's `TransactionAmount` standard deviation. It must be recomputed on the production training sample.

### 10.5 What does not change

The leakage-safe design, the feature definitions, the scaler choice and its measured justification, the sign convention, the two-family explanation layer, the percentile-aggregation scoring rule, and the review-only intervention model all transfer intact. **The reasoning survives the scale change; the fitted numbers do not.**

---

## 11. Handoff to Phase 16

The architecture above produces four things monitoring must watch, and Phase 16 addresses each: the 46 input features (drift), the cross-model agreement that stands in for this system's absent ground truth (concept drift), the flagged volume, and — because Phase 10 §2's bootstrap-stability check (now IF, LOF and GMM, the Hybrid Ensemble's three components) measured 41–72% flagged-set churn between retrains with no drift at all (Isolation Forest 47.3%, LOF 41.0%, GMM 71.9%) — the flagged-set *consistency*, not just its size.

*Next: `research/14_monitoring_framework.md` (Phase 16).*
