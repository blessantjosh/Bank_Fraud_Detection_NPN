# Research Pipeline Analysis — In-House 46-Feature Unsupervised Anomaly Detection

Scope: `src_research/01_data_understanding.py` through `13_threshold_optimization.py`, plus
`autoencoder_utils.py` / `vae_utils.py` / `config_research.py`, artifacts in
`artifacts_research/`. Every number below traces to a source file or artifact cited inline.
Where something could not be verified from available artifacts, that is stated explicitly
rather than estimated.

---

## 1. Dataset

Source: `data/bank_transactions_data_2.csv` (`config_research.py::RAW_CSV`).

| Fact | Value | Source |
|---|---|---|
| Rows | 2,512 | `artifacts_research/dataset_facts.json` |
| Raw columns | 18 (16 in the CSV header + 2 parsed datetime columns added at load) | `dataset_facts.json`, `config_research.py::load_raw` |
| Unique accounts | 495 | `dataset_facts.json` |
| Unique TransactionIDs | 2,512 (no duplicates) | `dataset_facts.json` |
| Missing cells | 0 / 45,216 | `data_quality_summary.json` |
| Exact/near-duplicate rows | 0 | `data_quality_summary.json` |
| Fraud label | **None exists anywhere in this project** | verified: `07_models_classical.py::load_and_split` asserts no `vote_count`/`risk_tier`/`is_fraud` columns in `features_v2.csv` |
| Dataset span | 364 days, avg 6.90 txns/day | `13_threshold_optimization.py` (`DATASET_SPAN_DAYS`) |

**Train/val split (not train/test):** row-level 80/20 via `train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)` → 2,009 train / 503 val rows, reproduced identically across `07_models_classical.py`, `08_models_deep.py`, `09_hyperparameter_optimization.py`, `10_evaluation.py`, `11_explainability.py`, and cross-checked against the Phase-7 autoencoder's own recorded split (`autoencoder_reconstruction_errors.csv["split"]`) — confirmed MATCH by an assertion in `load_and_split()`. **There is no held-out test set anywhere in this pipeline.** LSTM-AE alone uses a *different*, account-level 80/20 split (342/86 accounts) so no single account's chronological sequence is split across train/val.

Scaling: `RobustScaler` fit on `X_train` only, applied to train/val/all (`07_models_classical.py::load_and_split`). Chosen over Standard/MinMax/Quantile because skew/kurtosis are affine-invariant (mathematically identical across Standard/MinMax/Robust) and Phase 3 deliberately kept outliers as candidate fraud signal — `05_preprocessing.py::scale_sensitivity_analysis` measures how much each scaler's denominator (std / range / IQR) is dragged by the top-1% tail: `artifacts_research/scaler_sensitivity_to_outliers.csv`.

Identifiers explicitly excluded from modeling: `TransactionID`, `AccountID` (kept only for traceability, per `04_feature_engineering.py` header comment and `ID_COLS` in `07_models_classical.py`), plus raw `DeviceID`/`IP Address`/`MerchantID`/`Location` strings, which are converted into count/frequency/novelty features rather than used directly.

---

## 2. The 46 engineered features

Source of truth: `artifacts_research/autoencoder_config.json["feature_cols"]` (46 entries, verified `len == 46`), reused identically by every one of the 12 models so all cross-model comparisons are apples-to-apples (`07_models_classical.py` header comment).

| Category | Features | Notes |
|---|---|---|
| Raw numeric (5) | `TransactionAmount`, `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` | `LoginAttempts` skew 5.17, excess kurtosis 26.61 (`numeric_summary.csv`) — 95.1% of rows equal exactly 1, near-binary in practice |
| Categorical encoding (7) | `Location_enc`, `Location_Freq`, `TransactionType_Debit`, `Channel_Branch`, `Channel_Online`, `CustomerOccupation_Engineer/Retired/Student` | Both a label-encoded and frequency-encoded `Location` are kept simultaneously; `corr(Location_enc, Location_Freq) = 0.0029` (near-zero, not a meaningful double-count) — `07_models_classical.py` header |
| Behavioral / novelty (v1-reused, 5) | `Amount_vs_TypeAvg`, `Amount_vs_AccountAvg`, `DeviceNoveltyFlag`, `LocationNoveltyFlag`, `TimeSinceLastTxn` | Loaded from `artifacts/features.csv` (v1 pipeline), row-order-verified by recomputing `TimeSinceLastTxn` independently and asserting `np.allclose` (`04_feature_engineering.py::main`) |
| Frequency / prior-count (v1-reused, 3) | `DeviceTxnCount`, `IPTxnCount`, `MerchantTxnCount` | v1-computed, aligned onto this pipeline's sort order |
| Velocity (2) | `Velocity_1D_Count`, `Velocity_7D_Count` | Trailing rolling counts, `closed="left"` — strictly excludes the current row |
| Rolling/expanding statistics (7) | `Expanding_MeanAmount`, `Expanding_MedianAmount`, `Expanding_StdAmount`, `Expanding_MinAmount`, `Expanding_MaxAmount`, `Rolling3_MeanAmount`, `Rolling3_StdAmount` | Computed on `shift()`ed history — strictly prior rows only |
| Ratio / deviation (5) | `Amount_to_Balance_Ratio`, `Amount_to_RollingMean_Ratio`, `Amount_minus_ExpandingMean`, `Amount_minus_ExpandingMedian`, `Amount_ZScore_Account` | `Amount_ZScore_Account` uses a floored std denominator (5% of dataset-wide std, $14.60) to avoid divide-by-near-zero blowups — documented fix in `04_feature_engineering.py::ratio_and_deviation_features` |
| Cyclical time (4) | `Hour_sin`, `Hour_cos`, `DOW_sin`, `DOW_cos` | Sine/cosine encodings of hour-of-day and day-of-week |
| Behavioral flags (4) | `CustomerTxnCountSoFar`, `SpendCV_Account`, `ElevatedLoginFlag`, `ATM_Credit_InteractionFlag` | `CustomerTxnCountSoFar` = leakage-safe prior-count via `cumcount()` |
| Network proxy (3) | `DeviceSharedAccounts_Prior`, `IPSharedAccounts_Prior`, `MerchantSharedAccounts_Prior` | Count of *distinct other accounts* sharing the same device/IP/merchant at a strictly earlier timestamp |

Total: 5+7+5+3+2+7+5+4+4+3 = **46**, matching `input_dim` in `autoencoder_config.json`/`vae_config.json`.

**Leakage safety of the feature engineering itself:** every "history" feature (velocity, expanding/rolling stats, network-proxy counts, `CustomerTxnCountSoFar`) is built on `closed="left"` windows or `.shift()`/`cumcount()` constructions that structurally cannot see the current or a future row — verified by reading `04_feature_engineering.py` directly, not asserted.

---

## 3. Model-by-model

All 12 models share the identical RobustScaler-scaled 46-feature matrix. Score-sign convention (verified, stated in `07_models_classical.py` header): every `score_<model>` column is oriented **higher = more anomalous**; sklearn's native `decision_function` (opposite convention for IF/LOF/OCSVM/EE) is negated before saving.

### 1. Isolation Forest — `07_models_classical.py::model_isolation_forest`
- **Detects:** globally anomalous points via random-split path length (tree ensemble).
- **Hyperparameters tried:** 5 configs, `n_estimators ∈ {100,200,300}`, `max_samples ∈ {"auto",0.8,0.5}`, `contamination ∈ {0.03,0.05,0.10}`, `max_features ∈ {1.0,0.7}`. **Selected:** `n_estimators=300, max_samples=0.5, contamination=0.05, max_features=0.7` (best score-separation among the 5%-contamination configs by score std).
- **Output:** continuous `-decision_function`; native `.predict()==-1` flag.
- **Strength:** cheapest model here, native out-of-sample scoring, most production-ready.
- **Weakness:** contamination is a manual assumption; axis-aligned splits can miss rotated clusters.
- **Fraud pattern fit:** globally rare, high-magnitude outliers (e.g. extreme z-score amounts).

### 2. Local Outlier Factor — `model_lof`
- **Detects:** local density deviation vs. k nearest neighbors.
- **Hyperparameters tried:** `n_neighbors ∈ {10,20,35}`, `contamination ∈ {0.03,0.05,0.10}`. **Selected:** `n_neighbors=20, contamination=0.05` (`novelty=True` for out-of-sample scoring).
- **Output:** continuous `-decision_function`.
- **Strength:** catches local anomalies invisible to global methods.
- **Weakness:** O(n²) neighbor search (no ANN index used) — noted as a scaling limit past low tens-of-thousands of rows.
- **Fraud pattern fit:** transactions anomalous relative to a local peer cluster, not the whole dataset.

### 3. One-Class SVM — `model_ocsvm`
- **Hyperparameters tried:** kernel ∈ {rbf,linear,poly}, `nu ∈ {0.05,0.10}`, `gamma ∈ {scale,auto}`. **Selected:** `kernel="rbf", nu=0.05, gamma="scale"` (standard default, not empirically beaten by the alternatives tried).
- **Output:** continuous `-decision_function`.
- **Strength:** flexible non-linear boundary via kernel trick.
- **Weakness:** slowest classical model to fit here (QP solve, roughly O(n²)–O(n³)).
- **Fraud pattern fit:** transactions outside a learned non-linear "normal" envelope.

### 4. Elliptic Envelope — `model_elliptic_envelope`
- **Hyperparameters tried:** `contamination ∈ {0.05,0.10}`, `support_fraction ∈ {None,0.8}`. **Selected:** `contamination=0.05, support_fraction=None`.
- **Output:** continuous `-decision_function` (Mahalanobis distance from a Minimum Covariance Determinant estimate).
- **Assumption check (measured, not assumed):** Shapiro-Wilk on the 46 scaled features — code computes `frac_non_normal`; the project report states **100% reject normality at p<0.05**, and sklearn raised a rank-deficiency warning while fitting (`research/15_final_research_report.md`, line 165). Its Gaussian assumption is measurably violated on this data.
- **Fraud pattern fit:** deviation from a (here, poorly-fitting) multivariate-Gaussian "normal" region — kept for comparison, explicitly not recommended as the primary detector.

### 5. DBSCAN — `model_dbscan`
- **Hyperparameters:** `eps` from a k-distance elbow (`min_samples-1 = 9`), tried at elbow×{0.8,1.0,1.2}; `min_samples ∈ {5,10,15}`. **Selected:** eps at the elbow value, `min_samples=10`.
- **Output:** distance to nearest core point (0 for core points themselves); native noise flag (`label==-1`).
- **Weakness:** eps highly sensitive — noise rate swings across the 3×3 grid (`07_models_classical.py::model_dbscan` records the exact min/max noise rate at runtime); no native out-of-sample `.predict`.
- **Fraud pattern fit:** points in genuinely sparse regions of the 46-D space.

### 6. HDBSCAN — `model_hdbscan`
- **Hyperparameters tried:** 4 configs, `min_cluster_size ∈ {10,15,20,30}`, `min_samples ∈ {5,10,15,None}`. **Selected:** the config whose noise rate is closest to 5%.
- **Output:** GLOSH `outlier_scores_` (continuous, higher = more anomalous), no manual eps.
- **Honest finding (from the code's own comparison note):** HDBSCAN's noise rate here is *higher and no more stable* than DBSCAN's, and it only ever finds 2 clusters, assigning most points to neither — not a clean win over DBSCAN on this dataset.
- **Fraud pattern fit:** same sparse-region detection as DBSCAN, without manual eps tuning (though that flexibility did not translate into a better result here).

### 7. K-Means (distance-based) — `model_kmeans`
- **Hyperparameters tried:** `k ∈ [2,10]`, BIC/silhouette/inertia all computed. **Selected:** `k=4` from the inertia elbow — **not** from silhouette, because a documented degeneracy was found: every k in [2,10] carves out a <1%-of-training-rows micro-cluster containing the *same 3 extreme-outlier rows* every time (e.g. `Amount_ZScore_Account` = 92.56, 77.71), which makes naive silhouette-argmax pick k=2 as an artifact, not a real 2-population structure.
- **Score fix:** distance to nearest centroid **among clusters holding ≥1% of training rows only** — the micro-cluster centroids are excluded as distance targets, since points assigned to them are themselves the extreme outliers that should score high, not comparison points.
- **Fraud pattern fit:** transactions far from every legitimate behavioral cluster.

### 8. Gaussian Mixture Model — `model_gmm`
- **Hyperparameters tried:** `n_components ∈ [1,10]` × `covariance_type ∈ {full,diag,tied,spherical}`, `reg_covar=1e-5` fixed. **Selected by BIC:** best combination found via grid.
- **Output:** `-score_samples` (negative log-likelihood; low likelihood = anomaly).
- **Fraud pattern fit:** transactions with low probability density under a mixture-of-Gaussians model of normal behavior.

### 9. Autoencoder — reused from Phase 7, `08_models_deep.py::model_autoencoder_reused`
See §4 for architecture. **Not retrained** in Phase 8 ("no evidence of a problem with it"). Output: reconstruction MSE. Cheapest per-row inference cost of any of the 12 models once trained (single forward pass).

### 10. VAE — trained fresh, `model_vae`
See §4. Output: reconstruction MSE only (KL reported separately, not folded into the score, to avoid mixing an information-bits term with a squared-error term on different scales — a documented design choice).

### 11. LSTM Autoencoder — `model_lstm_autoencoder`
**Scoped, partial-coverage model, stated explicitly, not hidden:** 428/495 accounts (86.5%) have ≥3 transactions, covering 2,402/2,512 rows (95.6%); the remaining 110 rows (accounts with 1–2 transactions) get **no LSTM-AE score** (`lstm_ae_applicable=0`). See §4 for architecture.

### 12. Hybrid Ensemble — `model_hybrid_ensemble`
- **Rule:** majority vote (≥2 of 3) over Isolation Forest's native flag, LOF's native flag, and the Autoencoder's top-5% reconstruction-MSE flag.
- **Important correction found during this audit / already flagged in `research/15_final_research_report.md` line 327:** an earlier project document described the Hybrid Ensemble's flagged set as 253 rows ("majority-vote threshold"), but the actual majority (≥2-of-3) rule in the code flags a different, smaller count — the 253 figure is a ≥1-vote, tie-inflated cut, not the majority rule the name implies. **This report defers to the code** (`hybrid_vote_count >= 2`) as ground truth, per the task's own instruction to trust code over docs on disagreement.
- **Fraud pattern fit:** requires corroboration across a global-outlier detector (IF), a local-density detector (LOF), and a reconstruction-error detector (AE) — reduces single-model false positives (e.g. the report's own worked example `TX000566`, flagged top-1% by one model purely on a rare value of a near-constant engineered flag, not by the ensemble).

---

## 4. Deep-learning architectures

### Autoencoder (Model 9) — `autoencoder_utils.py`
- Input dim 46 → Dense(16, ReLU) → Dense(8, ReLU) → **bottleneck(4, ReLU)** → Dense(8, ReLU) → Dense(16, ReLU) → Dense(46, linear output).
- Optimizer: Adam, `lr=1e-3`, `weight_decay=1e-5`. Loss: `MSELoss`. Batch size 64. Epochs 200 (no early stopping — fixed budget, `autoencoder_training_history.json` has exactly 200 recorded epochs of `train_mse`/`val_mse`).
- Train/val: 2,009 / 503 rows, `random_state=42`.
- Reconstruction error: `mse = mean((out - x)**2, axis=1)` — per-row anomaly score.
- Results (`autoencoder_config.json`): train MSE 0.2896, val MSE 0.3280, val MSE P95 0.6433, P99 1.3718, max 6.7077.

### VAE (Model 10) — `vae_utils.py`
- Input 46 → Dense(16, ReLU) → Dense(8, ReLU) → parallel `fc_mu(8→4)`, `fc_logvar(8→4)` → **reparameterize:** `z = mu + eps * exp(0.5*logvar)`, `eps ~ N(0,I)` → Dense(4→8, ReLU) → Dense(8→16, ReLU) → Dense(16→46, linear).
- Loss: `recon_loss (MSE) + beta * KL`, where `KL = -0.5 * mean(1 + logvar - mu² - exp(logvar))` (closed-form KL of `q(z|x)` vs. `N(0,I)`). `beta=0.1`.
- Optimizer Adam, `lr=1e-3`, batch size 64, 200 epochs, no early stopping.
- Results (`vae_config.json`): train recon MSE 0.3426, val recon MSE 0.3790, val MSE P95 0.7476, P99 1.9461, max 4.3951, val KL mean 4.5198. Fit time 54.3s.
- Score used: reconstruction MSE only (KL reported separately — see §3).

### LSTM Autoencoder (Model 11) — `08_models_deep.py::LSTMAutoencoder`
- `LSTM(46→16)` encoder → `Linear(16→8 latent)` → `Linear(8→16)` (init hidden state) → `LSTM(46→16)` decoder (teacher-forced on the shifted input sequence) → `Linear(16→46)` output projection.
- Loss: masked MSE over real (non-padded) timesteps only.
- Optimizer Adam, `lr=1e-3`, `weight_decay=1e-5`, batch size 32, 150 epochs, no early stopping.
- **Split:** account-level 80/20 (342 train / 86 val accounts) — deliberately different from every other model's row-level split, because a chronological sequence must not be split across train/val.
- Results (`lstm_ae_config.json`): train MSE 1.6364, val MSE 1.2582 — markedly worse reconstruction than AE/VAE, attributed in the code/report to short (median 5), sparse per-account histories giving little repeated temporal pattern to learn from only 342 training sequences.

---

## 5. Hyperparameter optimization (`09_hyperparameter_optimization.py`)

Only **3 of the 12 models were actually tuned** with a formal search (Isolation Forest, GMM, VAE) — verified against `hyperparameter_optimization_results.json`. No fraud label exists, so each model's search objective is an internal, label-free heuristic, stated explicitly per model:

| Model | Objective | Method(s) compared | Search space | Trials | Baseline vs. tuned |
|---|---|---|---|---|---|
| Isolation Forest | Silhouette of top-5%-flagged vs. rest (1,000-row train subsample), maximize | Exhaustive grid (ground truth) vs. random search vs. Optuna/TPE | `n_estimators` 50–500, `max_samples` 0.1–1.0, `max_features` 0.3–1.0 | Grid: 60 combos; Random: 30; Optuna: 30 | Grid best silhouette **0.6092** (`n_estimators=300, max_samples=0.8, max_features=1.0`); Optuna reached **0.5981** (0.0111 below grid optimum — reported honestly as *not* beating the grid); Random reached 0.5884 |
| GMM | BIC on train split, minimize | Optuna/TPE only | `n_components` 2–10, `covariance_type` ∈ {full,diag,tied,spherical}, `reg_covar` 1e-6–1e-3 (log) | 40 | Optuna best BIC **-109,595.2** (`n_components=10, full, reg_covar≈1.19e-6`) vs. Phase-8 grid's fixed-`reg_covar=1e-5` best of -63,019.3 — improvement attributed mostly to additionally tuning `reg_covar`, not a genuinely better `n_components`/`covariance_type`, per the code's own caveat |
| VAE | Validation reconstruction MSE, minimize, 60-epoch search budget (vs. 200 deployed) | Optuna/TPE only | `latent_dim` ∈ {2,4,8}, `hidden1` ∈ {8,16,32}, `beta` 0.01–1.0 (log), `lr` 1e-4–1e-2 (log) | 20 | Search best (60 epochs): val MSE 0.3315 at `latent_dim=2, hidden1=32, beta≈0.0104, lr≈0.0031`. Deployed Model 10 (200 epochs, `latent_dim=4, hidden1=16, beta=0.1`): val MSE 0.3790 — **not directly comparable** (different epoch budgets), reported as such, not as a claim the search "beat" the deployed model. Main finding: `beta` above ~0.3 consistently worsens reconstruction MSE (classic beta-VAE tradeoff) |

Isolation Forest's `iforest_grid_search.csv` (60 rows) is the full grid-search result set backing the table above.

**Not tuned via any search:** LOF, OCSVM, Elliptic Envelope, DBSCAN, HDBSCAN, K-Means, Autoencoder, LSTM-AE, Hybrid Ensemble — these used the manual multi-config comparisons described in §3 (typically 3–5 hand-picked configs), not Optuna/grid/random search. This is stated plainly rather than implied otherwise.

---

## 6. Ensemble architecture (`12_ensemble_scoring.py`)

**11 of the 12 models are combined** (isolation_forest, lof, ocsvm, elliptic_envelope, dbscan, hdbscan, kmeans, gmm, autoencoder, vae, lstm_ae). The **Hybrid Ensemble (Model 12) is deliberately excluded as an input** — since it is itself already a majority vote of IF+LOF+AE, including it back in would double-count those three detectors. This is stated explicitly in the code, not silently done.

Four strategies were computed (all in `ensemble_scores.csv`):

1. **Weighted average** — each model's z-scored score, weighted by `1/(disagreement+0.05)` where `disagreement = mean over the other 10 models of (1-Spearman rho)/2`. Actual weights (`ensemble_weights.json`): HDBSCAN 0.1134, LOF 0.1069, Isolation Forest 0.1064, Autoencoder 0.1000, VAE 0.0982, LSTM-AE 0.0933, Elliptic Envelope 0.0930, K-Means 0.0907, OCSVM 0.0716, GMM 0.0689, DBSCAN 0.0577 (lowest — DBSCAN disagrees most with the other 10 models, mean disagreement 0.440, consistent with the Spearman heatmap's near-zero row for DBSCAN).
2. **Rank aggregation (Borda count)** — ordinal ranks summed across all 11 models; missing LSTM-AE ranks (110 rows) imputed with that model's own neutral/median rank, documented not silent.
3. **Percentile aggregation** — each model's own empirical CDF percentile (0–1), missing values **skipped** (not imputed) and averaged only over available models per row.
4. **PCA stacking proxy (unsupervised, NOT supervised stacking)** — first principal component of the 11 z-scored score columns (missing LSTM-AE rows zero-imputed for PCA's complete-matrix requirement), oriented to match the percentile-average direction. **PC1 explains 52.65% of variance** across the 11 models' z-scores (`ensemble_weights.json`).

**Selected as the final/recommended score: `ensemble_percentile_average`** (used directly as the input to `13_threshold_optimization.py`, and stated as the recommendation in `research/15_final_research_report.md` line 34) — chosen for being interpretable per-model-agnostic-CDF, bounded in (0,1), and robust to any single model's unbounded scale dominating the combination (unlike weighted-average, whose raw range in this data reaches 18.16 on the high end — see §7).

---

## 7. Threshold optimization (`13_threshold_optimization.py`)

Applied to `ensemble_percentile_average`. **No precision/recall or cost-minimizing threshold could be computed — there is no fraud label anywhere in this project**, stated explicitly in the code rather than worked around.

| Method | Threshold value | n flagged | % flagged |
|---|---|---|---|
| P95 | 0.8406 | 126 | 5.016% |
| P97 | 0.8700 | 76 | 3.025% |
| P99 | 0.9145 | 26 | 1.035% |
| P99.5 | 0.9443 | 13 | 0.518% |
| mean+3σ (on percentile-avg score) | 1.1088 | **0** | 0% |
| Q3+1.5·IQR (on percentile-avg score) | 1.1363 | **0** | 0% |

**Genuine methodological finding, not an error:** because `ensemble_percentile_average` is a bounded average of 11 percentiles in (0,1) with observed max 0.9988, both classic statistical thresholds (mean+3σ, Tukey IQR fence) exceed the score's maximum possible value and flag zero rows — a CLT-like tail-compression effect of averaging several roughly-independent percentiles. Confirmed as specific to percentile aggregation, not a dataset-wide effect, by applying the identical thresholds to two unbounded scores for context: raw Isolation Forest score (mean+3σ flags 25, IQR flags 43) and the Weighted-Average ensemble (mean+3σ flags 17, IQR flags 87) — both produce non-trivial thresholds. Practical conclusion stated in the artifact: sigma/IQR-style thresholds belong on an unbounded, z-scored ensemble score, not on the bounded percentile-average.

Illustrative-only business framing (v1's own stated, non-bank-real figures: $5/FP, $250/FN): at P95, upper-bound review cost *if every flagged row were a false positive* = $630 (126 × $5) — explicitly labeled a worst-case ceiling, not a real cost estimate, since no label exists to compute actual FP/FN counts.

---

## 8. Heatmap interpretations

All files below are in `audit/heatmaps/` (PNGs) and `audit/tables/` (backing CSVs).

**`research_model_spearman_correlation_heatmap.png`** — Strongest agreement: Autoencoder↔VAE (ρ=0.80, expected — near-identical architecture/objective), LOF↔HDBSCAN (ρ=0.84, both density-based). Weakest/near-zero: **DBSCAN vs. everything** (ρ 0.01–0.19) — DBSCAN's continuous "distance-to-nearest-core-point" score behaves almost orthogonally to the other 11 models, consistent with its own documented eps-sensitivity instability. GMM vs. OCSVM is the only negative pairing (ρ=-0.05), both effectively noise-level.

**`research_model_jaccard_agreement_heatmap.png` / `research_model_disagreement_heatmap.png`** — Mirrors the Spearman structure at the top-5%-flagged-set level: density/tree-based models (IF, LOF, HDBSCAN, AE, VAE) share substantial overlap in *which specific rows* they flag; DBSCAN and OCSVM flag largely disjoint sets from the rest — real diversity value for an ensemble, at the cost of DBSCAN's low individual weight (§6).

**`research_feature_correlation_heatmap.png` / `_top20.png`** — The `Amount_*` derived family (`Amount_vs_AccountAvg`, `Amount_ZScore_Account`, `Amount_to_RollingMean_Ratio`, `Amount_minus_ExpandingMean/Median`) forms a visibly correlated cluster, as expected since they are different transforms of the same underlying `TransactionAmount` vs. account-history relationship. The `Expanding_*` statistics correlate strongly with each other (shared account-history basis) but are near-independent of the cyclical time features (`Hour_sin/cos`, `DOW_sin/cos`) and the network-proxy features (`*SharedAccounts_Prior`), which form their own weakly-correlated block — evidence the 46-feature set spans genuinely different behavioral axes rather than redundant restatements of `TransactionAmount`.

**`research_feature_model_importance_heatmap.png`** — `Amount_vs_AccountAvg`, `TimeSinceLastTxn`, and `Amount_ZScore_Account` dominate both real-SHAP columns (IF and AE), confirming amount-deviation-from-own-history is the single strongest anomaly driver in this feature set for the two explainable models. The association-proxy columns (labeled `(assoc.)`, not SHAP) show LOF and HDBSCAN also weight `IPTxnCount`/`Location_Freq` heavily, consistent with their density-based sensitivity to rare categorical co-occurrence.

**`research_model_characteristics_heatmap.png`** — Engineering-judgment matrix (explicitly not a measured metric). Shows the expected split: tree/ensemble and centroid methods (IF, K-Means) score highest on real-time suitability and out-of-sample support; density methods without a native `.predict` (DBSCAN, HDBSCAN) score lowest on production readiness despite strong detection characteristics.

**`research_model_training_data_heatmap.png` (leakage audit)** — **"Test involved during fitting" is 0/No for all 12 models by construction**, because this pipeline has no held-out test split at all (only train/val, or an account-level train/val for LSTM-AE) — verified directly against `07_models_classical.py`/`08_models_deep.py`, not assumed. The one leakage-adjacent finding worth flagging: **DBSCAN and HDBSCAN fit directly on the full dataset (`X_all`)**, since neither has a native out-of-sample method — their "validation" rows are seen during fitting, unlike the other 10 models, which fit on `X_train` only and merely *score* validation rows afterward. This is not test-set leakage (no test set exists to leak into), but it is a real asymmetry in how "validation" rows are treated across the 12 models, documented here rather than glossed over.

**`research_top_transactions_model_score_heatmap.png`** — The top ~10 transactions by `ensemble_percentile_average` are flagged at or near the 95th+ percentile by essentially every model simultaneously (near-uniform dark red), i.e. genuine cross-model consensus at the extreme tail. Lower in the top-25, individual models start to disagree sharply — e.g. GMM drops to 0.36–0.38 percentile for several rows the ensemble still ranks highly, and TX002181 has a blank LSTM-AE cell (one of its 110 out-of-scope short-history accounts) — both real, visible limits of any single detector that the ensemble is designed to average over.

**`research_anomaly_model_evaluation_heatmap.png`** — Only 5 of the plan's candidate metrics actually exist in the artifacts: silhouette, Davies-Bouldin, Calinski-Harabasz (all 12 models, `internal_validity_metrics.csv`) and bootstrap-Jaccard stability (only Isolation Forest, LOF, Autoencoder — `stability_bootstrap_jaccard.csv`, Phase 10 explicitly limited the expensive bootstrap refit to these 3). No runtime/latency metric was consolidated into a single CSV (`cost_note` strings exist per-model in `model_summary_classical.json` but were not aggregated into a numeric table) — reported as **not verifiable from a single artifact**, not fabricated here.

**`research_cross_model_shap_importance_heatmap.png`** — Isolation Forest and Autoencoder SHAP importance vectors have Spearman ρ = **-0.1566 across all 46 features** (computed directly from `shap_global_importance_comparison.csv` in this audit) — i.e., essentially uncorrelated, meaning the tree-based and reconstruction-based explainers draw on largely different features to justify their anomaly scores. This is a genuine, structurally-expected finding (different model families, different mechanisms) and is exactly the kind of cross-check the project report cites as the practical value of running both (catching one detector's spurious signal, e.g. `TX000566`, against a structurally different second opinion).

---

## 9. Leakage-safety statement

This is an unsupervised pipeline with **no fraud label anywhere**, so classical train/test leakage (a model seeing test labels or test rows during fitting) cannot occur in the usual sense. What *can* leak here, and what was checked directly against the code:

1. **Feature engineering leakage** — every history-dependent feature (velocity, expanding/rolling stats, network-proxy counts) is built with `closed="left"` windows or `.shift()`/`cumcount()`, structurally excluding the current and future rows (§2). Verified by reading `04_feature_engineering.py` line-by-line.
2. **Scaler leakage** — `RobustScaler` is fit on `X_train` only and applied to val/all (`07_models_classical.py::load_and_split`), not fit on the combined matrix.
3. **Model-fitting leakage** — 10 of 12 models (all except DBSCAN, HDBSCAN) fit on `X_train` only; validation rows are only ever *scored*, never fit on. DBSCAN and HDBSCAN fit on the full dataset because they have no native out-of-sample `.predict` — this is a real asymmetry (flagged in §8), but since there is no test set anywhere in this pipeline, it does not constitute test-set leakage; it is best read as "these two models did not respect the train/val split," worth fixing if a genuine held-out evaluation set is ever introduced.
4. **Threshold/contamination selection circularity** — `contamination=0.05` (and `TOP_PCT=0.05` generally) is a fixed, a-priori assumption applied uniformly across models, not tuned against any outcome that would create circularity. The one place a threshold *is* optimized (Isolation Forest's Optuna/grid search, §5) optimizes an internal silhouette objective computed only on the training split, not against the ensemble or any held-out score — so model selection there does not leak into the final ensemble threshold decision in `13_threshold_optimization.py`, which uses fixed percentiles (P95/P97/P99/P99.5) with no fitting step at all.

**Overall assessment: leakage-safe by the applicable definition for this pipeline**, with one documented asymmetry (DBSCAN/HDBSCAN fitting on `X_all`) that is not test leakage but should be corrected before any future version of this pipeline introduces a genuine held-out test set.

---

## 10. Documentation discrepancies found

- The task brief references `research/12_final_model_selection.md` — **this file does not exist in the repository** (`research/` contains only `13_deployment_architecture.md` and `15_final_research_report.md`). `research/15_final_research_report.md` itself cites `research/12_final_model_selection.md` repeatedly (e.g. lines 21, 324) as if it exists, including "5 factual errors... recorded... in `research/12_final_model_selection.md` §5" — this referenced errata file could not be located or verified. Flagged rather than guessed at.
- `research/15_final_research_report.md` (line 327) itself documents and corrects an earlier factual error about the Hybrid Ensemble's flagged-row count (253 vs. the code's actual ≥2-of-3 majority count) — this audit defers to the code (`08_models_deep.py::model_hybrid_ensemble`, `vote_count >= 2`) per the task instruction to trust code over docs.
- `research/15_final_research_report.md` (line 320) flags that the "recommended online model set" (a 9-model variant excluding DBSCAN/HDBSCAN for incremental/out-of-sample scoring, since those two have no native out-of-sample method) **has not actually been computed anywhere in the artifacts** — only the published 11-model `ensemble_percentile_average` exists. This audit's ensemble section (§6) reports the 11-model version only, since that is what is actually in `ensemble_scores.csv`.

---

## Artifact index

**Heatmaps** (`audit/heatmaps/`, PNG, dpi=110): `research_model_spearman_correlation_heatmap.png`, `research_model_jaccard_agreement_heatmap.png`, `research_model_disagreement_heatmap.png`, `research_feature_correlation_heatmap.png`, `research_feature_correlation_heatmap_top20.png`, `research_feature_model_importance_heatmap.png`, `research_model_characteristics_heatmap.png`, `research_model_training_data_heatmap.png`, `research_top_transactions_model_score_heatmap.png`, `research_anomaly_model_evaluation_heatmap.png`, `research_cross_model_shap_importance_heatmap.png`.

**Tables** (`audit/tables/`, CSV): `research_model_spearman_correlation_matrix.csv`, `research_model_jaccard_agreement_matrix.csv`, `research_model_disagreement_matrix.csv`, `research_feature_correlation_matrix_full46.csv`, `research_feature_correlation_matrix_top20.csv`, `research_feature_model_importance_matrix.csv`, `research_model_characteristics_matrix.csv`, `research_model_training_data_matrix.csv`, `research_top_transactions_model_score_matrix.csv`, `research_anomaly_model_evaluation_matrix.csv`, `research_cross_model_shap_importance_matrix.csv`.

**Generating script:** written to the session scratchpad and executed against this repository's `artifacts_research/` directory (not committed to the repo — all outputs it produced are the files listed above).
