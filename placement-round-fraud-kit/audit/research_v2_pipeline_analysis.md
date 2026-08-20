# research_v2 Pipeline Analysis — 18-Feature Unsupervised Anomaly Detection

**Scope of this document.** This covers `src_research_v2/` only — the client/teammate-designated
18-feature pipeline that is served live by the dashboard (`dashboard/backend/api_server.py` reads
from `artifacts_research_v2/`). It runs the same 12-model unsupervised methodology as the in-house
`research` (46-feature) pipeline, but over a different, teammate-supplied feature set. Every number
below traces to a specific file under `artifacts_research_v2/` or is recomputed directly from those
files by `audit/heatmaps` / `audit/tables` generation script described in this report. Where code and
docs (`research_v2/12_final_model_selection.md`, `research_v2/15_final_research_report.md`) disagree,
code/artifacts are treated as ground truth and the discrepancy is noted.

---

## 1. Dataset used

- Raw source: `data/bank_transactions_data_2.csv` — 2,512 transactions, 495 unique accounts, 16 raw
  columns, **no fraud label anywhere in the file**.
- Canonical model input for this pipeline: `artifacts_research/features_teammate_merged.csv`
  (deliberately *not* in `artifacts_research_v2/` — it lives in the sister pipeline's artifact
  folder because it is the teammate's shared deliverable; `src_research_v2/config_research_v2.py`
  points directly at it and asserts the exact column layout on every load).
- Row alignment between the raw CSV and the feature file is asserted equal
  (`04_feature_verification.py`); `phase5_6_feature_verification.json` confirms **0 missing cells**
  (0/70,336) and **0 duplicate rows/TransactionIDs**.
- Account-level sequence structure (used by the LSTM-AE, re-verified directly on this file rather
  than assumed carried over from the in-house pipeline): 495 accounts, mean 5.075 txns/account,
  median 5, min 1, max 12; 428/495 accounts (86.5%) have ≥3 transactions, covering 2,402/2,512 rows
  (95.6%).

## 2. Full feature list (18 features, `FEATURE_COLS_V2`), verified count

`config_research_v2.py` and `phase5_6_feature_verification.json`/`phase7_dim_reduction_summary.json`
(`"autoencoder": {"input_dim": 18, ...}`) all independently confirm **18**, not the 46 of the
in-house pipeline.

| Feature | Category |
|---|---|
| TransactionAmount | Raw numeric (StandardScaler-scaled) |
| CustomerAge | Raw numeric (StandardScaler-scaled) |
| TransactionDuration | Raw numeric (StandardScaler-scaled) |
| LoginAttempts | Raw numeric (StandardScaler-scaled) |
| AccountBalance | Raw numeric (StandardScaler-scaled) |
| account_frequency | Frequency-encoded behavioural feature |
| device_frequency | Frequency-encoded behavioural feature |
| ip_frequency | Frequency-encoded behavioural feature |
| merchant_frequency | Frequency-encoded behavioural feature |
| amount_to_balance_ratio | Engineered ratio feature |
| high_amount_transaction | Engineered binary flag (verified: boundary 878.63/877.81, essentially the raw dataset's 95th percentile of TransactionAmount, 878.18; 126/2,512 = 5.02% flagged; corr with raw amount = 0.683) |
| TransactionType_Debit | Categorical one-hot (baseline: Credit) |
| Channel_Branch | Categorical one-hot (baseline: ATM) |
| Channel_Online | Categorical one-hot (baseline: ATM) |
| CustomerOccupation_Engineer | Categorical one-hot (baseline: Doctor) |
| CustomerOccupation_Retired | Categorical one-hot (baseline: Doctor) |
| CustomerOccupation_Student | Categorical one-hot (baseline: Doctor) |
| Location_FE | Frequency-encoded categorical (location) |

All 18 columns are already individually StandardScaler-scaled by the teammate/upstream process
(`04_feature_verification.py` checks all 18, not just the 5 called out in the brief). `06_models_classical.py`
then applies a **second scaling pass, RobustScaler, fit on the train split only** and applied to
train/val/all — justified in the code comments as consistency with the in-house Phase 6 recommendation
and robustness to the handful of extreme rows. This "StandardScaler-then-RobustScaler" double-scaling
is a real, stated design choice, not an oversight.

## 3. Model-by-model detail (12 models)

For every model: score sign convention is **higher = more anomalous** (sklearn `decision_function`
outputs are negated for IF/LOF/OCSVM/EE). Native-contamination models use their own flag; DBSCAN,
K-Means and GMM (no native contamination knob usable the same way here) use a standardized
**top-5%-by-score** flag for cross-model Jaccard comparability.

1. **Isolation Forest** (`06_models_classical.py`) — Detects: axis-aligned isolation via random
   splits; short average path length = anomalous. Input: RobustScaler-scaled 18-feature train split.
   Selected hyperparameters (of 5 tried, contamination fixed at 0.05, picked by max score-std among
   the contamination=0.05 configs): `n_estimators=100, max_samples="auto", contamination=0.05,
   max_features=1.0` (`model_summary_classical.json`). Native anomaly rate: **5.29%**. Fit+score of
   all 5 configs: 4.48s/2,512 rows/18 features. Strength: fast, no distributional assumption, scales
   well. Weakness: axis-aligned splits can miss diagonal/interaction anomalies; sensitive to
   `max_features`. Fraud pattern it would catch: an isolated single-column extreme value (e.g. a
   TransactionAmount far outside the account's usual range) more reliably than a coordinated
   multi-feature pattern.

2. **Local Outlier Factor (LOF)** (`06_models_classical.py`) — Detects: local density deviation
   relative to k nearest neighbors. Input: same. Selected: `n_neighbors=20, contamination=0.05`
   (novelty=True for out-of-sample scoring). Native rate: **4.46%**. Strength: catches local-density
   anomalies invisible to global methods (e.g. a transaction normal in aggregate but anomalous
   relative to its account's usual cluster). Weakness: O(n²) neighbor search by default, sensitive
   to `n_neighbors` choice, degrades with sparse high-dimensional neighborhoods.

3. **One-Class SVM** (`06_models_classical.py`) — Detects: transactions outside a learned RBF-kernel
   decision boundary around the bulk of "normal" training data. Selected: `kernel="rbf", nu=0.05,
   gamma="scale"`. Native rate: **5.73%**. Strength: flexible non-linear boundary via kernel trick.
   Weakness: QP solve cost (roughly O(n²)–O(n³)), `nu`/`gamma` sensitivity, boundary is not
   interpretable feature-by-feature.

4. **Elliptic Envelope** (`06_models_classical.py`) — Detects: Mahalanobis-distance outliers under
   an assumed multivariate-Gaussian fit (MCD). Selected: `contamination=0.05, support_fraction=None`.
   Native rate: **5.02%**. A **Shapiro-Wilk normality check was run on all 18 scaled train-split
   features and reported explicitly** — the code's own caveat: a large fraction of features reject
   normality at p<0.05, so "its MCD-based Mahalanobis distance should be read as a rough baseline,
   not a well-calibrated score" (direct quote from `model_summary_classical.json`'s
   `gaussian_assumption_note`). Strength: computationally cheap, captures correlated-feature outliers.
   Weakness: the stated Gaussian-assumption violation above.

5. **DBSCAN** (`06_models_classical.py`) — Detects: points that are not core/border members of any
   density-connected cluster (noise points). `eps` chosen via a k-distance elbow (perpendicular-distance
   method on the normalized k-NN curve, k=9 for min_samples=10 candidate); selected config:
   `eps=2.684, min_samples=10`. Native noise rate: **2.27%**. **Fit directly on the full dataset
   (`X_all`), not `X_train`** — DBSCAN has no native `.predict()` for genuinely new points in this
   pipeline's usage (explicitly flagged in `12_ensemble_scoring.py`'s docstring). Score for the
   ensemble = distance to nearest core point. Strength: no assumed cluster shape, finds
   density-contiguous "normal" regions. Weakness: single global `eps` struggles with clusters of
   different density; no out-of-sample scoring.

6. **HDBSCAN** (`06_models_classical.py`, `hdbscan` package) — Detects: hierarchical density-based
   noise, adaptive to varying density (no single global `eps`). 4 configs tried; selected by
   closest noise rate to 5%: `min_cluster_size=10, min_samples=5` → noise rate **8.88%**. Score =
   `outlier_scores_` (GLOSH). Also fit on `X_all` (same out-of-sample limitation as DBSCAN, noted
   identically in the code). Strength: handles variable-density clusters better than DBSCAN.
   Weakness: same lack of native out-of-sample scoring; noise-rate/config sensitivity.

7. **K-Means (distance-based)** (`06_models_classical.py`) — Detects: distance to nearest valid
   (non-micro) cluster centroid. `k` swept 2–10; inertia elbow rule (first k where marginal inertia
   drop < 15% of the k=2→3 drop) selected **k=10**; note the code explicitly records that the
   silhouette-argmax pick would have been k=2 — a real disagreement between the two selection
   criteria, reported rather than hidden. 1 micro-cluster (<1% of train rows) excluded from centroid
   targets at k=10. Top-5%-flagged rate: **5.02%**. Strength: fast, interpretable ("distance from
   the nearest normal behavioural group"). Weakness: assumes roughly spherical clusters; degenerate
   micro-clusters appear at higher k (documented: k=6..10 all produce micro-clusters of 15-20 points).

8. **Gaussian Mixture Model (GMM)** (`06_models_classical.py`) — Detects: low likelihood under a
   fitted mixture of Gaussians; score = negative log-likelihood. Swept n_components 1–10 × 4
   covariance types via BIC/AIC; selected **n_components=10, covariance_type="diag"** (BIC
   −27,620.9) — note this is `diag`, not `full`, i.e. the model prefers to *not* model
   cross-feature covariance even with only 18 features (`15_final_research_report.md` flags the
   BIC curve was still descending at the n=10 search boundary — an honest, unresolved caveat also
   true here). Top-5%-flagged rate: **5.02%**. Strength: soft/probabilistic membership, flexible
   density shape via multiple components. Weakness: BIC/covariance-type sensitivity; likelihood-based
   score can be dominated by a few high-variance dimensions.

9. **Autoencoder** (`autoencoder_utils.py`, reused from Phase 7 — **not retrained** in Phase 8) —
   Detects: high reconstruction error (MSE) after compressing through a 3-unit bottleneck. Architecture:
   `input(18) → Dense(8) → Dense(4) → bottleneck(3) → Dense(4) → Dense(8) → output(18)`, ReLU
   activations, Adam (`lr=1e-3, weight_decay=1e-5`), MSE loss, 200 epochs, batch size 64, trained on
   RobustScaler-scaled `X_train` (n=2,009) with `X_val` (n=503) tracked for monitoring only (no
   early stopping / no backprop on val). Results: `train_mse_mean=0.2858, val_mse_mean=0.2966,
   val_mse_p99=0.6339` (`autoencoder_config.json`, `phase7_dim_reduction_summary.json` — both agree).
   Top-5%-flagged rate: **5.0%** (by construction, top-5%-by-MSE). Strength: continuous, captures
   non-linear multi-feature reconstruction failure. Weakness: opaque — a raw MSE number has no
   feature-level meaning without SHAP (see §10 below).

10. **Variational Autoencoder (VAE)** (`vae_utils.py`, trained fresh in Phase 8, not reused) —
    Architecture: `input(18) → Dense(8) → Dense(4) → [μ(3), logvar(3)] → reparameterize → Dense(4) →
    Dense(8) → output(18)`. Encoder produces `μ` and `logvar` for a 3-dim latent Gaussian; sampling
    uses the reparameterization trick `z = μ + ε·exp(0.5·logvar)`, `ε ~ N(0,I)`; loss = reconstruction
    MSE + `β·KL(q(z|x) ‖ N(0,I))` with the closed-form KL term
    `KL = -0.5·Σ(1 + logvar - μ² - exp(logvar))`. Deployed config: `latent_dim=3, hidden1=8, β=0.1,
    200 epochs, lr=1e-3, batch_size=64`. Results (`vae_config.json`): `train_recon_mse_mean=0.3371,
    val_recon_mse_mean=0.3458, val_recon_mse_p99=0.8484, val_kl_mean=1.1958`. Score for ensemble =
    reconstruction MSE only (not KL). Top-5%-flagged rate: **5.02%**. Strength: regularized latent
    space (in principle more robust to overfitting the bottleneck than a plain AE). Weakness: in
    practice its val MSE (0.346) is *worse* than the plain Autoencoder's (0.297) at the deployed
    config — the added KL regularization costs reconstruction fidelity without an offsetting benefit
    this pipeline can currently measure (see §5 — Optuna found a much lower-β config would help).

11. **LSTM Autoencoder** (`07_models_deep.py`) — Detects: per-timestep reconstruction error within
    an account's transaction sequence (captures *sequential/temporal* deviation, not just cross-sectional
    outlierness). Architecture: `LSTM(18→12) encoder → Linear(12→6 latent) → Linear(6→12) → LSTM(18→12)
    decoder → Linear(12→18)`, teacher-forced, masked MSE over real (non-padded) timesteps only. Scope
    limited to accounts with ≥3 transactions: **428/495 accounts (86.5%), 2,402/2,512 rows (95.6%)** —
    re-verified directly on this feature file, not assumed. Uses an **account-level 80/20 split**
    (342 train accounts / 86 val accounts, `random_state=42`) — explicitly *not* the row-level split
    used by every other model in this pipeline. 150 epochs, Adam `lr=1e-3, weight_decay=1e-5`, batch
    size 32. Results: `train_mse_mean=0.4345, val_mse_mean=0.7907` — val MSE is markedly higher than
    train, and `reconstruction_metrics_summary_v2.json` states this is a genuine overfitting signature
    on short, sparse per-account sequences (val MSE rises after ~epoch 50), reported honestly rather
    than smoothed over. Top-5%-flagged rate within applicable rows: **5.04%**. Strength: only model
    here that uses temporal/sequence order within an account. Weakness: 4.4% of rows (accounts with
    1–2 transactions) get **no score at all**; demonstrated overfitting; smallest/sparsest of the
    deep models.

12. **Hybrid Ensemble** (`07_models_deep.py`) — Rule: majority vote (≥2 of 3) across Isolation
    Forest's native flag, LOF's native flag, and Autoencoder's top-5% MSE flag. Majority-flagged
    rate: computed directly from `vote_count` in `model_scores_all.csv` (0–3 discrete values only —
    this coarseness is exactly why Phase 10's business-evaluation walkthrough uses Isolation Forest
    instead, see §7). Pairwise agreement (from `07_models_deep.py`'s printed summary): IF↔LOF, IF↔AE,
    LOF↔AE agreement rates are all reported in `model_summary_classical.json`'s sibling output.
    Strength: simple, transparent, reduces any single model's false positives. Weakness: discrete
    0–3 score is too coarse for fine-grained percentile thresholds; **deliberately excluded from the
    4 ensemble strategies in Phase 12** to avoid triple-counting IF/LOF/AE (see §6).

## 4. Deep-learning architectures — summary table

| Model | Latent dim | Hidden layers | Epochs | Batch | LR | Optimizer | Loss |
|---|---|---|---|---|---|---|---|
| Autoencoder (Model 9) | 3 | 8 → 4 → 3 → 4 → 8 | 200 | 64 | 1e-3 | Adam (wd=1e-5) | MSE |
| VAE (Model 10) | 3 (μ, logvar) | 8 → 4 → [μ,logvar](3) → 4 → 8 | 200 | 64 | 1e-3 | Adam (wd=1e-5) | MSE + β·KL, β=0.1 |
| LSTM-AE (Model 11) | 6 | LSTM(12) enc → Linear(6) → LSTM(12) dec | 150 | 32 | 1e-3 | Adam (wd=1e-5) | Masked MSE (non-pad steps only) |

VAE reparameterization and KL term, verified directly in `vae_utils.py`:
```
mu, logvar = encode(x)
std = exp(0.5 * logvar)
z = mu + std * randn_like(std)
kl = -0.5 * mean(1 + logvar - mu.pow(2) - logvar.exp())
loss = recon_mse + beta * kl
```

## 5. Hyperparameter optimization findings (`09_hyperparameter_optimization.py`)

No fraud label exists, so each model's tuning objective is a stated, unsupervised proxy — not
accuracy/F1:

- **Isolation Forest**: objective = silhouette between the top-5%-by-score group and the rest, on a
  fixed 1,000-row subsample. Exhaustive grid (60 combos, 40.8s) found best silhouette
  **0.4154** at `n_estimators=50, max_samples=0.8, max_features=1.0`. Optuna/TPE (30 trials, 25.0s)
  reached **0.4107** — 0.0047 *below* the grid optimum, reported honestly as Optuna not quite
  matching the grid here (small, cheap-to-grid-search space) rather than rounded up to a "win."
  Random search (30 trials) reached 0.3840 — Optuna beat random search by +0.0267.
- **GMM**: objective = BIC on train split, minimize; also tunes `reg_covar` (fixed at 1e-5 in the
  Phase 8 grid). Optuna (40 trials, 7.5s) found **BIC = −47,044.7** at `n_components=10,
  covariance_type="diag", reg_covar≈1.01e-6` — a large apparent improvement over the Phase 8 grid's
  −27,620.9, but the code's own result JSON flags this is expected from adding a
  `reg_covar` search dimension, "not necessarily evidence Optuna found a better
  n_components/covariance_type."
- **VAE**: objective = validation reconstruction MSE, minimize, 60-epoch search budget (vs. 200 for
  the deployed artifact). Best found: `latent_dim=3, hidden1=8, β≈0.0113, lr≈0.00312`, val MSE
  **0.2757** — substantially better than the deployed 200-epoch/β=0.1 config's 0.3458, though the
  two are not directly comparable (different epoch budgets). The result JSON's own verdict
  recommends this as a real avenue to revisit β, not a proven final answer.

## 6. Ensemble architecture and actual weights (`12_ensemble_scoring.py`)

**11 of the 12 models are combined** — the Hybrid Ensemble is deliberately excluded as an input
because it is itself already a majority vote of IF+LOF+AE; including it back in would triple-count
those three detectors. Its `vote_count` is retained only as a comparison column.

Four unsupervised strategies computed, all NaN-aware for LSTM-AE's 110 inapplicable rows:

1. **Weighted average** (z-scored per model) — weights = `1 / (mean pairwise disagreement + 0.05)`,
   normalized to sum to 1, where `disagreement_m` = mean over the other 10 models of `(1 - Spearman
   rho)/2`, sourced from `model_pairwise_spearman.csv` (not recomputed). **Actual weights**
   (`ensemble_weights_v2.json`): K-Means 0.1090, HDBSCAN 0.1074, LOF 0.1038, OCSVM 0.1040, VAE
   0.0956, Elliptic Envelope 0.0934, Autoencoder 0.0911, Isolation Forest 0.0883, LSTM-AE 0.0866,
   GMM 0.0694, DBSCAN 0.0513 (lowest — DBSCAN has the highest disagreement with the other 10 models,
   0.3872, consistent with its outlier k-distance-elbow-driven noise definition being structurally
   different from the rest).
2. **Rank aggregation (Borda count)** — per-model rank sum, missing values imputed at the median rank.
3. **Percentile aggregation** — mean of per-model percentile ranks in (0,1); **this is the strategy
   Phase 13 (threshold optimization) actually recommends and uses** — bounded, easy to reason about
   operationally.
4. **PCA-stacking proxy** — first principal component of the z-scored score matrix (NaNs imputed to
   0), sign-aligned to the percentile-average via Spearman; explicitly labeled NOT supervised (no
   label exists to stack against). PC1 explains **54.9%** of the z-scored score-matrix variance
   (`pca_explained_variance_ratio_pc1`).

The 4 strategies agree strongly with each other (`ensemble_pairwise_comparison_v2.csv`): pairwise
Spearman ranges **0.983–1.000**, pairwise top-5% Jaccard ranges **0.615–0.968** — Rank(Borda) and
Percentile-Avg are nearly identical (ρ=0.9999, Jaccard=0.969), as expected since both are
rank-based.

## 7. Threshold analysis (`13_threshold_optimization.py`)

Applied to the recommended score, `ensemble_percentile_average`. **No fraud label exists, so a true
cost-minimizing threshold cannot be computed** — stated explicitly in the code and repeated here
rather than glossed over. What is shown instead:

| Method | Threshold | n flagged | % flagged | Illustrative upper-bound review cost (all-FP, $5/txn) |
|---|---|---|---|---|
| P95 | 0.8671 | 126 | 5.02% | $630 |
| P97 | 0.9109 | 76 | 3.03% | $380 |
| P99 | 0.9510 | 26 | 1.04% | $130 |
| P99.5 | 0.9646 | 13 | 0.52% | $65 |

Statistical thresholds (mean+3σ, Q3+1.5·IQR) on the *bounded* `ensemble_percentile_average` flag
**zero** transactions (thresholds 1.176/1.233 exceed the score's max of 0.995) — the code's own
finding is that averaging 11 roughly-independent percentile ranks compresses the tails (a CLT-like
effect), confirmed by applying the identical statistical thresholds to two *unbounded* scores for
context (raw Isolation Forest score, and the unbounded Weighted-Average ensemble) where they do flag
a non-zero, smaller set. Illustrative $5 FP / $250 FN cost figures are v1's own stated placeholders
carried through for comparability — not real bank numbers.

## 8. Heatmap interpretations

All images: `audit/heatmaps/v2_*.png`; all source tables: `audit/tables/v2_*.csv`.

1. **`v2_model_spearman_correlation_heatmap.png`** — All 12 models, direct from
   `model_pairwise_spearman.csv` (LSTM-AE pairs masked to its 2,402 applicable rows, matching how
   the source script computed it). K-Means/HDBSCAN/LOF form the most mutually agreeing cluster
   (ρ≈0.81–0.84); DBSCAN is the clear outlier, correlating weakly (ρ 0.15–0.33) with every other
   model — consistent with its lowest ensemble weight (§6). Hybrid Ensemble correlates only
   moderately (ρ≈0.39–0.49) with everything, including its own three input models, because its
   0–3 discrete vote count compresses information relative to any continuous score.

2. **`v2_model_jaccard_agreement_heatmap.png`** — Top-5%-flagged-set overlap. Values are
   systematically lower than the Spearman matrix for the same pairs (e.g. K-Means↔HDBSCAN ρ=0.83 but
   Jaccard is noticeably smaller) — a reminder that strong rank correlation does not guarantee the
   *specific* transactions flagged agree; broad ordering and precise top-5% set membership are
   different questions, as the in-house pipeline's report also found for its own Elliptic Envelope.

3. **`v2_model_disagreement_heatmap.png`** — Simply `1 - Jaccard`, recomputed here directly from the
   same source matrix; visually the mirror image of (2), included for readability when the point is
   "which pairs disagree most" rather than "which agree most" (DBSCAN again stands out).

4. **`v2_feature_correlation_heatmap.png`** — Full 18×18 Spearman matrix computed directly from
   `features_teammate_merged.csv`'s 18 `FEATURE_COLS_V2` columns (no pre-existing correlation CSV
   exists for this feature set in `artifacts_research_v2/`, so this was computed fresh by the audit
   script). `amount_to_balance_ratio` and `high_amount_transaction` correlate strongly with
   `TransactionAmount` by construction (they are derived from it); `device_frequency`,
   `ip_frequency`, `account_frequency`, `merchant_frequency` correlate with each other only weakly to
   moderately, indicating they carry substantially independent behavioural signal rather than being
   redundant restatements of one another.

5. **`v2_feature_model_importance_heatmap.png`** — Rows = 18 features, columns = 12 models.
   Isolation Forest and Autoencoder columns are **real SHAP** (mean|SHAP| per feature, normalized to
   [0,1] within each column, from `shap_isolation_forest_v2.csv` / `shap_autoencoder_v2.csv`); all
   other 10 columns are a **clearly-labeled proxy** (|Spearman rho| between that model's raw score
   and the raw feature value, also column-normalized) — explicitly not SHAP, since only IF and AE
   have SHAP computed in this pipeline (`11_explainability.py` only covers those two). The two real
   SHAP columns visibly disagree on which features matter most (see item 10 below), which the proxy
   columns for the remaining 10 models should be read in light of, not as independent confirmation.

6. **`v2_model_characteristics_heatmap.png`** — Engineering-judgment 0/1/2 matrix (explicitly labeled
   as such, not data-derived) built from what each model's actual sklearn/PyTorch call does in
   `06_models_classical.py`/`07_models_deep.py`. It surfaces, at a glance, that only the three deep
   models use a neural network, only LSTM-AE uses sequence information, and DBSCAN/HDBSCAN are the
   only two models flagged "no out-of-sample scoring" — matching the explicit code-level caveat in
   §3.5/§3.6 above.

7. **`v2_model_training_data_heatmap.png`** — Verified per-model from source rather than assumed
   uniform. The single "Test involved during fitting = 1" row is DBSCAN/HDBSCAN, both of which fit
   `.fit_predict()` directly on `X_all` (train+val combined) with no held-out split at all — every
   other model fits exclusively on `X_train`. See §9 for the full leakage-safety statement, including
   why this is a real but bounded limitation, not classic supervised leakage.

8. **`v2_top_transactions_model_score_heatmap.png`** — Top 25 transactions by
   `ensemble_percentile_average`, each model's score rank-normalized to [0,1] for visual
   comparability, plus the actual Final Ensemble Risk column. Two rows (TX000225, TX001423) have a
   blank LSTM-AE cell — real missing data (those transactions belong to accounts with <3
   transactions, outside the LSTM-AE's applicable scope), not an error. Agreement across the top 25
   is visibly very high (nearly all cells ≥0.9) — these are consensus extreme outliers most models
   independently flag, not artifacts of one dominant model.

9. **`v2_anomaly_model_evaluation_heatmap.png`** — Only metrics actually computed are shown:
   Silhouette/Davies-Bouldin/log10(Calinski-Harabasz) for all 12 models
   (`internal_validity_metrics_v2.csv`) and 5-run bootstrap Jaccard stability for only 3 models —
   Isolation Forest (0.602), LOF (0.512), Autoencoder (0.373) — `stability_bootstrap_jaccard_v2.csv`
   does not cover the other 9 models, and those cells are left blank rather than estimated. Elliptic
   Envelope has the best internal-validity silhouette (0.541) of all 12 models on this feature set;
   Autoencoder has the worst (0.172) despite being the pipeline's continuous flagship deep model —
   a genuine tension worth flagging, not resolved by this audit.

10. **`v2_cross_model_shap_importance_heatmap.png`** — Real SHAP mean|feature importance| for
    Isolation Forest (TreeExplainer, exact) vs. Autoencoder (GradientExplainer on a reconstruction-MSE
    wrapper), from `shap_global_importance_comparison_v2.csv`. **Recomputed Spearman rho between the
    two models' full 18-feature importance vectors: ρ = −0.3705**, matching `15_final_research_report.md`'s
    stated figure exactly (cross-checked, not just copied) and the same top-10-overlap-of-3
    (`TransactionAmount`, `account_frequency`, `amount_to_balance_ratio`) finding. This is a
    materially *negative* correlation — the tree-based and reconstruction-based explainers are
    picking up substantially different signal on this 18-feature set, a sharper disagreement than
    the in-house 46-feature pipeline's reported ρ=−0.157 for the same two-model comparison.

## 9. Leakage-safety statement

- **RobustScaler** is fit on `X_train` only (`06_models_classical.py`), then applied to val/all —
  correctly avoids leaking val/test statistics into the scaler.
- **Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, K-Means, GMM, Autoencoder, VAE, LSTM-AE**
  are all fit exclusively on their respective train split (`X_train`, or the account-level train
  split for LSTM-AE). Validation rows are used only for *monitoring* (loss/MSE tracked in training
  history, never backpropagated into weights, never used for early-stopping/model selection in this
  pipeline's actual code — a fixed epoch budget is used throughout).
- **DBSCAN and HDBSCAN are the one real exception**: both call `.fit_predict()` directly on `X_all`
  (train+val combined), because neither has a usable native out-of-sample `.predict()` in this
  pipeline's usage — a genuine limitation of the algorithms' sklearn/hdbscan API, explicitly flagged
  in `12_ensemble_scoring.py`'s own docstring, and identical to the same limitation in the in-house
  46-feature pipeline (not a smaller-feature-set artifact). This means val-split transactions did
  directly influence DBSCAN/HDBSCAN's cluster structure and noise scores.
- **There is no held-out "test" split anywhere in this pipeline** — only an 80/20 train/val split.
  Every model's final score reported to the business/dashboard is computed over `X_all` (train+val
  combined), because this is unsupervised anomaly *scoring*, not supervised generalization
  *estimation*: every transaction in the dataset must receive a risk score, so there is no
  equivalent of a supervised test set held back from scoring. This is standard practice for
  unsupervised deployment and is **not equivalent to classical supervised-learning test-set
  leakage** (no label-based decision was ever made using held-out rows, because no label exists).
  The one place this distinction matters and is called out honestly above is DBSCAN/HDBSCAN's model
  *fitting* (not just scoring) using val rows — a narrower and real caveat, isolated to those two
  models only.
- **Hyperparameter selection objectives** (Isolation Forest's/GMM's Optuna searches, §5) are computed
  on `X_train` only; the VAE search additionally uses `X_val` for its minimization objective (val
  reconstruction MSE) — this is standard, appropriate use of a validation split for hyperparameter
  selection, not leakage.
- **No SMOTE, no class weighting, no pseudo-labels** anywhere in this pipeline — verified by reading
  every model-fitting function in `06_models_classical.py`/`07_models_deep.py`; there is nothing to
  balance because there is no label.

## 10. Relationship to the in-house `research` (46-feature) pipeline

Both `research` and `research_v2` run the identical 12-model methodology (same model list, same
sign conventions, same ensemble-strategy code structure — `12_ensemble_scoring.py` in each tree is
explicitly a 1:1 mirror per its own docstring) over the **same 2,512 raw transactions**, but a
**different feature set**: 46 in-house engineered features (`research`) vs. this teammate's 18
frequency/behavioural features (`research_v2`).

**Re-verifying the prior audit claim of ρ≈−0.007 between the two pipelines' ensemble scores:**
directly comparing `artifacts_research/ensemble_scores.csv` and
`artifacts_research_v2/ensemble_scores_v2.csv` on their shared TransactionIDs (all 2,512 IDs match
across both files), computed fresh for this audit:

| Strategy pair (research vs. research_v2) | Spearman rho | p-value |
|---|---|---|
| ensemble_weighted_average | 0.5828 | ~1.7e-228 |
| ensemble_rank_borda | 0.6027 | ~2.7e-248 |
| **ensemble_percentile_average (both pipelines' recommended strategy)** | **0.6021** | ~1.2e-247 |
| ensemble_pca_stacking_proxy | 0.5950 | ~1.9e-240 |

**This does not reproduce the ρ≈−0.007 figure — the actual correlation is a moderate-to-strong
positive ρ≈0.58–0.60, not "essentially uncorrelated."** Tracing the ρ≈−0.007 figure to its actual
source: it comes from `artifacts_research_v2/ensemble_vs_v1_crosscheck_v2.json`
(`spearman_vs_v1_vote_count_ROUGH_PROXY_NOT_GROUND_TRUTH`, values −0.0065 to −0.0069 across the 4
strategies), which compares `research_v2`'s ensemble scores against **`artifacts/anomaly_votes.csv`'s
`vote_count`** — the vote count from a *third, separate, earlier* pipeline (the original supervised
`v1` model-comparison pipeline in `artifacts/`, distinct from both `research` and `research_v2`), not
against the in-house `research` (46-feature) pipeline's ensemble scores. `research_v2`'s own
`hybrid_vote_count_for_comparison` column correlates at a more moderate ρ≈0.50–0.51 against the same
4 strategies (also in that JSON), for additional context.

**Conclusion: the prior memory's ρ≈−0.007 figure does not hold up as a description of "research vs.
research_v2 ensemble score correlation."** The two pipelines' final ensemble scores are, in fact,
moderately positively correlated (ρ≈0.60) despite using entirely disjoint feature sets — a genuinely
different (and more reassuring) finding than "essentially uncorrelated." What *does* independently
verify from the docs is the **feature-importance-level** disagreement: this audit's own recomputation
confirms `15_final_research_report.md`'s reported ρ=−0.3705 between research_v2's own IF-vs-AE SHAP
importance vectors (§8, item 10) — that is a real, verified negative-correlation finding, just not
the same claim as the ensemble-score-level comparison the ρ≈−0.007 figure was misattributed to.

---

## Artifact index

**Heatmaps** (`audit/heatmaps/`): `v2_model_spearman_correlation_heatmap.png`,
`v2_model_jaccard_agreement_heatmap.png`, `v2_model_disagreement_heatmap.png`,
`v2_feature_correlation_heatmap.png`, `v2_feature_model_importance_heatmap.png`,
`v2_model_characteristics_heatmap.png`, `v2_model_training_data_heatmap.png`,
`v2_top_transactions_model_score_heatmap.png`, `v2_anomaly_model_evaluation_heatmap.png`,
`v2_cross_model_shap_importance_heatmap.png`.

**Tables** (`audit/tables/`): `v2_model_spearman_correlation_matrix.csv`,
`v2_model_jaccard_agreement_matrix.csv`, `v2_model_disagreement_matrix.csv`,
`v2_feature_correlation_matrix.csv`, `v2_feature_model_importance_matrix.csv`,
`v2_model_characteristics_matrix.csv`, `v2_model_training_data_matrix.csv`,
`v2_top_transactions_model_score_matrix.csv`, `v2_anomaly_model_evaluation_matrix.csv`,
`v2_cross_model_shap_importance_matrix.csv`, `v2_research_vs_research_v2_ensemble_crosscheck.json`.

## Unverifiable / not computed in this pipeline

- A true cost-minimizing or accuracy-based threshold — no fraud label exists (stated explicitly in
  `13_threshold_optimization.py` and honored here rather than estimated).
- Stability/bootstrap Jaccard for 9 of the 12 models (only IF/LOF/AE were run) — left blank in
  heatmap 9 rather than guessed.
- Real SHAP for the 10 non-IF/AE models — only a clearly-labeled association proxy exists (heatmap 5).
- Whether GMM's BIC curve would keep improving past `n_components=10` — the search boundary was
  reached while BIC was still descending (§3, item 8); not extended in this pipeline's code, so not
  resolved by this audit either.
