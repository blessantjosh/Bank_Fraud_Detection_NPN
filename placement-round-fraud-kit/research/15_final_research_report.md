# Phase 17 — Final Research Report

**Unsupervised anomaly detection for retail banking transactions**
Dataset: `data/bank_transactions_data_2.csv` — 2,512 transactions, 495 accounts, 16 raw columns, **no fraud label**.

This report is written to stand alone. Every claim traces to one of the sixteen phase reports listed below, cited inline by phase number so any figure can be checked at source.

| Phase | Subject | File |
|---:|---|---|
| 1 | Business understanding | `research/01_business_understanding.md` |
| 2 | Data understanding | `research/02_data_understanding.md` |
| 3–4 | Data quality and EDA | `research/03_data_quality_and_eda.md` |
| 5 | Feature engineering | `research/04_feature_engineering.md` |
| 6–7 | Preprocessing and dimensionality reduction | `research/05_feature_selection_and_preprocessing.md` |
| 8 | Model development (12 models) | `research/06_model_development.md` |
| 9 | Hyperparameter optimisation | `research/07_hyperparameter_optimization.md` |
| 10 | Evaluation framework | `research/08_evaluation.md` |
| 11 | Explainability | `research/09_explainability.md` |
| 12 | Ensemble scoring | `research/10_ensemble_scoring.md` |
| 13 | Threshold optimisation | `research/11_threshold_optimization.md` |
| 14 | Final model selection | `research/12_final_model_selection.md` |
| 15 | Production architecture | `research/13_deployment_architecture.md` |
| 16 | Monitoring framework | `research/14_monitoring_framework.md` |
| 17 | This report | `research/15_final_research_report.md` |

---

## 1. Executive Summary

Twelve unsupervised anomaly-detection models were built, tuned, evaluated, explained, ensembled and thresholded against an unlabelled retail banking transaction dataset, and a production architecture and monitoring framework were designed on top of the result.

**What the system is.** A 46-feature engineering pipeline with verified leakage safety, feeding nine to eleven independent anomaly detectors whose scores are combined by percentile aggregation into a single score in (0,1), thresholded into two human-review tiers, and explained to an investigator through two structurally different SHAP views.

**The recommendation.** Percentile aggregation over the detectors that can score out-of-sample, with Isolation Forest and Autoencoder SHAP attributions shown side by side, feeding a two-tier review queue at the 99th percentile (26 transactions, 1.04%) and the 95th percentile (126 transactions, 5.02%). **No automatic blocking.**

**The single most important technical finding.** Isolation Forest and the Autoencoder — the two best-performing individual models, jointly ranked first in Phase 14's decision matrix — agree on almost nothing about *why* a transaction is anomalous. Their global feature-importance rankings correlate at Spearman **ρ = −0.157**, sharing 1 of 10 top features (Phase 11). This is not noise; the mechanism is understood, and there is a worked case (`TX000566`) where Isolation Forest alone ranks a perfectly unremarkable $29.38 transaction in its top 1% for a spurious reason that the Autoencoder correctly ignores. That finding is why the recommendation is an ensemble despite a single model being cheaper, faster and easier to operate.

**What was not proven, stated plainly.** There is no fraud label in this dataset, so **no detection-performance claim is made anywhere in this project** — no precision, no recall, no AUC against fraud. What was measured is internal consistency: how cleanly each model partitions the feature space, how much the models agree with each other, how stable each is under retraining, and how plausible the top-scored transactions look to a human reasoning from documented fraud typologies. Three of the top-1% transactions examined match known fraud signatures; one does not, and that one is reported in as much detail as the three that do.

**The honest scale position.** This is a research prototype validated at 2,512 rows. The original brief this work descends from describes roughly 1M rows (`LIMITATIONS.md`). Phase 15 §10 sets out specifically what breaks at that scale and what must be rebuilt rather than resized. The reasoning transfers; the fitted numbers do not.

---

## 2. Problem Statement

Detect fraudulent transactions in a retail banking ledger that carries **no fraud label**.

That constraint drives every subsequent decision. With no label there is nothing to train a classifier against, nothing to compute precision or recall against, and no cost-optimal threshold to solve for. The problem therefore becomes: *identify transactions that deviate from established behavioural patterns, rank them by how strongly they deviate, and give a human enough context to judge them* — while being rigorous about the difference between "statistically unusual" and "fraudulent."

Phase 1 mapped six fraud typologies against what this schema can actually observe, before any modelling began:

| Scenario | Observable here? | Proxy features |
|---|---|---|
| Account takeover | **Yes — strongest fit** | `DeviceNoveltyFlag`, `LocationNoveltyFlag`, `LoginAttempts`, `Amount_vs_AccountAvg` |
| Transaction bursts / card testing | **Yes** | Per-account velocity features, `TimeSinceLastTxn` |
| Unusual spending / compromised card | **Yes** | Rolling deviation features, `Amount_vs_AccountAvg` |
| Mule accounts | **Partially** — shared infrastructure visible, fund-flow chains not | `DeviceTxnCount`, `IPTxnCount`, `MerchantTxnCount` |
| Synthetic identities | **Weakly** — no account-opening date or KYC field exists | `CustomerOccupation` / `CustomerAge` vs. behaviour |
| Money laundering (layering) | **No direct signal** — requires a counterparty ledger or transaction graph, neither of which exists | none |

Phase 1's instruction to downstream phases was explicit and was honoured: layering is out of scope and must not be claimed as detected anywhere, even if an anomaly model happens to flag a transaction a human might narrate that way.

The operating definition of "normal" is per-account, not global. A $14 debit is unremarkable for one customer and a red flag for another; a $5,000 transaction is routine for a high-balance account and alarming for one that never exceeds $200. Every meaningful feature in this system is therefore relative to the account's own history.

---

## 3. Dataset Overview

| Fact | Value |
|---|---:|
| Rows | 2,512 |
| Columns | 16 |
| Unique accounts | 495 (avg. 5.08 transactions each) |
| Unique transaction IDs | 2,512 (100% unique) |
| Missing cells | **0** |
| Duplicate rows | **0** |
| Date span | 2023-01-02 to 2024-01-01 (364 days) |
| Fraud labels | **none** |

Available fields: `TransactionID`, `AccountID`, `TransactionAmount`, `TransactionDate`, `TransactionType`, `Location`, `DeviceID`, `IP Address`, `MerchantID`, `Channel`, `CustomerAge`, `CustomerOccupation`, `TransactionDuration`, `LoginAttempts`, `AccountBalance`, `PreviousTransactionDate`.

**Two structural defects were found in the raw data and worked around rather than ignored** (Phase 2):

1. **`PreviousTransactionDate` is unusable as designed.** It contains only 7 distinct timestamps across all 2,512 rows, spanning 6.0 minutes on 2024-11-04 — roughly ten months *after* the latest `TransactionDate`. It is a single bulk-export moment stamped onto every row, not per-account history. Every recency and velocity feature is derived from `TransactionDate` sorted per account instead.
2. **`TransactionDate` is not realistic 24/7 timestamp data.** All 2,512 transactions fall Monday–Friday — **zero** weekend transactions — and **100% fall within a 16:00–18:21 window** (52.4% at 16:00, 32.6% at 17:00, 15.0% at 18:00), with Monday alone accounting for 42.6% of volume. This caps what time-of-day features can contribute: there is no overnight or weekend baseline, so classic "3am transaction" fraud signals are structurally invisible here.

Cardinality: `AccountID` 495 (24 accounts appear exactly once, so have no history to build a baseline against), `DeviceID` 681, `IP Address` 592, `MerchantID` 100, `Location` 43, `Channel` 3, `CustomerOccupation` 4, `TransactionType` 2.

---

## 4. EDA Findings

*Source: Phases 2–4. Plots in `research/plots/`.*

**Distributions.** `TransactionAmount` is strongly right-skewed (skew 1.74, excess kurtosis 3.63): min $0.26, median $211.14, mean $297.59, max $1,919.11 — the median well below the mean, confirming a small number of large transactions pull the average up. `AccountBalance` spans $101.25–$14,977.99 (mean $5,114.30, skew 0.60). `CustomerAge` is near-symmetric across 18–80 (skew 0.15). `TransactionDuration` runs 10–300 seconds (median 112.5s, skew 0.60).

**`LoginAttempts` is a rare-event flag disguised as a numeric column.** Skew 5.17, excess kurtosis 26.61: **95.14% of rows are exactly 1**, and only 122 rows (4.86%) take values 2–5. Phase 4 recommended engineering it as a near-binary indicator rather than feeding a smooth numeric value to distance-based models, and Phase 5 did so.

**The features carry near-independent information.** Every pairwise Pearson correlation among the five raw numeric features is below |0.03| except `CustomerAge` ↔ `AccountBalance` (Pearson 0.320, Spearman 0.404). The mutual-information matrix agrees non-linearly — MI(Age, Balance) ≈ 0.32 nats dominates, everything else ≤ 0.02 — so no hidden non-linear dependency is being missed. All VIFs sit at 1.001–1.115, far below the conventional concern threshold of 5: **no multicollinearity, no feature needs dropping for redundancy.**

**PCA confirms the same thing structurally.** The scree curve is nearly flat — 26.50%, 20.67%, 19.95%, 19.30%, 13.58% — so all five components are needed for 100% variance and the first two capture only 47.2%. With near-orthogonal inputs there is simply nothing for PCA to compress. The loadings read almost as a relabelling of the original features (PC3 ≈ `TransactionAmount` alone at 0.966).

**One statistically significant categorical association.** `Channel` × `TransactionType`: χ² = 136.91, dof = 2, **p = 1.87 × 10⁻³⁰**. ATM transactions are 91.2% Debit, while Branch and Online each carry a Credit share around 29–30%. The rare ATM+Credit combination (73 rows, 2.91%) became a dedicated interaction feature on the strength of this.

**Outliers were kept, deliberately.** Five detection methods were run on each numeric feature. They disagree substantially — IQR and Modified Z-score flag the identical 122 rows on `LoginAttempts` (Jaccard 1.0) but overlap only at Jaccard 0.187 with percentile bounds on `TransactionAmount`, and IQR/Z-score/Modified-Z return **zero** flags on `CustomerAge`, `TransactionDuration` and `AccountBalance`. A jointly-fit Isolation Forest at `contamination=0.05` flags 126 rows (5.02%) that overlap each univariate method at only Jaccard 0.16–0.24. That divergence is itself the finding: **real fraud is rarely extreme on one feature; it is an unusual combination.** No outliers were removed or capped — in an unsupervised fraud system they are the candidate signal, and deleting them would delete the population the system exists to find.

**Nonlinear projections.** t-SNE and UMAP both show one dominant diffuse population plus small isolated pockets — UMAP separates a tight 95-point cluster (3.8%) that PCA's linear projection cannot. Phase 7 checked it for an obvious single-feature driver and found none (its mean `LoginAttempts` is *lower* than the rest of the data, 1.042 vs 1.128), so it is reported as plausible local structure rather than a validated customer segment.

---

## 5. Feature Engineering

*Source: Phase 5. Output: `artifacts_research/features_v2.csv`, 2,512 × 48 (46 model features + 2 ID columns).*

The 46 features comprise v1's 20 features reused unchanged plus 26 new ones. They fall into six families: per-account velocity, expanding and rolling amount baselines, amount ratios and deviations, cyclical time encodings, behavioural flags, and network proxies.

**Leakage safety is verified, not assumed.** Every feature touching account history or shared-infrastructure history sees only strictly prior rows — `groupby().rolling(closed='left')` for time windows, `shift()` before every expanding window, `cumcount()` for tenure, and an explicit prior-only accumulator for the network features. Phase 5 confirmed alignment by recomputing `TimeSinceLastTxn` independently and checking it matched row-for-row against the v1 feature matrix (**confirmed: MATCH**).

**The load-bearing feature is `Amount_ZScore_Account`** — `(amount − account's prior expanding mean) / (prior expanding std)` — the most direct available answer to "how unusual is this for *this* customer." Its denominator is floored at **5% of the dataset-wide `TransactionAmount` standard deviation ($291.95 → $14.60)** rather than at an arbitrary epsilon. Phase 6 records why: an earlier build used a near-zero epsilon, and accounts with 2–3 near-identical prior transactions drove the feature into the hundreds of millions, swamping every other feature in the autoencoder's loss. Post-fix the feature ranges −92.60 to 102.80 — large, but explicable from genuine account-level variability.

**Three features were built and then honestly reported as near-useless on this dataset:**

- **Velocity.** `Velocity_1D_Count` is zero for **98.0%** of rows and `Velocity_7D_Count` for **91.1%** — the direct consequence of 495 accounts averaging 5.08 transactions across 364 days. On this data it is a rare high-precision flag, not a graded signal.
- **Cyclical time.** `Hour_sin` has standard deviation **0.040** and `DOW_sin`/`DOW_cos` take only 5 distinct values, because of the weekday/3-hour-window artifact in §3. Retained for pipeline completeness; carries no discriminative signal here.
- **Network proxies.** `DeviceSharedAccounts_Prior` > 0 for 72.8% of rows and `IPSharedAccounts_Prior` for 76.4%. Dataset-wide, **89.4% of the 681 devices and 93.2% of the 592 IPs are used by more than one account.** In a real bank that would be a mule epidemic; here it is unambiguously an artifact of quasi-random device/IP assignment during data generation. The leakage-safe logic is exactly what real mule detection needs, but on *this* dataset high values are the norm and must not be read as fraud evidence.

**Scaling: `RobustScaler`, chosen on measured grounds** (Phase 6). A finding worth stating because a naive comparison would have missed it: skewness is mathematically invariant under any affine transform, and this was confirmed empirically — `TransactionAmount` skew is **1.7391 raw and identically 1.7391** after Standard, MinMax *and* Robust scaling, to four decimals. A skew comparison therefore cannot distinguish the three. What does distinguish them is each scaler's denominator sensitivity, measured by trimming the top 1% of values: the standard deviation moves **13.94%** on average, the range **30.42%**, the IQR **2.40%**. For a system whose purpose is to retain and score outliers, the scaler's baseline must not be dictated by them. `QuantileTransformer` was rejected for the opposite reason — it reshapes skew to ≈0 and would compress the $1,919.11 maximum into "just the largest rank," discarding exactly the magnitude information an anomaly detector needs.

**Encoding.** Frequency encoding of `Location` beat label encoding against a rough proxy (R² 0.0027 vs 0.0000); one-hot beat both (0.0301) at a cost of 42 extra columns. All three numbers are tiny because the proxy has almost no variance to explain, and `Location` was never expected to be a strong standalone predictor. The recommendation — frequency over label encoding — rests on the structural argument as much as the numbers: label encoding's alphabetical integer codes impose a false ordinal relationship between cities.

---

## 6. Model Comparison

*Source: Phase 8. Twelve models, one shared 46-column feature matrix, one shared `RobustScaler` fit on the 2,009-row training split, all scores oriented higher = more anomalous.*

Models 1–4 and 7–10 were fit on the 2,009-row training split and scored all 2,512 rows out-of-sample. **DBSCAN and HDBSCAN have no out-of-sample `.predict` and had to be fit on the full dataset** — a methodological footnote in research that becomes an architectural constraint in production (§13). The LSTM Autoencoder required a separate account-level split, since a single account's chronological sequence cannot be split across train and validation without breaking it.

| Model | Flagged rate | Internal-validity Silhouette | ρ vs. independent v1 proxy |
|---|---:|---:|---:|
| HDBSCAN | 53.94% | **0.672** | 0.393 |
| One-Class SVM | 5.21% | 0.664 | 0.288 |
| K-Means | 5.02% | 0.663 | 0.280 |
| LSTM-AE | 4.82% | 0.643 | 0.329 |
| LOF | 4.86% | 0.617 | 0.428 |
| DBSCAN | 1.23% | 0.615 | 0.118 |
| Elliptic Envelope | 4.62% | 0.610 | 0.323 |
| Isolation Forest | 5.33% | 0.565 | 0.403 |
| Autoencoder | 5.02% | 0.496 | 0.386 |
| VAE | 5.02% | 0.496 | 0.385 |
| Hybrid Ensemble | 3.74% | 0.467 | **0.457** |
| GMM | 5.02% | 0.319 | 0.236 |

**This table is not a leaderboard, and reading it as one would be the single easiest way to reach a wrong conclusion.** Four qualifications:

1. **Silhouette structurally favours distance-based methods.** A top-5%-by-distance cut is almost guaranteed to separate well *in a distance metric*. The reconstruction-error models (Autoencoder, VAE) score lower not because they are worse but because an autoencoder's bottleneck can compress non-adjacent points to similar codes — low reconstruction error does not imply Euclidean proximity.
2. **HDBSCAN tops the ranking and is the least deployable model in the set.** Its *continuous GLOSH score* ranks well; its *native clustering output* calls 53.94% of the data noise (53.9%–75.0% across configurations) and never finds more than 2 clusters. Both findings are true and neither cancels the other.
3. **The v1 proxy is not ground truth.** It is the vote count from a separate, smaller 4-detector pipeline on a different feature set. The Hybrid Ensemble topping that column is mechanically driven — Isolation Forest and LOF appear in both — not independent confirmation.
4. **The flagged rates are largely set by construction.** Ten of the twelve cluster in 4.6%–5.3% because they either take a `contamination≈0.05` parameter or use the standardised top-5% convention. The rate carries little information; the *agreement on which rows* carries all of it.

**Cross-model agreement** is where the real structure is. Strongest pairs: LOF ↔ HDBSCAN ρ=0.840, Autoencoder ↔ VAE ρ=0.801, LOF ↔ Autoencoder ρ=0.787, Isolation Forest ↔ Elliptic Envelope ρ=0.758. Weakest: One-Class SVM ↔ GMM ρ=**−0.052** (the only negative pair), DBSCAN ↔ GMM ρ=0.010. **GMM and DBSCAN are consistently at the bottom of both the correlation and the flagged-set-overlap tables** — likelihood-based and single-cluster-density-based anomaly definitions diverge most from the distance/reconstruction-based majority. A system relying on either alone would be flagging a materially different population.

**Model-specific findings worth carrying forward:**

- **Elliptic Envelope's core assumption is measurably false.** Shapiro-Wilk on all 46 scaled features: **100% reject normality at p<0.05**, and sklearn raised a rank-deficiency warning during fitting. Kept for comparison; explicitly not recommended as a production detector.
- **K-Means needed a real fix, not a footnote.** Naive `argmax(silhouette)` picks k=2 with silhouette 0.9184 — but that "solution" is a 3-row micro-cluster against a 2,006-row majority. Every k from 2 to 10 produces a cluster holding <1% of training rows. Worse, naive nearest-centroid scoring would have made those extreme rows the *safest-looking* points in the dataset. The fix: k=4 from the inertia elbow, scoring distance only to clusters holding ≥1% of training rows.
- **The LSTM Autoencoder was feasibility-checked before being built.** 495 accounts, median 5 transactions, max 12; 428 accounts (86.5%) have ≥3, covering 2,402 rows (95.6%). It was built on that subset, and the remaining **110 rows (4.4%) get no score at all** — a permanent, reported coverage gap. Its reconstruction is markedly worse than the feedforward autoencoder's (val MSE 1.258 vs 0.328), which is what the sequence-length distribution predicts: ~342 training sequences of median length 5 give a recurrent model little repeated temporal structure to learn.
- **The VAE behaves exactly as beta-VAE theory predicts.** Higher reconstruction MSE than the plain autoencoder at every percentile except the maximum (val 0.379 vs 0.328; P99 1.946 vs 1.372; max 4.395 vs 6.708) — part of its loss budget goes to matching the latent prior, producing a smoother, less extreme tail.

---

## 7. Hyperparameter Tuning

*Source: Phase 9. Three models tuned with Optuna's TPE sampler: Isolation Forest, GMM, VAE.*

With no label, "optimise against what" is a genuine design decision and was stated per model rather than left implicit: silhouette between the top-5% group and the rest (Isolation Forest), BIC on the training split (GMM), validation reconstruction MSE (VAE).

| Model | Method | Result |
|---|---|---|
| Isolation Forest | Exhaustive grid, 60 combos, 63.9s | Silhouette **0.6092** |
| | Optuna TPE, 30 trials, 47.6s | 0.5981 |
| | Random search, 30 trials, 45.7s | 0.5884 |

**Reported plainly rather than spun: Bayesian optimisation did not win here.** The exhaustive grid found the best configuration; Optuna came 0.0111 short in half the evaluations, and random search was statistically indistinguishable from Optuna on the same budget (a 0.0097 gap). The honest read is that a 3-hyperparameter, cheap-to-evaluate search space is too small for Bayesian optimisation's main advantage to pay off. Optuna's TPE sampler *did* show its intended behaviour — locking onto a good region by trial 6 while random search was still improving at trial 12 — but the final quality gap across all three methods is under 0.02.

**GMM: the search found a better number and a worse model.** Adding `reg_covar` to the search space moved the best BIC from −63,019.3 to −109,595.2, an improvement of 46,575.9. It got there by pushing `n_components` to the boundary of the search range (10) and driving `reg_covar` to 1.19×10⁻⁶, more than 8× smaller than the grid's fixed value. A weaker floor under the covariance eigenvalues lets a full-covariance component — 1,081 free parameters, ten of them, against 2,009 training rows and 46 features — fit idiosyncratic structure faster than BIC's complexity penalty can catch up. The BIC curve keeps descending at the search boundary rather than showing an interior minimum. **Conclusion carried forward: raw BIC-driven search on full covariance should not be trusted in this dimension/sample-size regime**; the `tied` option remains the more defensible production choice even though it never wins on raw BIC.

**VAE: the one case where Optuna's efficiency argument genuinely applies**, because each trial is a full 60-epoch training run rather than a cheap fit (20 trials, 282.6s — the slowest search of the three). It surfaced a clear, interpretable result: **`beta` dominates the other three hyperparameters.** Every trial with beta above ~0.3 landed at validation MSE above 0.8; every trial below ~0.1 landed below 0.45, regardless of latent dimension or hidden width. The deployed model's beta=0.1 sits right at the edge of the good region.

**Net conclusion:** Bayesian optimisation's value was not uniform — most useful for the most expensive model to evaluate, least useful for the cheapest, which is the theoretically expected pattern rather than a project-specific surprise.

---

## 8. Evaluation Results

*Source: Phase 10. With no label, evaluation means four label-free things, none of which substitutes for precision and recall.*

### 8.1 Internal validity

Every model's top-5%-by-score partition was scored on Silhouette, Davies-Bouldin and Calinski-Harabasz in the shared 46-feature scaled space, using one consistent partition definition across all twelve so the comparison is like-for-like. Results are in §6's table, with the four qualifications that go with them. GMM is the clear low point on all three simultaneously (0.319 / 4.113 / 20.7) — independently confirming the divergence Phase 8 found in the correlation matrices.

### 8.2 Stability — the most operationally significant result in the project

Isolation Forest, LOF and the Autoencoder were each refit on 5 bootstrap resamples of the training split and rescored, and the top-5% flagged set recomputed each time.

| Model | Mean pairwise Jaccard (5 runs) | Min | Max |
|---|---:|---:|---:|
| LOF | **0.590** | 0.465 | 0.703 |
| Autoencoder | 0.533 | 0.448 | 0.658 |
| Isolation Forest | 0.527 | 0.448 | 0.565 |

**Roughly 41–47% of flagged transactions change between retrains on resamples of the same underlying data.** No drift, no new data, no hyperparameter change — just a different bootstrap sample. The mechanism is understood: the top-5% cut sits on a graded distribution rather than on a small set of stark outliers, so small shifts in the fitted boundary move a substantial number of borderline transactions across it. LOF is the most stable, but the 0.527–0.590 spread is too narrow to call any one of the three solved.

Two things follow, and both shaped the final design. **Operationally:** a fixed model artifact plus a monitored, versioned retraining process is the correct posture, not continuous retraining, and an operations team must be told the expected churn band in advance or they will reasonably conclude the system is broken (Phase 16 §5.2). **Methodologically:** any single-model result at the borderline should be treated as provisional, which is part of the argument for aggregating detectors.

### 8.3 Reconstruction quality

| Model | Train MSE | Val MSE | Val P95 | Val P99 | Val Max |
|---|---:|---:|---:|---:|---:|
| Autoencoder | 0.2896 | 0.3280 | 0.6433 | 1.3718 | 6.7077 |
| VAE | 0.3426 | 0.3790 | 0.7476 | 1.9461 | 4.3951 |
| LSTM-AE | 1.6364 | 1.2582 | — | — | — |

The autoencoder's train/validation gap (+13.1%) is small and expected for 2,009 rows through a 4-unit bottleneck, and its curves track a stable margin apart with no divergence. Its P99 (1.372) and max (6.708) sitting well above its P95 (0.643) is precisely the tail behaviour an anomaly detector needs: most rows reconstruct well, a small minority reconstruct much worse.

### 8.4 Business evaluation — reading real flagged transactions

Isolation Forest's top 1% (26 transactions) was examined by hand against Phase 1's scenario table. Five representative cases:

| Transaction | Amount | vs. account avg | Login attempts | Read |
|---|---:|---:|---:|---|
| `TX000275` | $1,176.28 | 3.4× | **5** (dataset max) | **Strongest match.** Transaction worth 3.6× the account's entire balance, combined with maximum observed login friction. The closest fit in the tier to Scenario 1 (account takeover) |
| `TX001354` | $1,510.71 | 149.0× | 1 | Extreme personal-baseline deviation, 73% of balance, and the transaction came 3,576 hours (~5 months) after the account's prior activity — a dormant-then-active pattern Phase 1 flags as an ATO signature. Equally consistent with a legitimate large one-off; **the data cannot distinguish these without a label**, and neither reading is forced |
| `TX000177` | $1,362.55 | 117.9× | 1 | Same shape. A pure amount-magnitude anomaly, no login friction, no burst |
| `TX002181` | $498.59 | 143.9× | 1 | **Ambiguous.** A moderate absolute amount that is extreme *relatively*, because this account's typical transaction is very small. No accompanying velocity burst, which argues against a card-testing reading |
| `TX000566` | $29.38 | **−0.96× (below average)** | 1 | **Does not look like fraud.** A small, unremarkable, below-average transaction with normal login behaviour and negligible balance impact. Included specifically to show that not every top-1% transaction is a fraud candidate |

A necessary caveat on reading these: `DeviceNoveltyFlag` is 1 for **99.52%** of all rows and `LocationNoveltyFlag` for **94.27%**, because with ~5 transactions per account almost nothing repeats. Both are near-constant and were not treated as independent signals in any of these narratives.

**The tail dilutes.** Spot-checking the bottom of the top-10% tier found small, round-number, near-account-average transactions with normal login counts and no velocity burst — ordinary transactions that enter the decile only because the score has a long, gradually-thinning tail rather than a sharp cliff. A review team using a 10th-percentile threshold should expect a meaningfully higher false-positive load than one using the 1st.

---

## 9. Explainability Results

*Source: Phase 11. `shap.TreeExplainer` (exact, 7.2s for all 2,512 rows) for Isolation Forest; `shap.GradientExplainer` on a wrapper returning per-row reconstruction MSE (128.8s) for the Autoencoder. Both run over the full dataset, not a subsample.*

**The central finding: the two best models explain their scores almost entirely differently.**

| Rank | Isolation Forest | mean\|SHAP\| | Autoencoder | mean\|SHAP\| |
|---:|---|---:|---|---:|
| 1 | `TransactionType_Debit` | 0.176 | `Amount_vs_AccountAvg` | 0.0390 |
| 2 | `CustomerOccupation_Retired` | 0.152 | `Amount_ZScore_Account` | 0.0367 |
| 3 | `CustomerOccupation_Engineer` | 0.143 | `Amount_to_Balance_Ratio` | 0.0367 |
| 4 | `CustomerOccupation_Student` | 0.139 | `Amount_to_RollingMean_Ratio` | 0.0360 |
| 5 | `LocationNoveltyFlag` | 0.125 | `TimeSinceLastTxn` | 0.0268 |

**Top-10 overlap: 1 feature of 10** (`TimeSinceLastTxn`). Spearman correlation between the two full 46-feature importance vectors: **ρ = −0.157** — essentially no agreement, slightly negative.

**The mechanism is understood, not mysterious.** Isolation Forest scores by how few random splits isolate a point, and a single split on a binary feature isolates an entire minority class in one step — so low-cardinality categoricals dominate its attributions. The Autoencoder scores by squared reconstruction error, dominated by whichever continuous features have the largest residuals after compression through a 4-dimensional bottleneck — and the amount-derived features have by far the widest dynamic range in this feature set. Phase 11's summary: **Isolation Forest is a "does this transaction's categorical and temporal shape look unusual" detector; the Autoencoder is an "is this transaction's amount unusual relative to this account's own scale" detector.** They are not redundant, and an operator relying on one is blind to the other's failure mode.

**The worked counterexample.** `TX000566` — the $29.38 below-average transaction from §8.4 — sits in Isolation Forest's top 1%. Its single largest SHAP driver, and the largest of any feature across all four local explanations examined, is `LocationNoveltyFlag` at **+0.960**. The raw value of that flag is **0**, meaning a *repeat* location. It carries that weight only because 0 is the rare value (5.73% of rows) in a flag that is 1 for 94.27% of the data — an artifact of the feature's construction, not a risk signal. Isolation Forest isolates rare categorical values quickly regardless of whether "rare" means "risky." The Autoencoder, scoring on amount-scale reconstruction error, assigned the same transaction SHAP magnitudes an order of magnitude smaller and did not share the read.

**Where they do agree, the agreement is informative.** On `TX000177`, every one of Isolation Forest's top drivers is amount-relative and all push the score up, with the Autoencoder independently reaching the same conclusion via `Amount_ZScore_Account` (+1.298) — a ~93-sigma jump from the account's own history. On `TX000275`, both models independently weight login friction and balance ratio, which is why that transaction is the project's strongest single fraud-signature match rather than merely its highest score.

**This is what makes the ensemble recommendation an evidence-based decision rather than a default.** Cross-checking a flagged transaction against a second model with a structurally different basis catches the `TX000566` class of false signal before it reaches a human reviewer.

---

## 10. Ensemble Scoring and Thresholds

*Source: Phases 12–13.*

Eleven of the twelve models were combined; the Hybrid Ensemble was deliberately excluded as an input, because it is itself a majority vote of Isolation Forest + LOF + Autoencoder and folding it back in would silently double-count those three.

Four aggregation strategies were built: **weighted average** (disagreement-inverse weights, ranging HDBSCAN 0.113 down to DBSCAN 0.058 — the scheme correctly down-weights the two models Phase 8 found most divergent), **rank aggregation** (Borda), **percentile aggregation**, and a **PCA stacking proxy** (explicitly not supervised stacking — there is no label to fit a meta-learner against; PC1 explains **52.65%** of the variance across the eleven standardised score columns).

**An honest finding rather than a manufactured distinction: Borda and percentile aggregation are near-identical here** — Spearman ρ = 0.9999, Jaccard 0.953 on the top-5% set. That is expected: summing ranks and averaging rank/N are the same operation up to a normalisation constant, and the two diverge only in how they handle LSTM-AE's 110 missing rows. They should not be presented as materially different signals. All four strategies correlate almost identically with the independent v1 proxy (0.442–0.444), so that cross-check does not separate them either.

**Percentile aggregation was selected** on three grounds: it is robust to eleven heterogeneous native scales (reconstruction MSE, a bounded kernel decision value, a GLOSH score, a negative log-likelihood, a centroid distance) with no assumption about their shapes; it has no tuned weights to defend to a reviewer, where the weighted average's weights come from a correlation matrix that will shift with more data; and it handles missing model scores by skipping and renormalising, which is the degradation behaviour a production system needs.

**Thresholds** (Phase 13). Score distribution: mean 0.5001, std 0.2029, min 0.0862, max 0.9988.

| Tier | Percentile | Score cut | Flagged |
|---|---|---:|---|
| Priority review | 99th | 0.9145 | 26 (1.04%) |
| Standard review | 95th | 0.8406 | 126 (5.02%) |

**A genuine methodological finding:** the classic statistical thresholds **flag zero transactions** on this score. mean+3σ = 1.1088 and Q3+1.5×IQR = 1.1363, both above the observed maximum of 0.9988 — because averaging eleven bounded percentiles compresses the tails, a CLT-like effect. This is not a defect in percentile aggregation; it is a mismatch between one thresholding convention and one score's shape, and it was confirmed by applying the identical rules to two *unbounded* scores, which both produced usable cut points (Isolation Forest's raw score: 25 and 43 flagged; the weighted-average ensemble: 17 and 87). If a "three standard deviations from typical" framing is wanted, apply it to the weighted average, not to the percentile score.

**Cost framing, bounded honestly.** v1's cost-optimal threshold methodology — sweeping a decision boundary to minimise `FP × $5 + FN × $250` — **cannot be reproduced here**, because a false-negative count requires knowing which *unflagged* transactions are fraud. What can be computed is an upper bound on review labour assuming every flagged transaction is a false positive: $630 at the 95th percentile, $130 at the 99th, using v1's own illustrative (not real-bank) figures. That is a ceiling, not an estimate. **A lower threshold is not "cheaper" in any total-cost sense** — it only reviews fewer transactions, trading off against catching less of whatever fraud is present, in a direction this project cannot quantify.

---

## 11. Business Insights

1. **Personal baseline beats global baseline, decisively.** The transactions that read most convincingly as fraud are not the largest in absolute terms. `TX002181` is $498.59 — unremarkable against a $1,919.11 maximum — but 143.9× its own account's average. Meanwhile, `Channel` barely separates transaction size at all (ATM $307.72, Online $297.21, Branch $288.23 mean). Any rule built on absolute amount thresholds would miss the former and generate noise on the latter.
2. **Login friction is the sharpest single available fraud signal, and it is rare enough to act on.** Only 4.86% of transactions have 2+ login attempts. The project's strongest fraud-signature match (`TX000275`) combines the dataset's maximum login attempts with a transaction worth 3.6× the account balance — and both explainability models independently rank login features among its top drivers. A simple operational rule (elevated login attempts *and* an amount above the account's normal range) is cheap, explainable, and grounded in the evidence here.
3. **Statistical rarity is not fraud, and the system must be operated on that basis.** `TX000566` is the proof: top 1% by score, driven entirely by a rare *value* of a near-constant engineered flag, and plainly not suspicious on inspection. This is why the recommendation includes a second, structurally different model as a mandatory cross-check, and why every output goes to a human rather than to an automatic block.
4. **Alert volume is a threshold policy decision, not a model property.** The 95th percentile flags 4.8× as many transactions as the 99.5th. The evidence supports a signal-strength gradient — the top 1% concentrates the clearest candidates, and the tail toward the 10th percentile dilutes into transactions that are merely mildly unusual — so the choice of tier is a review-capacity decision that this analysis can inform but not make.
5. **Retraining churns the queue by design, and this must be communicated before it happens.** 41–47% of flagged transactions change between retrains with no drift at all. An operations team that discovers this after a routine model refresh will reasonably assume something has broken.
6. **Two of this dataset's most intuitively appealing signals are artifacts and must be re-earned on real data.** Device/IP sharing across accounts (89.4% / 93.2%) and time-of-day patterns (a 3-hour weekday window) look like rich fraud signal and are neither. On real data both become genuinely informative — which is an upside, but only after refitting.

---

## 12. Limitations

Stated in full, because this section is the one a bank stakeholder should read most carefully. The first three carry forward from the v1 pipeline (`LIMITATIONS.md`) and **apply with more force here, not less.**

**12.1 No ground-truth fraud label exists — and this pipeline has no label at all.** v1 at least constructed a proxy label (3+ of 4 detectors agreeing) and trained a supervised model against it, reporting a **0.943** ROC-AUC that measured how well XGBoost reproduced the anomaly ensemble's own judgment, not fraud-catching accuracy. (`LIMITATIONS.md` cites this figure as 0.97; that number appears nowhere in v1's measured results — see §12.8(d).) **This pipeline does not even have that.** Its validation is entirely internal: cluster-validity metrics, cross-model agreement, retrain stability, and manual plausibility review. **No precision, recall, ROC-AUC, PR-AUC or false-positive rate against fraud is reported anywhere in this project, because none can be computed.** Every threshold, every "top 1%," every "plausible ATO pattern" is a statement about statistical unusualness plus human judgement, not about verified fraud.

**12.2 Circularity, in a sharper form than v1's.** v1's label was circular because the same features were engineered, then scored by detectors, then used to train the classifier — so strong SHAP attribution confirmed internal consistency, not causality. Here the circularity is more direct: **the system's definition of "anomalous" is the consensus of eleven detectors, and its primary validation is that those eleven detectors agree with each other.** Phase 8's cross-model correlations, Phase 12's PC1 variance and Phase 16's concept-drift proxy all measure the same consensus from different angles. Consistent detectors can be consistently wrong, and nothing in this project could tell the difference.

**12.3 Scale mismatch with the brief.** The problem statement describes ~1M rows; this dataset has 2,512 transactions across 495 accounts, spanning 364 days at 6.90 transactions/day. Nothing here has been stress-tested at production volume or against adversarial or evolving fraud patterns. Phase 15 §10 enumerates what breaks: One-Class SVM past ~50k–100k rows, LOF's O(n²) neighbour search, DBSCAN/HDBSCAN's full-refit requirement, pandas batch feature engineering, and the absence of a feature store for the real-time path.

**12.4 The dataset carries at least three generation artifacts.** The 16:00–18:21 weekday-only timestamp window; `PreviousTransactionDate`'s 7 values in a 6-minute export band; and 89.4%/93.2% of devices/IPs shared across accounts. Each was found, documented and worked around rather than modelled through — but each means the corresponding feature family is untested on realistic data.

**12.5 Some engineered features are near-constant here and therefore unvalidated.** `DeviceNoveltyFlag` (99.52% ones), `LocationNoveltyFlag` (94.27%), `Hour_sin` (std 0.040), `Velocity_1D_Count` (98.0% zeros). They are built correctly and would be informative on real data; on this data they contribute little, and one of them (`LocationNoveltyFlag`) actively produced a false signal.

**12.6 The recommended online model set has not been computed.** Phase 14 recommends percentile aggregation over the nine models that can score out-of-sample, but the published `ensemble_percentile_average` is the **11-model** score. The 9-model variant, and the 2-model fallback, must each be validated against the published score before their thresholds are used.

**12.7 No ensemble strategy has a measured stability figure.** Phase 10 bootstrap-tested three individual models. The assumption that aggregation reduces flagged-set churn below the measured 0.527–0.590 band is standard and plausible, and it is **untested here**. Phase 14 §5 identifies this as the highest-value missing measurement in the project.

**12.8 Five factual errors were found in earlier project documents while synthesising Phase 14.** They are recorded rather than quietly corrected, with full detail in `research/12_final_model_selection.md` §5:

(a) Phase 8 §3.1 labels the LSTM-AE's 4.82% flagged rate as being "of applicable rows" when it is of all 2,512 rows — §2.11 of the same report has the correct 5.04% figure (121 flagged of 2,402 applicable).
(b) Phase 10 §1 describes the Hybrid Ensemble's 253-row partition as the "majority-vote threshold," but the majority (≥2 of 3) rule flags **94** rows — 253 is the ≥1-vote set produced by a tie-inflated top-5% cut on a 4-valued score. Its Silhouette of 0.467 is therefore measured on a partition 2.7× larger than the ensemble's actual operating flag.
(c) Phase 10 §4 calls `TX000177` (z=92.56) "the single most extreme z-score in the whole dataset," and Phase 8 §1.7 makes a similar claim about `TX000177` and `TX002305` — the actual maximum is `TX001354` at **102.80**, with `TX000341` at 101.69 second, making `TX000177` third.
(d) `LIMITATIONS.md` cites v1's ROC-AUC as **0.97**; the measured values are 0.9428 (SMOTE), 0.9532 (class-weighted) and 0.943 (shipped model). The caveat's substance is right and important; the number overstates v1's performance.
(e) Phase 8's header lists `artifacts_research/vae.pt` and `artifacts_research/lstm_ae.pt`; both files are in `artifacts_research/models/`.

All five are labelling or transcription errors over correct underlying numbers. None changes a conclusion in this report, and all five should be corrected at source.

**12.9 Before production deployment.** Validate against a sample of real, investigator-labelled fraud cases from the bank's own case-management system; re-tune the 5% contamination assumption (documented as unverified in `src/config.py`) against the bank's actual historical fraud rate; and re-run the cost-based threshold analysis with the bank's real cost-of-fraud and cost-of-friction figures instead of the illustrative $5/$250 used here.

---

## 13. Future Improvements

**Near-term, days of work, high value:**

1. **Bootstrap-test the ensemble strategies** (Phase 14 §5). Re-run Phase 10's procedure end-to-end through the Phase 12 aggregation. This closes the largest evidential gap in the project and directly tests the assumption the final recommendation rests on.
2. **Compute and validate the 9-model online score** against the published 11-model version, using the Spearman and top-5% Jaccard measures Phase 12 already established. Until this exists, the recommended architecture has an unvalidated component at its centre.
3. **Build the canary reproducibility check and the `n_models_contributing` metric** (Phase 16 §6). Percentile aggregation degrades silently when a model goes missing — the score still lands in (0,1) and nothing errors. These two monitors take an afternoon and close every silent-failure mode the architecture has.
4. **Split and pin the dependency manifest.** `requirements.txt` describes the v1 pipeline only; the research pipeline additionally needs `torch`, `optuna` and the standalone `hdbscan` package, and the dashboard needs `fastapi` and `uvicorn`. A clean-machine install cannot currently run Phases 8, 9 or the dashboard.

**Medium-term, weeks:**

5. **Migrate the Argus dashboard onto the Phase 12 ensemble score** (Phase 15 §7.2). Steps 1, 2 and 4 of that migration are a day's work and would remove the supervised XGBoost layer entirely — eliminating the label circularity described in §12.2 from what an analyst actually sees.
6. **Show both SHAP views in the investigation UI.** Given ρ = −0.157 between the two models' importance rankings, showing one is showing half the picture.
7. **Persist analyst verdicts in a joinable schema.** The Argus Investigation Queue already writes Approve/Escalate/Block decisions to `dashboard/backend/queue_state.json` on first use. That is not a fraud label, but it is human, model-independent, and cumulative — and it is the cheapest available route out of the circularity problem.
8. **Enable HDBSCAN's `prediction_data=True`** and revalidate. It is the highest-weighted member of the weighted-average scheme (0.113) and is currently excluded from the online path solely because a fit-time flag was not set.

**Longer-term, requiring more data or infrastructure:**

9. **Validate against real investigator-labelled cases.** This is the only improvement that changes the project's epistemic position rather than its engineering. Everything in §12.1 and §12.2 dissolves the moment even a few hundred confirmed outcomes exist.
10. **Build the real-time feature store** (Phase 15 §10.2). The required per-account and per-device state is fully enumerated; `artifacts/reference.pkl` is already a partial implementation holding roughly 4 of 12 needed elements.
11. **Add graph features when a counterparty ledger exists.** Phase 1 identified money-laundering layering as structurally undetectable here for want of a beneficiary-account field. That is a data gap, not a modelling gap, and it is the largest single capability this schema is missing.
12. **Revisit the LSTM Autoencoder on longer histories.** Its poor showing (val MSE 1.258 vs 0.328) is well-explained by median-5 sequences and ~342 training examples. On real account histories the conclusion could reverse — but it should be re-tested, not assumed either way.

---

## 14. Final Recommendation

**Deploy percentile aggregation over the detectors that can score out-of-sample, with a mandatory two-model explanation layer, feeding a two-tier human review queue.**

| Component | Choice |
|---|---|
| **Score** | Percentile aggregation (Phase 12). Reference implementation: `ensemble_percentile_average` over 11 models for batch; a 9-model variant excluding DBSCAN and HDBSCAN for incremental scoring — **to be validated against the reference before use** |
| **Secondary score** | Phase 12's weighted average, computed in parallel at no extra model cost, because it is unbounded and therefore supports sigma/IQR thresholds that the percentile score cannot |
| **Explanation** | Isolation Forest SHAP (`TreeExplainer`, exact) **and** Autoencoder SHAP (`GradientExplainer`), both shown, never one alone |
| **Priority review** | Score ≥ 0.9145 (99th percentile) — 26 transactions, 1.04% |
| **Standard review** | Score ≥ 0.8406 (95th percentile) — 126 transactions, 5.02% |
| **Automatic blocking** | **None** |
| **Fallback** | If nine artifacts cannot be operated: percentile aggregation over Isolation Forest and the Autoencoder alone — also requiring its own validation |

### Why an ensemble, when a single model scores higher

Phase 14's decision matrix ranks Isolation Forest and the Autoencoder jointly first (4.20 weighted, 26/30 raw) and percentile aggregation third (3.40, 19/30). On cost, scalability and deployment readiness either single model beats any ensemble decisively — one artifact instead of nine, an exact 7.2-second explainer, a native out-of-sample decision function.

**The recommendation deliberately overrides its own matrix, and says so rather than adjusting the scores to hide it.** The reason is Phase 11: the two top-ranked models share 1 of 10 top features and correlate at ρ = −0.157 on feature importance, the mechanism behind that divergence is understood, and there is a documented case (`TX000566`) where Isolation Forest alone sends a plainly unremarkable transaction to a human reviewer for a spurious reason that the Autoencoder correctly ignores. In a system with no label, where cross-model consistency is the *only* validation available, discarding that consistency to save compute is the wrong trade — particularly when the compute saved is negligible and the cost avoided is analyst time spent on false positives.

### Why no blocking tier

v1 shipped one at probability ≥ 0.94, derived from a cost sweep against supervised proxy labels. Phase 13 established that the equivalent sweep cannot be reproduced without a label, because a false-negative count requires knowing which unflagged transactions are fraud. **Blocking a customer's transaction on a score whose false-negative behaviour has never been measured is not defensible.** Every output of this system should reach a human until it has been validated against real investigator-confirmed outcomes.

### What this system is, and what it is not

**It is** a rigorously-built, leakage-safe, fully-documented unsupervised anomaly detection pipeline that ranks transactions by how far they deviate from their own account's established behaviour, cross-checks that ranking across structurally different detector families, explains every alert in two independent ways, and knows precisely where its own evidence stops.

**It is not** a validated fraud detector. It has never been shown to catch fraud, because no fraud is identified in its data. It should be deployed as a triage and prioritisation layer for human investigators — which is a genuinely useful thing to be — and the first month of its operation should be spent gathering the investigator-labelled outcomes that would let its successor be evaluated properly.

The distinction between those two paragraphs is the most important thing in this report.

---

*End of Phase 17. Full evidence trail: `research/01_business_understanding.md` through `research/14_monitoring_framework.md`; numeric artifacts in `artifacts_research/`; plots in `research/plots/`; investigation dashboard in `dashboard/`.*
