# Phase 17 (v2) — Final Research Report

**Unsupervised anomaly detection for retail banking transactions, on the client-designated 18-feature dataset**

Raw data: `data/bank_transactions_data_2.csv` — 2,512 transactions, 495 accounts, 16 raw columns, **no fraud label**.
Feature matrix: `artifacts_research/features_teammate_merged.csv` — 18 engineered features + 10 ID/display columns.

This report is written to stand alone. Every claim traces to one of the phase reports listed below, cited inline by phase number so any figure can be checked at source.

| Phase | Subject | File |
|---:|---|---|
| 1 | Business understanding (raw data unchanged; carried over) | `research/01_business_understanding.md` |
| 5 | Feature engineering — the 18-feature set, verified | `research_v2/04_feature_engineering.md` |
| 6–7 | Preprocessing verification and dimensionality reduction | `research_v2/05_feature_selection_and_preprocessing.md` |
| 8 | Model development (9 models: 8 classical + Hybrid Ensemble) | `research_v2/06_model_development.md` |
| 9 | Hyperparameter optimisation | `research_v2/07_hyperparameter_optimization.md` |
| 10 | Evaluation framework | `research_v2/08_evaluation.md` |
| 11 | Explainability | `research_v2/09_explainability.md` |
| 12 | Ensemble scoring | `research_v2/10_ensemble_scoring.md` |
| 13 | Threshold optimisation | `research_v2/11_threshold_optimization.md` |
| 14 | Final model selection | `research_v2/12_final_model_selection.md` |
| 15 | Production architecture | `research_v2/13_deployment_architecture.md` |
| 16 | Monitoring framework | `research_v2/14_monitoring_framework.md` |
| 17 | This report | `research_v2/15_final_research_report.md` |

**A note on the two pipelines.** This project contains two complete, independently-built anomaly-detection pipelines over the same raw transactions: this one (18 population-level features, the client's designated final pipeline) and an in-house one (46 features including per-account personal-baseline statistics, retained as reference in `research/`, `src_research/`, `artifacts_research/`). **Both are real, both were fully executed, and their results differ in ways that are informative rather than contradictory.** §12 sets the two side by side in detail. Having built both is a genuine strength of this work: several findings cross-validated across two entirely different feature bases, and several others reversed — and knowing *which* did which is worth more than either pipeline alone.

---

## 1. Executive Summary

Nine unsupervised anomaly-detection models were built, tuned, evaluated, explained, ensembled and thresholded against an unlabelled retail banking transaction dataset using the client's designated 18-feature matrix, and a production architecture and monitoring framework were designed on top of the result. **No deep-learning model is part of this pipeline.** An earlier iteration trained an Autoencoder, a VAE and an LSTM-Autoencoder alongside the classical detectors; all three were removed (see Phase 14 §6 for the rationale) because none of the three could be justified against the classical field on this feature set — the LSTM-AE structurally could not score 4.4% of rows and overfit from ~epoch 50, and neither the plain Autoencoder nor the VAE ranked competitively on internal validity. Model 9, the **Hybrid Ensemble**, was redefined from "IF + LOF + Autoencoder majority vote" to **IF + LOF + GMM majority vote (≥2 of 3)** as a direct consequence.

**What the system is.** An 18-feature matrix — five raw behavioural fields, four global frequency counts, an amount-to-balance ratio, a global high-amount flag, and seven one-hot/frequency category encodings — feeding 8 independent classical anomaly detectors (Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, DBSCAN, HDBSCAN, K-Means, GMM) whose scores are combined by percentile aggregation into a single score in (0,1), thresholded into two human-review tiers, and explained to an investigator through Isolation Forest's SHAP attributions — the sole explainability output in this pipeline.

**The recommendation.** Percentile aggregation over the 6 detectors that can score out-of-sample (plus the batch-only DBSCAN/HDBSCAN when scoring a fixed historical set), with Isolation Forest SHAP attributions on every alert, feeding a two-tier review queue at the 99th percentile (**26 transactions, 1.04%, score ≥ 0.9627**) and the 95th percentile (**126 transactions, 5.02%, score ≥ 0.8852**). **No automatic blocking.**

**The single most important technical finding.** DBSCAN stands alone at the bottom of every cross-model agreement measure — its mean pairwise Spearman with the other 7 base models is **0.242**, against a 0.480–0.685 range for everyone else, and it correlates weakest of all with every single other model (lowest pair: DBSCAN ↔ LOF, ρ=0.161). This reproduces a finding from the in-house 46-feature pipeline almost exactly, which is strong independent corroboration that it is a property of DBSCAN's single-dense-cluster behaviour on this raw data, not an artifact of either feature-engineering choice. **That finding is why the recommendation is an 8-model ensemble rather than any single detector** — no one classical model's top-5% set can be trusted alone when even the *field* disagrees this sharply about one member.

**The strongest single fraud-signature match, and it was found twice, independently.** `TX000275` (account AC00454) — a $1,176.28 transaction against a $323.69 balance (**3.63×**) with **5 login attempts**, the dataset's observed maximum — ranks **2nd of 2,512** on this pipeline's ensemble score (`ensemble_percentile_average` = 0.9949; the current rank-1, `TX002192`, is a $879.25 transaction at 6.99× a $125.85 balance with only 1 login attempt — a large-ratio, no-friction profile, discussed in §7.4). The in-house 46-feature pipeline independently surfaced `TX000275` in its own top-1% tier, on completely different features. Isolation Forest's SHAP explains why: `LoginAttempts` (+1.714) and `amount_to_balance_ratio` (+1.607) are its two largest contributors for this row — the two features Phase 1 identified as the account-takeover signature, both firing on the same transaction, independently corroborated by a completely different feature-engineering pipeline.

**What this feature set structurally cannot do, stated up front rather than buried.** These 18 features contain **no personal-baseline statistics and no per-account novelty flags** (Phase 5 §3). There is no way to ask "is this amount unusual *for this customer*" — only "is this amount unusual in the population" (`high_amount_transaction`, a global $878.18 threshold) or "how does this transaction's size compare to this account's balance" (`amount_to_balance_ratio`, a within-row ratio). Account takeover — Phase 1's strongest-fit fraud scenario — is materially harder to detect here than on a feature set with personal-baseline and novelty features. Everything this pipeline detects is closer to "unusual in the population" than "unusual for this account".

**What was not proven, stated plainly.** There is no fraud label in this dataset, so **no detection-performance claim is made anywhere in this project** — no precision, no recall, no AUC against fraud. What was measured is internal consistency: how cleanly each model partitions the feature space, how much the models agree with each other, how stable each is under retraining, and how plausible the top-scored transactions look to a human reasoning from documented fraud typologies. Of the six top-1% transactions examined by hand, two are defensible fraud-signature matches, three are genuinely ambiguous, and one is a demonstrable false signal — and the false signal is reported in as much detail as the matches.

**The honest scale position.** This is a research prototype validated at 2,512 rows. Phase 15 §10 sets out specifically what breaks at bank scale and what must be rebuilt rather than resized. The reasoning transfers; the fitted numbers do not.

---

## 2. Problem Statement

Detect fraudulent transactions in a retail banking ledger that carries **no fraud label**.

That constraint drives every subsequent decision. With no label there is nothing to train a classifier against, nothing to compute precision or recall against, and no cost-optimal threshold to solve for. The problem therefore becomes: *identify transactions that deviate from established patterns, rank them by how strongly they deviate, and give a human enough context to judge them* — while being rigorous about the difference between "statistically unusual" and "fraudulent."

Phase 1 mapped six fraud typologies against what this schema can observe, before any modelling began. That mapping is unchanged (the raw data is the same), but **what this pipeline's features can actually reach differs from the in-house pipeline's**, and the table below states the v2 position specifically:

| # | Scenario | Observable in the raw schema? | Reachable with **these 18 features**? |
|---:|---|---|---|
| 1 | **Account takeover** | Yes — strongest fit | **Partially.** `LoginAttempts` is available and is a genuine signal. The novelty half of the signature (new device / new location for *this* account) is **not** — `device_frequency` and `ip_frequency` are global popularity counts, not per-account novelty flags |
| 2 | Transaction bursts / card testing | Yes | **No.** No velocity features, no time-since-last-transaction. `TX000395` ($6.30 with 5 login attempts and a 283-second duration) is offered in Phase 10 §4 as a *plausible* card-testing read, explicitly not an established one, because no follow-up-transaction linkage exists here |
| 3 | Mule accounts | Partially | **Partially, and less rigorously.** `device_frequency` / `ip_frequency` / `merchant_frequency` capture shared infrastructure, but as *global* counts rather than point-in-time prior-only ones |
| 4 | Unusual spending / compromised card | Yes | **Weakly.** Requires a personal baseline this feature set does not have |
| 5 | Synthetic identities | Weakly | **Weakly** — `CustomerOccupation` one-hots and `CustomerAge` vs. behaviour, same as the raw schema allows |
| 6 | Money laundering (layering) | **No direct signal** | **No.** Requires a counterparty ledger or transaction graph; neither exists. Never claimed as detected anywhere in this project |

**This table is the honest capability statement for this pipeline** and it should be given to the bank as-is. Scenario 1 is partially reachable and Scenario 3 is proxied; Scenarios 2, 4 and 6 are not reachable with these features.

---

## 3. Dataset and Feature Set

### 3.1 The raw data

2,512 transactions, 495 accounts, 16 columns, spanning 365 days at an average of **6.88 transactions/day**. Mean 5.075 transactions per account (min 1, max 12, median 5); 428 of 495 accounts (86.5%) have ≥3 transactions, covering 2,402 rows (95.6%) — re-verified directly on this feature file rather than assumed (Phase 5 §2.6).

One structural caveat inherited from Phase 1: `PreviousTransactionDate` is **not** a real per-account last-transaction timestamp — all values cluster within minutes of a single 2024-11-04 export moment. It is a snapshot artifact and is used by none of the 18 features, which is correct.

### 3.2 The 18 features

| Group | Features | Count |
|---|---|---:|
| Raw behavioural, StandardScaler-scaled | `TransactionAmount`, `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` | 5 |
| Global frequency counts, scaled | `account_frequency`, `device_frequency`, `ip_frequency`, `merchant_frequency` | 4 |
| Derived ratio | `amount_to_balance_ratio` | 1 |
| Global threshold flag | `high_amount_transaction` | 1 |
| One-hot categories | `TransactionType_Debit`; `Channel_Branch`, `Channel_Online`; `CustomerOccupation_Engineer/Retired/Student` | 6 |
| Frequency-encoded category | `Location_FE` | 1 |

### 3.3 What was verified rather than assumed

The features arrived pre-engineered. Phase 5–6 re-derived and checked them against the raw CSV (`src_research_v2/04_feature_verification.py` → `artifacts_research_v2/phase5_6_feature_verification.json`):

- **0 missing cells** (0/70,336), 0 duplicate rows, 0 duplicate `TransactionID`s.
- **Row alignment against the raw CSV confirmed exactly** — `(raw.TransactionID == df.TransactionID).all()` → True.
- **All 18 columns checked for scaling**, not just the 5 the brief named: the 11 continuous columns are mean≈0/std≈1.0002 (the 1.0002 is the `ddof=1` vs. `ddof=0` sample-vs-population convention, not a scaling error), and the 7 binary/dummy columns are correctly left as 0/1.
- **`amount_to_balance_ratio` was fully recovered.** Phase 5 reported r = 0.9467 against the raw `TransactionAmount / AccountBalance` ratio and concluded no exact formula was recoverable. Phase 14 §5 subsequently found one: it is exactly `StandardScaler(log1p(amount / (balance + 1)))` — max absolute error **0.0** across all 2,512 rows. The same check found `TransactionAmount` is `StandardScaler(log1p(raw amount))`, not the plain standardisation Phase 5 describes (max error 8.9 × 10⁻¹⁶ with the log, 3.57 z-units without it). **With those two settled, all 18 features are exactly reproducible from the 16 raw columns** — which turns the production feature specification (§10) from partly-inferred into complete, and is what lets the Bank Transaction Fraud & Anomaly Detection scenario simulator compute a real feature vector rather than an approximation. Raw ratio distribution, unchanged: mean 0.200, median 0.052, max 7.896.
- **`high_amount_transaction` was reverse-engineered empirically**, since the scaled file does not expose its threshold: 126/2,512 rows flagged (5.02%), minimum flagged raw amount **$878.63** versus maximum unflagged **$877.81** — an 82-cent gap sitting on the dataset's 95th percentile of **$878.18**. Conclusion: it is a **global** top-5%-by-amount flag, not a personalised one. A $900 transaction is flagged identically whether it comes from a $50,000-balance account or a $200-balance one.
- **One-hot baselines recovered** by cross-referencing raw category counts: `TransactionType` drops Credit (Debit 1,944 / Credit 568), `Channel` drops ATM (Branch 868 / ATM 833 / Online 811), `CustomerOccupation` drops Doctor (Student 657 / Doctor 631 / Engineer 625 / Retired 599).

A second, model-facing `RobustScaler` (fit on the 2,009-row training split only) is layered on top for every model, saved as `artifacts_research_v2/models/shared_robust_scaler.pkl`, so that distance-based methods get a scale reference that is not itself distorted by the extreme rows a mean/std scaler cannot be robust to.

### 3.4 A leakage caveat that matters for deployment, not for these results

The four frequency counts are computed over the **whole dataset**, which means each account's count includes its own future transactions. That is acceptable for offline anomaly scoring on a static dataset — no result in Phases 8–13 claims to be point-in-time — but it is **not leakage-safe for live scoring**, and the production form must be prior-only running counters (Phase 15 §3.2). This is stated in Phase 5 §1 and repeated at every point where it changes a decision.

---

## 4. Structure Found in the Feature Space

*Source: Phase 7. PCA, UMAP and t-SNE, all on the full 18-column matrix. No autoencoder-latent-space step is part of this analysis -- the pipeline's model suite is 8 classical unsupervised detectors plus the Hybrid Ensemble, and PCA/UMAP/t-SNE are independent of that.*

**PCA does not compress this feature space.** PC1 explains only **16.01%**, PC2 9.34%, PC3 8.87%, with a long declining tail; **9 components are needed for 80% cumulative variance and 11 for 90%**. The loadings are directly interpretable:

- **PC1** (16.0%) — `amount_to_balance_ratio` (+0.594), `AccountBalance` (−0.520), `CustomerAge` (−0.471), `TransactionAmount` (+0.299): a "large transaction relative to a smaller, younger account" axis.
- **PC2** (9.3%) — `TransactionAmount` (+0.716), `CustomerAge` (+0.408): a transaction-size axis, separate from PC1's ratio-driven one.
- **PC3** (8.9%) — `LoginAttempts` (+0.502), `ip_frequency` (−0.457), `Location_FE` (+0.406): a security/network-context axis.

**That `LoginAttempts` loads on a third, separate component is a genuinely useful finding**: "unusual amount behaviour" and "unusual login/network behaviour" are close to orthogonal in this space, not two views of the same anomaly. It is directly visible later — `TX001214` scores high on login friction with an entirely unremarkable 0.15× amount ratio.

**Two non-fraud structural drivers were found, and both are flagged so they are not mistaken for anomaly clusters.** UMAP resolves three groups, and t-SNE independently reproduces the same two drivers rather than merely looking similar:

| Group | UMAP | t-SNE | Defining feature |
|---|---|---|---|
| Main mass | 1,802 (71.7%) | — | no occupation dominance |
| Student segment | 614 (24.4%), 94.8% Student | 383 (15.25%), 94.5% Student | `CustomerAge` z-mean −1.16, `AccountBalance` z-mean −1.01 in both |
| Elevated-login pocket | 96 (3.8%), `LoginAttempts` z-mean **4.755** | 99 (3.94%), z-mean **4.655** | the high-login population, cleanly separated |

Two independent nonlinear projections agreeing this closely is much stronger evidence than either alone. **Neither group is a fraud population**: the student cluster is a demographic segment, and the elevated-login pocket is a population worth watching but not a verdict. Both reappear later — the student segment turns up in K-Means' k=2 split (§5) and in the weak tail of the alert queue (§7.3).

---

## 5. Model Comparison

*Source: Phase 8. Nine models (8 classical + the Hybrid Ensemble), one shared 18-column matrix, one shared `RobustScaler` fit on the 2,009-row training split, all scores oriented higher = more anomalous. An explicit assertion confirms no `vote_count`/`risk_tier`/`is_fraud` column exists anywhere in the input. No deep-learning model is trained in this pipeline -- an earlier iteration's Autoencoder, VAE and LSTM-AE were removed (Phase 14 §6); Model 9 (Hybrid Ensemble) was redefined from IF+LOF+Autoencoder to **IF + LOF + GMM majority vote (≥2 of 3)** as a direct consequence.*

Models 1–4 and 7–8 were fit on the training split and scored all 2,512 rows out-of-sample. **DBSCAN and HDBSCAN have no out-of-sample `.predict` and had to be fit on the full dataset** — a methodological footnote in research that becomes an architectural constraint in production (§10).

| Model | Flagged rate | Internal-validity Silhouette | Mean pairwise Spearman (self-excluded, 7 other base models) |
|---|---:|---:|---:|
| **Elliptic Envelope** | 5.02% | **0.5409** | 0.646 |
| HDBSCAN | 8.88% | 0.4302 | 0.685 |
| DBSCAN | 2.27% | 0.4280 | **0.242** |
| One-Class SVM | 5.73% | 0.4059 | 0.639 |
| K-Means | 5.02% | 0.4003 | **0.675** |
| Isolation Forest | 5.29% | 0.3975 | 0.586 |
| GMM | 5.02% | 0.2925 | 0.480 |
| LOF | 4.46% | 0.2765 | 0.627 |
| Hybrid Ensemble (vote_count, native ≥2-of-3) | 4.02% | 0.3064 | n/a (coarse 0–3 vote count) |

**This table is not a leaderboard, and reading it as one would be the easiest way to reach a wrong conclusion.** Three qualifications:

1. **Silhouette structurally favours distance-based methods.** A top-5%-by-distance cut is close to guaranteed to separate well *in a distance metric*. Density/likelihood-based methods (GMM, LOF) score lower not because they are worse but because their notion of "anomalous" is not the same as "far in Euclidean space."
2. **Elliptic Envelope tops the ranking and its core assumption is measurably false.** Shapiro-Wilk on all 18 scaled features: **100% reject normality at p<0.05** — every one of the 18 p-values rounds to 0.0000. Its MCD-based Mahalanobis distance is retained for comparison and explicitly not recommended as a primary detector.
3. **The flagged rates are largely set by construction.** Seven of the nine cluster in 4.0%–5.7% because they either take a `contamination≈0.05` parameter or use the standardised top-5% convention. The rate carries little information; the *agreement on which rows* carries all of it. DBSCAN (2.27%) and HDBSCAN (8.88%) are the two genuine outliers among the outlier-detectors, in opposite directions.

**Cross-model agreement is where the real structure is.** Strongest pairs among the 8 classical models: LOF ↔ K-Means **ρ=0.838**, HDBSCAN ↔ K-Means 0.832, HDBSCAN ↔ Elliptic Envelope 0.826, LOF ↔ HDBSCAN 0.814. Weakest: **every one of the seven lowest pairs involves DBSCAN**, bottoming at DBSCAN ↔ LOF ρ=**0.161**. **DBSCAN standing alone at the bottom is the single most reproducible cross-model finding in this project** — it reproduced independently on the in-house 46-feature set too, which makes it a property of DBSCAN's single-dense-cluster behaviour on this raw data rather than an artifact of either feature-engineering choice.

**A distinction the flagged-set overlap metric surfaces that rank correlation does not.** Elliptic Envelope has a comfortably mid-pack Spearman (0.646) but its top-5% set is only moderately shared with the field's densest cluster (LOF/HDBSCAN/K-Means, which mutually agree above ρ=0.81) — its broad *ranking* tracks the field more than its specific *top-5% set* does. This is why both Spearman and Jaccard are reported side by side (Phase 8 §3) rather than one being chosen — they answer different questions.

**Model-specific findings worth carrying forward:**

- **K-Means finds a real split, and it is demographic, not fraudulent.** Silhouette argmaxes at k=2 (0.491) and — unlike the in-house pipeline, where k=2 was a degenerate 3-row micro-cluster — this is a genuine **1,830/179** partition of the training split, with the 179-row minority **77.7% `CustomerOccupation_Student`** against 21.3% in the majority. It is the same segment Phase 7's UMAP and t-SNE both found. It is well-separated, real, and fraud-irrelevant. **The inertia elbow, however, is genuinely ambiguous**: no k in 2–10 clears the elbow test, so the rule mechanically returns the boundary, k=10. That ambiguity is reported rather than resolved by picking a more convenient answer.
- **HDBSCAN is far more usable here than in-house, and still fragile.** Its best config reaches 8.88% noise (in-house: far higher), plausibly because an 18-dimensional space suffers less mutual-reachability inflation than a 46-dimensional one. But other configs tried sit far higher — a cliff, not a gradient.
- **GMM prefers a simpler covariance structure here.** Best BIC is `n_components=10, covariance_type='diag'` (BIC −27,620.9) — **diag, not the in-house pipeline's full**. Even with only 18 free variance parameters per diagonal component, the data's per-component correlations do not justify a richer covariance structure. The BIC curve is still descending at the n=10 search boundary, so the true optimum may lie beyond it — an honest, unresolved caveat.
- **The Hybrid Ensemble (IF + LOF + GMM) agrees more with Isolation Forest and GMM than with LOF.** Pairwise agreement on the majority-vote flag: IF↔GMM 94.5%, LOF↔GMM 93.2%, IF↔LOF 93.2% (Phase 8 §2.9) — all three components agree with each other on 93%+ of rows, which is why the majority vote (4.02% flagged) sits close to, but not identical with, any single component's native rate.

---

## 6. Hyperparameter Tuning

*Source: Phase 9. Two models tuned with Optuna's TPE sampler: Isolation Forest and GMM. No deep-learning model is tuned here -- the VAE hyperparameter search was dropped along with the VAE model itself. With no label, "optimise against what" is a genuine design decision and was stated per model.*

| Model | Method | Result |
|---|---|---|
| **Isolation Forest** (objective: silhouette, top-5% vs. rest) | Exhaustive grid, 60 combos, 47.0s | **0.4154** |
| | Optuna TPE, 30 trials, 31.4s | 0.4107 |
| | Random search, 30 trials, 31.5s | 0.3840 |

**Reported plainly rather than spun: the exhaustive grid still won.** Optuna came **0.0047** short — a smaller gap than the in-house pipeline's, but still short. The practical conclusion carries over: a 3-hyperparameter, cheap-to-evaluate search space is too small for Bayesian optimisation's main selling point to pay off against a grid you can afford to enumerate.

**But one result genuinely differs from in-house, and it is the more interesting half.** Optuna beat random search here by **+0.0267** on an identical 30-trial budget. One plausible read: with fewer, more population-level features and no personal-baseline columns adding noise to the objective surface, the silhouette-vs-hyperparameter relationship here is smoother and more exploitable — exactly the setting where TPE's model-based proposals have more signal than uniform sampling.

**GMM: a materially more reassuring result than in-house.** Adding `reg_covar` to the search space moved best BIC from −27,620.9 to **−47,044.7** — and the winning covariance type **stayed `diag`** in both the fixed-`reg_covar` grid and the free-`reg_covar` search (best trial: `n_components=10, covariance_type=diag, reg_covar≈1.01e-6`). In-house, the equivalent search pushed toward an even more parameter-hungry `full`-covariance solution and reinforced an overfitting concern. Here it does not: a `diag` structure has 18 variances per component rather than a full covariance matrix, so a smaller regularisation floor is a far lower-risk route to a lower BIC. **`diag` covariance with a smaller `reg_covar` is a more defensible production choice here than the in-house pipeline's `full`-covariance result was for its own feature set.** The n_components boundary caveat still stands.

---

## 7. Evaluation Results

*Source: Phase 10. With no label, evaluation means four label-free things, none of which substitutes for precision and recall.*

### 7.1 Internal validity

Every model's top-5%-by-score partition (the Hybrid Ensemble's majority-vote partition for that one) was scored on Silhouette, Davies-Bouldin and Calinski-Harabasz in the shared 18-feature scaled space, using one consistent partition definition across all nine. Results are in §5's table with the qualifications that go with them.

**Elliptic Envelope leads by a wide margin** — Silhouette 0.5409, Davies-Bouldin 1.1529, Calinski-Harabasz **592.47**, more than 3× HDBSCAN's next-best 180.07. This is a real reversal from in-house, where it sat 7th of 12, and it *explains* the Jaccard anomaly from §5: EllipticEnvelope's MCD tail identifies a small, tightly-clustered, well-separated group in the scaled space — a coherent structural outlier population, just not the one everyone else converges on.

**LOF is last of the 8 base classical models** (Silhouette 0.2765, Davies-Bouldin 4.2348, Calinski-Harabasz 36.38), for the construction reason in §5 qualification 1 — density-based scoring does not imply the Euclidean separation these metrics reward. The Hybrid Ensemble's own partition (Silhouette 0.3064) scores lower still, because its coarse 0–3 vote count produces a 255-row (10.2%) "flagged" group rather than a clean top-5% cut, which is a direct consequence of its discreteness, not evidence it is a worse detector (§7.4 explains why it is not used for fine-grained tiering).

### 7.2 Stability — the most operationally significant result in this pipeline

Isolation Forest and LOF were each refit on 5 bootstrap resamples of the training split, rescored against all 2,512 rows, and the top-5% flagged set recomputed each time. (No deep-learning model remains to include here — the Autoencoder stability check that used to run alongside these two was removed with the model itself.)

| Model | Mean pairwise Jaccard (5 runs) | Min | Max |
|---|---:|---:|---:|
| **Isolation Forest** | **0.6021** | 0.5090 | 0.6689 |
| LOF | 0.5124 | 0.4651 | 0.5750 |

**Isolation Forest is the more retrain-stable of the two measured models on this feature set**, similar in direction to the in-house pipeline's finding for these same two models, though the in-house pipeline measured a narrower 0.527–0.590 spread against the tighter one here (0.512–0.602). Nothing changed except which rows the bootstrap happened to include — no drift, no new data, no hyperparameter change — and even in the worst observed pair, less than half the flagged set (46.5%, LOF's minimum) was shared between two runs.

Two things follow, and both shaped the final design. **Operationally:** a fixed model artifact plus a monitored, versioned retraining process is the correct posture, not continuous retraining, and an operations team must be told the expected churn band in advance (Phase 16 §4.2, §5.2). **Methodologically:** this is a direct argument for aggregating detectors rather than trusting a borderline single-model result — a conclusion this section supports even with only two models measured, since neither individually reaches 70% self-agreement under resampling.

### 7.3 Business evaluation — reading real flagged transactions

Isolation Forest's top 1% (26 transactions) was examined by hand against Phase 1's scenario table. **A necessary caveat, applied throughout:** `device_frequency` / `ip_frequency` / `account_frequency` are *global* population counts. A value near 0 means "about as common as the average device/IP/account in this dataset," **not** "unusual for this specific customer."

| Transaction | Amount | Balance | Ratio | Logins | Read |
|---|---:|---:|---:|---:|---|
| **`TX000275`** | $1,176.28 | $323.69 | **3.63×** | **5** | **Strongest match in either pipeline.** Maximum observed login friction combined with an amount worth 3.6× the account's entire balance — the clearest fit to Scenario 1 (account takeover). Independently flagged in the in-house pipeline's own top-1% tier |
| `TX001214` | $1,192.20 | $7,816.41 | 0.15× | **5** | **Plausible partial ATO.** The login-friction half of the signature is there; the amount is unremarkable relative to the balance. A concrete illustration of §4's PCA finding that login and amount signals are near-orthogonal here |
| `TX000935` | $1,022.75 | $207.74 | 4.92× | 1 | **Ambiguous.** Nearly 5× the balance, but no login friction. Could be a legitimate large purchase from a low-balance, frequently-used account (`account_frequency` z = 2.17) |
| `TX002192` | $879.25 | $125.85 | **6.99×** | 1 | **Ambiguous.** The most extreme ratio in the tier, but from a Student-occupation account whose chronically low balance is consistent with the demographic pattern §4 identified — a large one-off purchase reads as plausibly as fraud |
| `TX000395` | **$6.30** | $7,697.68 | 0.0008× | **5** | **Interesting ambiguity, not a simple false positive.** A trivial amount, but maximum login attempts and the tier's longest duration (283s) — superficially consistent with card-testing (Scenario 2). **Offered as plausible, not established**: this dataset has no follow-up-transaction linkage to confirm it |
| `TX001903` | $1,168.26 | $1,385.74 | 0.84× | 1 | **Does not look like fraud.** Amount below balance, normal logins, high `device_frequency` (a weak population-level signal at best). The only unusual value is a 37-second duration. Included specifically to show that a high Isolation Forest score does not always track an amount- or login-based narrative |

**The tail dilutes, and here we can say what it dilutes *into*.** Spot-checking the weakest 8 of the 252 transactions in the top-10% tier finds small-to-moderate transactions ($36–$918), almost all with `LoginAttempts=1` and balance ratios well under 1.5× — e.g. `TX002327` at $35.98 against an $11,147.34 balance (0.003×). **Five of the eight are Student-occupation accounts**, against 26.2% of the dataset overall: Isolation Forest's tail is partly picking up the legitimate demographic segment §4 identified, not fraud signal. A review team using a 10th-percentile threshold should expect a materially higher false-positive load than one using the 1st.

---

## 8. Explainability Results

*Source: Phase 11. `shap.TreeExplainer` (exact, ~8s for all 2,512 rows) for Isolation Forest — the **sole explainability output** in this pipeline. No other classical model here (LOF, OCSVM, Elliptic Envelope, DBSCAN, HDBSCAN, K-Means, GMM) is naturally SHAP-compatible without an expensive KernelExplainer, which this codebase does not use anywhere, so the cross-model SHAP comparison this pipeline used to run (Isolation Forest against the Autoencoder) was dropped entirely when the Autoencoder was removed. Sign convention verified directly, not assumed — TreeExplainer's raw output tracks `score_samples` (ρ=1.0000 on a 200-row check), the opposite of this project's convention, so values are negated before reporting.*

**Global feature importance (mean |SHAP|, Isolation Forest, all 2,512 rows):**

| Rank | Feature | mean\|SHAP\| |
|---:|---|---:|
| 1 | `TransactionType_Debit` | 0.391 |
| 2 | `CustomerOccupation_Retired` | 0.294 |
| 3 | `CustomerOccupation_Student` | 0.265 |
| 4 | `CustomerOccupation_Engineer` | 0.207 |
| 5 | `TransactionAmount` | 0.207 |

**The mechanism is understood, and it replicated across two independent feature sets.** Isolation Forest scores by how few random splits isolate a point, so a single split on a binary feature isolates an entire minority class in one step — low-cardinality categoricals dominate its attributions. **`TransactionType_Debit` is its #1 feature and `CustomerOccupation_Retired` its #2 in *both* pipelines**, on 18 and 46 features respectively. That is an independent replication of a mechanism, not a coincidence — and a caution for a reviewer: a high Isolation Forest score is frequently driven by *who the customer is* (occupation, transaction type) as much as by *what the transaction looks like*.

**Two worked local cases, both instructive:**

- **`TX000615`** — a $1,142 transaction (`high_amount_transaction` +1.375, `amount_to_balance_ratio` +1.115 at 3.82×) is read primarily as an amount anomaly, with `Location_FE` (+0.347) and `CustomerOccupation_Student` (+0.305) contributing materially as well — a case where the amount signal and the demographic/location signal reinforce each other rather than one dominating.
- **`TX001029`** — `TransactionType_Debit` (+1.046) and an extreme `merchant_frequency` value (+0.962, z=3.67) are the two largest drivers, ahead of `amount_to_balance_ratio` (+0.219, only 0.79× balance). Phase 10 independently judged this $516.47 transaction (0.40× balance, normal logins, Student account) not a plausible fraud pattern — a low-value flag driven mostly by a population-frequency outlier and a categorical feature, not by the transaction's dollar amount. With only one explained model in this pipeline, catching this class of flag now relies on cross-*detector* agreement (§9's ensemble, and specifically the fact that DBSCAN and GMM — see §5 — do not weight categorical one-hots the way Isolation Forest's splits do) rather than a second explainer's independent read.

**This is a narrower explainability surface than a two-model comparison would give an investigator, and that trade-off is stated plainly rather than smoothed over.** It is also the honest consequence of removing the Autoencoder: SHAP on the remaining classical models would require an approximate, expensive KernelExplainer with no existing precedent in this codebase, so Isolation Forest's exact TreeExplainer output is what ships.

---

## 9. Ensemble Scoring and Thresholds

*Source: Phases 12–13.*

The **8 classical models** were combined; the Hybrid Ensemble (Model 9) was deliberately excluded as an input, because it is itself a ≥2-of-3 vote of Isolation Forest + LOF + GMM and folding it back in would double-count those three.

Four strategies were built: **weighted average** (disagreement-inverse weights, HDBSCAN 0.151 down to DBSCAN 0.073 — the scheme correctly down-weights the model Phase 8 found most divergent), **rank aggregation** (Borda), **percentile aggregation**, and a **PCA stacking proxy** (explicitly *not* supervised stacking — there is no label to fit a meta-learner against; PC1 explains **59.70%** of the variance across the 8 standardised score columns).

**An honest finding rather than a manufactured distinction: Borda and percentile aggregation are near-identical here** — ρ = **1.0000**, Jaccard 1.000 on the top-5% set. That is expected: summing ranks and averaging rank/N are the same operation up to a normalisation constant, and with no model contributing missing scores in this 8-model combination, they no longer diverge at all. **Weighted Average and PCA Stacking also converge very tightly** (ρ = 0.9998, Jaccard 0.909).

**One cross-check returned nothing, and it is reported rather than dropped.** All four strategies correlate with v1's independent 4-detector `vote_count` proxy at **ρ between 0.0009 and 0.0047** — indistinguishable from zero. The explanation is not a defect in either pipeline: v1's proxy was built on personal-baseline/expanding-statistics features in the same style as the in-house 46-feature set, so a detector ensemble built entirely from population-level features and one built from personal-baseline features are **evidently identifying substantially different transactions as anomalous on this dataset**. That is a measured confirmation of the capability difference, not a contradiction to resolve.

**Percentile aggregation was selected** on three grounds: it is robust to 8 heterogeneous native scales (a bounded kernel decision value, a GLOSH score, a negative log-likelihood, a centroid distance) with no assumption about their shapes; it has no tuned weights to defend to a reviewer; and it handles missing model scores by skipping and renormalising, which is both the right degradation behaviour and the mechanism that makes a reduced-member production deployment possible at all (§10).

**Thresholds.** Score distribution: mean 0.5000, std 0.2242, min 0.0576, max 0.9967.

| Tier | Percentile | Score cut | Flagged | Daily load at this sample's rate |
|---|---|---:|---:|---:|
| **Priority review** | 99th | **0.9627** | 26 (1.04%) | ~0.07/day |
| **Standard review** | 95th | **0.8852** | 126 (5.02%) | ~0.35/day |

**A genuine methodological finding, reproduced independently on a completely different feature basis:** the classic statistical thresholds **flag zero transactions** on this score. mean+3σ = 1.1725 and Q3+1.5×IQR = 1.2120, both above the observed maximum of 0.9967 — because averaging 8 bounded percentiles compresses the tails, a CLT-like effect. The in-house pipeline found exactly the same thing on its own score. It is not a defect in percentile aggregation; it is a mismatch between one thresholding convention and one score's shape, confirmed by applying the identical rules to two **unbounded** scores, which both produced usable cut points (Isolation Forest's raw score: 16 and 25 flagged; the weighted-average ensemble: 32 and 75). **If a "three standard deviations" framing is wanted, apply it to the weighted average, not the percentile score** — and keep the weighted average computed in parallel for exactly that purpose, since it costs nothing once the members have run.

**Cost framing, bounded honestly.** A cost-optimal threshold sweep **cannot be reproduced here**, because counting false negatives requires knowing which *unflagged* transactions are fraud. What can be computed is an upper bound on review labour assuming every flagged transaction is a false positive: **$630 at the 95th percentile, $130 at the 99th**, using v1's own illustrative (not real-bank) figures of $5 per false positive. That is a ceiling, not an estimate. **A lower threshold is not "cheaper" in any total-cost sense** — it only reviews fewer transactions, trading off against catching less of whatever fraud is present, in a direction this project cannot quantify.

**Note on the flagged counts.** They are 126/76/26/13 across the four percentile tiers — mechanically identical to the in-house pipeline's, because both cut the same fixed percentiles of the same 2,512 rows. **This is not evidence the two pipelines agree on which transactions to flag.** The near-zero v1 cross-check above shows they largely do not.

---

## 10. Final Model Selection and Deployment

*Source: Phases 14–15.*

Thirteen candidates (9 models, including the Hybrid Ensemble, + 4 ensemble strategies) are scored on six dimensions — Detection Quality (0.25), Stability (0.20), Interpretability (0.15), Scalability (0.15), Deployment Readiness (0.15), Computational Cost (0.10) — using the identical rubric and weights as the in-house pipeline, so that differences in outcome are attributable to evidence rather than to a changed rubric. (The three deep-learning candidates that previously occupied slots in this matrix — Autoencoder, VAE, LSTM-AE — no longer exist in this pipeline and are removed from the matrix along with their scores; Phase 14's own document carries the full 13-candidate re-scoring.)

**Top of the matrix, directionally unchanged by the removal:**

| # | Candidate | Notes |
|---:|---|---|
| 1 | **Isolation Forest** | Highest-scoring single model, as before — highest internal-validity Silhouette among the 6 out-of-sample-capable base models, most retrain-stable of the two models measured (§7.2), exact TreeExplainer SHAP |
| 2 | **Percentile Aggregation** | The recommended ensemble strategy (§9) |
| 3= | K-Means / LOF / Hybrid Ensemble | Mid-pack on internal validity and cross-model agreement |

**Detection Quality is capped at 4 for every candidate.** With no label, nothing has been *shown* to detect fraud — only to partition the space more cleanly, agree more with the field, or produce more plausible examples. Awarding a 5 would misrepresent what was proven.

**The recommendation deliberately overrides a single-model top score, and the override is stated rather than engineered into the scores.** Isolation Forest ranks first individually, and the recommendation is still the 8-model ensemble — because in a system with no label, cross-model consistency is the only validation available, and §5's finding that DBSCAN is the consensus outlier among all 8 classical detectors (mean pairwise Spearman 0.242 vs. 0.480–0.685 for the rest of the field) demonstrates that even a well-performing single model's top-5% set is not something the rest of the field agrees with uniformly. At 6.88 transactions/day the compute saved by dropping to one model is negligible.

**The production constraint that decides the architecture.** DBSCAN and HDBSCAN cannot score a transaction they were not fit on. **Any 8-model ensemble score is therefore batch-only.** Phase 14 §3 sets out three options; **Option B** (drop both, aggregate over the remaining 6) is recommended, with **Option C** (re-enable HDBSCAN via `prediction_data=True`) a strong follow-up here — HDBSCAN is a more valuable member on this feature set than it was in-house (8.88% best-config noise, and a strong mean flagged-set agreement with the rest of the field). **A 6-model score is a different score, and Phase 13's thresholds do not transfer to it unrevalidated. That revalidation has not been run.**

**Reduced-member fallbacks.** The fallback ladder for this pipeline is: **(1) the Hybrid Ensemble alone** (IF + LOF + GMM majority vote — already an internal 3-model consensus, all three out-of-sample-capable), then **(2) Isolation Forest alone**, explicitly accepting the loss of the cross-model check that §5's DBSCAN finding argues for.

**Deployment posture: nightly batch.** The full ensemble is batch-only by construction; the volume does not demand real-time; the feature layer (not the models) is the blocker; and with no block tier nothing needs to complete inside a payment-authorisation window. The day real-time becomes necessary is the day a block tier is introduced — and that should not come before the system has been validated against investigator-labelled outcomes.

**The largest piece of unbuilt engineering** is the real-time feature layer for the five frequency-derived features, which must become prior-only running counters in a feature store rather than whole-dataset `groupby`s (Phase 15 §3.2). **A genuine advantage over the in-house pipeline, worth stating because it runs the other way:** the in-house 46-feature set needs a per-account *history scan* at inference (expanding means, rolling windows, novelty flags); this one needs only **counter reads**. Counters are dramatically cheaper to maintain, keep consistent and backfill. **The teammate's feature set is the more deployable of the two, and the less capable — both are true.**

**Investigator surface: Bank Transaction Fraud & Anomaly Detection** (`dashboard/`), a FastAPI + static-frontend console, reading this pipeline's artifacts: `ensemble_percentile_average` for the score, Phase 13's cutoffs for the tiers, and the precomputed Isolation Forest SHAP rows (the sole explainability output, §8). Nothing on it is hand-typed from a report — every number is read from an artifact at startup, so a stale artifact produces a visibly stale dashboard rather than a silently wrong one. Its What-if Simulator required a **redesign rather than a repoint**: free-form new-transaction scoring is not honestly possible on a feature set built from population statistics, so it was rebuilt as an *Account Scenario Simulator* anchored to a real account's true historical frequency values (Phase 15 §7.3).

**The one thing Bank Transaction Fraud & Anomaly Detection produces that nothing else in this project does: investigator decisions.** `queue_state.json` is the only mechanism here capable of generating labelled data, and every limitation in this report traces back to not having any.

---

## 11. Monitoring

*Source: Phase 16.*

**This feature set has a specific fragility with no in-house analogue, and the monitoring design is built around it.** Five of the eighteen features (`account_frequency`, `device_frequency`, `ip_frequency`, `merchant_frequency`, `Location_FE`) encode how common a category is *across the population*. Therefore:

> The encoded value of a category changes when the population changes, even if nothing about that category's own behaviour changed at all.

Onboard 40,000 new devices and `device_frequency` shifts for **every device already in the system**. Every model sees a shifted input. Every flagged set moves. No customer did anything different. This means feature drift on those five columns is **expected continuously**, and a monitoring design that alarms on it will be switched off within a month. The framework therefore triages the 18 features into three classes:

| Class | Features | Method | A breach means |
|---|---:|---|---|
| **A — frequency encodings** | 5 | PSI **paired with the population statistic that drives it** (distinct device/IP/merchant/account counts, unseen-category rate) | Usually: refresh the encoding on schedule. Sometimes: a genuine sharing-behaviour shift |
| **B — binary/one-hot** | 7 | Positive-rate proportion test (PSI is degenerate on two-valued columns) | Mix shift. Check `high_amount_transaction` first — a rate pinned at exactly 5.02% is evidence the frozen $878.18 threshold is being silently recomputed, which is a bug |
| **C — behavioural** | 6 | PSI + KS | Genuine behavioural change — **this is the only class that should page someone** |

**The unseen-category rate for Class A is the metric that matters most**, more than PSI itself: a category the frozen lookup table has never seen has no encoded value at all. No baseline for it exists (it was 0% by construction at training), so the proposed >2% alarm is labelled as an uncalibrated starting value rather than presented as derived.

**Retraining is measurably expensive here**, and the operating procedure must say so: §7.2's bootstrap numbers mean a retrain changes **40%–49%** of the flagged set (Isolation Forest, LOF; the only two models bootstrap-tested) with no drift and no new data. Every trigger except an anchor-correlation collapse therefore requires two consecutive batches before firing. Isolation Forest (0.6021 mean pairwise Jaccard) is the more retrain-stable of the two measured models and the natural anchor to evaluate a retrain against; LOF (0.5124) needs somewhat tighter change control. Extending this measurement to the remaining 6 base models (§14 item 9) would sharpen this picture.

**Three hard-failure integrity checks** (the run stops, rather than a signal being raised): the ensemble member count must match the manifest (percentile aggregation's skip-and-renormalise property means a silently-dropped member still produces a plausible score with no error); a frozen canary set must reproduce exactly — **including `TX002192`, the dataset's highest-scoring transaction and its best single canary**; and the Phase 5–6 feature-engineering assertions must all pass, above all the row-alignment check, since a misaligned merge produces a fully-populated, correctly-typed, entirely wrong feature matrix that every other metric would read as normal.

**The honest limit.** Fraud that deliberately uses common devices, common merchants and common locations is **invisible to a feature set built entirely from population frequencies**, and no amount of monitoring on these 18 columns will surface it (Phase 16 §3.3, §9).

---

## 12. This Pipeline vs. the In-House 46-Feature Pipeline

Both pipelines are real, both were executed end to end on the same 2,512 raw transactions, and both are valid. This section sets their results side by side. **The comparison is a strength of this work, not an embarrassment to be smoothed over**: where two entirely different feature bases reached the same conclusion, that conclusion is much better supported than either pipeline alone could make it; where they diverged, the divergence is traceable to a specific, understood property of the feature sets.

### 12.1 What the two feature sets are

| | In-house (46 features) | This pipeline (18 features) |
|---|---|---|
| Personal-baseline statistics | `Expanding_Mean/Median/Std/Max/Min`, `Rolling3_Mean/Std`, `Amount_ZScore_Account`, `Amount_vs_AccountAvg`, `SpendCV_Account` | **None** |
| Velocity / temporal | `Velocity_1D/7D_Count`, `TimeSinceLastTxn`, `CustomerTxnCountSoFar`, cyclical hour/DOW encodings | **None** |
| Per-account novelty | `DeviceNoveltyFlag`, `LocationNoveltyFlag` | **None** — only *global* frequency counts |
| Cross-account sharing | Point-in-time, prior-only shared-account counts | Global (not point-in-time) frequency counts |
| Amount-to-balance ratio | `Amount_to_Balance_Ratio` | `amount_to_balance_ratio` — **same idea, independently arrived at** |
| Location encoding | `Location_enc` + `Location_Freq` | `Location_FE` — **same idea, independently arrived at** |
| Global high-amount flag | Not built standalone | `high_amount_transaction` — **the teammate set has one the in-house set flagged as "worth adopting" and never built** |

**Convergent validation worth noting:** two independent teams, working separately, both built an amount-to-balance ratio and a location frequency encoding. Those two ideas are corroborated by independent arrival, not just by one team's judgement.

### 12.2 Findings that cross-validated — the same answer on two different feature bases

These are the most trustworthy findings in the entire project, because each was reached twice from completely different inputs:

| Finding | In-house | This pipeline |
|---|---|---|
| **`TX000275` is one of the clearest fraud-signature matches in the data** | In its top-1% tier (Phase 10) | **Rank 2 of 2,512** on the ensemble score (0.9949) |
| **DBSCAN is the consensus outlier among the detectors** | Lowest agreement, lowest ensemble weight | Lowest by a wide margin: mean ρ 0.242 vs. 0.480–0.685 for the rest of the field; weight 0.073 |
| **Isolation Forest's SHAP is dominated by low-cardinality one-hots** | #1 `TransactionType_Debit` (0.176), #2 `CustomerOccupation_Retired` (0.152) | **#1 `TransactionType_Debit` (0.391), #2 `CustomerOccupation_Retired` (0.294)** — same two features, same order |
| **Borda ≈ percentile aggregation** | ρ = 0.9999, Jaccard 0.953 | ρ = **1.0000**, Jaccard **1.000** (no members produce missing scores in this 8-model combination) |
| **mean+3σ and Q3+1.5×IQR flag zero on a percentile-averaged score** | Zero flagged (thresholds 1.1088 / 1.1363 vs. max 0.9988) | **Zero flagged** (1.1725 / 1.2120 vs. max 0.9967) |
| **Elliptic Envelope's Gaussian assumption is violated** | 100% of 46 features reject Shapiro-Wilk | **100% of 18 features reject** |
| **Optuna does not beat an affordable exhaustive grid on a 3-hyperparameter space** | Grid 0.6092 vs. Optuna 0.5981 | Grid 0.4154 vs. Optuna 0.4107 |
| **Retraining substantially changes the flagged set** | 41–47% churn | **33–53% churn** (Isolation Forest, LOF only — no deep-learning model to test) |
| **The top-10% tail dilutes into unremarkable transactions** | Confirmed by spot check | Confirmed, **and attributed** — largely the Student demographic segment |
| **The recommended ensemble strategy** | Percentile aggregation | Percentile aggregation |
| **No automatic block tier is defensible without a label** | Confirmed | Confirmed |
| **The Hybrid Ensemble's Phase 10 top-5%-cut partition mechanically selects the ≥1-vote set, not the model's own ≥2-of-3 majority** | 253 rows described as majority-vote; actually ≥1-vote (94 = true majority) | **255 rows selected by the top-5%-of-vote_count cut; actually ≥1-vote (101 = true majority, matching the model's own reported `majority_flagged_rate` of 4.02%)** — the same mechanical effect reproduced independently, which points at the shared top-5%-cut convention applied to a discrete 0–3 score, not a transcription slip |

### 12.3 Findings that reversed — and what each reversal is attributable to

| Finding | In-house | This pipeline | Attributable to |
|---|---|---|---|
| **Internal-validity leader** | HDBSCAN 0.672; Elliptic Envelope 7th (0.610) | **Elliptic Envelope 0.5409 by a wide margin**; CH 592.5 vs. next-best 180.1 | A lower-dimensional, more Gaussian-ish scaled space gives an MCD ellipsoid a coherent tail to find. Tempered by its field-lowest mean Jaccard (0.206) — it finds a real group, just not the one anyone else finds |
| **Most retrain-stable model (of the models measured)** | LOF 0.590; IF least at 0.527 (narrow spread) | **IF 0.6021; LOF 0.5124** (wider spread) | Isolation Forest's univariate splits are less resample-sensitive than LOF's local-density estimate on a feature set this size |
| **HDBSCAN usability** | Best config 53.94% noise | **Best config 8.88% noise** | Lower dimensionality reduces mutual-reachability inflation |
| **K-Means k=2** | A degenerate 3-row micro-cluster artifact | **A genuine 1,830/179 demographic split** (77.7% Student) | Real structure in the occupation one-hots that the in-house space did not surface as cleanly |
| **K-Means elbow** | Clean k=4 | **No elbow in k=2–10**; rule mechanically returns the boundary | Reported as an unresolved ambiguity rather than resolved conveniently |
| **GMM covariance** | `full` wins; the reg_covar search reinforced an overfitting warning | **`diag` wins and stays `diag`** under a free-`reg_covar` search — no overfitting warning | Far fewer free parameters (18 variances vs. a full covariance matrix) leaves much less room for a shrinking regularisation floor to overfit |
| **Optuna vs. random search** | Statistically tied | **Optuna wins clearly (+0.0267)** | A smoother, more exploitable objective surface without personal-baseline noise |
| **Winning `n_estimators` for IF** | 300 (largest in grid) | **50 (smallest in grid)** | 18 lower-dimensional features need fewer trees to isolate the tail |
| **Cross-check against v1's independent proxy** | ρ ≈ 0.442–0.444 (modest, consistent) | **ρ ≈ 0.000–0.005 (nothing)** | v1's proxy is itself built on personal-baseline features. **A measured confirmation that the two feature philosophies flag different transactions** — and a real reduction in the corroborating evidence available to this pipeline |
| **Explainability surface** | Two SHAP explainers (Isolation Forest + Autoencoder), compared | **One SHAP explainer (Isolation Forest only)** — no cross-model comparison, since no other remaining classical model is naturally SHAP-compatible without an expensive KernelExplainer | This pipeline's deep-learning model was removed; the cross-model comparison it enabled went with it |
| **Recommended minimal fallback** | 2 models: IF + Autoencoder | **The Hybrid Ensemble (IF+LOF+GMM) alone, then IF alone** | No Autoencoder exists in this pipeline to pair with IF; the Hybrid Ensemble is already an internal 3-model consensus |

### 12.4 The bottom line for the bank

**Neither pipeline is strictly better. They trade capability against deployability, and the trade is real in both directions:**

- **The in-house 46-feature set is more capable.** It can ask "is this unusual *for this customer*" — the question Phase 1 identified as the primary signal for account takeover, the strongest-fit fraud scenario in this schema. It has per-account novelty flags. This pipeline can do none of that.
- **The teammate's 18-feature set is more deployable.** Its inference-time features are counter reads and frozen lookups, not per-account history scans. It trains and scores faster (Isolation Forest: ~4.1s vs. ~9.7s for the same 5 configs). It is easier to reason about, easier to explain to a reviewer, and materially cheaper to keep consistent in production. It also contributes one idea the in-house pipeline flagged as worth having and never built (`high_amount_transaction`).
- **Where they agree, believe it.** §12.2 lists twelve findings reached twice, independently. `TX000275` being one of the clearest fraud candidates in the data, DBSCAN being the odd detector out, percentile aggregation being the right combination strategy, and no block tier being defensible — all of these are now supported by two independent lines of evidence rather than one.
- **Where they disagree, the disagreement is explained, not hand-waved.** Every reversal in §12.3 has a stated mechanism traceable to a specific structural difference between the feature sets.
- **The one genuinely uncomfortable number is worth stating plainly.** The two ensembles correlate at essentially **zero** on which transactions they rank as anomalous. That is not a bug in either; it is the clearest possible measurement of how much a feature-engineering philosophy determines what an unsupervised system detects. **A bank running only one of these two feature philosophies is seeing one view of its transaction risk, not the whole of it** — and the strongest single practical recommendation this project can make from having built both is that the frequency-based features here and the personal-baseline features in the in-house set are **complementary, and the eventual production feature set should contain both.**

---

## 13. Limitations

Stated in order of how much they constrain what can be claimed.

1. **No fraud label.** The binding constraint. No precision, no recall, no AUC, no cost-optimal threshold, no validated detection claim anywhere in this project. Every "evaluation" here is internal consistency or human plausibility. This is not a caveat to the results; it is the boundary of what the results can be.
2. **No personal baseline and no per-account novelty in this feature set.** Phase 1's strongest-fit scenario (account takeover) is only *partially* reachable, and Scenarios 2 and 4 are not reachable at all (§2). This is a capability statement, not a modelling shortfall.
3. **The frequency features are not leakage-safe for live scoring.** Global counts include each account's own future transactions. Fine for offline scoring on a static dataset, wrong for a point-in-time backtest, and requiring a rebuild as prior-only counters before deployment (§3.4, Phase 15 §3.2).
4. **No ensemble-level stability measurement exists.** Two individual models were bootstrap-tested (Isolation Forest, LOF); none of the four ensemble strategies was. The recommendation rests partly on the assumption that aggregation damps the measured 0.512–0.602 churn, and that assumption is untested here. **This is the single highest-value missing measurement** (Phase 14 §5, Phase 16 §5.2).
5. **The recommended production score has not been fully revalidated at reduced membership.** Option B's 6-model score (dropping DBSCAN and HDBSCAN) is a different score from the published 8-model one, and Phase 13's thresholds do not transfer to it unrevalidated (§10).
6. **2,512 rows, 495 accounts, 365 days, 6.88 transactions/day.** A research sample. The relative comparisons generalise; the fitted constants and the absolute daily volumes do not.
7. **The 5% contamination assumption is an assumption**, used throughout as a modelling convention. It is not a measured fraud rate for this population.
8. **Two of the eight ensemble members (DBSCAN, HDBSCAN) cannot score unseen rows.**
9. **Alert-volume and unseen-category alarm bands are proposals, not measurements** — there is only one batch of data in this project (Phase 16 §5.1, §2.3).
10. **No deep-learning model is part of this pipeline.** An earlier iteration's Autoencoder, VAE and LSTM-Autoencoder were removed after evaluation showed none was competitive against the classical field on internal validity or (for the two measured) stability, and the LSTM-AE structurally could not score 4.4% of rows. This narrows the explainability surface to a single explained model (§8) and the ensemble to 8 classical detectors — both trade-offs stated, not hidden.
11. **`TX000395`-style readings are plausibility, not evidence.** Where Phase 10 offers a card-testing or ATO narrative for an ambiguous transaction, it says so explicitly. None of those readings is established.

---

## 14. Future Improvements

In priority order, with the reasoning for the order:

1. **Capture investigator decisions from day one.** Bank Transaction Fraud & Anomaly Detection's `queue_state.json` is the only label-generating mechanism in this project. A year of it makes a supervised model, a real precision/recall number, and a genuine cost-optimised threshold possible. **Everything in §13's list of limitations either dissolves or shrinks once labels exist.**
2. **Measure ensemble-level bootstrap stability.** Cheap (re-run Phase 10 §2's procedure through Phase 12's aggregation) and it closes the largest evidential gap under the current recommendation.
3. **Compute and validate the Option B 6-model score**, with Spearman/Jaccard against the published 8-model score and re-derived thresholds. Required before any deployment.
4. **Merge the two feature philosophies.** §12.4's strongest finding. The production feature set should carry this pipeline's frequency encodings *and* the in-house pipeline's personal-baseline and per-account novelty features. Neither alone sees the whole picture, and the near-zero correlation between the two ensembles measures exactly how much is being missed by picking one.
5. **Rebuild the frequency features as prior-only running counters.** Prerequisite for real-time scoring and for any honest point-in-time backtest.
6. **Resolve K-Means' k and GMM's `n_components`.** Both are currently boundary artifacts of their search ranges (§5). Extending the search range is a few minutes of compute.
7. **Enable `prediction_data=True` for HDBSCAN** and evaluate Option C. HDBSCAN has a strong flagged-set agreement with the field here and is worth more than it was in-house.
8. **Build the Phase 16 monitoring in the order given in §8 of that report** — the integrity assertions first, because the code already exists and they catch the invisible failures.
9. **Extend bootstrap stability testing to the remaining 6 out-of-sample-capable base models** (OCSVM, Elliptic Envelope, K-Means, GMM), not just Isolation Forest and LOF, so the retrain-churn picture covers the whole ensemble rather than 2 of 8 members.

---

## 15. Final Recommendation

**Deploy: percentile aggregation over the 8 classical out-of-sample/batch-capable detectors (Phase 14 Option B for real-time), with Isolation Forest SHAP attributions — the sole explainability output — on every alert, feeding a two-tier human review queue.**

| Tier | Threshold | Score cut | Volume in this sample |
|---|---|---:|---|
| **Priority review** | 99th percentile | **0.9627** | 26 (1.04%) |
| **Standard review** | 95th percentile | **0.8852** | 126 (5.02%) |

**Secondary score, computed in parallel at no extra model cost:** the weighted-average ensemble, which is unbounded and therefore supports sigma/IQR-style thresholds that the percentile score structurally cannot (§9).

**Fallbacks if fewer artifacts can be operated:** the Hybrid Ensemble alone (IF + LOF + GMM majority vote, already a 3-model consensus), then Isolation Forest alone — explicitly accepting, at that last step, the loss of the cross-model check §5's DBSCAN-agreement finding argues for.

**Run it nightly in batch.** Real-time waits for the frequency-counter feature store, and it is not needed until a block tier exists.

### Why an ensemble, when a single model scores highest individually

Because with no label, cross-model agreement is the only validation available, and this pipeline measured that even a strong individual detector's top-5% set is not something the rest of the classical field agrees with uniformly — DBSCAN's mean pairwise Spearman with the other 7 base models is 0.242 against a 0.480–0.685 range for everyone else (§5), and no model's flagged set fully explains another's. Aggregating damps any one detector's idiosyncrasies against a field that includes at least one clear outlier. Discarding that check to save seconds of compute at 6.88 transactions/day is the wrong trade.

### Why no blocking tier

Because a cost-optimal threshold requires counting false negatives, and counting false negatives requires knowing which *unflagged* transactions are fraud — which is unknowable without a label. Blocking a customer's transaction on a score whose false-negative behaviour has never been measured is not something to hand a bank. Every output of this system goes to a human until it has been validated against real investigator-labelled outcomes.

### What this system is, and what it is not

**It is** a ranked, explained, thresholded queue of transactions that deviate from population-level patterns, built on a verified feature pipeline, with every model's behaviour measured, every disagreement documented, and every unmeasured claim marked as unmeasured. **It is** the more deployable of two pipelines built over the same data, and it independently corroborated twelve of that other pipeline's findings.

**It is not** a fraud detector with a known hit rate. It has never been shown to catch fraud, because there is no fraud in this dataset to be shown against. **It is not** capable of asking whether a transaction is unusual *for a specific customer* — that requires features this set does not contain. And **it is not** finished: the single most valuable next step is not a better model, it is capturing the investigator decisions that would let anyone, finally, measure whether any of this works.
