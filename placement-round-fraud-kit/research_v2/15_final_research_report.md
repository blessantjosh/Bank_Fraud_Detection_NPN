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
| 8 | Model development (12 models) | `research_v2/06_model_development.md` |
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

Twelve unsupervised anomaly-detection models were built, tuned, evaluated, explained, ensembled and thresholded against an unlabelled retail banking transaction dataset using the client's designated 18-feature matrix, and a production architecture and monitoring framework were designed on top of the result.

**What the system is.** An 18-feature matrix — five raw behavioural fields, four global frequency counts, an amount-to-balance ratio, a global high-amount flag, and seven one-hot/frequency category encodings — feeding nine to eleven independent anomaly detectors whose scores are combined by percentile aggregation into a single score in (0,1), thresholded into two human-review tiers, and explained to an investigator through two structurally different SHAP views.

**The recommendation.** Percentile aggregation over the detectors that can score out-of-sample, with Isolation Forest and Autoencoder SHAP attributions shown side by side, feeding a two-tier review queue at the 99th percentile (**26 transactions, 1.04%, score ≥ 0.9510**) and the 95th percentile (**126 transactions, 5.02%, score ≥ 0.8671**). **No automatic blocking.**

**The single most important technical finding.** Isolation Forest and the Autoencoder disagree almost completely about *why* a transaction is anomalous. Their global feature-importance rankings correlate at Spearman **ρ = −0.3705** (Phase 11) — a sharper disagreement than the in-house pipeline's ρ = −0.157, on a feature set less than half the size. The mechanism is understood and replicated independently across both pipelines: Isolation Forest's attributions are dominated by low-cardinality one-hots that a single split isolates cheaply (`TransactionType_Debit` is its #1 feature and `CustomerOccupation_Retired` its #2 in *both* pipelines), while the Autoencoder's are dominated by whichever continuous features resist compression through a small bottleneck. There is a worked case (`TX001029`) where a $516.47 transaction at 0.40× its account's balance, with normal login behaviour, lands high on Isolation Forest purely on an extreme `merchant_frequency` value. **That finding is why the recommendation is an ensemble despite a single model being cheaper, faster and easier to operate.**

**The strongest single fraud-signature match, and it was found twice, independently.** `TX000275` (account AC00454) — a $1,176.28 transaction against a $323.69 balance (**3.63×**) with **5 login attempts**, the dataset's observed maximum — is the **highest-scoring transaction in this pipeline's entire ensemble ranking** (`ensemble_percentile_average` = 0.9951, rank 1 of 2,512). The in-house 46-feature pipeline independently surfaced the same transaction in its own top-1% tier, on completely different features. Both this pipeline's SHAP models agree on why: Isolation Forest attributes +1.714 to `LoginAttempts` and +1.607 to `amount_to_balance_ratio`; the Autoencoder attributes +1.570 to `amount_to_balance_ratio`. Two independent feature-engineering philosophies and two structurally different explainers converging on one transaction is the strongest corroboration available in a project with no label.

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

*Source: Phase 7. PCA, UMAP, t-SNE and an autoencoder bottleneck, all on the full 18-column matrix.*

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

Two independent nonlinear projections agreeing this closely is much stronger evidence than either alone. **Neither group is a fraud population**: the student cluster is a demographic segment, and the elevated-login pocket is a population worth watching but not a verdict. Both reappear later — the student segment turns up in K-Means' k=2 split (§5) and in the weak tail of the alert queue (§7.4).

**Autoencoder.** `input(18) → 8 → 4 → bottleneck(3) → 4 → 8 → output(18)`, Adam (lr 1e-3), 200 epochs, MSE loss, on the RobustScaler-scaled matrix. Train MSE **0.2858**, validation **0.2966** — a **+3.8%** train/val gap, proportionally tighter than the in-house 46-feature autoencoder's +13.2%, which is what an easier (lower-dimensional) reconstruction task predicts. Val P95 0.5551, P99 0.6339, max 0.9633. The tail is proportionally much less extreme than the in-house autoencoder's (whose val max was over 10× its P95) — a direct consequence of this feature set having no personal-baseline column capable of producing a very large single-row deviation.

---

## 5. Model Comparison

*Source: Phase 8. Twelve models, one shared 18-column matrix, one shared `RobustScaler` fit on the 2,009-row training split, all scores oriented higher = more anomalous. An explicit assertion confirms no `vote_count`/`risk_tier`/`is_fraud` column exists anywhere in the input.*

Models 1–4 and 7–10 were fit on the training split and scored all 2,512 rows out-of-sample. **DBSCAN and HDBSCAN have no out-of-sample `.predict` and had to be fit on the full dataset** — a methodological footnote in research that becomes an architectural constraint in production (§10). The LSTM Autoencoder required a separate account-level split.

| Model | Flagged rate | Internal-validity Silhouette | Mean pairwise Spearman (self-excluded) |
|---|---:|---:|---:|
| **Elliptic Envelope** | 5.02% | **0.5409** | 0.602 |
| HDBSCAN | 8.88% | 0.4302 | 0.665 |
| DBSCAN | 2.27% | 0.4280 | **0.235** |
| One-Class SVM | 5.73% | 0.4059 | 0.650 |
| K-Means | 5.02% | 0.4003 | **0.670** |
| Isolation Forest | 5.29% | 0.3975 | 0.579 |
| LSTM-AE (of 2,402 applicable rows) | 5.04% | 0.3746 | 0.565 |
| GMM | 5.02% | 0.2925 | 0.449 |
| Hybrid Ensemble | 3.30% native | 0.2774 | 0.423 |
| LOF | 4.46% | 0.2765 | 0.646 |
| VAE | 5.02% | 0.2012 | 0.609 |
| Autoencoder | 5.02% | **0.1724** | 0.588 |

**This table is not a leaderboard, and reading it as one would be the easiest way to reach a wrong conclusion.** Four qualifications:

1. **Silhouette structurally favours distance-based methods.** A top-5%-by-distance cut is close to guaranteed to separate well *in a distance metric*. The reconstruction-error models score lowest not because they are worse but because low reconstruction error does not imply Euclidean proximity — and an 18→3 compression leaves even less geometric residual than the in-house 46→4 did.
2. **Elliptic Envelope tops the ranking and its core assumption is measurably false.** Shapiro-Wilk on all 18 scaled features: **100% reject normality at p<0.05** — every one of the 18 p-values rounds to 0.0000. Its MCD-based Mahalanobis distance is retained for comparison and explicitly not recommended as a primary detector.
3. **The flagged rates are largely set by construction.** Eight of the twelve cluster in 4.5%–5.7% because they either take a `contamination≈0.05` parameter or use the standardised top-5% convention. The rate carries little information; the *agreement on which rows* carries all of it. DBSCAN (2.27%) and HDBSCAN (8.88%) are the two genuine outliers among the outlier-detectors, in opposite directions.
4. **The self-excluded Spearman column above is not the figure Phase 8 published.** Phase 8 §3.2/§3.3's "mean pairwise" values include each model's self-correlation of 1.0 in a 12-way average, inflating every figure by the same monotone transform. Rankings and conclusions are unaffected; the corrected values are given here and the discrepancy is logged in Phase 14 §5.

**Cross-model agreement is where the real structure is.** Strongest pairs: LOF ↔ VAE **ρ=0.839**, LOF ↔ K-Means 0.838, Autoencoder ↔ VAE 0.837, HDBSCAN ↔ K-Means 0.832. Weakest: **every one of the six lowest pairs involves DBSCAN**, bottoming at DBSCAN ↔ Autoencoder ρ=**0.153**. **DBSCAN standing alone at the bottom is the single most reproducible cross-model finding in this project** — it reproduced independently on the in-house 46-feature set too, which makes it a property of DBSCAN's single-dense-cluster behaviour on this raw data rather than an artifact of either feature-engineering choice.

**A distinction the flagged-set overlap metric surfaces that rank correlation does not.** Elliptic Envelope has the **lowest mean Jaccard overlap of any model** (0.170 self-excluded) despite a comfortably mid-pack Spearman (0.602). Its broad *ranking* tracks the field; its specific *top-5% set* barely overlaps anyone's. This is why both metrics are reported side by side rather than one being chosen — they answer different questions and here they disagree.

**Model-specific findings worth carrying forward:**

- **K-Means finds a real split, and it is demographic, not fraudulent.** Silhouette argmaxes at k=2 (0.491) and — unlike the in-house pipeline, where k=2 was a degenerate 3-row micro-cluster — this is a genuine **1,830/179** partition of the training split, with the 179-row minority **77.7% `CustomerOccupation_Student`** against 21.3% in the majority. It is the same segment Phase 7's UMAP and t-SNE both found. It is well-separated, real, and fraud-irrelevant. **The inertia elbow, however, is genuinely ambiguous**: no k in 2–10 clears the elbow test, so the rule mechanically returns the boundary, k=10. That ambiguity is reported rather than resolved by picking a more convenient answer.
- **HDBSCAN is far more usable here than in-house, and still fragile.** Its best config reaches 8.88% noise (in-house: 53.94%), plausibly because an 18-dimensional space suffers less mutual-reachability inflation than a 46-dimensional one. But the other three configs tried sit at **86.1%–90.4%** noise — a cliff, not a gradient.
- **GMM prefers a simpler covariance structure here.** Best BIC is `n_components=10, covariance_type='diag'` (BIC −27,620.9) — **diag, not the in-house pipeline's full**. Even with only 171 free covariance parameters per full-covariance component (vs. 1,081 at 46 features), the data's per-component correlations do not justify them. The BIC curve is still descending at the n=10 search boundary, so the true optimum may lie beyond it — an honest, unresolved caveat.
- **The LSTM Autoencoder overfits, and it was reported rather than early-stopped away.** Validation MSE bottoms out near **0.49 around epoch 50** and then climbs steadily to **0.79 by epoch 150** while training MSE keeps falling — a textbook overfitting signature, on 342 training *sequences* of median length 5. It also structurally cannot score 110/2,512 rows (4.4%). Phase 8's own conclusion: it is even less justified on this feature set than the in-house one was.
- **The VAE buys nothing here.** It reconstructs worse than the plain autoencoder at **every percentile including the maximum** (val 0.3458 vs. 0.2966; P99 0.8484 vs. 0.6339; max 1.2137 vs. 0.9633). On the in-house feature set the VAE at least smoothed the extreme tail; here that effect does not appear at all.

---

## 6. Hyperparameter Tuning

*Source: Phase 9. Three models tuned with Optuna's TPE sampler: Isolation Forest, GMM, VAE. With no label, "optimise against what" is a genuine design decision and was stated per model.*

| Model | Method | Result |
|---|---|---|
| **Isolation Forest** (objective: silhouette, top-5% vs. rest) | Exhaustive grid, 60 combos, 40.8s | **0.4154** |
| | Optuna TPE, 30 trials, 25.0s | 0.4107 |
| | Random search, 30 trials, 26.8s | 0.3840 |

**Reported plainly rather than spun: the exhaustive grid still won.** Optuna came **0.0047** short in half the evaluations — a smaller gap than the in-house pipeline's 0.0111, but still short. The practical conclusion carries over: a 3-hyperparameter, cheap-to-evaluate search space is too small for Bayesian optimisation's main selling point to pay off against a grid you can afford to enumerate.

**But one result genuinely differs from in-house, and it is the more interesting half.** Optuna beat random search here by **+0.0267** on an identical 30-trial budget, where in-house the two were statistically indistinguishable. One plausible read: with fewer, more population-level features and no personal-baseline columns adding noise to the objective surface, the silhouette-vs-hyperparameter relationship here is smoother and more exploitable — exactly the setting where TPE's model-based proposals have more signal than uniform sampling. Also notable: the winning config uses `n_estimators=50`, the **smallest** value in the grid — the opposite of the in-house pipeline's `n_estimators=300`. With only 18 lower-dimensional features, fewer and shallower isolation trees already separate the top-5% tail as well as a much larger forest.

**GMM: a materially more reassuring result than in-house.** Adding `reg_covar` to the search space moved best BIC from −27,620.9 to **−47,044.7** — and the winning covariance type **stayed `diag`** in both the fixed-`reg_covar` grid and the free-`reg_covar` search. In-house, the equivalent search pushed toward an even more parameter-hungry `full`-covariance solution and reinforced an overfitting concern. Here it does not: a `diag` structure has 18 variances per component rather than 171 covariance entries, so a smaller regularisation floor is a far lower-risk route to a lower BIC. **`diag` covariance with a smaller `reg_covar` is a more defensible production choice here than the in-house pipeline's `full`-covariance result was for its own feature set.** The n_components boundary caveat still stands.

**VAE: replicates the in-house finding almost exactly.** 20 trials, 192.2s (the slowest search — each trial is a full 60-epoch training run). Best: `latent_dim=3, hidden1=8, beta=0.0113, lr=0.00312`, val MSE 0.2757 at 60 epochs. The search landed on exactly the deployed model's architecture, differing only in `beta` and `lr`, and **`beta` is again clearly the dominant hyperparameter** — the search's best beta sits an order of magnitude below the deployed model's 0.1. This is consistent with, and mechanistically explains, §5's finding that the deployed VAE reconstructs worse than the plain autoencoder at every percentile. (The 0.2757 and the deployed model's 0.3458 are **not directly comparable** — different epoch budgets — and are not presented as if they were.)

---

## 7. Evaluation Results

*Source: Phase 10. With no label, evaluation means four label-free things, none of which substitutes for precision and recall.*

### 7.1 Internal validity

Every model's top-5%-by-score partition was scored on Silhouette, Davies-Bouldin and Calinski-Harabasz in the shared 18-feature scaled space, using one consistent partition definition across all twelve. Results are in §5's table with the qualifications that go with them.

**Elliptic Envelope leads by a wide margin** — Silhouette 0.5409, Davies-Bouldin 1.1529, Calinski-Harabasz **592.47**, the last more than 3× HDBSCAN's next-best 180.07. This is a real reversal from in-house, where it sat 7th of 12, and it *explains* the Jaccard anomaly from §5: EllipticEnvelope's MCD tail identifies a small, tightly-clustered, well-separated group in the scaled space — a coherent structural outlier population, just not the one everyone else converges on.

**The Autoencoder is last** (0.1724 / 5.6806 / 15.73), more sharply so than in-house, for the construction reason in §5 qualification 1.

### 7.2 Stability — the most operationally significant result in this pipeline

Isolation Forest, LOF and the Autoencoder were each refit on 5 bootstrap resamples of the training split, rescored against all 2,512 rows, and the top-5% flagged set recomputed each time.

| Model | Mean pairwise Jaccard (5 runs) | Min | Max |
|---|---:|---:|---:|
| **Isolation Forest** | **0.6021** | 0.5090 | 0.6689 |
| LOF | 0.5124 | 0.4651 | 0.5750 |
| **Autoencoder** | **0.3726** | **0.2115** | 0.5849 |

**Read the Autoencoder row plainly: in the worst observed retrain pair, fewer than one in four flagged transactions was shared between two runs.** Nothing changed except which rows the bootstrap happened to include — no drift, no new data, no hyperparameter change.

**The ranking is inverted relative to the in-house pipeline**, where LOF was most stable (0.590) and Isolation Forest least (0.527) within a narrow 0.527–0.590 spread. Here the spread is wide (0.373–0.602) and the order is different: **Isolation Forest is clearly the most retrain-stable model on this feature set, and the Autoencoder clearly the least.** A plausible mechanism, stated as such: with only 18 largely population-level features and a 3-dimensional bottleneck, the Autoencoder has little redundant structure to anchor its notion of "normal" on, whereas Isolation Forest's splits are simple univariate thresholds less sensitive to exactly which rows are present.

Two things follow, and both shaped the final design. **Operationally:** a fixed model artifact plus a monitored, versioned retraining process is the correct posture, not continuous retraining, and an operations team must be told the expected churn band in advance (Phase 16 §4.2, §5.2). **Methodologically:** this is a direct argument for aggregating detectors rather than trusting a borderline single-model result.

### 7.3 Reconstruction quality

| Model | Train MSE | Val MSE | Val P95 | Val P99 | Val Max |
|---|---:|---:|---:|---:|---:|
| Autoencoder | 0.2858 | **0.2966** | 0.5551 | 0.6339 | 0.9633 |
| VAE | 0.3371 | 0.3458 | 0.6394 | 0.8484 | 1.2137 |
| LSTM-AE | 0.4345 | 0.7907 | — | — | — |

Same ordering as in-house (AE < VAE < LSTM-AE) for the same reasons, but with two differences worth noting: the plain autoencoder reconstructs **better** here than the in-house one did (0.2966 vs. 0.3280 val MSE) on an easier, lower-dimensional task, and the LSTM-AE's train/val gap is proportionally larger (1.82×) because of the overfitting documented in §5.

### 7.4 Business evaluation — reading real flagged transactions

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

*Source: Phase 11. `shap.TreeExplainer` (exact, 8.3s for all 2,512 rows) for Isolation Forest; `shap.GradientExplainer` on a wrapper returning per-row reconstruction MSE (112.3s) for the Autoencoder. Both run over the full dataset, not a subsample. Both sign conventions verified directly, not assumed — TreeExplainer's raw output tracks `score_samples` (ρ=1.0000 on a 200-row check), the opposite of this project's convention, so its values are negated before reporting.*

**The central finding: the two explained models attribute their scores almost entirely differently.**

| Rank | Isolation Forest | mean\|SHAP\| | Autoencoder | mean\|SHAP\| |
|---:|---|---:|---|---:|
| 1 | `TransactionType_Debit` | 0.391 | `Location_FE` | 0.0354 |
| 2 | `CustomerOccupation_Retired` | 0.294 | `account_frequency` | 0.0304 |
| 3 | `CustomerOccupation_Student` | 0.265 | `merchant_frequency` | 0.0302 |
| 4 | `CustomerOccupation_Engineer` | 0.207 | `TransactionAmount` | 0.0279 |
| 5 | `TransactionAmount` | 0.207 | `TransactionDuration` | 0.0257 |

**Top-10 overlap: 3 of 10** (`TransactionAmount`, `account_frequency`, `amount_to_balance_ratio`). Spearman correlation between the two full 18-feature importance vectors: **ρ = −0.3705** — a sharper disagreement than the in-house pipeline's ρ = −0.157.

**The mechanism is understood, and it replicated across two independent feature sets.** Isolation Forest scores by how few random splits isolate a point, so a single split on a binary feature isolates an entire minority class in one step — low-cardinality categoricals dominate its attributions. **`TransactionType_Debit` is its #1 feature and `CustomerOccupation_Retired` its #2 in *both* pipelines**, on 18 and 46 features respectively. That is an independent replication of a mechanism, not a coincidence.

**What the Autoencoder became on this feature set, and why it matters to the bank.** In-house, the Autoencoder's top four features were all *personal-baseline amount* features (`Amount_vs_AccountAvg`, `Amount_ZScore_Account`, `Amount_to_Balance_Ratio`, `Amount_to_RollingMean_Ratio`), and Phase 11 there read it as an "is this amount unusual for this account" detector. **Here, `amount_to_balance_ratio` ranks only 7th**; the top three are all frequency-encoded (`Location_FE`, `account_frequency`, `merchant_frequency`). This is the direct, structural consequence of the capability gap in §3: with no personal-baseline columns for the reconstruction error to concentrate on, the hardest-to-reconstruct residual shifts to the frequency features. **On this feature set the Autoencoder must be read as "is this transaction's popularity profile unusual", not "is this amount unusual for this account."** The in-house reading does not transfer, and presenting it as if it did would be a material misrepresentation to an investigator.

**Two worked local cases, both instructive:**

- **`TX000615`** — the clearest example of the two models disagreeing about *which aspect* of the same transaction is anomalous. Isolation Forest reads it as an amount anomaly (`high_amount_transaction` +1.375, `amount_to_balance_ratio` +1.115); the Autoencoder reads it primarily as a **location-frequency** anomaly (`Location_FE` +0.086, its largest driver for this row, above the amount ratio). Neither is wrong; relying on one would give an investigator an incomplete picture.
- **`TX001029`** — the most important case in this section. Both models **agree**, and what they agree on is that the oddity is an extreme `merchant_frequency` value (z = 3.67), not an unusual amount, login pattern or balance-relative size. Phase 10 independently judged this $516.47 transaction (0.40× balance, normal logins, Student account) not a plausible fraud pattern. **Cross-checking against a second, structurally different model is exactly what catches this class of low-value flag before it reaches a human reviewer** — and on this feature set it did.

**This is what makes the ensemble recommendation an evidence-based decision rather than a default.**

---

## 9. Ensemble Scoring and Thresholds

*Source: Phases 12–13.*

Eleven of the twelve models were combined; the Hybrid Ensemble was deliberately excluded as an input, because it is itself a ≥2-of-3 vote of Isolation Forest + LOF + Autoencoder and folding it back in would double-count those three.

Four strategies were built: **weighted average** (disagreement-inverse weights, K-Means 0.109 down to DBSCAN 0.051 — the scheme correctly down-weights the model Phase 8 found most divergent), **rank aggregation** (Borda), **percentile aggregation**, and a **PCA stacking proxy** (explicitly *not* supervised stacking — there is no label to fit a meta-learner against; PC1 explains **54.90%** of the variance across the eleven standardised score columns).

**An honest finding rather than a manufactured distinction: Borda and percentile aggregation are near-identical here** — ρ = **0.9999**, Jaccard 0.969 on the top-5% set. That is expected: summing ranks and averaging rank/N are the same operation up to a normalisation constant, and they diverge only in how they handle the LSTM-AE's 110 missing rows. **On this feature set, Weighted Average and PCA Stacking also converge very tightly** (ρ = 0.9990, Jaccard 0.924) — more tightly than in-house — so there are two strategy clusters, not four independent signals.

**One cross-check returned nothing, and it is reported rather than dropped.** All four strategies correlate with v1's independent 4-detector `vote_count` proxy at **ρ between −0.0069 and −0.0065** — indistinguishable from zero. In-house, the same cross-check returned a modest but consistently positive ρ ≈ 0.442–0.444. The explanation is not a defect in either pipeline: v1's proxy was built on personal-baseline/expanding-statistics features in the same style as the in-house 46-feature set, so a detector ensemble built entirely from population-level features and one built from personal-baseline features are **evidently identifying substantially different transactions as anomalous on this dataset**. That is a measured confirmation of the capability difference, not a contradiction to resolve — but it does mean this pipeline has one fewer piece of corroborating evidence than the in-house one, and Phase 14 explicitly declines to lean on it.

**Percentile aggregation was selected** on three grounds: it is robust to eleven heterogeneous native scales (reconstruction MSE, a bounded kernel decision value, a GLOSH score, a negative log-likelihood, a centroid distance) with no assumption about their shapes; it has no tuned weights to defend to a reviewer; and it handles missing model scores by skipping and renormalising, which is both the right degradation behaviour and the mechanism that makes a reduced-member production deployment possible at all (§10).

**Thresholds.** Score distribution: mean 0.5007, std 0.2250, min 0.0458, max 0.9951.

| Tier | Percentile | Score cut | Flagged | Daily load at this sample's rate |
|---|---|---:|---:|---:|
| **Priority review** | 99th | **0.9510** | 26 (1.04%) | ~0.07/day |
| **Standard review** | 95th | **0.8671** | 126 (5.02%) | ~0.35/day |

**A genuine methodological finding, reproduced independently on a completely different feature basis:** the classic statistical thresholds **flag zero transactions** on this score. mean+3σ = 1.1757 and Q3+1.5×IQR = 1.2332, both above the observed maximum of 0.9951 — because averaging eleven bounded percentiles compresses the tails, a CLT-like effect. The in-house pipeline found exactly the same thing on its own score. It is not a defect in percentile aggregation; it is a mismatch between one thresholding convention and one score's shape, confirmed by applying the identical rules to two **unbounded** scores, which both produced usable cut points (Isolation Forest's raw score: 16 and 25 flagged; the weighted-average ensemble: 29 and 79). **If a "three standard deviations" framing is wanted, apply it to the weighted average, not the percentile score** — and keep the weighted average computed in parallel for exactly that purpose, since it costs nothing once the members have run.

**Cost framing, bounded honestly.** A cost-optimal threshold sweep **cannot be reproduced here**, because counting false negatives requires knowing which *unflagged* transactions are fraud. What can be computed is an upper bound on review labour assuming every flagged transaction is a false positive: **$630 at the 95th percentile, $130 at the 99th**, using v1's own illustrative (not real-bank) figures of $5 per false positive. That is a ceiling, not an estimate. **A lower threshold is not "cheaper" in any total-cost sense** — it only reviews fewer transactions, trading off against catching less of whatever fraud is present, in a direction this project cannot quantify.

**Note on the flagged counts.** They are 126/76/26/13 across the four percentile tiers — mechanically identical to the in-house pipeline's, because both cut the same fixed percentiles of the same 2,512 rows. **This is not evidence the two pipelines agree on which transactions to flag.** The near-zero v1 cross-check above shows they largely do not.

---

## 10. Final Model Selection and Deployment

*Source: Phases 14–15.*

Sixteen candidates (12 models + 4 ensemble strategies) were scored on six dimensions — Detection Quality (0.25), Stability (0.20), Interpretability (0.15), Scalability (0.15), Deployment Readiness (0.15), Computational Cost (0.10) — using the identical rubric and weights as the in-house pipeline, so that differences in outcome are attributable to evidence rather than to a changed rubric.

**Top of the matrix:**

| # | Candidate | Raw /30 | Weighted |
|---:|---|---:|---:|
| 1 | **Isolation Forest** | 27 | **4.40** |
| 2= | **Percentile Aggregation** | 19 | **3.40** |
| 2= | **Autoencoder** | 21 | **3.40** |
| 4= | K-Means / LOF / Hybrid Ensemble | 20 / 19 / 19 | 3.15 |
| … | … | … | … |
| 16 | LSTM Autoencoder | 9 | 1.50 |

**Detection Quality is capped at 4 for every candidate.** With no label, nothing has been *shown* to detect fraud — only to partition the space more cleanly, agree more with the field, or produce more plausible examples. Awarding a 5 would misrepresent what was proven.

**The recommendation deliberately overrides the matrix, and the override is stated rather than engineered into the scores.** Isolation Forest ranks first by a full point, and the recommendation is still the ensemble — because in a system with no label, cross-model consistency is the only validation available, §8's ρ = −0.3705 shows a single model is blind to the other family's failure mode, and `TX001029` is a worked instance of that failure mode being caught. At 6.88 transactions/day the compute saved by dropping to one model is negligible.

**The production constraint that decides the architecture.** DBSCAN and HDBSCAN cannot score a transaction they were not fit on. **Any 11-model ensemble score is therefore batch-only.** Phase 14 §3 sets out three options; **Option B** (drop both, aggregate over the remaining nine) is recommended, with **Option C** (re-enable HDBSCAN via `prediction_data=True`) a strong follow-up here — HDBSCAN is a more valuable member on this feature set than it was in-house (8.88% best-config noise vs. 53.94%, and the field's highest mean flagged-set agreement). **A 9-model score is a different score, and Phase 13's thresholds do not transfer to it unrevalidated. That revalidation has not been run.**

**Reduced-member fallbacks, and where they differ from in-house.** The in-house pipeline recommended a 2-model IF + Autoencoder fallback. **That is not recommended here**, because on this feature set those two are not peers: Isolation Forest is the most retrain-stable model measured (0.6021) and the Autoencoder is the least (0.3726, min 0.2115), and the Autoencoder is last of 12 on internal validity. The v2 fallback ladder is: **(1) three models — IF + LOF + Autoencoder, percentile-aggregated** (the only three-model subset where all three have measured stability and all three score out-of-sample natively), then **(2) Isolation Forest alone**, explicitly accepting the loss of the cross-family check and the `TX001029` failure mode it exists to catch.

**Deployment posture: nightly batch.** The full ensemble is batch-only by construction; the volume does not demand real-time; the feature layer (not the models) is the blocker; and with no block tier nothing needs to complete inside a payment-authorisation window. The day real-time becomes necessary is the day a block tier is introduced — and that should not come before the system has been validated against investigator-labelled outcomes.

**The largest piece of unbuilt engineering** is the real-time feature layer for the five frequency-derived features, which must become prior-only running counters in a feature store rather than whole-dataset `groupby`s (Phase 15 §3.2). **A genuine advantage over the in-house pipeline, worth stating because it runs the other way:** the in-house 46-feature set needs a per-account *history scan* at inference (expanding means, rolling windows, novelty flags); this one needs only **counter reads**. Counters are dramatically cheaper to maintain, keep consistent and backfill. **The teammate's feature set is the more deployable of the two, and the less capable — both are true.**

**Investigator surface: Bank Transaction Fraud & Anomaly Detection** (`dashboard/`), a FastAPI + static-frontend console, now reading this pipeline's artifacts: `ensemble_percentile_average` for the score, Phase 13's cutoffs for the tiers, and the precomputed Isolation Forest and Autoencoder SHAP rows shown side by side. Nothing on it is hand-typed from a report — every number is read from an artifact at startup, so a stale artifact produces a visibly stale dashboard rather than a silently wrong one. Its What-if Simulator required a **redesign rather than a repoint**: free-form new-transaction scoring is not honestly possible on a feature set built from population statistics, so it was rebuilt as an *Account Scenario Simulator* anchored to a real account's true historical frequency values (Phase 15 §7.3).

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

**Retraining is measurably expensive here**, and the operating procedure must say so: §7.2's bootstrap numbers mean a retrain changes **40%–63%** of the flagged set with no drift and no new data. Every trigger except an anchor-correlation collapse therefore requires two consecutive batches before firing. Isolation Forest (0.6021) is the retrain-stable anchor to evaluate a retrain against; the Autoencoder (0.3726) needs the tightest change control of any model here — a direct reversal of the in-house pipeline.

**Three hard-failure integrity checks** (the run stops, rather than a signal being raised): the ensemble member count must match the manifest (percentile aggregation's skip-and-renormalise property means a silently-dropped member still produces a plausible score with no error); a frozen canary set must reproduce exactly — **including `TX000275`, the dataset's highest-scoring transaction and its best single canary**; and the Phase 5–6 feature-engineering assertions must all pass, above all the row-alignment check, since a misaligned merge produces a fully-populated, correctly-typed, entirely wrong feature matrix that every other metric would read as normal.

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
| **`TX000275` is the clearest fraud-signature match in the data** | In its top-1% tier (Phase 10) | **Rank 1 of 2,512** on the ensemble score (0.9951) |
| **DBSCAN is the consensus outlier among the detectors** | Lowest agreement, lowest ensemble weight | Lowest by a wide margin: mean ρ 0.235 vs. next-lowest 0.423; weight 0.051 |
| **Isolation Forest's SHAP is dominated by low-cardinality one-hots** | #1 `TransactionType_Debit` (0.176), #2 `CustomerOccupation_Retired` (0.152) | **#1 `TransactionType_Debit` (0.391), #2 `CustomerOccupation_Retired` (0.294)** — same two features, same order |
| **Isolation Forest and the Autoencoder explain scores almost oppositely** | ρ = −0.157, 1/10 top-10 overlap | ρ = **−0.3705**, 3/10 overlap |
| **Borda ≈ percentile aggregation** | ρ = 0.9999, Jaccard 0.953 | ρ = 0.9999, Jaccard 0.969 |
| **mean+3σ and Q3+1.5×IQR flag zero on a percentile-averaged score** | Zero flagged (thresholds 1.1088 / 1.1363 vs. max 0.9988) | **Zero flagged** (1.1757 / 1.2332 vs. max 0.9951) |
| **Elliptic Envelope's Gaussian assumption is violated** | 100% of 46 features reject Shapiro-Wilk | **100% of 18 features reject** |
| **Optuna does not beat an affordable exhaustive grid on a 3-hyperparameter space** | Grid 0.6092 vs. Optuna 0.5981 | Grid 0.4154 vs. Optuna 0.4107 |
| **`beta` dominates the VAE's hyperparameter space** | Confirmed | Confirmed, on a smaller architecture |
| **Retraining substantially changes the flagged set** | 41–47% churn | **40–63% churn** |
| **The top-10% tail dilutes into unremarkable transactions** | Confirmed by spot check | Confirmed, **and attributed** — largely the Student demographic segment |
| **The recommended ensemble strategy** | Percentile aggregation | Percentile aggregation |
| **No automatic block tier is defensible without a label** | Confirmed | Confirmed |
| **The Hybrid Ensemble's Phase 10 partition is mislabelled** | 253 rows described as majority-vote; actually ≥1-vote (94 = majority) | **269 rows described as majority-vote; actually ≥1-vote (83 = majority)** — the same error reproduced independently, which points at the shared top-5%-cut convention on a discrete score, not a transcription slip |

### 12.3 Findings that reversed — and what each reversal is attributable to

| Finding | In-house | This pipeline | Attributable to |
|---|---|---|---|
| **Internal-validity leader** | HDBSCAN 0.672; Elliptic Envelope 7th (0.610) | **Elliptic Envelope 0.5409 by a wide margin**; CH 592.5 vs. next-best 180.1 | A lower-dimensional, more Gaussian-ish scaled space gives an MCD ellipsoid a coherent tail to find. Tempered by its field-lowest Jaccard (0.170) — it finds a real group, just not the one anyone else finds |
| **Most retrain-stable model** | LOF 0.590; IF least at 0.527 (narrow spread) | **IF 0.6021; Autoencoder least at 0.3726** (wide spread) | With 18 mostly population-level features and a 3-D bottleneck, the Autoencoder has little redundant structure to anchor "normal" on; IF's univariate thresholds are less resample-sensitive |
| **What the Autoencoder detects** | "Is this amount unusual *for this account*" (top 4 features all personal-baseline) | **"Is this transaction's popularity profile unusual"** (top 3 all frequency-encoded) | Directly caused by the absence of personal-baseline features. **This is the single most important interpretive difference for an investigator** |
| **HDBSCAN usability** | Best config 53.94% noise | **Best config 8.88% noise** | Lower dimensionality reduces mutual-reachability inflation |
| **K-Means k=2** | A degenerate 3-row micro-cluster artifact | **A genuine 1,830/179 demographic split** (77.7% Student) | Real structure in the occupation one-hots that the in-house space did not surface as cleanly |
| **K-Means elbow** | Clean k=4 | **No elbow in k=2–10**; rule mechanically returns the boundary | Reported as an unresolved ambiguity rather than resolved conveniently |
| **GMM covariance** | `full` wins; the reg_covar search reinforced an overfitting warning | **`diag` wins and stays `diag`** under a free-`reg_covar` search — no overfitting warning | Far fewer free parameters (18 variances vs. 171 covariance entries) leaves much less room for a shrinking regularisation floor to overfit |
| **VAE tail behaviour** | Smooths the extreme tail (lower max than the plain AE) | **Does not smooth** — worse at every percentile including the max | Less redundant structure for KL regularisation to trade against |
| **LSTM-AE training** | Smooth curves, no overfitting | **Overfits from ~epoch 50** (val 0.49 → 0.79) | Fewer, more redundant features to encode; the same tiny 342-sequence training set |
| **Optuna vs. random search** | Statistically tied | **Optuna wins clearly (+0.0267)** | A smoother, more exploitable objective surface without personal-baseline noise |
| **Winning `n_estimators` for IF** | 300 (largest in grid) | **50 (smallest in grid)** | 18 lower-dimensional features need fewer trees to isolate the tail |
| **Cross-check against v1's independent proxy** | ρ ≈ 0.442–0.444 (modest, consistent) | **ρ ≈ −0.007 (nothing)** | v1's proxy is itself built on personal-baseline features. **A measured confirmation that the two feature philosophies flag different transactions** — and a real reduction in the corroborating evidence available to this pipeline |
| **Decision-matrix leader** | IF and Autoencoder tied 4.20 | **IF alone at 4.40; Autoencoder 3.40** | The Autoencoder's measured collapse on internal validity and stability |
| **Recommended minimal fallback** | 2 models: IF + Autoencoder | **3 models (IF + LOF + AE), then IF alone** | The IF/AE stability gap makes them poor peers here |

### 12.4 The bottom line for the bank

**Neither pipeline is strictly better. They trade capability against deployability, and the trade is real in both directions:**

- **The in-house 46-feature set is more capable.** It can ask "is this unusual *for this customer*" — the question Phase 1 identified as the primary signal for account takeover, the strongest-fit fraud scenario in this schema. It has per-account novelty flags. Its Autoencoder detects amount anomalies relative to personal history. This pipeline can do none of that.
- **The teammate's 18-feature set is more deployable.** Its inference-time features are counter reads and frozen lookups, not per-account history scans. It trains and scores faster (Isolation Forest: 4.48s vs. 9.72s for the same 5 configs). It is easier to reason about, easier to explain to a reviewer, and materially cheaper to keep consistent in production. It also contributes one idea the in-house pipeline flagged as worth having and never built (`high_amount_transaction`).
- **Where they agree, believe it.** §12.2 lists fourteen findings reached twice, independently. `TX000275` being the clearest fraud candidate in the data, DBSCAN being the odd detector out, Isolation Forest and the Autoencoder being complementary rather than redundant, percentile aggregation being the right combination strategy, and no block tier being defensible — all of these are now supported by two independent lines of evidence rather than one.
- **Where they disagree, the disagreement is explained, not hand-waved.** Every reversal in §12.3 has a stated mechanism traceable to a specific structural difference between the feature sets.
- **The one genuinely uncomfortable number is worth stating plainly.** The two ensembles correlate at essentially **zero** on which transactions they rank as anomalous. That is not a bug in either; it is the clearest possible measurement of how much a feature-engineering philosophy determines what an unsupervised system detects. **A bank running only one of these two feature philosophies is seeing one view of its transaction risk, not the whole of it** — and the strongest single practical recommendation this project can make from having built both is that the frequency-based features here and the personal-baseline features in the in-house set are **complementary, and the eventual production feature set should contain both.**

---

## 13. Limitations

Stated in order of how much they constrain what can be claimed.

1. **No fraud label.** The binding constraint. No precision, no recall, no AUC, no cost-optimal threshold, no validated detection claim anywhere in this project. Every "evaluation" here is internal consistency or human plausibility. This is not a caveat to the results; it is the boundary of what the results can be.
2. **No personal baseline and no per-account novelty in this feature set.** Phase 1's strongest-fit scenario (account takeover) is only *partially* reachable, and Scenarios 2 and 4 are not reachable at all (§2). This is a capability statement, not a modelling shortfall.
3. **The frequency features are not leakage-safe for live scoring.** Global counts include each account's own future transactions. Fine for offline scoring on a static dataset, wrong for a point-in-time backtest, and requiring a rebuild as prior-only counters before deployment (§3.4, Phase 15 §3.2).
4. **No ensemble-level stability measurement exists.** Three individual models were bootstrap-tested; none of the four strategies was. The recommendation rests partly on the assumption that aggregation damps the measured 0.373–0.602 churn, and that assumption is untested here. **This is the single highest-value missing measurement** (Phase 14 §5, Phase 16 §5.2).
5. **The recommended production score has not been computed.** Option B's 9-model score is a different score from the published 11-model one, and Phase 13's thresholds do not transfer to it unrevalidated (§10).
6. **2,512 rows, 495 accounts, 365 days, 6.88 transactions/day.** A research sample. The relative comparisons generalise; the fitted constants and the absolute daily volumes do not.
7. **The 5% contamination assumption is an assumption**, used throughout as a modelling convention. It is not a measured fraud rate for this population.
8. **Two of the eleven ensemble members cannot score unseen rows**, and one (LSTM-AE) structurally cannot score 4.4% of rows at all.
9. **Alert-volume and unseen-category alarm bands are proposals, not measurements** — there is only one batch of data in this project (Phase 16 §5.1, §2.3).
10. **Five discrepancies were found in this pipeline's own reports while synthesising**, all logged in Phase 14 §5 rather than silently corrected: the self-inclusive "mean pairwise" agreement figures in Phase 8 §3.2–3.3 (rankings unaffected); the mislabelled Hybrid Ensemble partition in Phase 10 §1 (269 rows is the ≥1-vote set, not the ≥2-of-3 set of 83); the unqualified LSTM-AE rate key in `model_comparison_summary.json` (4.82% over all rows vs. the correctly-labelled 5.04% over applicable rows); **`TransactionAmount` being `StandardScaler(log1p(amount))` rather than the plain `StandardScaler(amount)` Phase 5 §1 states**; and **`amount_to_balance_ratio` being exactly `StandardScaler(log1p(amount / (balance + 1)))`, contradicting Phase 5 §2.3's conclusion that no exact formula was recoverable**. The last two change no modelling result — every phase used the columns as given — but they are the difference between a partly-inferred and a fully-specified production feature layer, and the log transform on `TransactionAmount` changes how a SHAP contribution on that column should be read (unusual on a *log*-amount scale, not in dollars).
11. **`TX000395` and `TX002192`-style readings are plausibility, not evidence.** Where Phase 10 offers a card-testing or ATO narrative for an ambiguous transaction, it says so explicitly. None of those readings is established.

---

## 14. Future Improvements

In priority order, with the reasoning for the order:

1. **Capture investigator decisions from day one.** Bank Transaction Fraud & Anomaly Detection's `queue_state.json` is the only label-generating mechanism in this project. A year of it makes a supervised model, a real precision/recall number, and a genuine cost-optimised threshold possible. **Everything in §13's list of limitations either dissolves or shrinks once labels exist.**
2. **Measure ensemble-level bootstrap stability.** Cheap (re-run Phase 10 §2's procedure through Phase 12's aggregation) and it closes the largest evidential gap under the current recommendation.
3. **Compute and validate the Option B 9-model score**, with Spearman/Jaccard against the published 11-model score and re-derived thresholds. Required before any deployment.
4. **Merge the two feature philosophies.** §12.4's strongest finding. The production feature set should carry this pipeline's frequency encodings *and* the in-house pipeline's personal-baseline and per-account novelty features. Neither alone sees the whole picture, and the near-zero correlation between the two ensembles measures exactly how much is being missed by picking one.
5. **Rebuild the frequency features as prior-only running counters.** Prerequisite for real-time scoring and for any honest point-in-time backtest.
6. **Resolve K-Means' k and GMM's `n_components`.** Both are currently boundary artifacts of their search ranges (§5). Extending the search range is a few minutes of compute.
7. **Refit the VAE at the searched `beta` (~0.011) rather than the deployed 0.1**, or drop it — on current evidence it is a strictly worse Autoencoder that correlates 0.837 with one (§5, §6).
8. **Enable `prediction_data=True` for HDBSCAN** and evaluate Option C. HDBSCAN has the field's highest flagged-set agreement here and is worth more than it was in-house.
9. **Build the Phase 16 monitoring in the order given in §8 of that report** — the integrity assertions first, because the code already exists and they catch the invisible failures.
10. **Drop the LSTM-AE.** It overfits, it cannot score 4.4% of rows, and it scores last of sixteen candidates (§10). Keeping it costs a model artifact and a data-preparation pipeline for no measurable return.

---

## 15. Final Recommendation

**Deploy: percentile aggregation over the out-of-sample-capable detectors (Phase 14 Option B), with Isolation Forest and Autoencoder SHAP attributions shown side by side on every alert, feeding a two-tier human review queue.**

| Tier | Threshold | Score cut | Volume in this sample |
|---|---|---:|---|
| **Priority review** | 99th percentile | **0.9510** | 26 (1.04%) |
| **Standard review** | 95th percentile | **0.8671** | 126 (5.02%) |

**Secondary score, computed in parallel at no extra model cost:** the weighted-average ensemble, which is unbounded and therefore supports sigma/IQR-style thresholds that the percentile score structurally cannot (§9).

**Fallbacks if fewer artifacts can be operated:** three models (IF + LOF + Autoencoder, percentile-aggregated), then Isolation Forest alone — explicitly accepting, at that last step, the loss of the cross-model check and the failure mode it catches.

**Run it nightly in batch.** Real-time waits for the frequency-counter feature store, and it is not needed until a block tier exists.

### Why an ensemble, when a single model scores a full point higher

Because with no label, cross-model agreement is the only validation available, and this pipeline measured that Isolation Forest and the Autoencoder agree on almost nothing about *why* a transaction is anomalous (ρ = −0.3705). `TX001029` is the worked proof: a $516.47 transaction at 0.40× its account's balance with normal login behaviour, high on Isolation Forest for an extreme merchant-frequency value that has nothing to do with fraud. A second model with a structurally different basis is what stops that reaching a reviewer. Discarding that check to save seconds of compute at 6.88 transactions/day is the wrong trade.

### Why no blocking tier

Because a cost-optimal threshold requires counting false negatives, and counting false negatives requires knowing which *unflagged* transactions are fraud — which is unknowable without a label. Blocking a customer's transaction on a score whose false-negative behaviour has never been measured is not something to hand a bank. Every output of this system goes to a human until it has been validated against real investigator-labelled outcomes.

### What this system is, and what it is not

**It is** a ranked, explained, thresholded queue of transactions that deviate from population-level patterns, built on a verified feature pipeline, with every model's behaviour measured, every disagreement documented, and every unmeasured claim marked as unmeasured. **It is** the more deployable of two pipelines built over the same data, and it independently corroborated fourteen of that other pipeline's findings.

**It is not** a fraud detector with a known hit rate. It has never been shown to catch fraud, because there is no fraud in this dataset to be shown against. **It is not** capable of asking whether a transaction is unusual *for a specific customer* — that requires features this set does not contain. And **it is not** finished: the single most valuable next step is not a better model, it is capturing the investigator decisions that would let anyone, finally, measure whether any of this works.
