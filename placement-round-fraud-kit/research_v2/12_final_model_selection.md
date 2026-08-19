# Phase 14 (v2) — Final Model Selection (Teammate's 18-Feature Matrix)

**Nothing is computed in this phase.** Every score below is a judgement call over evidence already produced in Phases 5–13 of this pipeline (`research_v2/04_feature_engineering.md` through `research_v2/11_threshold_optimization.md`), and every number cited is traceable to the report that produced it or to the artifact that report points at. Where a dimension could not be measured for a candidate, the score is marked **(inferred)** and the reasoning states what it was inferred from — it is never presented as if it had been measured.

Sixteen candidates are scored: the 12 models built in Phase 8 (v2) (`research_v2/06_model_development.md`) and the 4 ensemble-scoring strategies built in Phase 12 (v2) (`research_v2/10_ensemble_scoring.md`).

This phase is deliberately structured to be comparable, dimension for dimension and weight for weight, with the in-house 46-feature pipeline's own Phase 14 (`research/12_final_model_selection.md`) — so that where this pipeline reaches a *different* conclusion, the difference is attributable to the evidence rather than to a changed rubric. Several of the conclusions do differ, and Section 4 says so plainly.

---

## 0. Scoring Scheme

Six dimensions, each scored 1–5 (5 = best). The rubric is fixed before scoring, not fitted to the answer, and is identical to the in-house pipeline's:

| Dimension | What it measures | Primary evidence in this pipeline |
|---|---|---|
| **Detection Quality** | Internal cluster-validity of the model's own top-5% partition, plus how plausible its flags looked under manual business review, plus how much its ranking agrees with the rest of the field | Phase 10 (v2) §1 (Silhouette / Davies-Bouldin / Calinski-Harabasz), Phase 10 (v2) §4 (business walkthrough), Phase 8 (v2) §3.2–3.3 (mean Spearman / mean Jaccard) |
| **Stability** | Would a retrain flag the same transactions? | Phase 10 (v2) §2 (bootstrap Jaccard, measured for 3 models only), Phase 8 (v2) §1.5–1.8 and Phase 9 (v2) (config-to-config sensitivity as a proxy for the rest) |
| **Interpretability** | Is there a working, defensible per-feature attribution path — one that actually exists in this codebase? | Phase 11 (v2) (`research_v2/09_explainability.md`) |
| **Computational Cost** | Measured fit/score/search cost at n=2,512 × 18 features, plus lifecycle burden | Phase 8 (v2) per-model timings (`artifacts_research_v2/model_summary_classical.json`), Phase 9 (v2) search timings, Phase 11 (v2) explainer timings |
| **Scalability** | Can it score a transaction it was not fit on, and how does that cost grow? | Phase 8 (v2) §0 (split methodology) and per-model notes; Phase 12 (v2) §0 (the DBSCAN/HDBSCAN out-of-sample constraint) |
| **Deployment Readiness** | Judgement call: could this ship as-is, given the artifacts that actually exist in `artifacts_research_v2/`? | The saved artifacts in `artifacts_research_v2/models/`, plus every caveat above |

**Weighting** used for the weighted total: Detection Quality 0.25, Stability 0.20, Interpretability 0.15, Scalability 0.15, Deployment Readiness 0.15, Computational Cost 0.10 — carried over unchanged from the in-house rubric, for the reason given there and equally true here: for a bank fraud-review system, being wrong and being unstable cost more than being slow, and at this dataset's 6.88 transactions/day (Phase 13 v2 §4) compute is close to free. The unweighted total out of 30 is reported alongside so a reader who disagrees with the weights can use it.

**Three ceiling rules, stated up front rather than discovered in the scores.**

1. **No candidate can score 5 on Detection Quality.** There is no fraud label anywhere in this project (Phases 10, 12 and 13 v2 each state this independently), so no candidate has been *shown* to detect fraud better than any other — only to partition the feature space more cleanly, agree more with the rest of the field, or produce more plausible-looking examples under manual reading. A 4 is the ceiling the available evidence supports.
2. **This feature set's own structural gap bounds what Detection Quality can mean here.** Phase 5 (v2) §3 established that these 18 features contain no personal-baseline (expanding/rolling) statistics and no per-account novelty flags, so every model scored below is, structurally, detecting "unusual in the population" rather than "unusual for this specific account" (Phase 10 v2 §4 restates this before reading any example). A high Detection Quality score here is a score *within that narrower capability*, not against it.
3. **The one semi-independent external check available in the in-house pipeline is not available here.** In-house, all four ensemble strategies correlated with v1's independent 4-detector `vote_count` proxy at ρ ≈ 0.442–0.444, and that cross-check contributed real evidence to its Detection Quality scores. Here, Phase 12 (v2) §2.1 measured that same correlation at **ρ between −0.0069 and −0.0065 — indistinguishable from zero** for all four strategies, because v1's proxy was itself built on personal-baseline features this set does not have. That check therefore carries **no** evidential weight in this phase, in either direction, and no score below leans on it. This is a genuine reduction in the evidence available to this pipeline relative to the in-house one, and it is recorded as such rather than substituted for with something weaker.

**A note on the agreement figures quoted below.** Phase 8 (v2) §3.2/§3.3 report "mean pairwise Spearman/Jaccard per model" using a mean that *includes each model's own self-correlation of 1.0* in a 12-way average. Recomputed from `artifacts_research_v2/model_pairwise_spearman.csv` and `model_pairwise_jaccard.csv` while preparing this phase, the true self-excluded pairwise means are lower (e.g. DBSCAN 0.235 rather than 0.299; K-Means 0.670 rather than 0.698). The transform is monotone and identical for every model, so **the ranking Phase 8 (v2) drew from these figures is unaffected and every qualitative conclusion built on it stands** — but the numbers differ, so both are given below (as `self-excluded / as-published`) rather than one silently replacing the other. This is logged as Inconsistency 1 in §5.

---

## 1. Candidate-by-Candidate Scoring

### 1.1 Isolation Forest — 27/30 (weighted 4.40)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | Mid-table on internal validity (Silhouette 0.398, **6th of 12** — Phase 10 v2 §1), and mid-table on cross-model agreement (mean Spearman 0.579/0.614, 5th-lowest). What earns the 4 is that it is the **only model in this pipeline that was put through a manual business review** (Phase 10 v2 §4): of the six top-1% transactions examined, `TX000275` is a clean Phase 1 Scenario 1 match (5 login attempts — the dataset maximum — plus an amount 3.63× the account's balance) and `TX001214` a defensible partial ATO match (5 login attempts, unremarkable amount); `TX000935`, `TX002192` and `TX000395` are genuinely ambiguous; and `TX001903` is a demonstrable false signal (a 0.84× balance ratio, normal logins, flagged essentially on a 37-second duration). Docked from higher by that failure mode and by Phase 11 (v2) §2's `TX001029` case, where its high score traces to `merchant_frequency` rarity rather than anything fraud-relevant. |
| Stability | 4 | **Measured, and the best of the three tested**: mean bootstrap Jaccard **0.6021** across 5 refits (min 0.509, max 0.669 — Phase 10 v2 §2). Scored 4 rather than 5 because ~40% of its flagged set still changes between retrains, which is a genuine operational problem; scored above the in-house pipeline's Isolation Forest (which took a 3 at 0.527) because 0.602 is both higher in absolute terms and clearly separated from its peers here (LOF 0.512, Autoencoder 0.373) rather than being inside a spread too narrow to call, as it was in-house. |
| Interpretability | 4 | The best explainer path in this pipeline: `shap.TreeExplainer`, **exact** (not an approximation), no background sample required, **8.3s for all 2,512 rows** (Phase 11 v2 §0), with the score-sign convention verified directly rather than assumed (ρ=1.0000 against `score_samples` on a 200-row spot-check). Not a 5 because Phase 11 (v2) §1 showed *what* it explains is frequently unhelpful: its top four global drivers are all low-cardinality one-hots (`TransactionType_Debit` 0.391, `CustomerOccupation_Retired` 0.294, `CustomerOccupation_Student` 0.265, `CustomerOccupation_Engineer` 0.207), because a single binary split isolates an entire minority class cheaply. An exact explanation of a mechanically-cheap split is still a weak explanation. |
| Computational Cost | 5 | 4.48s to fit and score all 5 configs at n=2,512 × 18 features (Phase 8 v2 §1.1) — 2.2× faster than the in-house 46-feature run's 9.72s, as expected from the narrower input. Phase 9 (v2) enumerated a full 60-combination grid in 40.8s. Cheapest classical model to *search*, and its explainer is 13.5× cheaper than the Autoencoder's. |
| Scalability | 5 | Native `decision_function`, used as intended: fit on the 2,009-row train split, scored all 2,512 rows out-of-sample (Phase 8 v2 §0). Scoring is a tree-path traversal, effectively O(log n) per row per tree. Notably, Phase 9 (v2) §1 found the *smallest* `n_estimators` in the grid (50) won on this 18-feature space — so the deployable forest can be smaller here than in-house, not larger. |
| Deployment Readiness | 5 | Fitted artifact saved (`artifacts_research_v2/models/isolation_forest.pkl`), a native `contamination` parameter that yields a threshold without a separate calibration step, an exact and fast explainer, best measured stability in the pipeline, and it is the model Phase 10 (v2) already used to generate reviewable examples. The closest thing in this pipeline to something that could ship this week. |

### 1.2 Percentile Aggregation — 19/30 (weighted 3.40)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | Aggregates 11 of the 12 models (Phase 12 v2 §0 excludes the Hybrid Ensemble to avoid double-counting IF/LOF/AE), so no single detector's blind spot dominates — the direct remedy for the ρ = −0.3705 SHAP divergence Phase 11 (v2) §1 measured between Isolation Forest and the Autoencoder. Its top-ranked transaction across all 2,512 rows is **`TX000275`** (score 0.9951, rank 1 of 2,512, verified against `artifacts_research_v2/ensemble_scores_v2.csv`) — the same transaction the in-house 46-feature pipeline independently surfaced in its own top-1% tier, and the clearest Phase 1 Scenario 1 match either pipeline produced. Held at 4, not above, because the external cross-check that supported the in-house ensemble's score is worthless here (ceiling rule 3) and because its own internal-validity partition was never measured — Phase 10 (v2) §1 scored the 12 models, not the 4 strategies. |
| Stability | 4 | **(inferred)** — no bootstrap study was run for any ensemble strategy. Averaging 11 percentile ranks should damp the per-model resample churn that Phase 10 (v2) §2 measured at 0.373–0.602, and doing so is standard practice, but *this pipeline did not test it*. This inferred 4 is the single largest load-bearing assumption in the whole table, exactly as it was in-house; §5 records it as the first measurement to add. |
| Interpretability | 4 | Every input is a per-model empirical percentile in (0,1), so "this transaction sits at the 96th percentile on 9 of 11 detectors" is a sentence an investigator can act on without a statistics background, and the contribution of each model is directly inspectable. Not a 5 because the *features* behind the score are only explainable through the two models that have SHAP layers (Isolation Forest, Autoencoder), not through the ensemble itself. |
| Computational Cost | 1 | Requires all 11 member models to be fit, maintained, versioned and scored on every batch. The measured member costs are individually small (0.6s–44.3s each), but the lifecycle burden of eleven artifacts is the dominant cost, and it is a recurring one. |
| Scalability | 2 | Bounded by its weakest member. Phase 12 (v2) §0 records the constraint explicitly: **DBSCAN and HDBSCAN have no out-of-sample `.predict()`** in this build, so the published 11-model score can only be produced in batch by refitting both over the full history. Percentile aggregation's own skip-missing rule (§1.3) is what makes a reduced-member version possible at all — see §3. |
| Deployment Readiness | 4 | `artifacts_research_v2/ensemble_scores_v2.csv` exists and covers all 2,512 rows with no missing values; Phase 13 (v2) has already thresholded it (0.9510 / 0.8671); it is bounded in (0,1) so a frozen reference distribution is straightforward to monitor. Docked one point for the member-set constraint above, which must be resolved before real-time scoring. |

### 1.3 Autoencoder — 21/30 (weighted 3.40)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | **Last of 12 on internal validity** — Silhouette 0.172, Davies-Bouldin 5.681, Calinski-Harabasz 15.73 (Phase 10 v2 §1), a much sharper bottom-placement than the in-house Autoencoder's 0.496. Phase 10 (v2) explains the mechanism (a top-5% cut by squared reconstruction error is not optimised to separate well in Euclidean distance, and an 18→3 bottleneck leaves less geometric residual than the in-house 46→4 did), so this is not purely a demerit — but it is not evidence of quality either. Earns a 3, not lower, on two real strengths: the best reconstruction fidelity of the three deep models (val MSE 0.2966 vs. VAE 0.3458 vs. LSTM-AE 0.7907) and a genuinely different feature basis from Isolation Forest, which is what makes it useful in an ensemble. |
| Stability | 2 | **Measured, and the worst of the three tested by a wide margin**: mean bootstrap Jaccard **0.3726**, min **0.2115** (Phase 10 v2 §2) — in the worst observed retrain pair, fewer than one in four flagged transactions was shared. This is a straight reversal of the in-house result (where the Autoencoder measured 0.533 and sat mid-pack), and Phase 10 (v2) gives a plausible mechanism: with only 18 largely population-level features and a 3-dimensional bottleneck there is little redundant structure to anchor the learned "normal" on. Scored 2 on measured evidence, not inferred. |
| Interpretability | 4 | A working `shap.GradientExplainer` path over the `AEErrorWrapper` module whose forward pass returns the scalar per-row reconstruction MSE, run over all 2,512 rows (Phase 11 v2 §0), with additivity spot-checked (ρ=0.9879 over 10 rows) rather than assumed. Not a 5 because expected-gradients is an approximation, not an exact decomposition, and because Phase 11 (v2) §1 found its attributions are dominated by frequency-encoded features (`Location_FE` 0.0354, `account_frequency` 0.0304, `merchant_frequency` 0.0302) — so on this feature set it must be read as a "is this transaction's popularity profile unusual" detector, **not** the "is this amount unusual for this account" detector the in-house Autoencoder was. |
| Computational Cost | 3 | Training is cheap (200 epochs on 2,009 rows × 18 features). The explainer is not: **112.3s for all 2,512 rows**, 13.5× the Isolation Forest TreeExplainer's 8.3s, and that cost recurs on every refresh of the explanation layer. |
| Scalability | 5 | A forward pass through a five-layer network with a 3-unit bottleneck; scoring is O(1) per row and trivially batchable. Artifacts (`autoencoder.pt`, `autoencoder_scaler.pkl`, `autoencoder_config.json`) reload through `src_research_v2/autoencoder_utils.py::load_autoencoder()`. |
| Deployment Readiness | 4 | All artifacts saved and reloadable, a working explainer, out-of-sample scoring — but docked for the measured stability figure above, which is the weakest such number in the pipeline and would need a monitored, versioned retrain process (Phase 16 v2 §4) before this model could be trusted to drive an investigator queue on its own. |

### 1.4 Local Outlier Factor — 19/30 (weighted 3.15)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Silhouette 0.277, **10th of 12** (Phase 10 v2 §1) — a real drop from the in-house pipeline, where LOF sat 5th at 0.617. It does sit in the well-agreeing core of the field (mean Spearman 0.646/0.676, 4th-highest; strongest single pairings VAE ρ=0.839 and K-Means ρ=0.838, Phase 8 v2 §3.2), which is what keeps it at 3 rather than 2. Never put through a business-evaluation walkthrough, so its flags have not been read by a human. |
| Stability | 4 | **Measured**: mean bootstrap Jaccard 0.5124 (min 0.465, max 0.575) — 2nd of the three tested. Note this is another reversal from in-house, where LOF was the *most* stable of the three at 0.590; here Isolation Forest clearly beats it. Scored 4 for being measured and mid-range, not 5, since ~49% flagged-set churn per retrain is still substantial. |
| Interpretability | 2 | No native or exact SHAP path was built in Phase 11 (v2) — only Isolation Forest and the Autoencoder were explained. `novelty=True` was set at fit time, so a model-agnostic `KernelExplainer` route is possible in principle, but it was not run. Scored on what exists, not what could be built. |
| Computational Cost | 4 | 4.00s for 5 configs at n=2,512 (Phase 8 v2 §1.2) — cheap here. Docked one point because the default neighbour search is O(n²) and that cost is already inside the fit, not merely a future concern. |
| Scalability | 3 | Has out-of-sample scoring (`novelty=True`), which puts it well ahead of DBSCAN/HDBSCAN, but an approximate-nearest-neighbour index would be needed well before six figures of rows — a real engineering task, not a config change. |
| Deployment Readiness | 3 | Artifact saved (`models/lof.pkl`), measured stability, out-of-sample capable — but no explainability path and a known indexing rework ahead of it. Deployable as an ensemble member, not as the single model behind an investigator-facing alert. |

### 1.5 K-Means — 20/30 (weighted 3.15)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Silhouette 0.400, 5th of 12, and the **highest mean cross-model agreement in the field** (mean Spearman 0.670/0.698 — Phase 8 v2 §3.2), which is why Phase 12 (v2) §1.1 gives it the largest consensus weight (0.109). Held at 3 because the structure it actually finds is demographic, not fraud-relevant: at k=2 the strong silhouette of 0.491 comes from a 1,830/179 split that is 77.7% `CustomerOccupation_Student` in the minority cluster — the same student segment Phase 7 (v2) §7.2/7.3 identified in UMAP *and* t-SNE and explicitly warned should not be mistaken for an anomaly population. |
| Stability | 2 | **(inferred)** — not bootstrap-tested. The inference is well-grounded rather than speculative: Phase 8 (v2) §1.7 found **no clear inertia elbow anywhere in k=2–10**, so the operating k is genuinely unresolved (the elbow rule mechanically returns the search boundary, k=10), and micro-clusters holding <1% of training rows appear from k=6 onward. A model whose principal hyperparameter is not settled cannot be assumed to produce a stable flagged set across retrains. |
| Interpretability | 3 | Distance to nearest centroid decomposes naturally per feature (which centroid coordinates the row is far from), so a defensible attribution exists without any explainer library. No SHAP layer was built, and the centroids themselves are only interpretable once the student-cluster caveat above is attached. |
| Computational Cost | 5 | Among the cheapest in the set; the full k=2–10 sweep with silhouette on a 1,000-row subsample is seconds of work at this scale. |
| Scalability | 4 | Native `predict` — assigning a new row costs O(k·p), 10×18 multiply-adds here. Docked one point because refitting for a settled k is still outstanding. |
| Deployment Readiness | 3 | Artifact saved and out-of-sample capable, but shipping a detector whose k was chosen by a rule its own author flagged as mechanically forced (Phase 8 v2 §1.7) is not defensible without resolving that first. |

### 1.6 Hybrid Ensemble (IF + LOF + AE, ≥2-of-3 majority vote) — 19/30 (weighted 3.15)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Requires two of three structurally different detectors to agree, which is the right instinct given Phase 11 (v2)'s ρ = −0.3705 divergence — its native flag is 83/2,512 rows (3.30%), the most conservative in the pipeline. Its Phase 10 (v2) §1 Silhouette of 0.277 (9th of 12) is **not** evaluated on that 83-row set: it is evaluated on a 269-row partition, which is the ≥1-vote set, not the ≥2-of-3 majority-vote set (verified directly against `model_scores_all.csv`: vote counts 0→2,243, 1→186, 2→64, 3→19). See Inconsistency 2 in §5. The internal-validity number is therefore measured on a partition 3.2× larger than the ensemble's real operating flag and should not be read as a verdict on the flag itself. |
| Stability | 3 | **(inferred)** — the arithmetic mean of its three components' *measured* stability is (0.602 + 0.512 + 0.373)/3 = 0.496, and a ≥2-of-3 vote should be somewhat more stable than that average since a flag survives one component flipping. Inferred one band above its weakest member, not above its best. |
| Interpretability | 3 | Two of its three components have working SHAP layers, and "which detectors voted" is itself a legible explanation — the same detector-chip idea the Bank Transaction Fraud & Anomaly Detection dashboard already renders. Capped at 3 because the third component (LOF) has no attribution path, so an explanation is always partial. |
| Computational Cost | 3 | Three models to fit, version and score, one of which carries the 112.3s explainer. |
| Scalability | 4 | All three components score out-of-sample natively — a genuine advantage over any strategy that includes DBSCAN or HDBSCAN. |
| Deployment Readiness | 3 | Buildable today from three saved artifacts, but its 4-valued score is too coarse to carve percentile tiers (Phase 10 v2 §4 gives exactly this reason for not using it in the business walkthrough), so it can gate but not rank. |

### 1.7 Weighted Average (consensus-weighted z-scores) — 17/30 (weighted 3.10)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | Same 11-model input set as Percentile Aggregation and a near-identical ordering to the PCA stacking proxy (ρ=0.9990, Jaccard 0.924 — Phase 12 v2 §2). Its one genuine advantage over the recommended score is that it is **unbounded**, so Phase 13 (v2) §3's sigma/IQR thresholds actually work on it (mean+3σ = 2.1937 → 29 flagged; Q3+1.5×IQR = 1.5462 → 79 flagged) where they flag zero on the percentile score. |
| Stability | 4 | **(inferred)**, same basis as Percentile Aggregation, with one extra caution: its weights are derived from a Spearman matrix that will itself shift as data accumulates, so the score definition is not fixed across retrains the way a percentile average's is. |
| Interpretability | 3 | The weights are principled (inverse mean disagreement) and the derivation is documented in `ensemble_weights_v2.json`, but they are an additional layer of judgement to defend to a reviewer — and one that a reader cannot verify without the pairwise matrix in front of them. |
| Computational Cost | 1 | Identical to Percentile Aggregation: eleven models, plus a pairwise-correlation pass to derive the weights. |
| Scalability | 2 | Same DBSCAN/HDBSCAN out-of-sample constraint, plus the weights must be recomputed whenever the member set changes. |
| Deployment Readiness | 3 | Computed and saved (`ensemble_scores_v2.csv`), but shipping it means shipping and defending a weight vector; the unbounded-score advantage is real but narrow. |

### 1.8 Rank Aggregation (Borda count) — 16/30 (weighted 2.95)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | Phase 12 (v2) §2 measured it as **near-mathematically identical** to Percentile Aggregation (ρ=0.9999, Jaccard 0.969) — as expected, since summing raw ranks and averaging rank/N differ only by a per-model normalisation constant. It detects what Percentile Aggregation detects. |
| Stability | 4 | **(inferred)**, same basis. |
| Interpretability | 3 | A Borda sum (here ranging into the thousands) has no intrinsic scale a reviewer can interpret without being told the row count. |
| Computational Cost | 1 | Same eleven-model burden. |
| Scalability | 2 | Same out-of-sample constraint, plus a specific operational defect: **a Borda sum's scale depends on how many rows are ranked together**, so it is not comparable across production batches of differing size, where a percentile is. |
| Deployment Readiness | 2 | Computed and saved, but strictly dominated by Percentile Aggregation — same signal, worse operational properties. Its only distinguishing choice (substituting a median rank for LSTM-AE's 110 missing rows) is less conservative than percentile aggregation's skip-and-renormalise. |

### 1.9 Elliptic Envelope — 18/30 (weighted 2.90)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | **The internal-validity leader by a wide margin** — Silhouette 0.5409, Davies-Bouldin 1.1529, Calinski-Harabasz 592.47, the last more than 3× the next-best model's 180.07 (Phase 10 v2 §1). This is a real, measured result and a genuine reversal from in-house, where it sat 7th. It is scored 3 rather than 4 for two documented reasons, both of which cut against reading that lead as detection quality: (a) a top-5% cut by MCD Mahalanobis distance is close to *mechanically guaranteed* to look well-separated under distance-based validity indices computed in the same scaled space — the same construction-artifact caveat the in-house Phase 10 attached to One-Class SVM; and (b) it has the **lowest mean Jaccard overlap of any model** (0.170/0.239, Phase 8 v2 §3.3) despite mid-pack rank correlation, meaning the specific 126 transactions it flags are almost nobody else's — and with no label, cross-model agreement is the only corroboration available. Also never put through a business walkthrough. |
| Stability | 2 | **(inferred)** — not bootstrap-tested. MCD selects a support subset of the data and estimates covariance from it; that subset is precisely the population a bootstrap resample perturbs most. No reason to expect it beats the measured 0.373–0.602 band; scored at its lower edge. |
| Interpretability | 2 | No explainer built in Phase 11 (v2). A Mahalanobis distance does decompose into per-feature standardised contributions in principle, but nothing in this codebase does so, and its underlying assumption is violated (below). |
| Computational Cost | 4 | 3.419s for 3 configs at n=2,512 (Phase 8 v2 §1.4). Cheap at this scale; MCD's subset search grows worse than linearly. |
| Scalability | 4 | Native `decision_function`, fit on train and scored out-of-sample like the other classical models. |
| Deployment Readiness | 3 | Artifact saved and out-of-sample capable, but Phase 8 (v2) §1.4 measured **Shapiro-Wilk rejection of normality on 100% of the 18 features (every p-value rounds to 0.0000)** — the multivariate-Gaussian assumption its Mahalanobis distance rests on is not merely approximate here, it is violated on every single input column. Phase 8 (v2) itself declines to recommend it as a primary detector for this reason, and that judgement is upheld here rather than overturned by its internal-validity lead. |

### 1.10 Variational Autoencoder — 17/30 (weighted 2.70)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 2 | Silhouette 0.201, 11th of 12 (Phase 10 v2 §1), and it reconstructs **worse than the plain Autoencoder at every percentile including the maximum** (val MSE 0.3458 vs. 0.2966; val max 1.2137 vs. 0.9633 — Phase 8 v2 §2.10). On the in-house feature set the VAE at least bought tail-smoothing for that loss of fidelity; here it buys nothing measurable. It also correlates ρ=0.837 with the Autoencoder, so it is largely redundant with a model already in the set. |
| Stability | 2 | **(inferred)** from the Autoencoder's measured 0.3726 — same architecture family, same split, same 18-feature input, plus a stochastic sampling step in the forward pass that can only add variance. |
| Interpretability | 2 | No explainer built. The `AEErrorWrapper` pattern would extend to it, but that work was not done. |
| Computational Cost | 3 | 44.3s to train 200 epochs (Phase 8 v2 §2.10), and Phase 9 (v2) §3's 20-trial search took 192.2s — the slowest of the three searches run. |
| Scalability | 5 | A forward pass, same as the Autoencoder. |
| Deployment Readiness | 3 | Artifact saved (`models/vae.pt`) and reloadable, but Phase 9 (v2) §3 found the deployed `beta=0.1` is materially worse than the search's best `beta=0.0113`, so the shipped artifact is known to be sub-optimally configured and would want a refit before use. |

### 1.11 One-Class SVM — 16/30 (weighted 2.55)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Silhouette 0.406, 4th of 12, and 3rd-highest mean Jaccard (0.306/0.364) — it sits comfortably inside the consensus core. Same construction caveat as Elliptic Envelope: a top-5% cut by an RBF decision function is partly guaranteed to separate well in a distance metric. Never business-reviewed. |
| Stability | 2 | **(inferred)** — not bootstrap-tested. Its boundary is defined by the support vectors nearest the margin, exactly the population a resample perturbs most. Scored one band below the measured trio. |
| Interpretability | 2 | An RBF kernel decision value has no natural per-feature decomposition, and no model-agnostic explainer was built. |
| Computational Cost | 5 | The **fastest measured fit in the pipeline**: 0.727s for all 5 configs at n=2,512 (Phase 8 v2 §1.3). Judged on measured cost at this scale it earns a 5; its growth problem is scored under Scalability rather than double-counted here. |
| Scalability | 2 | A QP solve, roughly O(n²)–O(n³) in the number of support vectors — a real scalability concern past ~50k–100k rows without subsampling or an approximate variant. It can score out-of-sample, which keeps it off the floor. |
| Deployment Readiness | 2 | Artifact saved and out-of-sample capable, but no explanation path and a known scaling wall. |

### 1.12 Gaussian Mixture Model — 16/30 (weighted 2.50)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 2 | Silhouette 0.293, 8th of 12, and the **second-lowest cross-model agreement** in the field (mean Spearman 0.449/0.495 — Phase 8 v2 §3.2). More fundamentally, its principal hyperparameter is unresolved: the winning `diag` BIC curve is **still decreasing at n_components=10**, the search boundary, in both Phase 8 (v2) §1.8's grid and Phase 9 (v2) §2's Optuna search, so the selected 10 components is a boundary artifact rather than an optimum. |
| Stability | 2 | **(inferred)** — EM is sensitive to initialisation and to which rows are present; with the component count itself unsettled, two refits could plausibly land on materially different mixtures. |
| Interpretability | 2 | Negative log-likelihood under a 10-component diagonal mixture has no per-feature story a reviewer can use, and no explainer was built. |
| Computational Cost | 4 | The full 10×4 BIC grid is seconds of work at this scale; Phase 9 (v2)'s 40-trial Optuna search was likewise cheap. |
| Scalability | 4 | Native `score_samples` on new rows; a diagonal-covariance mixture is inexpensive to evaluate — cheaper here than the in-house pipeline's `full`-covariance winner. |
| Deployment Readiness | 2 | Artifact saved, but shipping a detector whose component count is a search-boundary artifact is not defensible. Phase 9 (v2) §2 is a genuine *improvement* on the in-house finding (the search did not push toward an over-parameterised covariance structure here — `diag` stayed `diag`, so the in-house overfitting warning does not replicate), which is why this scores 2 rather than the in-house pipeline's 1.95-weighted floor territory, but "less alarming" is not "ready". |

### 1.13 PCA Stacking Proxy — 13/30 (weighted 2.35)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Nearly indistinguishable from Weighted Average (ρ=0.9990, Jaccard 0.924 — Phase 12 v2 §2), so it detects essentially what that strategy detects. **PC1 explains only 54.90%** of the variance across the 11 standardised score columns, so ~45% of the cross-model signal is discarded into components nobody downstream can interrogate. |
| Stability | 3 | **(inferred)**. A PCA rotation refit on new data can change both loadings and the sign of PC1, which must be manually re-oriented on every refresh — a stability hazard the other three strategies do not have. |
| Interpretability | 2 | A "consensus axis" is hard to explain to a non-technical reviewer, and the sign convention is an implementation detail that would confuse one. |
| Computational Cost | 1 | Same eleven-model burden, plus a PCA fit. |
| Scalability | 2 | Same out-of-sample constraint. |
| Deployment Readiness | 2 | Computed and saved, but Phase 12 (v2) §3 explicitly declines to recommend it as the primary production score, and nothing in this phase overturns that. |

### 1.14 HDBSCAN — 12/30 (weighted 1.90)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Silhouette 0.430, 2nd of 12, and the **highest mean Jaccard overlap in the field** (0.318/0.375) — it agrees with more of the field on *which rows* to flag than any other model. Genuinely more usable on this feature set than in-house, where its best config never got below 53.94% noise; here the best config reaches 8.88%. |
| Stability | 1 | The evidence against is direct, not inferred: of the 4 configs tried, one reaches 8.88% noise and **the other three sit at 86.1%–90.4%** (Phase 8 v2 §1.6). That is a cliff, not a gradient — the single usable configuration is surrounded by unusable ones, which is the clearest possible signal that small changes to the data or the hyperparameters will move the flagged set drastically. |
| Interpretability | 1 | A GLOSH outlier score has no per-feature decomposition and no explainer was built. |
| Computational Cost | 5 | 1.311s for all 4 configs at n=2,512 (Phase 8 v2 §1.6). |
| Scalability | 1 | **Cannot score a transaction it was not fit on.** `prediction_data=True` was not set at fit time, so even `approximate_predict` is unavailable without a refit (Phase 12 v2 §0). |
| Deployment Readiness | 1 | Its highest-agreement result is real, but a model that cannot score new rows and whose usable hyperparameter region is a single point on a cliff cannot ship as anything other than a batch-only research signal. |

### 1.15 DBSCAN — 11/30 (weighted 1.65)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 2 | Silhouette 0.428, 3rd of 12 — but this is the model the entire cross-model comparison singles out: **the lowest mean Spearman agreement by a wide margin** (0.235/0.299, next-lowest 0.423), the lowest ensemble weight (0.051), and every one of the six weakest pairs in the whole Spearman matrix involves it (lowest of all: DBSCAN ↔ Autoencoder, ρ=0.153). Phase 8 (v2) §3.2 calls this "the single clearest, most reproducible cross-model finding" — and it reproduced independently on the in-house 46-feature set too, which makes it a property of DBSCAN's behaviour on this raw data rather than an artifact of either feature-engineering choice. Without a label, being the model nobody else agrees with is a demerit, not a differentiator. |
| Stability | 1 | Direct evidence: across the 3×3 `eps`/`min_samples` grid the noise rate ranges from **0.4% to 11.4%** and the cluster count from 1 to 3 (Phase 8 v2 §1.5). The operating point is one choice on a steep surface. |
| Interpretability | 1 | Distance-to-nearest-core-point has no per-feature story; no explainer built. |
| Computational Cost | 5 | 0.637s for all 9 grid configs at n=2,512 — the cheapest grid in the pipeline. |
| Scalability | 1 | **No out-of-sample `.predict()`** (Phase 12 v2 §0). Its distance-to-nearest-core-point pseudo-score does technically generalise to a new point without refitting the clustering, which is the only reason this is not a hard zero. |
| Deployment Readiness | 1 | Batch-only, hyperparameter-fragile, unexplainable, and the least-agreeing model in the set. |

### 1.16 LSTM Autoencoder — 9/30 (weighted 1.50)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 2 | Silhouette 0.375, 7th of 12, but computed on only 2,402 of 2,512 rows. The decisive evidence is the training curve: Phase 8 (v2) §2.11 reports that **validation MSE bottoms out near 0.49 around epoch 50 and then climbs steadily to 0.79 by epoch 150** while training MSE keeps falling — a textbook overfitting signature, reported honestly rather than early-stopped away. Its final val MSE (0.7907) is 2.7× the plain Autoencoder's. |
| Stability | 1 | **(inferred)**, on strong grounds: 342 training *sequences* (not rows), a median of 5 transactions per account, and a demonstrated tendency to overfit within 150 epochs. |
| Interpretability | 1 | No explainer; a masked sequence reconstruction error has no per-feature story. |
| Computational Cost | 2 | 31.7s to train, plus per-account sequence assembly, padding and masking — the most involved data-preparation step in the pipeline for the least return. |
| Scalability | 2 | Structurally cannot score 110/2,512 rows (4.4%) at all — accounts with fewer than 3 transactions have no sequence to reconstruct. In production this is worse, not better: new customers are exactly the population with short histories. |
| Deployment Readiness | 1 | Phase 8 (v2) §2.11's own stated conclusion is that "the LSTM-AE on this 18-feature set is even less justified than the in-house one was". Nothing here disputes that. |

---

## 2. Consolidated Decision Matrix

Ordered by weighted total. **DQ** = Detection Quality, **St** = Stability, **In** = Interpretability, **Co** = Computational Cost, **Sc** = Scalability, **De** = Deployment Readiness.

| # | Candidate | DQ (.25) | St (.20) | In (.15) | Co (.10) | Sc (.15) | De (.15) | Raw /30 | **Weighted** |
|---:|---|:--:|:--:|:--:|:--:|:--:|:--:|---:|---:|
| 1 | **Isolation Forest** | 4 | 4 | 4 | 5 | 5 | 5 | 27 | **4.40** |
| 2= | **Percentile Aggregation** | 4 | 4 | 4 | 1 | 2 | 4 | 19 | **3.40** |
| 2= | **Autoencoder** | 3 | 2 | 4 | 3 | 5 | 4 | 21 | **3.40** |
| 4= | K-Means | 3 | 2 | 3 | 5 | 4 | 3 | 20 | 3.15 |
| 4= | Local Outlier Factor | 3 | 4 | 2 | 4 | 3 | 3 | 19 | 3.15 |
| 4= | Hybrid Ensemble (IF+LOF+AE) | 3 | 3 | 3 | 3 | 4 | 3 | 19 | 3.15 |
| 7 | Weighted Average | 4 | 4 | 3 | 1 | 2 | 3 | 17 | 3.10 |
| 8 | Rank Aggregation (Borda) | 4 | 4 | 3 | 1 | 2 | 2 | 16 | 2.95 |
| 9 | Elliptic Envelope | 3 | 2 | 2 | 4 | 4 | 3 | 18 | 2.90 |
| 10 | Variational Autoencoder | 2 | 2 | 2 | 3 | 5 | 3 | 17 | 2.70 |
| 11 | One-Class SVM | 3 | 2 | 2 | 5 | 2 | 2 | 16 | 2.55 |
| 12 | Gaussian Mixture Model | 2 | 2 | 2 | 4 | 4 | 2 | 16 | 2.50 |
| 13 | PCA Stacking Proxy | 3 | 3 | 2 | 1 | 2 | 2 | 13 | 2.35 |
| 14 | HDBSCAN | 3 | 1 | 1 | 5 | 1 | 1 | 12 | 1.90 |
| 15 | DBSCAN | 2 | 1 | 1 | 5 | 1 | 1 | 11 | 1.65 |
| 16 | LSTM Autoencoder | 2 | 1 | 1 | 2 | 2 | 1 | 9 | 1.50 |

### 2.1 Where this matrix differs from the in-house pipeline's, and why

The rubric and weights are identical, so every difference below is a difference in *evidence*, not in method. Four are material:

| Candidate | In-house weighted | v2 weighted | What moved it |
|---|---:|---:|---|
| **Isolation Forest** | 4.20 | **4.40** | Stability 3 → 4. In-house it measured 0.527 inside a 0.527–0.590 spread too narrow to separate the three tested models; here it measures **0.6021** and is clearly separated from LOF (0.512) and the Autoencoder (0.373). |
| **Autoencoder** | 4.20 (tied 1st) | **3.40** (tied 2nd) | Detection Quality 4 → 3 and Stability 3 → 2. It falls from mid-pack to **last of 12** on internal validity (0.496 → 0.172) and from mid-pack to **worst measured** on retrain stability (0.533 → 0.373). Both are measured, not inferred. |
| **Local Outlier Factor** | 3.20 | **3.15** | Almost unchanged in total, but the composition inverted: it *gains* on Stability (3 → 4, now measured 2nd-best rather than best) and *loses* on Detection Quality (4 → 3, internal validity 0.617/5th → 0.277/10th). |
| **Elliptic Envelope** | 2.35 | **2.90** | Detection Quality 2 → 3 and Deployment Readiness 2 → 3, on the back of a genuine reversal: it goes from 7th to **1st** on internal validity (0.610 → 0.541 in absolute terms, but from mid-pack to a clear lead over the field). Held below the leaders by the violated Gaussian assumption and its field-lowest flagged-set overlap. |

**Three things this matrix does not capture, stated so nobody over-reads it:**

1. **It cannot see redundancy.** The VAE ranks 10th here, which is closer to its real merit than the in-house matrix's 4th, but the principle still bites: the matrix has no column for "correlates ρ=0.837 with a model already in the set" (Phase 8 v2 §3.2). Read on its own, the VAE on this feature set is a strictly worse Autoencoder — worse at every reconstruction percentile including the max — and it should be discounted below its rank.
2. **Stability is measured for 3 of 16 candidates.** Only Isolation Forest (0.6021), LOF (0.5124) and the Autoencoder (0.3726) have bootstrap Jaccard numbers (Phase 10 v2 §2). Every other Stability score is inferred, including all four ensemble strategies — and the ensembles' inferred 4s are the largest single load-bearing assumption in this table. §5 records this as the first measurement that should be added.
3. **Detection Quality is capped at 4, and bounded further by the feature set itself.** No candidate has been shown to catch fraud, because there is no fraud to check against; and per ceiling rule 2, everything scored here detects population-level unusualness, not personal-baseline deviation. The matrix ranks *engineering and evidential quality within a known capability envelope*, not detection performance.

---

## 3. The Constraint That Cuts Across Every Ensemble Strategy

All four ensemble strategies take the same 11 models as inputs (Phase 12 v2 §0). **Two of those 11 — DBSCAN and HDBSCAN — cannot score a transaction they were not fit on.** Phase 8 (v2) §0 records the consequence directly: while models 1–4 and 7–10 were fit on the 2,009-row training split and scored all 2,512 rows out-of-sample, DBSCAN and HDBSCAN were fit directly on the full dataset because neither has a native out-of-sample `.predict`. HDBSCAN's library does offer `approximate_predict`, but only when `prediction_data=True` is set at fit time, which this build does not do.

Unlike the in-house pipeline, this constraint was **identified during Phase 12 rather than first surfacing at Phase 14** (`research_v2/10_ensemble_scoring.md` §0 flags it explicitly and pre-emptively), so it is not a new finding here. What this phase adds is the decision.

At 2,512 rows this is invisible — refitting everything on the full dataset takes seconds. In production it is structural: **any ensemble score built over all 11 models can only be produced in batch, by refitting the two clustering models over the full transaction history on every run.** That decides the architecture (Phase 15 v2 §4).

| Option | What it costs | What it buys |
|---|---|---|
| **A. Keep all 11, run batch-only** | No real-time scoring at all; refit cost grows with total history, not batch size | Uses the exact score Phase 12 (v2) recommended and Phase 13 (v2) thresholded, with no revalidation needed |
| **B. Drop DBSCAN and HDBSCAN, aggregate over 9** | The resulting score is **not the one Phase 12 (v2) computed**, so Phase 13 (v2)'s thresholds (0.9510 / 0.8671) would not transfer unrevalidated. Loses HDBSCAN, which has the *highest* mean flagged-set agreement in the field (0.318/0.375) — a more meaningful loss here than in-house | Restores incremental scoring; percentile aggregation supports this natively, since it skips missing models and renormalises over those available (Phase 12 v2 §1.3) |
| **C. Enable `prediction_data=True` for HDBSCAN, drop DBSCAN only** | A refit and a revalidation; `approximate_predict` is an approximation, so the score changes | Retains the highest-agreement member; loses only DBSCAN, the model with the weakest evidence in the whole set (§1.15) |

**Option B is the recommended path, with Option C as a strong follow-up.** The v2-specific argument for prioritising Option C higher than the in-house pipeline did: on the in-house feature set HDBSCAN was barely usable (best config 53.94% noise) and dropping it cost little; here its best config reaches 8.88% noise and it has the highest mean Jaccard of any model, so it is a more valuable member to retain. Option A is acceptable only for the research prototype at its current scale.

Whichever is chosen, the 9- or 10-model score must be compared against the published 11-model `ensemble_percentile_average` (Spearman and top-5% Jaccard, the same two measures Phase 12 v2 §2 used) before any threshold is carried over. **That comparison has not been run, and no number in this report should be read as if it had been.**

---

## 4. Final Recommendation

**Production score: Percentile Aggregation, computed over the subset of models that can score out-of-sample (Option B above), with `ensemble_percentile_average` from `artifacts_research_v2/ensemble_scores_v2.csv` as the reference implementation and batch/backfill score.**

**Explanation layer, non-optional: Isolation Forest SHAP and Autoencoder SHAP, both shown, side by side** (`artifacts_research_v2/shap_isolation_forest_v2.csv`, `shap_autoencoder_v2.csv`).

**Operating points: Phase 13 (v2)'s two tiers, adopted unchanged.**

| Tier | Threshold | Score cut | Volume in this sample |
|---|---|---:|---|
| **Priority review** | 99th percentile | 0.9510 | 26 transactions (1.04%) |
| **Standard review** | 95th percentile | 0.8671 | 126 transactions (5.02%) |

**No automatic block tier is recommended**, for the same reason as in-house: Phase 13 (v2) §1 established that a cost-minimising sweep cannot be reproduced without a label, because counting false negatives requires knowing which *unflagged* transactions are fraud. Blocking a customer's transaction on a score whose false-negative behaviour has never been measured is not defensible. Everything this system produces should land in a human review queue until it has been validated against real investigator-labelled outcomes.

### Why not Isolation Forest alone, when the matrix ranks it first by a full point

The matrix ranks Isolation Forest at 4.40, a full point clear of anything else — a wider margin than in-house, where it was tied at 4.20. **The recommendation still deliberately overrides the matrix, and the override is stated rather than engineered into the scores.**

The reason is Phase 11 (v2), which is the strongest single piece of evidence in this pipeline:

- Isolation Forest and the Autoencoder share **3 of 10** top global features, and their full 18-feature mean|SHAP| importance vectors correlate at **ρ = −0.3705** — a *more sharply negative* disagreement than the in-house pipeline's ρ = −0.157, on a feature set less than half the size.
- The mechanism is understood, not mysterious, and it **replicated independently across both pipelines**: Isolation Forest isolates points with random splits, so low-cardinality binary features are cheap split points and dominate its attributions (`TransactionType_Debit` is the #1 feature and `CustomerOccupation_Retired` the #2 in *both* the in-house and this pipeline — an independent replication of the same mechanism on two entirely different feature sets). The Autoencoder's score is squared reconstruction error through a compressed bottleneck, dominated here by the frequency-encoded features (`Location_FE`, `account_frequency`, `merchant_frequency`).
- There are two worked counterexamples in this pipeline, not one. `TX000615` shows the two models disagreeing on *which aspect* of the same transaction is anomalous (Isolation Forest reads it as an amount anomaly, the Autoencoder as a `Location_FE` anomaly). `TX001029` shows the more important case: both models agree its high score comes from an extreme `merchant_frequency` value (z=3.67) rather than anything fraud-relevant, and Phase 10 (v2) independently judged it not a plausible fraud pattern — a $516.47 transaction at 0.40× its account's balance with normal login behaviour. Cross-checking against a second, structurally different model is what catches that before it reaches a human.

So the choice is not "one good model versus a more expensive ensemble." In a system with no label, cross-model consistency is the only validation available, and deliberately discarding it to save compute is the wrong trade. At 6.88 transactions/day (Phase 13 v2 §4) the compute saved is negligible.

### Where this recommendation genuinely differs from the in-house pipeline's

The in-house Phase 14 named a two-model fallback — percentile aggregation over Isolation Forest and the Autoencoder alone — as its defensible minimal system, on the basis that both were jointly ranked first, both had SHAP layers, and both had measured stability. **That fallback is less balanced on this feature set, and the difference should be stated rather than copied across.**

Here, Isolation Forest and the Autoencoder are not peers: Isolation Forest is the most stable model measured (0.6021) while the Autoencoder is the *least* (0.3726, min 0.2115), and the Autoencoder is last of 12 on internal validity. A two-model score pairing the pipeline's most stable detector with its least stable one inherits the weaker member's churn.

**Recommended fallback for this pipeline, if fewer artifacts must be operated:**

1. **First fallback (3 models): percentile aggregation over Isolation Forest + LOF + Autoencoder.** This is the Hybrid Ensemble's member set, and it is the only three-model subset in which *all three* members have measured bootstrap stability (0.602 / 0.512 / 0.373) and all three score out-of-sample natively. Using percentile aggregation over their continuous scores rather than the Hybrid Ensemble's ≥2-of-3 vote also fixes that model's real defect — a 4-valued score cannot carve percentile tiers (§1.6).
2. **Last resort (1 model): Isolation Forest alone**, explicitly accepting the loss of the cross-family check, and explicitly accepting the `TX001029`-style failure mode that check exists to catch. This is a materially more defensible single-model choice on this feature set than it was in-house — best stability, exact 8.3s explainer, native out-of-sample scoring, saved artifact — but it is a downgrade in evidential quality, not a simplification with no cost.

**The two-model IF + Autoencoder fallback the in-house pipeline recommended is not recommended here.** Same reasoning, different evidence, different answer.

Like Option B, **neither of these reduced-member scores was computed in Phase 12 (v2)** and each would need its own Spearman/Jaccard comparison against the published 11-model score, and its own thresholds, before use.

### Why Percentile Aggregation over the other three strategies

Phase 12 (v2) §3's reasoning is adopted and reinforced by the deployment dimensions scored here:

- **Over Rank (Borda)**: Phase 12 (v2) showed they are near-identical on a fixed dataset (ρ=0.9999, Jaccard 0.969) and explicitly warned against treating them as materially different signals. The tiebreaker is operational, from §1.8: a Borda sum's scale depends on how many rows are ranked together, so it is not comparable across production batches of differing size, while a percentile is bounded in (0,1) and can be evaluated against a frozen reference distribution.
- **Over Weighted Average**: the weights are defensible but they are an additional judgement — a vector derived from a correlation matrix that will shift with more data. The one place Weighted Average clearly wins is thresholding: it is unbounded, so sigma/IQR rules work on it (Phase 13 v2 §3: mean+3σ = 2.1937 → 29 flagged; Q3+1.5×IQR = 1.5462 → 79 flagged) where they flag **zero** on the percentile score. **Recommendation: keep the Weighted Average as a secondary, computed-in-parallel score for exactly that purpose** — it costs nothing extra once all the member models have run, and it gives stakeholders who want a "three sigma" framing something honest to point at.
- **Over PCA Stacking**: 54.90% of PC1 variance means nearly half the cross-model signal is discarded into a component nobody downstream can interrogate, and the sign orientation needs manual re-verification on every refit. Note also that on *this* feature set, Weighted Average and PCA Stacking converge to ρ=0.9990 (Phase 12 v2 §2) — even tighter than in-house's 0.9947 — so PCA Stacking adds essentially nothing that Weighted Average does not already provide more legibly.

---

## 5. Gaps and Inconsistencies Found While Synthesising

Recorded here rather than smoothed over. The first is a missing measurement; the rest are discrepancies found by checking the `research_v2/` reports' claims against the artifacts they cite.

**Gap — no stability measurement for any ensemble strategy.** Phase 10 (v2) §2 bootstrap-tested three individual models. No equivalent exists for the four Phase 12 (v2) strategies, yet the recommendation above rests substantially on the assumption that aggregation reduces flagged-set churn below the measured 0.373–0.602 range. That assumption is plausible and standard, but it is untested here — and it matters *more* on this feature set than in-house, because the measured floor is lower (0.373 vs. 0.527). **This is the single highest-value measurement to add next**, and it is cheap: re-run the Phase 10 (v2) §2 bootstrap procedure end-to-end through the Phase 12 (v2) aggregation.

**Inconsistency 1 — "mean pairwise" Spearman and Jaccard in Phase 8 (v2) include each model's self-correlation.** Phase 8 (v2) §3.2 states "DBSCAN is lowest by a wide margin (mean ρ=0.299 …), followed by Hybrid Ensemble (0.471) … GMM (0.495). K-Means (0.698) and HDBSCAN (0.693) are the most 'consensus' models", and §3.3 states Elliptic Envelope has "the lowest mean Jaccard overlap of any model (0.239) despite a middling … mean Spearman correlation (0.635)". Recomputed directly from `artifacts_research_v2/model_pairwise_spearman.csv` and `model_pairwise_jaccard.csv`, each of those figures is the mean of a full 12-entry matrix row **including the diagonal value of 1.0**, i.e. `(true_pairwise_mean × 11 + 1) / 12`. The self-excluded pairwise means are:

| Model | Spearman (self-excluded) | Spearman (as published) | Jaccard (self-excluded) | Jaccard (as published) |
|---|---:|---:|---:|---:|
| DBSCAN | 0.235 | 0.299 | 0.218 | 0.283 |
| Hybrid Ensemble | 0.423 | 0.471 | 0.295 | 0.354 |
| GMM | 0.449 | 0.495 | 0.186 | 0.254 |
| Elliptic Envelope | 0.602 | 0.635 | **0.170** | **0.239** |
| HDBSCAN | 0.665 | 0.693 | 0.318 | 0.375 |
| K-Means | **0.670** | **0.698** | 0.310 | 0.367 |

The transform is monotone and identical for every model, so **every ranking and every qualitative conclusion Phase 8 (v2) drew from these figures is unaffected** — DBSCAN is still lowest by a wide margin, K-Means and HDBSCAN are still the consensus leaders, Elliptic Envelope still has the lowest Jaccard of any model. Only the absolute values are inflated. Importantly, the **ensemble weights are not affected**: `artifacts_research_v2/ensemble_weights_v2.json`'s `disagreements` were verified to be computed correctly over the 10 *other* models with self and the Hybrid Ensemble both excluded (recomputed and matched to 6 decimal places for all 11 members). The defect is confined to the narrative tables in Phase 8 (v2) §3.2–3.3 and should be corrected there.

**Inconsistency 2 — the Hybrid Ensemble partition in Phase 10 (v2) is not the majority-vote partition.** Phase 8 (v2) §2.12 defines the Hybrid Ensemble's flag as a **≥2-of-3 majority vote**, giving 83 rows (3.30%). Phase 10 (v2) §1 lists its partition as 269 rows and describes it as "not exactly a top-5% cut — 269/2,512 rows, 10.7%, **per Phase 8 v2 Section 2.12's ≥2-of-3 majority-vote rule**". Checked against `model_scores_all.csv`: the vote distribution is 0→2,243, 1→186, 2→64, 3→19, so ≥2 votes = **83 rows** and ≥1 vote = **269 rows**, and `flag_hybrid_ensemble` sums to exactly 83. Phase 10 (v2)'s 269-row partition is therefore the **≥1-vote** set produced by applying a top-5% cut to a 4-valued score (89.3% of rows score zero, so the cut lands at 1 and ties pull in all 269) — it is *not* the ≥2-of-3 set, and the attribution to §2.12's majority-vote rule is wrong. The row count and the resulting Silhouette of 0.2774 are correct; the description is not. This matters for interpretation: the Hybrid Ensemble's internal-validity number is evaluated on a partition **3.2× larger** than its actual operating flag, which is part of why it scores 9th of 12. **This is the same class of error the in-house Phase 14 found in its own Phase 10 (there: 253 rows vs. 94), reproduced independently in this pipeline** — which suggests the underlying cause is the shared top-5%-cut convention in `10_evaluation.py`'s partition definition applied to a discrete-valued score, not a one-off transcription slip in either report.

**Inconsistency 3 — `model_comparison_summary.json` reports the LSTM-AE rate on a different denominator than the report does.** Phase 8 (v2) §2.11 and §3.1 both state the LSTM-AE's top-5% flagged rate as **5.04%**, correctly qualified as "within the 95.6% applicable rows" / "(of applicable rows)". `artifacts_research_v2/model_comparison_summary.json` records `"lstm_ae": 4.82` with no denominator qualifier alongside eleven other models whose rates *are* over all 2,512 rows. Checked directly: the model flags **121** rows and has non-null scores for **2,402** rows; 121/2,402 = 5.04% and 121/2,512 = 4.82%. **Both numbers are real and the report is the one that is correctly labelled** — the artifact's unqualified key is the problem, and it is the mirror image of the in-house pipeline's Inconsistency 1 (where the report was mislabelled and the artifact was not). Nothing downstream depends on it, since Phase 10 (v2) onward consistently restricts the LSTM-AE to its 2,402 applicable rows. Recommendation: rename the JSON key to `lstm_ae_all_rows` or add the applicable-rows figure beside it.

**Inconsistency 4 — `TransactionAmount` is log-transformed before scaling, and Phase 5 (v2) does not say so.** Phase 5 (v2) §1's feature dictionary describes it as `StandardScaler(TransactionAmount)`. Checked directly against `data/bank_transactions_data_2.csv`: standardising the raw dollar amount reproduces the column only to a **maximum absolute error of 3.57 z-units** — nowhere near a match. Standardising `log1p(TransactionAmount)` reproduces it to **8.9 × 10⁻¹⁶**, i.e. exactly. The correct formula is:

```
TransactionAmount = StandardScaler( log1p(raw TransactionAmount) )
```

This does not change a single modelling result — every phase used the column as it was given — but it changes two things that matter downstream. First, the feature's relationship to dollars is logarithmic, so a SHAP contribution on this column should be read as "unusual on a log-amount scale", not "unusual in dollars". Second, and more practically, it means the feature is **exactly** reproducible from raw inputs, which is a prerequisite for any honest real-time feature layer (Phase 15 v2 §3.2) or scenario simulator (Phase 15 v2 §7.3).

**Inconsistency 5 — `amount_to_balance_ratio` *is* exactly recoverable, contrary to Phase 5 (v2) §2.3.** That section reports r = 0.9467 against the raw `TransactionAmount / AccountBalance` ratio and concludes the exact formula is "not exactly recoverable", speculating about capping or a denominator floor. Both the speculation and the conclusion are wrong. Checked directly:

| Candidate formula | Max abs. error vs. the actual column |
|---|---:|
| `StandardScaler(a / b)` | 6.720 |
| `StandardScaler(log1p(a) − log1p(b))` | 5.328 |
| `StandardScaler(log1p(a / b))` | 0.0168 |
| **`StandardScaler(log1p(a / (b + 1)))`** | **0.0** |

where `a` = raw `TransactionAmount`, `b` = raw `AccountBalance`. The `+1` denominator floor Phase 5 (v2) hypothesised is real, and combined with the same `log1p` transform found in Inconsistency 4 it reproduces the column exactly. Phase 5 (v2)'s substantive conclusion — that this feature measures the same underlying quantity as the in-house `Amount_to_Balance_Ratio`, independently arrived at — is **strengthened**, not weakened, by the exact recovery.

**Consequence of Inconsistencies 4 and 5 taken together: all 18 features are now exactly reproducible from the 16 raw columns.** The remaining sixteen were verified exact while checking these two: `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` (plain `StandardScaler`, max error ≤ 8.9 × 10⁻¹⁶); the four frequency counts (`StandardScaler` of a global `groupby` size, ≤ 1.3 × 10⁻¹⁵); `Location_FE` (`StandardScaler` of the location's proportion — note that scaling the *count* instead gives an identical column, since the two differ by a constant factor of N); `high_amount_transaction` (`raw amount > $878.179`, the exact 95th percentile — reproduces the flag for all 2,512 rows); and the seven one-hots. **This upgrades the deployment position materially**: Phase 15 (v2) §3.2's real-time feature layer is now a fully specified transformation rather than a partly-inferred one, and the frozen constants it must store are exactly enumerable.

**Not an inconsistency, but worth recording: Phase 13 (v2) §2's flagged counts are mechanically identical to the in-house pipeline's (126/76/26/13).** Phase 13 (v2) already states plainly why — both pipelines cut the same fixed percentiles of the same N=2,512 rows — and Phase 12 (v2) §2.1's near-zero cross-check against v1's `vote_count` independently confirms the two scores do not agree on *which* transactions are flagged. Flagged here only because a reader comparing the two reports side by side will notice the identical counts and could reasonably misread them as agreement between the pipelines. They are not.

---

## 6. Handoff to Phase 15

- **Production score**: percentile aggregation, Option B model set (§3), with `ensemble_percentile_average` (`artifacts_research_v2/ensemble_scores_v2.csv`) as the reference/batch implementation.
- **Explanation layer**: Isolation Forest (`shap.TreeExplainer`, exact, 8.3s/2,512 rows) and Autoencoder (`shap.GradientExplainer` via `AEErrorWrapper`, 112.3s/2,512 rows), both shown side by side, both already computed and saved.
- **Secondary score for sigma-style thresholding**: Phase 12 (v2)'s Weighted Average, computed in parallel at no additional model cost.
- **Operating points**: 0.9510 (priority review, 26 rows) and 0.8671 (standard review, 126 rows). No block tier.
- **Reduced-member fallbacks**: 3-model (IF + LOF + AE, percentile-aggregated) then 1-model (Isolation Forest). **Not** the in-house pipeline's 2-model IF + Autoencoder fallback — see §4.
- **Open measurement blocking full confidence**: bootstrap stability of the ensemble strategies themselves (§5).

