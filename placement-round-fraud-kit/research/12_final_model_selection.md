# Phase 14 — Final Model Selection

**Nothing is computed in this phase.** Every score below is a judgement call over evidence already produced in Phases 3–13, and every number cited is traceable to the report that produced it (`research/03_data_quality_and_eda.md` through `research/11_threshold_optimization.md`) or to the artifact that report points at. Where a dimension could not be measured for a candidate, the score is marked **(inferred)** and the reasoning says what it was inferred from — it is never presented as if it had been measured.

Sixteen candidates are scored: the 12 models built in Phase 8 (`research/06_model_development.md`) and the 4 ensemble-scoring strategies built in Phase 12 (`research/10_ensemble_scoring.md`).

---

## 0. Scoring Scheme

Six dimensions, each scored 1–5 (5 = best). The rubric is fixed before scoring, not fitted to the answer:

| Dimension | What it measures | Primary evidence |
|---|---|---|
| **Detection Quality** | Internal cluster-validity of the model's own top-5% partition, plus how plausible its flags looked under manual business review | Phase 10 §1 (Silhouette / Davies-Bouldin / Calinski-Harabasz), Phase 10 §4 (business walkthrough), Phase 8 §3.4 (ρ vs. v1's independent 4-detector proxy) |
| **Stability** | Would a retrain flag the same transactions? | Phase 10 §2 (bootstrap Jaccard, measured for 3 models only), Phase 8 §1.5–1.8 (config-to-config sensitivity as a proxy for the rest) |
| **Interpretability** | Is there a working, defensible per-feature attribution path? | Phase 11 (`research/09_explainability.md`) |
| **Computational Cost** | Measured fit/score/search cost at n=2,512, plus lifecycle burden | Phase 8 per-model timings, Phase 9 search timings, Phase 11 explainer timings |
| **Scalability** | Can it score a transaction it was not fit on, and how does that cost grow? | Phase 8 §0 (split methodology) and per-model notes |
| **Deployment Readiness** | Judgement call: could this ship as-is, given the artifacts that actually exist? | The saved artifacts in `artifacts_research/models/`, plus every caveat above |

**Weighting** used for the weighted total: Detection Quality 0.25, Stability 0.20, Interpretability 0.15, Scalability 0.15, Deployment Readiness 0.15, Computational Cost 0.10. Rationale: for a bank fraud-review system, being wrong and being unstable cost more than being slow — at this dataset's 6.90 transactions/day (Phase 13 §4) compute is close to free, so cost gets the lowest weight. The unweighted total out of 30 is reported alongside so a reader who disagrees with the weights can use it.

**A ceiling rule, stated up front.** No candidate can score 5 on Detection Quality. There is no fraud label anywhere in this project (Phases 10, 12 and 13 each state this independently), so no candidate has been *shown* to detect fraud better than any other — only to partition the feature space more cleanly, agree more with the rest of the field, or produce more plausible-looking examples under manual reading. A 4 is the ceiling the available evidence supports, and awarding a 5 would misrepresent what was proven.

---

## 1. Candidate-by-Candidate Scoring

### 1.1 Isolation Forest — 26/30 (weighted 4.20)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | Mid-table on internal validity (Silhouette 0.565, 8th of 12 — Phase 10 §1), but it is the only model that was put through a manual business review: of the five top-1% transactions examined in Phase 10 §4, three are defensible Scenario 1/4 matches (`TX000275`: 5 login attempts plus an amount 3.6× the account balance; `TX000177` and `TX001354`: amounts 117.9× and 149.0× their accounts' own averages), one is ambiguous, and one (`TX000566`, a $29.38 below-average transaction) is a demonstrable false signal. It also ranks 3rd of 12 on agreement with v1's independent 4-detector proxy (ρ=0.403, Phase 8 §3.4). Docked from a 4-plus by the `TX000566` failure mode, which Phase 11 traced to a real mechanism, not bad luck. |
| Stability | 3 | Measured: mean bootstrap Jaccard 0.527 across 5 refits (min 0.448, max 0.565) — the lowest of the three models tested (Phase 10 §2). Scored 3 rather than 2 because Phase 10's own read is that the 0.527–0.590 spread across IF, LOF and the Autoencoder is too narrow to separate them; roughly 47% of its flagged set changes between retrains, which is a genuine problem but not one it has more of than its peers. |
| Interpretability | 4 | The best explainer path in the project: `shap.TreeExplainer`, **exact** (not an approximation), no background sample required, 7.2s for all 2,512 rows (Phase 11 §0). Not a 5 because Phase 11 §1 showed *what* it explains is often unhelpful — its top global drivers are low-cardinality categoricals (`TransactionType_Debit` 0.176, the `CustomerOccupation_*` one-hots) because a single binary split isolates a whole minority class cheaply, and on `TX000566` the single largest SHAP contribution in the entire local-explanation set came from `LocationNoveltyFlag=0` being statistically rare rather than risky. An exact explanation of a misleading signal is still misleading. |
| Computational Cost | 5 | 9.72s to fit and score 5 configs at n=2,512 (Phase 8 §1.1); Phase 9 enumerated a full 60-combination grid in 63.9s. Cheapest classical model to search. |
| Scalability | 5 | Native `decision_function`, used exactly as intended in Phase 8: fit on the 2,009-row train split, scored all 2,512 rows out-of-sample. Scoring is a tree-path traversal, effectively O(log n) per row per tree. Phase 8 §1.1 names it "the only one of the 8 classical models with a native, well-understood out-of-sample `decision_function`". |
| Deployment Readiness | 5 | Fitted artifact saved (`artifacts_research/models/isolation_forest.pkl`), a native `contamination` parameter that yields a threshold without a separate calibration step, an exact and fast explainer, and it is the model Phase 10 already used to generate reviewable examples. The closest thing in this project to something that could ship this week. |

### 1.2 Local Outlier Factor — 19/30 (weighted 3.20)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | Silhouette 0.617 (5th of 12), and 2nd of 12 on the v1 proxy cross-check (ρ=0.428, Phase 8 §3.4). It sits in the well-agreeing core of the model field — its strongest pairings are HDBSCAN (ρ=0.840, the highest in the whole matrix) and the Autoencoder (ρ=0.787), so it is not an idiosyncratic detector. Never put through a business-evaluation walkthrough, so its flags have not been read by a human the way Isolation Forest's have. |
| Stability | 3 | Measured: the **most stable** of the three tested, mean bootstrap Jaccard 0.590 (min 0.465, max 0.703) — Phase 10 §2 attributes this to bootstrap resampling perturbing k-nearest-neighbour sets less than it perturbs random tree splits or learned network weights. Still ~41% flagged-set churn per retrain, which is why this is a 3 and not a 4. |
| Interpretability | 2 | No native or exact SHAP path was built in Phase 11 — only Isolation Forest and the Autoencoder were explained. `novelty=True` gives it a `decision_function`, so a model-agnostic `KernelExplainer` or permutation-importance route is possible in principle, but it was not run, and KernelExplainer on 2,512 rows × 46 features is materially more expensive than the 7.2s TreeExplainer path. Scored on what exists, not what could be built. |
| Computational Cost | 4 | 6.29s for 5 configs at n=2,512 (Phase 8 §1.2) — cheap here. Docked one point because the default neighbour search is O(n²) and that cost is already implicit in the fit, not just a future concern. |
| Scalability | 3 | Has out-of-sample scoring (`novelty=True` was set deliberately for this reason, `src_research/07_models_classical.py:183`), which puts it well ahead of DBSCAN/HDBSCAN. But Phase 8 §1.2 states it "will need an ANN index well before six figures of rows" — a real engineering task, not a config change. |
| Deployment Readiness | 3 | Artifact saved (`models/lof.pkl`), best measured stability, out-of-sample capable — but no explainability path and a known indexing rework ahead of it. Deployable as an ensemble member, not as the single model behind an investigator-facing alert. |

### 1.3 One-Class SVM — 16/30 (weighted 2.55)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | 2nd-highest Silhouette (0.664) — but Phase 10 §1 warns explicitly that a top-5%-by-distance cut is close to guaranteed to separate well *in a distance metric*, so this is partly a construction artifact rather than evidence of quality. Against the one semi-independent check available it does poorly: ρ=0.288 vs. v1's vote count, 10th of 12 (Phase 8 §3.4). It is also one half of the only negative pairing in the entire cross-model matrix (OCSVM ↔ GMM, ρ=−0.052). |
| Stability | 2 | **(inferred)** — not bootstrap-tested. Its decision boundary is defined by the support vectors sitting nearest the margin, exactly the population a resample perturbs most, and its `nu=0.05` cut lands on the same graded score distribution that Phase 10 §2 identified as the mechanism behind 41–47% churn in the models that *were* tested. No reason to expect it beats the tested three; scored one band below them. |
| Interpretability | 2 | Same position as LOF — a `decision_function` exists so a model-agnostic explainer is possible, but none was built in Phase 11, and an RBF kernel decision value has no natural per-feature decomposition to fall back on. |
| Computational Cost | 5 | The fastest measured fit in Phase 8: 1.27s for all 5 configs at n=2,512 (§1.3). Judged on measured cost at this scale, it earns a 5; its growth problem is scored under Scalability, not double-counted here. |
| Scalability | 2 | Phase 8 §1.3 singles it out: "the one model in this set flagged as a real scalability concern past ~50k–100k rows without subsampling or an approximate variant" — a QP solve roughly O(n²)–O(n³) in the number of support vectors. It can score out-of-sample, which keeps it off the floor. |
| Deployment Readiness | 2 | Artifact saved (`models/ocsvm.pkl`), but it combines a hard scale ceiling, no explainability, and the weakest agreement with the rest of the field of any non-DBSCAN model. Nothing here argues for putting it in front of an investigator. |

### 1.4 Elliptic Envelope — 15/30 (weighted 2.35)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 2 | Respectable on paper (Silhouette 0.610, ρ=0.323 vs. the v1 proxy, and ρ=0.758 with Isolation Forest — one of the stronger pairings). Scored down hard because its core assumption is measurably false: Phase 8 §1.4 ran Shapiro-Wilk on all 46 scaled features and **100% reject normality at p<0.05**, and sklearn itself raised a "covariance matrix is not full rank" warning during fitting. Its Mahalanobis distances are being computed under an assumption the data does not satisfy, so the score is a rough baseline, not a calibrated one — Phase 8 says exactly this. |
| Stability | 2 | **(inferred)** — not bootstrap-tested. MCD works by selecting a robust subset of observations, which is inherently resample-sensitive, and the rank-deficiency warning means its covariance estimate is already near the edge of numerical conditioning. v1 hit the same problem from the other direction: `DOCUMENTATION.md` Stage 2 records that the near-constant `DeviceNoveltyFlag` made MCD's covariance ill-conditioned on some resamples. That is direct evidence of resample fragility, even though the Jaccard number was never measured. |
| Interpretability | 2 | Mahalanobis distance is decomposable per dimension in principle — a genuine explanatory route that DBSCAN/HDBSCAN do not have. But decomposing a quadratic form built on a rank-deficient covariance matrix estimated under a violated Gaussian assumption produces attributions nobody should defend to a reviewer. Credit for the route, discount for its trustworthiness. |
| Computational Cost | 4 | 3 configs run in Phase 8 §1.4; no per-model timing was reported (noted as a gap). MCD is iterative and more expensive than a single covariance estimate, but at n=2,512 × 46 features this is trivially cheap. Scored 4 rather than 5 only because the number is missing rather than good. |
| Scalability | 3 | Native `decision_function`, and scoring a new row is a cheap quadratic form. The binding constraint is statistical, not computational — more rows will not repair a violated distributional assumption. |
| Deployment Readiness | 2 | Artifact saved (`models/elliptic_envelope.pkl`), but Phase 8 §1.4's verdict is explicit and is adopted here unchanged: "kept for completeness and comparison, **not recommended as a primary production detector**." |

### 1.5 DBSCAN — 9/30 (weighted 1.45)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 2 | Silhouette 0.615 looks fine in isolation, but every other signal is bad. All **9** grid combinations found exactly **1 cluster** (Phase 8 §1.5) — it never discovers structure, only how much of the fringe to call noise (0.7%–2.9%). Its selected config flags 1.23% of rows, an anomaly rate set by density geometry rather than any business assumption. It has the weakest agreement with v1's proxy of all 12 models (ρ=0.118) and its strongest correlation with *any* of the other 11 models is only ρ=0.17. Phase 8 §3.3 pairs it with HDBSCAN at Jaccard 0.023 — the two density-based methods barely agree with each other on which rows are outliers. |
| Stability | 1 | **(inferred)** — not bootstrap-tested, but the Phase 8 grid provides a harder proxy: noise rate swings from 0.7% to 2.9% (a factor of ~4) across a modest 3×3 grid around the k-distance elbow at eps=8.428. A flagged population that quadruples under small parameter perturbation is not a stable basis for an alert queue. |
| Interpretability | 1 | Its pseudo-score is distance to the nearest core point. There is no per-feature attribution, no explainer was run, and "this row is far from the dense mass" is not an explanation an investigator can act on. |
| Computational Cost | 3 | The 9-combination grid plus a k-distance elbow computation (`research/plots/dbscan_kdistance_elbow.png`) is more setup than Isolation Forest needs, and — critically — it must be refit over the *entire* dataset every time rather than fit once and reused. |
| Scalability | 1 | **No native out-of-sample `.predict`.** Phase 8 §0 had to fit it on the full 2,512 rows rather than the 2,009-row training split for exactly this reason, and §1.5 calls it out as "a real production limitation versus IF/LOF/OCSVM/EE." Every new batch requires a refit over all history. |
| Deployment Readiness | 1 | One cluster, an unstable noise fraction, no out-of-sample scoring, no explanation path. It earned its place in the comparison; it has no place in the deployed system. |

### 1.6 HDBSCAN — 10/30 (weighted 1.70)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Genuinely split evidence, so the score splits the difference. Its GLOSH outlier score tops the internal-validity table (Silhouette 0.672, the best of 12) and ranks 4th on the v1 proxy (ρ=0.393), and it is one half of the strongest pairing in the whole matrix (LOF ↔ HDBSCAN, ρ=0.840). Its *native clustering output*, however, is unusable: 53.94% noise on the selected config, 53.9%–75.0% across the four tried, and only ever 2 clusters (Phase 8 §1.6). Phase 10 §1 is careful that these two findings do not contradict — the ranking is useful, the partition is not. Only the ranking is scoreable here. |
| Stability | 1 | **(inferred)** — noise rate ranging 53.9%–75.0% across four configurations is the clearest parameter-sensitivity signal of any model in the set. Phase 8 §1.6's verdict is that it "traded manual `eps`-tuning brittleness for a different failure mode," not that it solved it. |
| Interpretability | 1 | GLOSH is a density-ratio outlier score with no per-feature decomposition and no explainer built. Same position as DBSCAN. |
| Computational Cost | 3 | Four configurations run; no timing reported (a gap). Like DBSCAN it must be refit over the full dataset, which is the dominant cost, not the fit itself. |
| Scalability | 1 | **No out-of-sample scoring as built.** Phase 8 §0 fit it on the full dataset for this reason. The `hdbscan` library does expose `approximate_predict`, but it requires `prediction_data=True` at fit time, which this build does not set (`src_research/07_models_classical.py:469` records exactly this) — so as it stands the artifact cannot score a new transaction. This is a fixable gap, but it is a gap today. |
| Deployment Readiness | 1 | The highest-ranking score in the internal-validity table attached to the least deployable model in the set. Nothing about the saved `models/hdbscan.pkl` can be pointed at a live transaction without a rebuild. |

### 1.7 K-Means — 19/30 (weighted 3.05)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Silhouette 0.663 (3rd of 12), but with the same distance-cut caveat as OCSVM, and a weak 9th-of-12 on the v1 proxy (ρ=0.280). The more important qualification is that its anomaly score only works because of a documented repair: Phase 8 §1.7 found that naive nearest-centroid distance would have made the three most extreme rows in the dataset the *safest-looking* points, because each had become its own micro-cluster. The fix — score distance only to clusters holding ≥1% of training rows — is sound, but a scoring rule that inverts without a hand-applied correction earns a middling grade, not a high one. |
| Stability | 2 | **(inferred)** — not bootstrap-tested. Two concrete instability sources are documented: k-means++ initialisation variance, and the finding that **every k from 2 to 10 produced at least one cluster holding <1% of the 2,009 training rows** (Phase 8 §1.7). Since the score depends on which clusters clear the ≥1% inclusion rule, a refit can change not just the centroids but the *set of valid reference centroids*, which is a sharper discontinuity than the tested models face. |
| Interpretability | 3 | The one bright spot among the clustering models. Distance to a centroid decomposes exactly and cheaply into per-feature squared differences, giving a real "which features push this transaction away from its nearest normal centroid" story with no explainer library involved. Not scored higher because it was not actually built or validated in Phase 11. |
| Computational Cost | 4 | Cheap to fit at this scale. Docked one point because model selection was not free: the silhouette-based selection had to run on a 1,000-row subsample due to silhouette's O(n²) cost, and both an inertia elbow and a silhouette curve were needed (`research/plots/kmeans_elbow_silhouette.png`) before the silhouette result could be rejected as an artifact. |
| Scalability | 4 | Native `.predict`, O(k·d) per row — genuinely cheap and out-of-sample capable. One point off because the ≥1%-of-training-rows cluster filter is an extra piece of state that must be pinned and shipped with the pickle; the artifact alone is not sufficient to reproduce the score. |
| Deployment Readiness | 3 | Artifact saved (`models/kmeans.pkl`), fast, predictable, out-of-sample capable — but it needs a custom scoring wrapper carrying the micro-cluster exclusion rule, and its detection evidence is the weakest of the operationally-viable models. |

### 1.8 Gaussian Mixture Model — 13/30 (weighted 1.95)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 1 | The clear low point of the internal-validity table on all three metrics simultaneously: Silhouette 0.319, Davies-Bouldin 4.113, Calinski-Harabasz 20.7 (Phase 10 §1) — Calinski-Harabasz is roughly a sixth of the next-worst model's. It is also the model most structurally divergent from the field: near-zero or negative correlation with 10 of the other 11 (OCSVM ρ=−0.052, DBSCAN ρ=0.010, K-Means ρ=0.052), and Jaccard as low as 0.047–0.072 against several (Phase 8 §3.2–3.3). Phase 8 §3.3 names likelihood-based scoring as one of the two definitions that "diverge the most from the distance/reconstruction-based majority." Divergence alone would not be fatal — Phase 11 argues divergence is *useful* — but divergence combined with the worst partition quality in the set is. |
| Stability | 1 | **(inferred)**, and the inference is unusually well-supported. Phase 9 §2 added one hyperparameter (`reg_covar`) to Phase 8's grid and the "best" BIC moved from −63,019.3 to −109,595.2, a swing of 46,575.9, by pushing `n_components` to the boundary of the search range (10) and `reg_covar` down to 1.19×10⁻⁶. A selection criterion that moves that far on a single added degree of freedom, with the BIC curve still descending at the search boundary, is not identifying a stable optimum. Both Phase 8 §1.8 and Phase 9 §2 flag this independently. |
| Interpretability | 2 | A negative log-likelihood under a diagonal or spherical covariance decomposes readably per feature. Under the selected `covariance_type='full'` — 1,081 free covariance parameters per component, ten components — it does not, and any attribution would rest on a covariance structure that both tuning phases flagged as probably overfit. |
| Computational Cost | 3 | Phase 8 searched `n_components` 1–10 × 4 covariance types; Phase 9 added a 40-trial Optuna study. Scoring is cheap; selection is not, and the selection is the part that has been shown to be untrustworthy. |
| Scalability | 4 | Native `score_samples`, cheap per row, and unusually this is a model that would genuinely *benefit* from more data — 2,009 training rows against 46 features is precisely the regime where full-covariance components overfit. |
| Deployment Readiness | 2 | Artifact saved (`models/gmm.pkl`) and cheap to score, but two independent phases recommended against the configuration that the selection criterion actually chose, and Phase 9 §2's carried-forward conclusion is that the `tied` covariance option "remains the more numerically defensible production choice even though it never wins on raw BIC." Shipping the model would mean shipping the configuration nobody in the pipeline endorsed. |

### 1.9 Autoencoder — 26/30 (weighted 4.20)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | Silhouette 0.496 is 9th of 12 — but Phase 10 §1 explains why that number should not be read as a ranking against the distance-based models: "a low reconstruction error does not necessarily correspond to being close to the bulk of points in Euclidean distance, since the autoencoder's bottleneck can compress non-adjacent points to similar codes." Judged on evidence that is not stacked against it, it does well: the best reconstruction fit of the three deep models (val MSE 0.328 vs. VAE 0.379 vs. LSTM-AE 1.258), 5th of 12 on the v1 proxy (ρ=0.386), and Phase 11 §1 shows its score is driven by exactly the features Phase 1 identified as this dataset's real signal — the four amount-relative features occupy its top four SHAP ranks (`Amount_vs_AccountAvg` 0.0390, `Amount_ZScore_Account` 0.0367, `Amount_to_Balance_Ratio` 0.0367, `Amount_to_RollingMean_Ratio` 0.0360). Decisively, it is the model that got `TX000566` *right*: where Isolation Forest ranked a $29.38 below-average transaction in the top 1% on the strength of a near-constant flag, the Autoencoder assigned it SHAP magnitudes an order of magnitude smaller than the genuine outliers (Phase 11 §2). |
| Stability | 3 | Measured: mean bootstrap Jaccard 0.533 (min 0.448, max 0.658) across 5 refits (Phase 10 §2) — statistically indistinguishable from Isolation Forest's 0.527 and within the band Phase 10 declined to separate. Same ~47% churn caveat. |
| Interpretability | 4 | A working explainer exists and was run over all 2,512 rows: `shap.GradientExplainer` on an `AEErrorWrapper` module whose forward pass returns the scalar per-row reconstruction MSE, i.e. the exact quantity being thresholded (Phase 11 §0). Not a 5 for two honest reasons: it is an expected-gradients *approximation*, not an exact decomposition, and its additivity spot-check came back at Spearman ρ=0.95 rather than 1.0; and it took 128.8s versus TreeExplainer's 7.2s. Both are acceptable; neither is best-in-class. |
| Computational Cost | 5 | Phase 8 §2.9: "cheapest per-row inference cost of any model in this comparison: a single forward pass, O(1) per row independent of dataset size." The model is already trained and was reused, not retrained, in Phase 8 — there is no fitting cost to pay again. |
| Scalability | 5 | Stateless given the scaler, batchable, and O(1) per row. Nothing about the forward pass changes at 1M rows; only training time does. |
| Deployment Readiness | 5 | The best-packaged artifact in the project: weights (`autoencoder.pt`), the **exact** fitted `RobustScaler` used at training time (`autoencoder_scaler.pkl` — not a refit copy), the architecture and the ordered 46-name `feature_cols` schema (`autoencoder_config.json`), per-row errors with a train/val split flag (`autoencoder_reconstruction_errors.csv`), and loader functions that reload it without re-deriving anything (`src_research/autoencoder_utils.py::load_autoencoder` / `reconstruction_errors`). This is what every other model's packaging should look like. |

### 1.10 Variational Autoencoder — 20/30 (weighted 3.30)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | Silhouette indistinguishable from the plain Autoencoder (both 0.496 as reported in Phase 10) and a near-identical v1-proxy correlation (0.385 vs. 0.386), but it reconstructs consistently less precisely at every percentile except the maximum (val MSE 0.379 vs 0.328; val P99 1.946 vs 1.372) — the expected beta-VAE tradeoff, since part of the loss budget goes to the KL term (Phase 8 §2.10). The problem for a selection decision is redundancy, not quality: at ρ=0.801 with the Autoencoder it is one of the most correlated pairs in the entire matrix. It is a slightly worse copy of a model already in the set. |
| Stability | 3 | **(inferred)** from the Autoencoder's measured 0.533 — same architecture family, same 2,009/503 split, same `random_state=42`, same optimiser regime. Its KL regularisation arguably smooths the latent space (its worst row scores 4.395 vs the AE's 6.708), which could make it marginally more stable, but that was not measured and no credit is given for it. |
| Interpretability | 3 | The same `AEErrorWrapper` + `GradientExplainer` route that worked for the Autoencoder applies directly, since the score used is reconstruction MSE only, not MSE+KL (a documented Phase 8 §2.10 design choice specifically so the two are comparable). It was not actually run in Phase 11, so it is scored one band below the model where it was demonstrated. |
| Computational Cost | 3 | 40.7s to train for 200 epochs (Phase 8 §2.10), and it was the most expensive model to search in Phase 9: 20 trials, 282.6s, because each trial is a full 60-epoch training run rather than a single cheap fit. |
| Scalability | 5 | Same as the Autoencoder — an O(1) forward pass per row, stateless given the scaler. |
| Deployment Readiness | 3 | Well-packaged (`models/vae.pt`, `vae_config.json`, `vae_training_history.json`) and cheap to score, but it is a second model to version, monitor and retrain in exchange for a signal that correlates 0.80 with one already deployed. Phase 9 §3 also showed its most consequential hyperparameter, `beta`, drives val MSE from below 0.45 to above 1.2 across the searched range — the deployed `beta=0.1` sits right at the edge of the good region, which is fine in hindsight but means the configuration is not comfortably inside a stable plateau. |

### 1.11 LSTM Autoencoder — 11/30 (weighted 1.85)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 2 | Silhouette 0.643 (4th of 12) is the one number in its favour. Everything else argues against: it reconstructs far less precisely than the feedforward autoencoder (val MSE 1.258 vs 0.328), it correlates only ρ=0.57 with the plain AE — "unremarkable, no higher than several classical models'" (Phase 8 §2.11) — and it ranks 7th of 12 on the v1 proxy (ρ=0.329). Phase 8's own diagnosis is that this dataset gives a recurrent model nothing to learn: median 5 transactions per account, maximum 12, only ~342 training sequences. |
| Stability | 2 | **(inferred)** — not bootstrap-tested. ~342 training sequences is by far the smallest effective training set of any model here, and its train/val MSE relationship (1.636 train vs 1.258 val) is unlike the other deep models', which is itself a sign the fit is not well-determined. |
| Interpretability | 2 | Per-timestep × per-feature attribution is technically possible but was not built, and there is a conceptual problem beneath the engineering one: attributing a *sequence* reconstruction error to the *single transaction* an investigator is looking at is genuinely ambiguous when the sequence is 5 transactions long. |
| Computational Cost | 2 | 31.4s for 150 epochs, but Phase 8 §2.11 is explicit that the per-epoch cost is "meaningfully slower to train per epoch than the feedforward AE/VAE (sequential, non-parallelizable across timesteps)" and concludes the added cost "is not justified by a corresponding gain in either coverage or score distinctiveness." |
| Scalability | 2 | Scoring requires assembling the account's full padded sequence (up to `max_len=12`) at inference time, which means per-account sequence state, not a stateless row transform. It also carries a **permanent coverage gap**: 110 rows (4.4%), the accounts with 1–2 transactions, get no score at all — reported by Phase 8 as permanent, not as something more data fixes for those specific accounts. |
| Deployment Readiness | 1 | Worst reconstruction, no explanation path, sequence state required at inference, and a structural coverage hole. It was worth building to answer the feasibility question honestly; the answer was no. |

### 1.12 Hybrid Ensemble (IF + LOF + AE, ≥2-of-3 majority vote) — 19/30 (weighted 3.15)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | It has the highest correlation with v1's independent proxy of all 12 models (ρ=0.457) — but Phase 8 §3.4 pre-empts any credit for that: it is "an expected, mechanically-driven result... not independent confirmation of anything," since both are majority votes over overlapping detector families and Isolation Forest and LOF appear in both. On internal validity it is 11th of 12 (Silhouette 0.467), which Phase 10 §1 attributes to a real effect: combining detectors that disagree about *which specific rows* to flag produces a less cleanly-separated partition than any component alone (except the AE). Requiring 2-of-3 agreement drops the flagged rate to 3.74% (94 rows) from each component's ~5%. |
| Stability | 3 | **(inferred)** — requiring two of three detectors to agree filters out each component's idiosyncratic borderline flags, which should reduce churn relative to any single member. But it inherits all three members' instability (measured at 0.527, 0.533 and 0.590) and was never bootstrap-tested itself. Scored level with its components, not above them, because the argument is theoretical. |
| Interpretability | 3 | "Two of three independent detectors flagged this" is one of the clearest one-line justifications available to an investigator, and two of the three members (IF, AE) have working SHAP layers underneath. The vote count itself, however, carries no graded per-feature attribution. |
| Computational Cost | 3 | Compute is just the sum of three inference costs, all cheap. Phase 8 §2.12 makes the more important point: "the real production cost of an ensemble like this is maintaining and monitoring 3 models, not raw compute." |
| Scalability | 4 | All three members score out-of-sample. The binding constraint is LOF's O(n²) neighbour search, inherited whole. |
| Deployment Readiness | 3 | Simple and explainable, but with one disqualifying property for the score itself: `hybrid_vote_count` takes only 4 distinct values (0–3). Phase 10 §4 rejected it as the basis for the business walkthrough for exactly this reason — "cutting clean, distinct top-1% / top-2% / top-10% tiers from a 4-valued score is not meaningful (most rows within a tier would be tied on the same vote count with no principled way to rank within it)." A review queue needs an ordering, and this score cannot supply one inside a tier. |

### 1.13 Weighted Average (consensus-weighted z-scores) — 17/30 (weighted 3.10)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | Aggregates 11 detectors and is the only strategy that acts on the field's known problem cases, giving DBSCAN and GMM the two lowest weights (0.058 and 0.069 against HDBSCAN's 0.113) via a disagreement-inverse rule derived from Phase 8's Spearman matrix. It ties for the best correlation with v1's independent proxy of the four strategies (ρ=0.444, a difference from the others "well within noise" per Phase 12 §2.1). It reaches the 4 ceiling for the same reason all the sound aggregation strategies do: it directly addresses the Phase 11 finding that no single detector family is safe alone. |
| Stability | 4 | **(inferred, and flagged as a gap)** — no bootstrap Jaccard was measured for *any* ensemble strategy, which is the most important missing measurement in this phase. The inference rests on two things: averaging 11 partially-independent scores is the standard variance-reduction argument, and Phase 12 §2 showed the four strategies agree with each other at ρ=0.982–0.995, i.e. the aggregate is robust to the choice of aggregation rule even where the rules differ substantially in construction. That is evidence of robustness to *method* choice, not to *resampling*, and the distinction is stated rather than blurred. |
| Interpretability | 3 | Explainable in one sentence as a weighted average of standardised detector scores, and the per-model contributions are directly inspectable. Docked because the weights themselves need defending: Phase 12 §3 makes precisely this argument against it — the weight vector is "computed from a correlation matrix that could shift with more data or a different train/val split," so it is an extra modelling decision a compliance reviewer can question. |
| Computational Cost | 1 | Requires running all 11 models. In compute terms that is still only minutes at n=2,512, but the lifecycle cost is the real one: 11 artifacts to version, retrain, monitor and roll back together, plus a weight vector (`ensemble_weights.json`) that is itself derived from a matrix that must be recomputed. This is the single strongest argument for a single model instead, and it is scored honestly at the floor. |
| Scalability | 2 | Inherits its members' worst properties. Two of the 11 inputs — DBSCAN and HDBSCAN — cannot score a transaction they were not fit on (§3 below), so the strategy as built cannot run incrementally at all; and OCSVM's O(n²)–O(n³) ceiling binds the rest. |
| Deployment Readiness | 3 | One real operational advantage over the recommended Percentile Average: its output is **unbounded**, so classical statistical thresholds work on it. Phase 13 §3 found mean+3σ = 2.1516 flags 17 transactions and Q3+1.5×IQR = 0.9413 flags 87 — usable cut points, where the same rules on the percentile score flag zero. Against that: the weight vector is an extra versioned artifact with its own drift surface, and the 11-model dependency stands. |

### 1.14 Rank Aggregation (Borda count) — 16/30 (weighted 2.95)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | ρ=0.442 against v1's proxy, indistinguishable from the other three strategies. Sound construction: raw ordinal ranks summed across 11 models, with a missing model's contribution set to the neutral median rank so it neither helps nor hurts. Same 4 ceiling, same reason. |
| Stability | 4 | **(inferred)** — identical argument to the Weighted Average, with one addition in its favour: pure rank aggregation is insensitive to the *magnitude* of any single model's score excursion, so one detector producing an extreme value cannot pull the aggregate. Same unmeasured caveat. |
| Interpretability | 3 | "The sum of where 11 detectors rank this transaction" is explainable, and the per-model rank vector is inspectable. Docked because a raw rank sum has no natural units or reference point — a score of 18,400 means nothing without knowing N. |
| Computational Cost | 1 | Same as the Weighted Average: all 11 models. |
| Scalability | 2 | Same 11-model dependency, including the two that cannot score out-of-sample. |
| Deployment Readiness | 2 | One point below Percentile Aggregation for a specific and, as far as this project's reports go, previously unstated reason: **a Borda sum's scale depends on N**, the number of rows being ranked together. Phase 12 §2 correctly found the two strategies near-identical (ρ=0.9999, Jaccard 0.953) — but that comparison was made on one fixed 2,512-row dataset. In production, where batches arrive with different row counts, a raw rank sum is not comparable across batches without renormalisation, whereas a percentile is bounded in (0,1) by construction and can be evaluated against a frozen training-time reference distribution. This is a deployment consideration that only appears once scoring becomes incremental, and it is the one concrete thing that separates two otherwise mathematically equivalent strategies. |

### 1.15 Percentile Aggregation — 19/30 (weighted 3.40)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 4 | ρ=0.442 against v1's proxy — statistically level with all three alternatives, so this dimension does not separate the strategies and is not pretended to. It reaches the ceiling on the strength of the same argument as the others: Phase 11 §3 established mechanistically that Isolation Forest and the Autoencoder respond to different failure modes (SHAP rank correlation ρ=−0.157, 1 of 10 top features shared), and an aggregate over 11 detectors is the direct answer to that. Its specific construction advantage is that it makes no assumption about the shape or relative variance of the 11 native scales it is combining — reconstruction MSE, a bounded kernel decision value, a GLOSH score, a negative log-likelihood and a centroid distance all become percentiles of their own distributions. |
| Stability | 4 | **(inferred)** — same aggregation argument as above, plus the most conservative missing-data handling of the four strategies: a missing model is skipped and the remaining models renormalised, rather than imputed to a median rank (Borda) or zero (PCA Stacking). That is exactly the degradation behaviour a production system needs when one model is unavailable, and it means the score is defined for all 2,512 rows with no imputation anywhere. Unmeasured for bootstrap Jaccard, like all four strategies. |
| Interpretability | 4 | The best of the four, and Phase 12 §3 argues it directly: "the average of what percentile each detector places this transaction at" is a one-sentence, defensible explanation. It is also genuinely interrogable — the 11-element per-model percentile vector *is* the explanation, and it can be shown to an investigator alongside the IF and AE SHAP breakdowns that already exist for two of its members. Not a 5 only because it has no per-*feature* attribution of its own; that has to come from the member models. |
| Computational Cost | 1 | All 11 models. Identical to the other strategies, and the honest cost of the recommendation made below. |
| Scalability | 2 | Same structural blocker: DBSCAN and HDBSCAN cannot score out-of-sample (§3). |
| Deployment Readiness | 4 | The most production-shaped of the four. Bounded in (0,1); Phase 13 already derived concrete operating points from it (99th percentile = score 0.9145 = 26 transactions = 1.04%; 95th percentile = 0.8406 = 126 transactions = 5.02%); it degrades gracefully when a member model is missing; and it needs no tuned weight vector to version alongside it. Held to 4 rather than 5 by two real limits: the 11-model dependency, and Phase 13 §3's finding that sigma- and IQR-based thresholds flag **zero** transactions on it (mean+3σ = 1.1088 against an observed maximum of 0.9988), because averaging bounded percentiles compresses the tail. If a stakeholder wants a "three standard deviations from typical" framing, this score cannot supply it. |

### 1.16 PCA Stacking Proxy — 13/30 (weighted 2.35)

| Dimension | Score | Reasoning |
|---|:--:|---|
| Detection Quality | 3 | ρ=0.442 against v1's proxy, level with the rest. Scored a band lower than the other three strategies because of what its own diagnostic says: **PC1 explains only 52.65%** of the variance across the 11 standardised score columns (Phase 12 §1.4). Just over half. Choosing this strategy means discarding nearly half of the cross-model signal in exchange for a single axis, and Phase 12 is explicit that this is "an honest, moderate number," not evidence of a sharply-defined consensus direction. |
| Stability | 3 | **(inferred)** — one band below the other strategies for a concrete reason they do not share: the PC1 direction is itself estimated from the score covariance matrix and can rotate between refits. It also requires a sign-orientation check on every refit (Phase 12 §1.4 did check, and no flip was needed *that time*) — a manual verification step that is a genuine operational hazard if it is ever skipped. |
| Interpretability | 2 | Phase 12 §3 rules it out on exactly this ground: its "consensus axis" framing is "harder to explain to a non-technical reviewer than a percentile average," and a downstream user has "no direct way to interrogate" it feature-by-feature. A loading vector over 11 model scores is not an explanation an investigator can use. |
| Computational Cost | 1 | All 11 models, plus a PCA fit (negligible in itself). |
| Scalability | 2 | Same 11-model dependency, and additionally it is the one strategy that cannot tolerate a missing model: PCA requires a complete matrix, so LSTM-AE's 110 missing rows had to be zero-imputed (a documented Phase 12 choice). In production, any unavailable model forces an imputation decision rather than a graceful renormalisation. |
| Deployment Readiness | 2 | The PC1 loading vector must be pinned as a versioned artifact, its sign re-verified on every refit, and its 52.65% variance explained to anyone who asks why half the signal was dropped. Phase 12's recommendation section states plainly that it "is not recommended as the primary production score"; that judgement is adopted here unchanged. |

---

## 2. Consolidated Decision Matrix

Ordered by weighted total. **DQ** = Detection Quality, **St** = Stability, **In** = Interpretability, **Co** = Computational Cost, **Sc** = Scalability, **De** = Deployment Readiness.

| # | Candidate | DQ (.25) | St (.20) | In (.15) | Co (.10) | Sc (.15) | De (.15) | Raw /30 | **Weighted** |
|---:|---|:--:|:--:|:--:|:--:|:--:|:--:|---:|---:|
| 1 | **Isolation Forest** | 4 | 3 | 4 | 5 | 5 | 5 | 26 | **4.20** |
| 1= | **Autoencoder** | 4 | 3 | 4 | 5 | 5 | 5 | 26 | **4.20** |
| 3 | **Percentile Aggregation** | 4 | 4 | 4 | 1 | 2 | 4 | 19 | **3.40** |
| 4 | VAE | 3 | 3 | 3 | 3 | 5 | 3 | 20 | 3.30 |
| 5 | Local Outlier Factor | 4 | 3 | 2 | 4 | 3 | 3 | 19 | 3.20 |
| 6 | Hybrid Ensemble (IF+LOF+AE) | 3 | 3 | 3 | 3 | 4 | 3 | 19 | 3.15 |
| 7 | Weighted Average | 4 | 4 | 3 | 1 | 2 | 3 | 17 | 3.10 |
| 8 | K-Means | 3 | 2 | 3 | 4 | 4 | 3 | 19 | 3.05 |
| 9 | Rank Aggregation (Borda) | 4 | 4 | 3 | 1 | 2 | 2 | 16 | 2.95 |
| 10 | One-Class SVM | 3 | 2 | 2 | 5 | 2 | 2 | 16 | 2.55 |
| 11= | Elliptic Envelope | 2 | 2 | 2 | 4 | 3 | 2 | 15 | 2.35 |
| 11= | PCA Stacking Proxy | 3 | 3 | 2 | 1 | 2 | 2 | 13 | 2.35 |
| 13 | Gaussian Mixture Model | 1 | 1 | 2 | 3 | 4 | 2 | 13 | 1.95 |
| 14 | LSTM Autoencoder | 2 | 2 | 2 | 2 | 2 | 1 | 11 | 1.85 |
| 15 | HDBSCAN | 3 | 1 | 1 | 3 | 1 | 1 | 10 | 1.70 |
| 16 | DBSCAN | 2 | 1 | 1 | 3 | 1 | 1 | 9 | 1.45 |

**Three things this matrix does not capture, stated so nobody over-reads it:**

1. **It cannot see redundancy.** The VAE ranks 4th, above LOF and every ensemble strategy except Percentile Aggregation. That is an artifact of additive scoring: it scores well on Scalability and Cost because it is an easy model to *run*, and the matrix has no column for "correlates 0.801 with a model already in the set" (Phase 8 §3.2). Read on its own merits the VAE is a slightly worse Autoencoder, and its ranking should be discounted accordingly.
2. **Stability is measured for 3 of 16 candidates.** Only Isolation Forest (0.527), the Autoencoder (0.533) and LOF (0.590) have bootstrap Jaccard numbers (Phase 10 §2). Every other Stability score is inferred, including all four ensemble strategies — and the ensembles' inferred 4s are the largest single load-bearing assumption in this table. §5 records this as the first measurement that should be added.
3. **Detection Quality is capped at 4 for everyone.** No candidate has been shown to catch fraud, because there is no fraud to check against. The matrix ranks *engineering and evidential quality*, not detection performance.

---

## 3. The Constraint That Cuts Across Every Ensemble Strategy

This did not surface in Phase 12 and is the most consequential finding of this phase.

All four ensemble strategies take the same 11 models as inputs (Phase 12 §0). **Two of those 11 — DBSCAN and HDBSCAN — cannot score a transaction they were not fit on.** Phase 8 §0 records the consequence directly: while models 1–4 and 7–10 were fit on the 2,009-row training split and scored all 2,512 rows out-of-sample, "DBSCAN and HDBSCAN have no native out-of-sample `.predict`, so are fit directly on the full dataset." HDBSCAN's library does offer `approximate_predict`, but only when `prediction_data=True` is set at fit time, which this build does not do (`src_research/07_models_classical.py:469`).

At 2,512 rows this is invisible — refitting everything on the full dataset takes minutes. In production it is structural: **any ensemble score built over all 11 models can only be produced in batch, by refitting the two clustering models over the full transaction history on every run.** That is not a tuning detail; it decides the architecture (Phase 15 §4).

There are three ways out, and the tradeoffs are real:

| Option | What it costs | What it buys |
|---|---|---|
| **A. Keep all 11, run batch-only** | No real-time scoring at all; refit cost grows with total history, not batch size — untenable well before 1M rows | Uses the exact score Phase 12 recommended and Phase 13 thresholded, with no revalidation needed |
| **B. Drop DBSCAN and HDBSCAN, aggregate over 9** | Loses the highest-weighted model in the Weighted Average scheme (HDBSCAN, 0.113) and the lowest (DBSCAN, 0.058); the resulting score is **not the one Phase 12 computed**, so Phase 13's thresholds (0.8406 / 0.9145) would not transfer unrevalidated | Restores incremental scoring; percentile aggregation already supports this natively, since it skips missing models and renormalises over those available (Phase 12 §1.3) |
| **C. Enable `prediction_data=True` for HDBSCAN, drop DBSCAN only** | A refit and a revalidation; `approximate_predict` is an approximation, so the score changes | Retains the highest-weighted member; loses only the model with the weakest evidence in the whole set (§1.5) |

**Option B is the recommended path**, with Option C as a follow-up if HDBSCAN's contribution proves worth the revalidation. Option A is acceptable only for the research prototype at its current scale. Whichever is chosen, the 9- or 10-model score must be compared against the published 11-model `ensemble_percentile_average` (Spearman and top-5% Jaccard, the same two measures Phase 12 §2 used) before any threshold is carried over. **That comparison has not been run, and no number in this report should be read as if it had been.**

---

## 4. Final Recommendation

**Production score: Percentile Aggregation, computed over the subset of models that can score out-of-sample (Option B above), with `ensemble_percentile_average` from `artifacts_research/ensemble_scores.csv` as the reference implementation and batch/backfill score.**

**Explanation layer, non-optional: Isolation Forest SHAP and Autoencoder SHAP, both shown, side by side.**

### Why not Isolation Forest alone, when the matrix ranks it first

The matrix ranks Isolation Forest and the Autoencoder jointly first at 4.20, ahead of Percentile Aggregation at 3.40. That ranking is not being disputed — on cost, scalability and deployment readiness, either single model beats any ensemble decisively, and an ensemble means eleven artifacts to version and monitor instead of one. **The recommendation deliberately overrides the matrix, and the override is stated rather than engineered into the scores.**

The reason is Phase 11, which is the strongest piece of evidence in this project:

- Isolation Forest and the Autoencoder share **1 of 10** top global features, and their full 46-feature mean|SHAP| importance vectors correlate at **ρ = −0.157** — essentially no agreement, slightly negative.
- The mechanism is understood, not mysterious. Isolation Forest isolates points with random splits, so low-cardinality binary features are cheap split points and dominate its attributions; the Autoencoder's score is squared reconstruction error, dominated by the widest-dynamic-range continuous features, which here are the amount-relative ones. Phase 11 §1: "Isolation Forest is best read here as a 'does this transaction's categorical/temporal shape look unusual' detector, and the Autoencoder as a 'is this transaction's amount unusual relative to this account's own scale' detector."
- There is a worked counterexample. `TX000566` — a $29.38 transaction, *below* its account's own average, normal login count, negligible balance impact — landed in Isolation Forest's **top 1%**. Phase 11 §2 traced it to `LocationNoveltyFlag = 0` carrying the single largest SHAP contribution of any feature across all four local explanations, purely because that value is rare (5.73% of rows) as an artifact of how the flag is constructed, not because a repeat location is suspicious. The Autoencoder assigned the same transaction SHAP magnitudes an order of magnitude smaller and did not share the read.

So the choice is not "one good model versus a more expensive ensemble." It is: **Isolation Forest alone has a demonstrated, reproducible failure mode that sends a plainly unremarkable transaction to a human reviewer, and the fix for it is already in the codebase.** Phase 11 §3 draws this conclusion itself — the divergence is "a direct, mechanistic argument for ensembling rather than relying on a single detector family." In a system with no label, where the only available validation is cross-model consistency, deliberately discarding that consistency to save compute is the wrong trade. At 6.90 transactions/day (Phase 13 §4) the compute saved is negligible; at bank scale the false-positive cost of the `TX000566` failure mode is not.

### Why Percentile Aggregation over the other three strategies

Phase 12 §3's reasoning is adopted and reinforced by the deployment dimensions scored here:

- **Over Rank (Borda)**: Phase 12 showed they are near-identical on a fixed dataset (ρ=0.9999, Jaccard 0.953) and explicitly warned against treating them as materially different signals. The tiebreaker is operational, from §1.14: a Borda sum's scale depends on how many rows are ranked together, so it is not comparable across production batches of differing size, while a percentile is bounded in (0,1) and can be evaluated against a frozen reference distribution.
- **Over Weighted Average**: the weights are defensible but they are an additional judgement — a vector derived from a correlation matrix that will shift with more data — and they need defending to anyone who reviews the system. Percentile aggregation has no tuned parameter to defend. The one place Weighted Average wins is thresholding: it is unbounded, so sigma/IQR rules work on it (Phase 13 §3: mean+3σ = 2.1516 → 17 flagged), where they flag zero on the percentile score. **Recommendation: keep the Weighted Average as a secondary, computed-in-parallel score for exactly that purpose** — it costs nothing extra once all the member models have run, and it gives stakeholders who want a "three sigma" framing something honest to point at.
- **Over PCA Stacking**: 52.65% of PC1 variance means nearly half the cross-model signal is discarded into a component nobody downstream can interrogate, and the sign orientation needs manual re-verification on every refit.

### Operating point

Phase 13's two-tier recommendation is adopted unchanged and is part of this recommendation:

| Tier | Threshold | Score cut | Volume in this sample |
|---|---|---:|---|
| **Priority review** | 99th percentile | 0.9145 | 26 transactions (1.04%) |
| **Standard review** | 95th percentile | 0.8406 | 126 transactions (5.02%) |

**No automatic block tier is recommended.** v1 shipped one (probability ≥ 0.94 → block, `DOCUMENTATION.md` Stage 7), but that threshold came from a cost sweep against supervised proxy labels, and Phase 13 §1 established that the equivalent sweep cannot be reproduced here — computing a false-negative count requires knowing which *unflagged* transactions are fraud, which is unknowable without a label. Blocking a customer's transaction on a score whose false-negative behaviour has never been measured is not defensible. Everything this system produces should land in a human review queue until it has been validated against real investigator-labelled outcomes.

### Fallback if 9–11 models cannot be operated

If the operating team cannot maintain nine model artifacts, the defensible minimal system is **percentile aggregation over Isolation Forest and the Autoencoder alone** — the two models jointly ranked first in this matrix, the two with working SHAP layers, the two with measured stability, and precisely the two whose divergence Phase 11 documented. That is a two-artifact system that keeps the cross-family check the recommendation exists to preserve. Like Option B, **this two-model score was not computed in Phase 12** and would need its own Spearman/Jaccard comparison against the published 11-model score, and its own thresholds, before use.

---

## 5. Gaps and Inconsistencies Found While Synthesising

Recorded here rather than smoothed over. The first is a missing measurement; the rest are errors in earlier reports found by checking their claims against the artifacts.

**Gap — no stability measurement for any ensemble strategy.** Phase 10 §2 bootstrap-tested three individual models. No equivalent exists for the four Phase 12 strategies, yet the recommendation above rests substantially on the assumption that aggregation reduces flagged-set churn below the measured 0.527–0.590 range. That assumption is plausible and standard, but it is untested here. **This is the single highest-value measurement to add next**, and it is cheap: re-run the Phase 10 §2 bootstrap procedure end-to-end through the Phase 12 aggregation.

**Inconsistency 1 — LSTM-AE flagged rate is labelled against the wrong denominator.** Phase 8 §3.1's comparison table reads "LSTM-AE (of applicable rows) | 4.82%", but §2.11 of the same report states "Top-5%-flagged rate (within the 95.6% applicable rows): 5.04%". Checked against `artifacts_research/model_scores_all.csv`: the model flags **121** rows and has non-null scores for **2,402** rows. 121/2,402 = 5.04%; 121/2,512 = 4.82%. **§2.11 is correct and §3.1's label is wrong** — the 4.82% figure is computed over all 2,512 rows, not over applicable rows, and the same 4.82% appears in `model_comparison_summary.json`. The numbers are both real; only the label is wrong. Nothing downstream depends on it, since Phase 10 onward consistently restricts LSTM-AE to its 2,402 applicable rows.

**Inconsistency 2 — the Hybrid Ensemble partition in Phase 10 is not the majority-vote partition.** Phase 8 §2.12 defines the Hybrid Ensemble's flag as a **≥2-of-3 majority vote**, giving 94 rows (3.74%). Phase 10 §1 describes its partition as "majority-vote threshold, 253 rows / 10.07%". Checked against `model_scores_all.csv`: the vote distribution is 0→2,259, 1→159, 2→59, 3→35, so ≥2 votes = **94 rows** and ≥1 vote = **253 rows**. Phase 10's 253-row partition is therefore the **≥1-vote** set produced by applying a top-5% cut to a 4-valued score (89.9% of rows score zero, so the cut lands at 1 and ties pull in all 253). The row count and the resulting Silhouette of 0.467 are correct; the description "majority-vote threshold" is not. This matters for interpretation: Phase 10's Hybrid Ensemble silhouette is evaluated on a partition **2.7× larger** than the ensemble's actual operating flag, which is part of why it scores 11th of 12.

**Inconsistency 3 — `TX000177` is not the most extreme z-score in the dataset.** Phase 10 §4 describes `TX000177` (`Amount_ZScore_Account` = 92.56) as "the single most extreme z-score in the whole dataset," and Phase 8 §1.7 describes `TX000177` (92.56) and `TX002305` (77.71) as "the two most extreme z-scores in the whole dataset." Checked against `artifacts_research/features_v2.csv`, the actual top of the ranking is:

| Rank | TransactionID | AccountID | `Amount_ZScore_Account` |
|---:|---|---|---:|
| 1 | TX001354 | AC00312 | 102.80 |
| 2 | TX000341 | AC00107 | 101.69 |
| 3 | **TX000177** | AC00363 | **92.56** |
| 4 | TX001985 | AC00303 | 83.26 |
| 5 | TX001953 | AC00350 | 82.14 |

`TX000177` is **3rd**, not 1st — and `TX001354` (102.80), which Phase 10 lists in the very same table two rows below, is the actual maximum. Both source reports have the individual values right; the superlatives attached to them are wrong. The accurate statement is that `TX000177` and `TX002305` were the rows K-Means isolated into a persistent micro-cluster (which is what Phase 8 §1.7 was actually demonstrating), not that they were the dataset's two largest z-scores. This does not change any conclusion — `TX001354` was already analysed alongside `TX000177` in both Phase 10 and Phase 11 and reads the same way — but the claim as written is false and should be corrected in those two reports.

**Inconsistency 4 — `LIMITATIONS.md` cites a v1 ROC-AUC that v1 never measured.** `LIMITATIONS.md` states "The 0.97 ROC-AUC measures how well XGBoost reproduces the anomaly ensemble's own judgment." No 0.97 appears anywhere in the v1 results: `DOCUMENTATION.md` Stage 5 reports ROC-AUC **0.9428** (SMOTE) and **0.9532** (class-weighted), and Stage 6 reports **0.943** for the shipped model. The caveat's substance — that the figure measures reproduction of the ensemble's own judgment rather than fraud-catching accuracy — is correct and important; only the number is wrong, and it is wrong in the direction of overstating v1's performance. It should read 0.943.

**Inconsistency 5 (minor) — two artifact paths in Phase 8 are wrong.** Phase 8's header lists "Fitted artifacts: ... `artifacts_research/vae.pt`, `artifacts_research/lstm_ae.pt`". Both files actually live in `artifacts_research/models/`. `artifacts_research/autoencoder.pt` *is* at the top level, which is probably how the error crept in. Noted because Phase 15's scoring service loads these by path.

---

## 6. Handoff to Phase 15

- Production score: percentile aggregation, Option B model set (§3), with `ensemble_percentile_average` as the reference/batch implementation.
- Explanation layer: Isolation Forest (`shap.TreeExplainer`, exact, 7.2s/2,512 rows) and Autoencoder (`shap.GradientExplainer` via `AEErrorWrapper`, 128.8s/2,512 rows), both shown.
- Secondary score for sigma-style thresholding: Phase 12's Weighted Average, computed in parallel at no additional model cost.
- Operating points: 0.9145 (priority review) and 0.8406 (standard review). No block tier.
- Open validation items carried into Phase 15 and Phase 16: the Option B score has not been compared to the 11-model score; no ensemble strategy has a measured bootstrap stability figure.

*Next: `research/13_deployment_architecture.md` (Phase 15).*
