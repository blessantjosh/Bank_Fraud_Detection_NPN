# Phase 16 — Monitoring Framework

Every baseline value in this framework is a measured number from Phases 2–13. Every threshold is either an industry rule of thumb (cited as such) or derived from one of those baselines by stated arithmetic. Nothing here is a placeholder waiting for a real number.

---

## 0. Why Monitoring Carries More Weight Here Than Usual

In a supervised fraud system, monitoring is a safety net: precision and recall against confirmed outcomes tell you whether the model still works, and drift metrics tell you *why* when it stops. **This system has no such backstop.** There is no fraud label anywhere in the project — Phases 10, 12 and 13 each state it independently — so there is no metric that can ever say "the model is still catching fraud."

That has two consequences the framework is built around:

1. **Drift monitoring is not a supporting signal; it is the primary signal.** With no outcome metric, changes in the input distribution and in the models' agreement with each other are the only evidence available that the system's behaviour has changed.
2. **The system's own "ground truth" can drift.** What this system calls anomalous is the consensus of eleven unsupervised detectors. If that consensus decays — if the detectors stop agreeing with each other — the score's meaning changes even when nothing about the score's *distribution* looks different. §3 makes measuring that decay an explicit, first-class monitor rather than an afterthought.

One structural point that shapes everything: **the ensemble score's own distribution is useless as a drift signal.** `ensemble_percentile_average` is the mean of eleven percentile ranks, so it sits near mean 0.5 (measured: 0.5001, std 0.2029, Phase 13) essentially regardless of what the input data does. Monitoring the score distribution would produce a permanently green dashboard over a system that had completely changed. The inputs and the cross-model agreement are what must be watched.

---

## 1. The Four Monitoring Layers

| Layer | Question | Primary metric | §|
|---|---|---|---|
| **Feature drift** | Do incoming transactions still look like what the models were fit on? | PSI, KS D-statistic on the 46 engineered features | §2 |
| **Concept drift** | Has the definition of "normal" moved, given that our definition *is* an ensemble consensus? | Mean pairwise Spearman across detector pairs; PC1 explained variance | §3 |
| **Model drift** | Do the fitted artifacts still fit? Should we retrain? | Composite triggers from the layers above | §4 |
| **Alert-volume and flagged-set drift** | Is the review queue behaving, and is it *stable*? | Flagged rate vs. control band; run-to-run flagged-set Jaccard | §5 |

Plus a fifth, non-statistical layer that is easy to omit and expensive to omit: **pipeline integrity** (§6) — because percentile aggregation is designed to degrade silently when a model goes missing.

---

## 2. Feature Drift

### 2.1 PSI — method and thresholds

Population Stability Index, computed per feature against the frozen training distribution:

```
PSI = Σ_bins ( actual% − expected% ) × ln( actual% / expected% )
```

Binning: 10 bins at the training distribution's deciles, fixed at model-training time and versioned with the model (§9 of Phase 15). Bin edges must **not** be recomputed per monitoring window — recomputing them makes PSI structurally incapable of detecting the shift it exists to detect. Empty bins are floored at 0.0001 to keep the logarithm finite; that floor should be recorded alongside the metric, since it inflates PSI for sparse features.

**Interpretation bands** — the standard credit-scorecard rule of thumb, cited as a convention rather than a derivation:

| PSI | Reading | Action |
|---|---|---|
| < 0.10 | No significant shift | None |
| 0.10 – 0.20 | Moderate shift | Investigate; log; watch the next window |
| > 0.20 | Significant shift | Investigate immediately; candidate retraining trigger (§4) |

Two adaptations for this system, both necessary rather than decorative:

- **PSI > 0.20 alone does not trigger a retrain.** It triggers an investigation. §4.2 explains why: retraining is not free here — it churns 41–47% of the flagged set on its own.
- **PSI is not valid on every one of the 46 features.** Several are near-constant or near-binary, where decile binning is degenerate. §2.3 sorts them.

### 2.2 KS two-sample statistic

The Kolmogorov–Smirnov D-statistic (maximum absolute difference between the training and current empirical CDFs) is run alongside PSI on the continuous features, because the two catch different shapes: PSI is a binned divergence sensitive to mass moving between bins anywhere in the distribution, KS is sensitive to the largest single displacement of the CDF and is therefore better at catching a shift concentrated in one region — for instance a fraud campaign clustering in one amount band.

**Use D, not the p-value.** At production scale the KS p-value is worthless: with a 1M-row current window against a 2,512-row reference, essentially every feature rejects the null and the test reports "significant drift" on all 46 forever. The effect size is the usable quantity. Suggested bands, aligned in spirit to the PSI ones:

| KS D | Reading |
|---|---|
| < 0.10 | Comparable distributions |
| 0.10 – 0.20 | Moderate displacement — investigate |
| > 0.20 | Substantial displacement — investigate immediately |

If p-values are reported at all (they are useful at small window sizes), correct for the 46 simultaneous tests — Benjamini–Hochberg is the appropriate choice, since these features are correlated and Bonferroni would be needlessly conservative.

### 2.3 Feature triage — which of the 46 to monitor, how, and with what expectation

Monitoring all 46 features identically would bury the real signals under features that are *guaranteed* to alarm. The 46 sort into four classes.

#### Class A — Expected to break on day one. Gate, do not alert.

These features are near-constant in the training data **because of documented artifacts in this dataset**, not because the world is like that. In any real deployment they will breach every PSI threshold immediately, and that breach is information about the training data, not about the world.

| Feature | Training-data value | Why it will break |
|---|---|---|
| `Hour_sin` | std **0.040** | All 2,512 transactions fall in a 16:00–18:21 window (Phase 2 §4) — 52.4% at 16:00, 32.6% at 17:00, 15.0% at 18:00 |
| `Hour_cos` | std **0.200** | Same |
| `DOW_sin` | std 0.461, **5 distinct values** | Zero weekend transactions; Monday alone is 42.6% of volume (Phase 2 §4) |
| `DOW_cos` | std 0.825, 5 distinct values | Same |
| `DeviceSharedAccounts_Prior` | > 0 for 72.8% of rows | 89.4% of the 681 distinct devices are used by more than one account — Phase 5 §2.4 identifies this as quasi-random device assignment during data generation, not a mule epidemic |
| `IPSharedAccounts_Prior` | > 0 for 76.4% of rows | 93.2% of the 592 distinct IPs shared across accounts, same cause |
| `DeviceNoveltyFlag` | **1 for 99.52%** of rows | With ~5 transactions per account almost no device repeats (Phase 10 §4) |
| `LocationNoveltyFlag` | **1 for 94.27%** of rows | Same mechanism |

**Handling:** these eight get a **one-time deployment reality-check gate**, not a recurring alert. On the first production window, measure them and expect large PSI. If it appears, the correct response is not "the data drifted" but **"the training data was unrepresentative on these features, and the models must be refit on production data before the system is trusted."** That is a go/no-go gate before launch, and it is the single most likely finding of the first monitoring run.

This is worth stating positively too: Phase 11 §2 traced Isolation Forest's `TX000566` false signal directly to `LocationNoveltyFlag = 0` being statistically rare (5.73% of rows) rather than risky. Refitting on data where that flag is genuinely informative removes the mechanism behind that failure mode.

**Temporal features are a HIGH monitoring priority, not an afterthought** — precisely because they are near-dead in training. A real 24/7 book makes `Hour_sin`/`Hour_cos` genuinely discriminative, which means the classic overnight-activity fraud signals that Phase 2 §4 called "structurally invisible in this data" become visible for the first time. That is a capability gain, and it will only be realised if someone is watching for the distribution change that signals it.

#### Class B — Degenerate under PSI. Monitor the rate directly.

Binary or near-binary features where decile binning collapses. Monitor the **positive rate** with a two-proportion z-test or a p-chart against the training rate, not PSI.

| Feature | Training rate | Alert on |
|---|---:|---|
| `ElevatedLoginFlag` | 4.86% (122/2,512) | A doubling or halving of rate. This is the closest thing in the feature set to a direct credential-attack indicator (Phase 1 Scenario 1), so a genuine rise is a security signal in its own right, independent of model drift |
| `ATM_Credit_InteractionFlag` | 2.91% (73/2,512) | Rate change. Phase 4 established the `Channel`×`TransactionType` association is real (χ²=136.91, dof=2, p=1.87×10⁻³⁰); a shift in this rare combination's frequency means the channel mix moved |
| `Velocity_1D_Count` | **98.0% zero** | The **nonzero rate**, not the binned distribution. Phase 5 §2.1 is explicit that on this dataset the feature is "a rare, high-precision flag" rather than a graded signal. A real transaction stream will produce far more nonzero mass — expected, and worth measuring |
| `Velocity_7D_Count` | **91.1% zero** | Same |
| `LoginAttempts` | 95.14% at exactly 1 | Same treatment — Phase 2 §2 found skew 5.17 and excess kurtosis 26.61 and concluded it "behaves more like a rare-event flag than a continuous variable" |

#### Class C — Core PSI/KS monitoring. These are the ones that matter.

Genuinely continuous, well-spread in training, and load-bearing for the score. This is where the daily monitoring attention belongs.

| Feature group | Members | Training reference |
|---|---|---|
| **Amount-relative (highest priority)** | `Amount_ZScore_Account`, `Amount_vs_AccountAvg`, `Amount_to_Balance_Ratio`, `Amount_to_RollingMean_Ratio`, `Amount_minus_ExpandingMean`, `Amount_minus_ExpandingMedian`, `Amount_vs_TypeAvg` | `Amount_ZScore_Account` spans −92.60 to 102.80; `Amount_to_Balance_Ratio` mean 0.200, median 0.052, max 7.896 (Phase 5 §2.2) |
| **Raw monetary/behavioural** | `TransactionAmount`, `AccountBalance`, `TransactionDuration`, `CustomerAge` | Amount: min 0.26, median 211.14, mean 297.59, max 1,919.11, skew 1.74; Balance: 101.25–14,977.99, mean 5,114.30; Duration: 10–300s, median 112.5; Age: 18–80, mean 44.67 (Phase 2 §2) |
| **Account-history baselines** | `Expanding_{Mean,Median,Std,Min,Max}Amount`, `Rolling3_{Mean,Std}Amount`, `SpendCV_Account`, `CustomerTxnCountSoFar` | `SpendCV_Account` median 0.541 (Phase 5 §2.2) |
| **Recency** | `TimeSinceLastTxn` | Ranked 7th by Isolation Forest SHAP (0.100) and 5th by Autoencoder SHAP (0.0268) — the **only** feature in both models' top 10 (Phase 11 §1) |
| **Frequency/count** | `DeviceTxnCount`, `IPTxnCount`, `MerchantTxnCount`, `Location_Freq`, `MerchantSharedAccounts_Prior` | 681 devices, 592 IPs, 100 merchants, 43 locations (Phase 2 §3); `MerchantSharedAccounts_Prior` mean 12.3, max 43 (Phase 5 §2.4) |

**Within Class C, the amount-relative group is the top monitoring priority in the entire framework.** Three independent reasons: they occupy the Autoencoder's top four SHAP ranks (0.0390, 0.0367, 0.0367, 0.0360 — Phase 11 §1); they are what actually drove the plausible fraud reads in Phase 10's business walkthrough (`TX000177` at 117.9× its account average, `TX001354` at 149.0×, `TX000275` at 3.6× its account balance); and they depend on a training-derived constant that silently breaks if amount scale moves (§2.4).

#### Class D — Categorical encodings. Monitor category distribution.

`TransactionType_Debit` (77.39% Debit), `Channel_Branch` / `Channel_Online` (Branch 34.55%, and the three channels at ATM 833 / Online 811 / Branch 868), `CustomerOccupation_{Engineer,Retired,Student}` (Student 26.15% of 4 categories), `Location_enc`. Monitor with a chi-square goodness-of-fit test against the training category proportions, and separately alert on **any category value never seen in training** — a new `Location`, a new `CustomerOccupation` — which is an encoding failure, not drift. `Location_enc` deserves specific attention: it is a label encoding whose integer codes are alphabetical, so a new city inserted mid-alphabet shifts the codes of existing cities. Phase 6 §6.3 already recommended frequency encoding over label encoding for exactly this class of reason; `Location_Freq` is the safer of the two to rely on and both are present in the 46.

### 2.4 Training-derived constants — the silent failure class

These are not features and will not show up in any feature-drift dashboard, but each is a number computed from the training data and baked into the scoring path. If the underlying data moves, they become wrong quietly.

| Constant | Value | Failure mode if data drifts |
|---|---|---|
| `Amount_ZScore_Account` denominator floor | **$14.60** (5% of the training `TransactionAmount` std of $291.95, `04_feature_engineering.py:119`) | If typical amounts grow, a floor calibrated to a $291.95 std becomes far too small, and the divide-by-near-zero blow-up Phase 6 §7.3 documented (values reaching the hundreds of millions before the fix) can recur |
| `RobustScaler` centre/IQR per feature | Fit on the 2,009-row train split | Every model's input space shifts. Phase 6 §6.2 chose RobustScaler precisely because its IQR denominator is stable (2.40% mean movement under a top-1% trim vs. 13.94% for std), but stable is not immune |
| `Location_Freq` frequency table | Global proportions over 43 cities, top city 2.79% | New cities get no frequency; existing ones become stale |
| `type_avg` (Debit/Credit means) | Credit $306.50, Debit $294.99 | `Amount_vs_TypeAvg` drifts with the population mean |
| K-Means valid-centroid set | Clusters holding ≥1% of the 2,009 training rows | If invalidated, the score inverts (Phase 8 §1.7) |

**Monitor:** recompute each constant on the current window monthly and alert if it moves more than 20% from the versioned value. This is cheap and it catches a failure class that feature-level PSI structurally cannot see.

---

## 3. Concept Drift Without a Label

### 3.1 The problem, stated precisely

Concept drift normally means P(y|X) changes — the relationship between features and the outcome moves. **That is unmeasurable here**, because y does not exist. What this system actually has instead of y is the agreement of eleven unsupervised detectors: a transaction is anomalous because the detector field collectively ranks it so.

So the operational question becomes: **is the detector field still the same field?** If the models begin to disagree with each other more than they did at training time, the consensus that defines "anomalous" has weakened, and the score means something different from what it meant on day one — even if every individual feature's PSI is green and the score distribution is unchanged.

Phase 8 §3 measured that agreement in full, which gives a real baseline to monitor against rather than an invented one.

### 3.2 The proxy metrics and their baselines

**Metric 1 — Mean pairwise Spearman across the 55 detector pairs.**

Derived from Phase 12 §1.1's published disagreement column, where `disagreement_m` is the mean of `(1 − ρ)/2` between model *m* and the other ten. The eleven values are 0.199, 0.215, 0.216, 0.233, 0.238, 0.253, 0.254, 0.262, 0.345, 0.360, 0.440; their mean is **0.2741**, so the mean pairwise Spearman across all 55 pairs is `1 − 2 × 0.2741 =` **0.452**. (Cross-checked against the 55 upper-triangle entries of `artifacts_research/model_pairwise_spearman.csv`: mean 0.4519, min −0.0524, max 0.8402 — the derivation and the matrix agree.)

**Baseline: mean pairwise ρ = 0.452.**

**Metric 2 — PC1 explained variance across the eleven standardised score columns.** Phase 12 §1.4 measured **52.65%**. This is a single number summarising how much of the detector field lines up on one axis. Phase 12 read it honestly at the time — "just over half, not the dominant share that would suggest a single, sharply-defined consensus anomaly axis" — and that moderate value is exactly what makes it a useful monitor: it has room to move in both directions.

**Metric 3 — Specific pairwise anchors.** Aggregate metrics hide which relationship broke. Track these individually against their Phase 8 §3.2/§3.3 values:

| Pair | Spearman baseline | What a drop would mean |
|---|---:|---|
| LOF ↔ HDBSCAN | 0.840 | The strongest agreement in the field; the two density-based views diverging is a structural change in the data's density profile |
| Autoencoder ↔ VAE | 0.801 | Same architecture family, same split — this pair *should* stay tightly coupled. Divergence here means a training or artifact problem, not data drift |
| LOF ↔ Autoencoder | 0.787 | Density and reconstruction agreeing is the cross-family check the system leans on |
| Isolation Forest ↔ Elliptic Envelope | 0.758 | — |
| Isolation Forest ↔ Autoencoder | 0.643 | The two explained models. Phase 11 gave this pair a mechanistic reading; a change here directly affects the two SHAP views an analyst sees |
| OCSVM ↔ GMM | −0.052 | The only negative pair in the matrix. If this turns positive, the field has re-aligned substantially |
| DBSCAN ↔ GMM | 0.010 | — |

And on flagged sets (Jaccard, Phase 8 §3.3): LOF ↔ Hybrid 0.577, Autoencoder ↔ Hybrid 0.497, Isolation Forest ↔ Hybrid 0.452, LOF ↔ OCSVM 0.446, OCSVM ↔ K-Means 0.444; at the bottom, DBSCAN ↔ HDBSCAN 0.023 and DBSCAN ↔ GMM 0.047.

**Metric 4 — Agreement among the four ensemble strategies.** Cheap, and it isolates whether a change is in the detectors or in the aggregation. Baselines from Phase 12 §2: Borda ↔ Percentile ρ=0.9999 / Jaccard 0.953; Weighted Average ↔ PCA Stacking 0.9947 / 0.881; the remaining pairs 0.9818–0.9827 / 0.636–0.691. **Borda ↔ Percentile at 0.9999 is effectively an identity check** — Phase 12 showed the two are the same operation up to normalisation and differ only in how they handle LSTM-AE's missing rows. Any material drop in *that* pair is a pipeline bug, not drift.

### 3.3 Concept-drift triggers

| Trigger | Threshold | Rationale |
|---|---|---|
| Mean pairwise Spearman drops below **0.40** | ~11% relative below the 0.452 baseline | A field-wide loss of agreement; the consensus that defines the score is weakening |
| Mean pairwise Spearman rises above **0.55** | ~22% relative above baseline | Investigated with equal seriousness, and this is the counter-intuitive one worth building in: if every detector starts agreeing, the ensemble's diversity — the entire reason Phase 14 recommended it over a single model — has collapsed, and eleven models are now doing one model's job at eleven models' cost |
| PC1 explained variance leaves **[45%, 60%]** | Against the 52.65% baseline | Same logic in a single number: below 45% the field has fragmented, above 60% it has homogenised |
| Any tracked pairwise ρ moves by more than **0.15** absolute | Against its Phase 8 anchor | Localises a field-wide change to a specific relationship |
| Autoencoder ↔ VAE ρ drops below **0.70** | From 0.801 | Treated as a **pipeline alert, not a drift alert** — these two share an architecture and a split and should not diverge from data alone |

**Cadence:** monthly, or on each scoring batch large enough for a stable rank correlation. Spearman over eleven score vectors needs a reasonable number of rows to be meaningful; a nightly batch of a handful of transactions cannot support it, so accumulate to a fixed window (a suggested floor of 500 rows, roughly the size of the 503-row validation split these models were assessed against) rather than computing it on whatever arrived.

---

## 4. Model Drift and Retraining Triggers

### 4.1 Trigger criteria

A retrain is warranted when **any one** of the following holds. Each is a composite designed not to fire on a single noisy window.

| # | Trigger | Threshold | Source |
|---:|---|---|---|
| 1 | PSI > 0.20 on **any Class C amount-relative feature**, sustained over **2 consecutive** monitoring windows | Industry PSI band (§2.1) | Phase 11 §1 — these features dominate the Autoencoder's score |
| 2 | PSI > 0.20 on **three or more** Class C features simultaneously in one window | Same band, breadth rather than persistence | A broad shift needs less persistence evidence than a narrow one |
| 3 | Mean pairwise Spearman outside **[0.40, 0.55]** | §3.3, against the 0.452 baseline | Phase 8 §3.2, Phase 12 §1.1 |
| 4 | PC1 explained variance outside **[45%, 60%]** | §3.3, against 52.65% | Phase 12 §1.4 |
| 5 | Flagged rate outside **[3.0%, 8.0%]** at the fixed 0.8406 threshold, for **3 consecutive** windows | Against the 5.02% design point | §5.1 |
| 6 | Any training-derived constant moves > **20%** | §2.4 | The `$14.60` floor especially |
| 7 | **Time-based floor: retrain at least quarterly regardless.** | — | The 5% contamination assumption is documented as unverified (`src/config.py`, `LIMITATIONS.md`); a system resting on an unverified assumption should not go a year without being re-examined |
| 8 | Any Class A feature's PSI **fails to** breach on the first production window | one-time | Inverted on purpose: if `DeviceNoveltyFlag` is still 1 for 99.5% of production rows, that is evidence of a feature-engineering bug, not of a genuinely matching distribution |

### 4.2 The counter-trigger: retraining is not free

This is the part most retraining policies omit, and Phase 10 §2 measured the reason.

Isolation Forest, LOF and the Autoencoder were each refit on 5 bootstrap resamples of the same 2,009-row training split and rescored. Mean pairwise Jaccard across the flagged sets: **LOF 0.590** (min 0.465, max 0.703), **Autoencoder 0.533** (0.448–0.658), **Isolation Forest 0.527** (0.448–0.565).

**Roughly 41–47% of the flagged transactions change between retrains trained on resamples of the same underlying data, with no drift at all.** Phase 10's diagnosis is that the top-5% cut sits on a graded distribution rather than on a small set of stark outliers, so small shifts in the fitted boundary move a substantial number of borderline transactions across it.

Three policy consequences:

1. **Do not retrain on a marginal trigger.** A single window at PSI 0.21 costs less to leave alone than the ~45% queue churn a retrain imposes. That is why triggers 1 and 5 require persistence.
2. **A fixed model artifact plus a monitored, versioned retraining process is the correct posture** — Phase 10 §2 says exactly this — rather than continuous or automatic retraining.
3. **Every retrain must be measured against its predecessor before promotion.** Compute the flagged-set Jaccard between the candidate and incumbent models. If it falls **below 0.45** — outside the range bootstrap resampling alone produced — the new model is doing something materially different and needs review before it reaches the queue, not after.

### 4.3 Retraining protocol

1. Refit on a window that includes the drift, preserving the pipeline invariants: chronological sort, `closed='left'` and `shift()` leakage safety, `RobustScaler` fit on the training split only, the 46-column ordered schema.
2. Recompute every training-derived constant in §2.4 — especially the `Amount_ZScore_Account` floor, which is 5% of the *training* amount std and must not be inherited.
3. Recompute the full pairwise Spearman/Jaccard matrices and the PC1 variance; these become the new concept-drift baselines, replacing 0.452 and 52.65%.
4. Recompute the percentile thresholds. The 0.9145 and 0.8406 cut points are percentiles of a *specific* score from a *specific* model set and do not survive a refit unexamined.
5. Compare candidate against incumbent (§4.2 point 3), and version both so a rollback is a pointer change (Phase 15 §9).
6. Re-run the Phase 10 §2 bootstrap on the new artifacts to refresh the stability baseline — and, per Phase 14 §5's open gap, run it **through the ensemble aggregation** this time, which has never been measured.

---

## 5. Alert-Volume Drift and Flagged-Set Consistency

Volume alone is an insufficient monitor here, and Phase 10 §2 is the reason: a queue can hold a perfectly steady 5% while consisting of almost entirely different transactions week to week. **Both must be monitored, and they are different alarms.**

### 5.1 Volume

Design points, from Phase 13 §2: at the fixed threshold 0.8406 the training data flags **126 transactions (5.02%)**; at 0.9145, **26 (1.04%)**.

| Monitor | Band | Reading if breached |
|---|---|---|
| Flagged rate at 0.8406 | Alert outside **[3.0%, 8.0%]**; retraining trigger if outside for 3 consecutive windows | A drop toward 3% means the score has compressed and genuine anomalies are being missed; a rise toward 8% means the queue is filling with transactions the training distribution would have considered normal |
| Flagged rate at 0.9145 | Alert outside **[0.5%, 2.0%]** | The priority tier is small enough that absolute counts matter as much as rates — 26 transactions over 364 days here |
| Ratio of standard-tier to priority-tier volume | Baseline **126 / 26 ≈ 4.8**; alert outside [3.5, 6.5] | Shape-sensitive: this moves when the score's tail changes even if the total flagged count does not |

**Read the absolute numbers as this sample's numbers only.** Phase 13 §4 is explicit: 2,512 transactions over 364 days is 6.90 transactions/day, so the "0.35 flagged transactions/day" figure at the 95th percentile describes this research sample and nothing else. **The rates generalise; the daily counts do not.** At 1M rows, 5.02% is 50,000 alerts, which is a staffing question rather than a monitoring one (Phase 15 §10.4).

### 5.2 Flagged-set consistency — an explicit requirement, not a nice-to-have

Two distinct measurements that are easy to conflate:

**(a) Same model, consecutive windows — expected Jaccard ≈ 1.0 on the overlapping rows.** A deployed model scoring the same transaction twice must produce the same score. Any change in an already-scored row's flag status, with no retrain, is **a bug** — a refit scaler, a reordered feature matrix, a changed constant — not drift. This is the cheapest and most valuable check in the framework and it needs no statistics at all (§6.2).

**(b) Across a retrain — expected Jaccard 0.53–0.59.** This is the measured band from Phase 10 §2. Its purpose is calibration: when a retrain changes 45% of the queue, that is *normal*, and an operations team that has not been told this will reasonably conclude the system is broken. Publishing the expected band alongside the retrain is a communication requirement as much as a technical one.

| Monitor | Baseline | Alert |
|---|---|---|
| Run-to-run Jaccard, same model, overlapping rows | 1.0 | **Any deviation** → pipeline integrity incident |
| Retrain-to-retrain Jaccard, priority + standard tiers | 0.53–0.59 (Phase 10 §2) | Below **0.45** → candidate model held for review before promotion (§4.2) |
| Retrain-to-retrain Jaccard, priority tier alone | Not measured — establish on first retrain | The top 1% should be *more* stable than the 5% tier, since it sits further from the graded boundary. Worth measuring; do not assume it |

### 5.3 Composition monitoring

Beyond count and identity, watch what the queue is *made of*, against Phase 10 §4's characterisation of the tiers:

- **Mean `Amount_ZScore_Account` among flagged transactions.** Phase 10's top-1% tier was dominated by extreme personal-baseline deviations (118×, 149×, 144× account averages). If the flagged set's mean z-score falls sharply, the score has stopped keying on the amount-magnitude signal that produced its most plausible cases.
- **Share of flagged transactions with `ElevatedLoginFlag = 1`.** Against a 4.86% population base rate. `TX000275` — 5 login attempts plus a transaction worth 3.6× the account balance — was Phase 10's and Phase 11's strongest single Scenario 1 match. A rising share here is a genuine account-takeover signal, distinct from model drift.
- **Share of flagged transactions driven by a single detector rather than consensus.** Computed from the per-model percentile vector each alert already carries (Phase 15 §6). A rising share of single-detector flags is the `TX000566` failure mode returning at scale, and it is measurable directly.

---

## 6. Pipeline Integrity

### 6.1 Silent model dropout — the failure this architecture is most exposed to

Percentile aggregation skips missing models and renormalises over those available (Phase 12 §1.3). That property is a deliberate strength — it is why the score is defined for all 2,512 rows despite LSTM-AE covering only 2,402 — but it has a sharp edge in production: **if a model fails to load, times out, or throws, the score is still produced, still lands in (0,1), and still looks entirely normal.** Nothing errors. An ensemble quietly running on six of nine models produces a plausible dashboard.

**Required monitor:** emit `n_models_contributing` with every scored row and alert on any value below the expected count. Expected is 9 for accounts with ≥3 transactions and 8 for the ~4.4% of rows LSTM-AE cannot score (Phase 8 §2.11) — so the alert condition is "below 8", plus a separate check that the share of 8-contributor rows stays near the 4.4% baseline. Without this metric, model dropout is invisible.

### 6.2 Canary set — score reproducibility

Freeze a fixed sample of scored transactions (500 rows is sufficient and matches the 503-row validation split size) with their scores under the current model version. Re-score the canary on **every** run and assert exact reproduction.

This catches, at near-zero cost, the entire class of failures that statistical monitoring cannot: a refit rather than reloaded scaler, a permuted feature matrix, a stale artifact, a library upgrade changing a numerical path, a changed constant. It is the direct implementation of §5.2(a) and it should be the first thing built (§8).

### 6.3 Feature-engineering assertions

Phase 5 verified its leakage safety by recomputing `TimeSinceLastTxn` independently and confirming a row-for-row match against `artifacts/features.csv`. That check should run on every batch, not once. Add the schema assertions Phase 8 §0 already relies on: the feature matrix has exactly 46 columns in the `autoencoder_config.json` order, and it contains no `vote_count` / `risk_tier` / `is_fraud` column (`07_models_classical.py::load_and_split` asserts this today — keep it in production, where the consequence of a label leaking into the feature matrix is worse).

---

## 7. Monitoring Scorecard

One page, produced per window. Every baseline below is a measured number from this project.

| Metric | Baseline | Green | Amber | Red |
|---|---|---|---|---|
| Max PSI, Class C features | 0 (self) | < 0.10 | 0.10–0.20 | > 0.20 |
| Count of Class C features with PSI > 0.20 | 0 | 0 | 1–2 | ≥ 3 |
| Max KS D, Class C features | 0 (self) | < 0.10 | 0.10–0.20 | > 0.20 |
| `ElevatedLoginFlag` rate | 4.86% | 3–7% | 7–10% | > 10% |
| Mean pairwise Spearman (55 pairs) | 0.452 | 0.42–0.50 | 0.40–0.42 or 0.50–0.55 | < 0.40 or > 0.55 |
| PC1 explained variance | 52.65% | 48–58% | 45–48% or 58–60% | < 45% or > 60% |
| AE ↔ VAE Spearman | 0.801 | > 0.75 | 0.70–0.75 | < 0.70 (pipeline alert) |
| Flagged rate @ 0.8406 | 5.02% | 4–6% | 3–4% or 6–8% | < 3% or > 8% |
| Flagged rate @ 0.9145 | 1.04% | 0.7–1.5% | 0.5–0.7% or 1.5–2.0% | < 0.5% or > 2.0% |
| Canary reproduction | exact | exact | — | any mismatch |
| `n_models_contributing` below 8 | 0 rows | 0 | — | any row |
| Rows with 8 contributors (LSTM-AE gap) | 4.4% | 3–6% | 6–10% | > 10% |
| Training-derived constant drift | 0% | < 10% | 10–20% | > 20% |
| Retrain-to-retrain flagged Jaccard | 0.53–0.59 | ≥ 0.50 | 0.45–0.50 | < 0.45 |

---

## 8. What to Build First

If only one monitor ships, it should be **PSI on the amount-relative feature group** — `Amount_ZScore_Account`, `Amount_vs_AccountAvg`, `Amount_to_Balance_Ratio`, `Amount_to_RollingMean_Ratio` — evaluated weekly against the frozen training deciles, alerting at PSI > 0.20 on any of the four.

The reasoning, in order:

1. **These four features are the Autoencoder's top four SHAP drivers** (0.0390, 0.0367, 0.0367, 0.0360 — Phase 11 §1). Drift here moves the score more than drift anywhere else.
2. **They produced every plausible fraud read the project has.** Phase 10 §4's defensible cases were all amount-relative: 117.9×, 149.0× and 143.9× account averages, and a transaction worth 3.6× an account balance.
3. **They sit on top of a training-derived constant that fails silently.** The `$14.60` z-score floor is 5% of the training amount standard deviation. If amount scale moves, the floor is wrong, and Phase 6 §7.3 documents exactly how badly that can go — the pre-fix version produced feature values in the hundreds of millions and swamped the autoencoder's entire loss. Feature-level PSI is the only monitor positioned to catch the onset of that.
4. **They are the features most likely to genuinely drift.** Amounts move with inflation, seasonality, product mix and customer base. Unlike the Class A temporal and network features, drift here would be real rather than an artifact correction.

**Build second, and immediately after:** the canary reproducibility check (§6.2) and `n_models_contributing` (§6.1). Neither requires any statistics, both take an afternoon, and between them they catch every silent-failure mode this architecture has.

---

## 9. Honest Limits of This Framework

Stated so nobody mistakes a well-instrumented system for a validated one.

- **None of these monitors can detect that the system is failing to catch fraud.** They detect that inputs changed, that detectors stopped agreeing, that volume moved, or that the pipeline broke. A system that never caught any fraud would show green across this entire scorecard. Only validation against real investigator-labelled outcomes closes that gap, and `LIMITATIONS.md` already names it as the first pre-deployment requirement.
- **The concept-drift proxy is circular, by necessity.** Measuring whether the ensemble's own consensus has weakened uses the ensemble as its own reference. It detects *change*, not *error*. It cannot tell you the consensus was wrong on day one.
- **The ensemble strategies have no measured stability baseline.** Phase 10 §2's 0.527–0.590 band is for three individual models. The retrain-to-retrain Jaccard band in §5.2 is inherited from those three and applied to an aggregate that has never been bootstrap-tested — Phase 14 §5 flags this as the highest-value missing measurement in the project, and §4.3 step 6 folds it into the retraining protocol.
- **Every threshold here is a starting point, not a calibration.** The PSI bands are an industry convention; the Spearman and PC1 bands are reasoned margins around measured baselines. All of them should be re-set after the first few production windows establish what normal variation actually looks like — and the Class A gate (§2.3) means the first window is likely to invalidate several of these baselines outright, by design.

*Next: `research/15_final_research_report.md` (Phase 17).*
