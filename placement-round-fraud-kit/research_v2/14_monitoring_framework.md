# Phase 16 (v2) — Monitoring Framework (Teammate's 18-Feature Pipeline)

This phase specifies what to monitor once the Phase 15 (v2) architecture is running, what thresholds to alarm on, and — as important — which of the 18 features will produce *false* drift alarms if monitored naively, so that a monitoring rollout does not spend its first month being ignored.

**Nothing here is computed.** Every baseline quoted is a number already measured in Phases 5–13 (v2) and cited to the report that produced it. Where a baseline does not exist, the section says so and states what would have to be measured to establish one.

---

## 0. Why Monitoring Carries More Weight on *This* Feature Set Than on the In-House One

Both pipelines need monitoring. This one needs a **different** monitoring design, and the reason is structural rather than a matter of degree.

The in-house 46-feature set is built from **per-account history**: expanding means, rolling windows, time-since-last-transaction, first-seen-device novelty flags. Those features are self-referential — a customer's `Amount_ZScore_Account` is computed against that same customer's own past, so it stays meaningful even as the surrounding population changes completely. The in-house Phase 16's headline fragility was a *temporal-export artifact* (`PreviousTransactionDate` clustering at a single 2024-11-04 moment) and the general risk of training-derived constants going stale.

**This feature set is built from population statistics.** Five of the eighteen features — `account_frequency`, `device_frequency`, `ip_frequency`, `merchant_frequency`, `Location_FE` — encode *how common a category is across the whole dataset*. That has a consequence with no in-house analogue:

> **The encoded value of a category changes when the population changes, even if nothing about that category's own behaviour changed at all.**

If a bank onboards 40,000 new devices next quarter, `device_frequency` shifts for **every device already in the system**, including devices whose usage was completely stable. Every downstream model sees a shifted input distribution. Every model's flagged set moves. No customer did anything different.

This is not a hypothetical failure mode; it is the arithmetic of frequency encoding. It means that on this pipeline:

1. **Feature drift on the frequency columns is expected continuously, not exceptionally.** A monitoring design that alarms on it will alarm constantly.
2. **The right question is not "did the frequency distribution shift" but "did it shift more than population growth alone explains".** §2.3 Class A builds that comparison explicitly.
3. **Retraining is the *normal* remedy here, not the emergency one.** On the in-house feature set a PSI breach is a signal something went wrong; here, on the frequency columns, it is mostly a signal that the encoding is due for a refresh.

Three further reasons monitoring carries unusual weight, shared with the in-house pipeline but sharper here:

- **There is no fraud label**, so there is no accuracy metric to degrade visibly. Every warning this system will ever give is a proxy warning.
- **Retrain-to-retrain instability is measured and worse here.** Phase 10 (v2) §2 measured mean bootstrap flagged-set Jaccard at **0.6021 (Isolation Forest), 0.5124 (LOF), 0.3726 (Autoencoder)** — the Autoencoder's floor of 0.2115 in the worst observed pair means fewer than 1 in 4 flagged transactions survived a retrain. The in-house pipeline's equivalent range was a tighter 0.527–0.590. **Retraining this pipeline changes who gets reviewed, substantially, and that must be a monitored, deliberate event.**
- **The ensemble score is defined by its member list** (Phase 15 v2 §9). An 11-member and a 9-member `ensemble_percentile_average` are different numbers under the same column name, and a silently-dropped member produces a plausible score with no error.

---

## 1. The Five Monitoring Layers

| Layer | Question it answers | Can it be measured without a label? |
|---|---|---|
| **1. Feature drift** | Are the inputs still shaped like the training data? | **Yes**, fully — PSI and KS on the 18 columns |
| **2. Concept drift** | Has "normal" itself moved, so that the model's notion of anomalous is stale? | **Only by proxy** — §3 |
| **3. Model drift** | Is a specific model behaving differently from when it was validated? | **Yes**, via score-distribution and canary reproducibility |
| **4. Alert-volume drift** | Is the review queue still the size the operating plan assumed? | **Yes**, directly |
| **5. Pipeline integrity** | Is the pipeline actually doing what it thinks it is doing? | **Yes**, and this is the cheapest and most valuable layer |

Layers 1, 4 and 5 are objectively measurable and should be built first. Layer 2 is proxy-only and must be labelled as such wherever it is displayed. Layer 3 sits between the two.

---

## 2. Feature Drift

### 2.1 PSI — method and thresholds

Population Stability Index, computed per feature against a frozen reference distribution taken from the 2,009-row training split (`artifacts_research_v2/models/shared_robust_scaler.pkl` was fit on exactly this split, so it is the correct reference population):

```
PSI = Σ_bins (actual% − expected%) × ln(actual% / expected%)
```

Ten equal-frequency bins derived from the **reference** distribution, frozen, with edges stored as an artifact. Bins must never be recomputed from the current batch — recomputing them makes PSI structurally incapable of detecting a shift, which is a classic and silent monitoring failure.

Industry-standard action bands, adopted unchanged:

| PSI | Reading | Action |
|---|---|---|
| < 0.10 | No material shift | None |
| 0.10 – 0.25 | Moderate shift | Investigate; note in the weekly review |
| > 0.25 | Significant shift | Escalate; treat as a retraining candidate |

**One adjustment specific to this feature set**, stated up front so it does not look like grade inflation later: for the five frequency-derived columns (§2.3 Class A) the >0.25 band is a *retraining* trigger, not an *incident* trigger. For the six behavioural columns (Class C) it is an incident trigger. The same number means two different things depending on which column produced it, because the underlying mechanism is different.

### 2.2 KS two-sample statistic

The Kolmogorov–Smirnov statistic (max absolute difference between empirical CDFs) is run **alongside** PSI on the continuous features, not instead of it, because the two fail differently:

- PSI is binned, so it is blind to movement *within* a bin and sensitive to bin-edge placement.
- KS is unbinned and detects any distributional difference, but its p-value is essentially useless at production batch sizes — with tens of thousands of rows, statistically significant differences that are operationally meaningless are guaranteed.

**Use the KS statistic, ignore the KS p-value.** Alarm at D > 0.10 for the Class C behavioural features, with the same Class A caveat as above. KS is not meaningful for the seven binary/dummy columns (§2.3 Class B).

### 2.3 Feature triage — which of the 18 to monitor, how, and with what expectation

The 18 features do not all behave the same way under drift monitoring, and treating them uniformly is the fastest way to get a monitoring dashboard ignored.

#### Class A — Frequency encodings. Expected to drift continuously; monitor the *mechanism*, not just the number.

`account_frequency`, `device_frequency`, `ip_frequency`, `merchant_frequency`, `Location_FE` — **5 features**

These are the defining fragility of this pipeline (§0). Monitor all five with PSI, but **pair every PSI reading with the population statistic that drives it**, so a shift can be attributed rather than merely detected:

| Feature | Pair its PSI with | What the pairing distinguishes |
|---|---|---|
| `device_frequency` | count of distinct `DeviceID`s in the window; share of transactions on devices unseen in the reference period | Population growth (expected, retrain) vs. a genuine shift in device-sharing behaviour (investigate — this is the mule-account signal, Phase 1 Scenario 3) |
| `ip_frequency` | count of distinct `IP Address`es; share on unseen IPs | Same, plus IP-range reallocation, which looks identical to churn but is an infrastructure event |
| `merchant_frequency` | count of distinct `MerchantID`s; merchant onboarding/offboarding counts | A merchant closing shifts every other merchant's encoded value; that is bookkeeping, not fraud |
| `account_frequency` | count of distinct `AccountID`s; mean transactions per account | Mean transactions/account was **5.075** (min 1, max 12, median 5) at training time (Phase 5 v2 §2.6). A shift in this single number moves the whole feature |
| `Location_FE` | count of distinct `Location` values (**43** at training time); share of transactions at unseen locations | Branch/geography expansion vs. genuine geographic shift in customer behaviour |

**The unseen-category rate is the metric that actually matters for this class**, more than PSI itself. A category the frozen lookup table has never seen has no encoded value. The policy for it (§2.4) is a training-derived constant, and the *rate* at which it fires is the direct early-warning signal that the encoding is stale. **Alarm at >2% of rows in a batch hitting the unseen-category default for any of the five.** No baseline for this exists — by construction it was 0% on the training data — so 2% is a starting value to be tuned in the first quarter of operation, and it is labelled as such rather than presented as derived.

#### Class B — Binary and one-hot. PSI is degenerate; monitor the rate directly.

`high_amount_transaction`, `TransactionType_Debit`, `Channel_Branch`, `Channel_Online`, `CustomerOccupation_Engineer`, `CustomerOccupation_Retired`, `CustomerOccupation_Student` — **7 features**

A 10-bin PSI over a two-valued column is meaningless. Monitor the positive rate against its training value with a simple proportion test:

| Feature | Training rate | Note |
|---|---:|---|
| `high_amount_transaction` | **5.02%** (126/2,512) | **The single most important rate in this class.** It is a *frozen $878.18 dollar threshold*, not a recomputed percentile (Phase 5 v2 §2.4). Under amount inflation its rate must rise — that is the feature working correctly. A rate that stays pinned near 5.02% is evidence the threshold is being silently recomputed per batch, which is a bug (§2.4) |
| `TransactionType_Debit` | **77.4%** (1,944/2,512) | Credit is the dropped baseline |
| `Channel_Branch` / `Channel_Online` | 34.6% / 32.3% | ATM (33.2%) is the dropped baseline; monitor all three shares together |
| `CustomerOccupation_*` | Student 26.2%, Doctor 25.1%, Engineer 24.9%, Retired 23.8% | Monitor all four shares. **Watch the Student share specifically**: Phase 7 (v2) §7.2/7.3 found the student segment forms its own distinct region in both UMAP and t-SNE, and Phase 10 (v2) §4 found the weak tail of Isolation Forest's top-10% is disproportionately Student. A shift in this demographic mix will move flagged-set composition without any change in fraud |

Alarm at a ±5 percentage-point absolute move sustained over two consecutive batches — sustained, because a single batch of a few thousand rows moves these rates by chance.

#### Class C — Core behavioural features. These are the ones a genuine drift alarm should come from.

`TransactionAmount`, `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance`, `amount_to_balance_ratio` — **6 features**

These are the columns where PSI > 0.25 means *something has actually changed about customer behaviour*, and they are the only class where a breach should page someone.

| Feature | Reference facts from this pipeline | What a breach would mean |
|---|---|---|
| `TransactionAmount` | Frozen `StandardScaler` reference; the raw 95th percentile is $878.18 | Amount inflation, product-mix change, or a new customer segment |
| `amount_to_balance_ratio` | Raw ratio mean 0.200, median 0.052, max 7.896 (Phase 5 v2 §2.3); the strongest single loading on PC1 (+0.594) | The feature both SHAP models agreed was the dominant driver of `TX000275` and `TX000935` (Phase 11 v2 §2). Drift here directly moves the top of the alert queue |
| `LoginAttempts` | Near-discrete, mostly 1, up to 6.43 in z-score units; loads on **PC3** (+0.502), a security axis nearly orthogonal to the amount axes (Phase 7 v2 §7.1) | Either an authentication-system change (investigate the system, not the customers) or a genuine credential-attack shift — Phase 1 Scenario 1. **The most operationally interesting single feature to monitor in the whole set** |
| `TransactionDuration` | Phase 10 (v2) §4 found `TX001903` flagged essentially on a 37-second duration alone | Channel/UX changes move this feature wholesale and will move the flagged set with it |
| `AccountBalance` | Loads −0.520 on PC1 | Portfolio composition change |
| `CustomerAge` | Loads −0.471 on PC1; z-mean −1.162 within the student cluster | Demographic drift; interacts with the Class B occupation shares |

#### Summary

| Class | Count | Method | Breach means |
|---|---:|---|---|
| A — frequency encodings | 5 | PSI + paired population statistic + unseen-category rate | Usually: refresh the encoding. Sometimes: a real sharing-behaviour shift |
| B — binary/one-hot | 7 | Positive-rate proportion test | Mix shift; check `high_amount_transaction` first as a bug detector |
| C — behavioural | 6 | PSI + KS | Genuine behavioural change — this is the class that should page someone |

### 2.4 Training-derived constants — the silent failure class

This pipeline contains constants that were derived from the training data and **must be frozen**. Each one fails silently — producing plausible numbers with no error — if it is recomputed per batch instead:

| Constant | Value / source | Failure mode if recomputed |
|---|---|---|
| `high_amount_transaction` threshold | **$878.18** (raw 95th percentile, recovered empirically in Phase 5 v2 §2.4) | The flag becomes "top 5% of *this batch*", so its rate is pinned at 5% forever and it can never detect amount inflation. **Detection: monitor the flag's rate; a rate that never moves is the symptom** |
| `Location_FE` lookup table | 43 locations → training proportions | Recomputing per batch means the same location gets a different encoded value in every batch. **Detection: score a canary set (§6.2) and diff** |
| `StandardScaler` means/stds (18 columns) | The teammate's upstream fit | Recomputing re-centres every batch on itself, destroying any ability to see a distribution move. **Detection: the scaled mean of every column will sit at exactly 0.0000 in every batch — that is the tell** |
| `RobustScaler` medians/IQRs | `models/shared_robust_scaler.pkl`, train-split fit | Same failure, one layer down |
| One-hot category lists | Credit / ATM / Doctor as dropped baselines | An unseen category silently encodes as all-zeros, which is indistinguishable from the dropped baseline |
| Percentile thresholds | 0.9510 / 0.8671 (Phase 13 v2 §2) | If recomputed per batch, the alert volume is pinned at exactly 1%/5% by construction and alert-volume monitoring (§5) becomes structurally incapable of detecting anything |

**The last row is the one to build a test for first.** It is the most tempting shortcut in the whole pipeline ("just take the top 1% of today's batch"), and taking it would disable an entire monitoring layer while looking completely reasonable in code review.

---

## 3. Concept Drift Without a Label

### 3.1 The problem, stated precisely

Concept drift is a change in the relationship between features and the target. **There is no target.** So concept drift is not directly measurable here, and any dashboard that claims to show it is claiming something it cannot support.

What *is* measurable is whether the model's own notion of "normal" has moved. Three proxies are available, all of which must be labelled as proxies wherever they are displayed.

### 3.2 The proxy metrics and their baselines

| Proxy | Baseline from this pipeline | Trigger |
|---|---|---|
| **Autoencoder reconstruction error distribution** | Train MSE **0.2858**, val MSE **0.2966**, val P95 **0.5551**, val P99 **0.6339**, val max **0.9633** (Phase 7 v2 §7.4 / Phase 10 v2 §3) | Batch mean MSE > 0.40 (≈ +35% over val) for two consecutive batches, **or** batch P99 exceeding the val maximum of 0.9633. Rising reconstruction error on data the model was not trained on is the cleanest label-free signal that "normal" has moved |
| **VAE reconstruction error** | Train 0.3371, val 0.3458, val P95 0.6394, val P99 0.8484, val max 1.2137 | Same shape of trigger. Useful mainly as a corroborating second opinion — note it reconstructs worse than the AE at *every* percentile on this feature set (Phase 8 v2 §2.10), so it is a confirmatory signal, not an independent one |
| **Cross-model rank agreement** | The full Spearman matrix, `artifacts_research_v2/model_pairwise_spearman.csv`. Anchor pairs: LOF↔VAE **0.839**, LOF↔K-Means **0.838**, AE↔VAE **0.837**, HDBSCAN↔K-Means **0.832**. Anchor low: DBSCAN↔AE **0.153** | A drop of >0.15 in any anchor pair, or the mean pairwise agreement moving materially, indicates the models are no longer seeing the same structure. **This is the most powerful proxy available**, because it needs no label at all — it asks whether eleven independently-motivated detectors still agree with each other |

**A methodological note that matters for implementing the third proxy.** Phase 8 (v2) §3.2/§3.3's published "mean pairwise" figures include each model's self-correlation of 1.0 in a 12-way average (documented as Inconsistency 1 in Phase 14 v2 §5). A monitoring implementation must use the **self-excluded** pairwise mean, and must compare against the self-excluded baselines (DBSCAN 0.235, K-Means 0.670, HDBSCAN 0.665, Elliptic Envelope 0.602), not the published inflated ones. Comparing a correctly-computed production value against an inflated baseline would show a permanent phantom drop of roughly 0.03–0.06.

### 3.3 What concept drift would actually look like here

Given this feature set's specific structure, the two most plausible real concept-drift scenarios are worth naming so they are recognised rather than debugged from scratch:

1. **Fraud migrates to a channel the frequency features cannot see.** These 18 features have no per-account novelty flags (Phase 5 v2 §3). If attackers move to using *high*-frequency devices and *high*-frequency merchants — deliberately blending into the population — every population-level feature in this set reads them as ordinary. **Nothing in this monitoring framework would detect that**, and no proxy metric would move. This is the honest limit of a population-statistics feature set, and it is why §8's first recommendation is to capture investigator labels.
2. **The student demographic segment grows or shrinks.** Phase 7 (v2) identified it in two independent projections; Phase 8 (v2) §1.7 found K-Means splits on it at k=2; Phase 10 (v2) §4 found it disproportionately represented in the weak tail of the alert queue. A shift in this segment moves flagged-set *composition* without any change in fraud, and it will look like drift. Monitoring the Class B occupation shares (§2.3) is what distinguishes it.

---

## 4. Model Drift and Retraining Triggers

### 4.1 Trigger criteria

Retraining is triggered by **any** of:

| # | Trigger | Threshold | Source of the baseline |
|---:|---|---|---|
| 1 | PSI > 0.25 on any **Class C** behavioural feature | Sustained over two batches | §2.3 |
| 2 | PSI > 0.25 on any **Class A** frequency feature | Sustained over two batches — a *scheduled-refresh* trigger, not an incident | §2.3, §0 |
| 3 | Unseen-category rate > 2% for any Class A feature | Single batch | §2.3 (starting value, not yet calibrated) |
| 4 | Autoencoder batch mean MSE > 0.40 | Two consecutive batches | §3.2 |
| 5 | Anchor Spearman pair drops > 0.15 | Single batch, investigate immediately | §3.2 |
| 6 | Alert volume outside its band | §5.1 | §5.1 |
| 7 | Scheduled | Quarterly, regardless of the above | — |

Trigger 7 exists specifically because of §0: on a frequency-encoded feature set, the encodings go stale as a *matter of course*, not as a matter of failure. A quarterly refresh is the baseline maintenance cadence, and triggers 2 and 3 are early signals that it should happen sooner.

### 4.2 The counter-trigger: retraining is not free, and here it is measurably expensive

**This is the single most important operational fact in this framework**, and it is stronger on this feature set than on the in-house one.

Phase 10 (v2) §2 measured, directly, what a retrain does to the flagged set — refitting the same model with the same hyperparameters on a bootstrap resample of the same data:

| Model | Mean pairwise Jaccard across 5 refits | Min | Max |
|---|---:|---:|---:|
| Isolation Forest | 0.6021 | 0.5090 | 0.6689 |
| LOF | 0.5124 | 0.4651 | 0.5750 |
| **Autoencoder** | **0.3726** | **0.2115** | 0.5849 |

Read the Autoencoder row plainly: **in the worst observed retrain pair, fewer than one in four flagged transactions was shared between the two runs.** Nothing changed except which rows the resample happened to include. The in-house pipeline's equivalent spread was 0.527–0.590 — noticeably tighter.

Consequences, all of which should be written into the operating procedure:

1. **A retrain is a change to who gets investigated.** It is not a maintenance action; it is a policy change with a human-workload consequence.
2. **Do not retrain on a single trigger firing once.** Every trigger above except #5 requires two consecutive batches for exactly this reason.
3. **Isolation Forest is the retrain-stable anchor on this feature set** (0.6021, the best measured). If a retrain must be evaluated quickly, evaluate it on Isolation Forest's flagged-set overlap first — it has the least intrinsic churn to see through.
4. **The Autoencoder needs the tightest change control of any model here.** This is a direct reversal of the in-house pipeline, where the Autoencoder sat mid-pack and LOF was the stable anchor.

### 4.3 Retraining protocol

1. **Freeze the incumbent.** Version the full artifact set (Phase 15 v2 §9), including scalers, the frequency lookup tables, the frozen constants and the ensemble member manifest.
2. **Refit on the new window** using the identical configuration recorded in `model_summary_classical.json` / `autoencoder_config.json` / `vae_config.json`.
3. **Recompute the ensemble** with the same member list, using the same skip-and-renormalise rule (Phase 12 v2 §1.3).
4. **Measure the change before shipping it.** Spearman and top-5% Jaccard of the new score against the incumbent's, on the overlapping rows — the same two measures Phase 12 (v2) §2 already uses. **Compare against the retrain-churn baselines in §4.2**: a Jaccard around 0.60 for Isolation Forest is *ordinary resample noise*, not evidence of drift. A Jaccard well below the model's measured floor is the signal.
5. **Re-derive the thresholds.** The percentile cuts are properties of the score distribution and do not transfer automatically.
6. **Diff the alert queue and have it reviewed by a human before cutover** — specifically, the transactions that *leave* the priority tier, which nobody would otherwise look at.
7. **Keep the incumbent scoreable in parallel for one cycle** so the two can be compared on live data.

---

## 5. Alert-Volume Drift and Flagged-Set Consistency

### 5.1 Volume

The operating plan is 1.04% priority / 5.02% standard (Phase 13 v2 §2). With **frozen** thresholds (§2.4), the realised rate is free to move — and that movement is the signal.

| Metric | Expected | Alarm |
|---|---:|---|
| Priority-tier rate (≥0.9510) | 1.04% | Outside 0.5%–2.0% for two consecutive batches |
| Standard-tier rate (≥0.8671) | 5.02% | Outside 3.0%–8.0% for two consecutive batches |
| Absolute daily priority volume | ~0.07/day in this 6.88 txn/day sample | Any value that exceeds review capacity, regardless of rate |

**The bands above are a starting proposal, not a measured result** — no batch-to-batch variance was measured for this pipeline, because there is only one batch. They should be replaced with empirical bands after the first quarter of operation. Saying so is more useful than presenting an unvalidated number as calibrated.

**If the realised rate is *exactly* 1.04% and 5.02% every single batch, that is a bug, not a success** — it means the thresholds are being recomputed per batch (§2.4).

### 5.2 Flagged-set consistency — an explicit requirement, not a nice-to-have

Volume alone can be perfectly stable while the *identity* of the flagged transactions churns completely. Given §4.2's measured numbers, this is not a theoretical concern here.

**Requirement: on every batch, compute the top-5% Jaccard overlap between consecutive scoring runs on the overlapping row set, per model and for the ensemble.** Interpret against the measured retrain baselines (Isolation Forest 0.6021, LOF 0.5124, Autoencoder 0.3726) — these are the *floor of ordinary noise*, not targets.

**Known gap, carried forward from Phase 14 (v2) §5:** no equivalent baseline exists for any ensemble strategy. The recommendation to use Percentile Aggregation rests partly on the assumption that aggregating 9–11 models damps this churn, and **that assumption has not been tested in this pipeline**. Establishing it is the single highest-value measurement outstanding, and it is cheap — re-run the Phase 10 (v2) §2 bootstrap procedure end-to-end through the Phase 12 (v2) aggregation.

### 5.3 Composition monitoring

Track the composition of the priority tier over time, because a stable count can hide a complete change in what is being reviewed:

- **Occupation mix** — with the Student-segment caveat from §2.3 Class B and §3.3 firmly attached.
- **Channel and transaction-type mix.**
- **Which model drove each flag** — from the per-model percentiles in `ensemble_scores_v2.csv` / `model_scores_all.csv`. If one member starts dominating the top tier, that is a member-level failure the aggregate score will hide.
- **SHAP driver mix** — the distribution of "top driver" across flagged transactions, from `shap_isolation_forest_v2.csv` and `shap_autoencoder_v2.csv`. Baseline from Phase 11 (v2) §1: Isolation Forest's global top drivers are `TransactionType_Debit` (0.391), `CustomerOccupation_Retired` (0.294), `CustomerOccupation_Student` (0.265); the Autoencoder's are `Location_FE` (0.0354), `account_frequency` (0.0304), `merchant_frequency` (0.0302). **If the Autoencoder's driver mix shifts away from the frequency features, the encodings have moved** — this is a second, independent detector for the §0 failure mode, and a good one, because it observes the consequence rather than the cause.

---

## 6. Pipeline Integrity

The cheapest layer to build and the most likely to catch a real incident.

### 6.1 Silent model dropout — the failure this architecture is most exposed to

Percentile aggregation **skips missing members and renormalises** (Phase 12 v2 §1.3). That property is what makes Phase 14 (v2)'s Option B implementable — and it is also a hazard: if a model fails to load or produces `NaN`, **the ensemble score is still produced, still bounded in (0,1), and still looks completely normal.** There is no error and no obvious symptom.

**Requirement: every scoring run must assert its member count and log the member manifest with the score.** A run that aggregated 8 members instead of 9 must fail loudly, not quietly produce a different score under the same column name.

### 6.2 Canary set — score reproducibility

Freeze a fixed set of ~50 transactions with their expected scores under the current artifact set. Re-score them on every run. Any deviation beyond floating-point tolerance means an artifact, a scaler, a frozen constant or a lookup table changed.

**Include `TX000275` in the canary set.** It is the highest-scoring transaction in the dataset (`ensemble_percentile_average` = **0.9951**, rank 1 of 2,512), it was independently surfaced by the in-house 46-feature pipeline's own top-1% tier, and Phase 11 (v2) confirmed both SHAP models agree on why it scores high (`LoginAttempts` +1.714 and `amount_to_balance_ratio` +1.607 for Isolation Forest; `amount_to_balance_ratio` +1.570 for the Autoencoder). If `TX000275` stops ranking at or near the top, something upstream has broken — it is the single best canary in the dataset.

Also include a spread across the score range, and at least one row from each of the three Phase 7 (v2) structural groups (main mass, student cluster, elevated-`LoginAttempts` pocket), so that a break confined to one region of the feature space is still caught.

### 6.3 Feature-engineering assertions — port these, they already exist

`src_research_v2/04_feature_verification.py` already implements every check below, and its outputs are in `phase5_6_feature_verification.json`. They should run in production as **hard assertions on every batch**, not as a one-off research verification:

| Assertion | Reference value | What it catches |
|---|---|---|
| Column layout matches `ID_COLS + FEATURE_COLS_V2` exactly | `config_research_v2.py::load_features_v2()` | Column reorder, rename, or drop |
| 0 missing cells | 0/70,336 at training | Upstream join failure |
| 0 duplicate `TransactionID`s | 0 at training | Double ingestion |
| Row alignment against the raw CSV | `(raw.TransactionID == df.TransactionID).all()` → True | Misaligned merge — the failure that would silently attach every transaction's features to the wrong transaction |
| Continuous columns scaled to the **frozen** reference, not re-centred | mean ≈ 0, std ≈ 1.0002 at training | A refit scaler (§2.4) |
| Binary columns strictly in {0, 1} | 7 columns | A scaler accidentally applied to the one-hots |
| `amount_to_balance_ratio` correlates with raw amount/balance | r = **0.9467** at training | The ratio silently changing definition |
| `high_amount_transaction` boundary sits at the frozen dollar threshold | min flagged $878.63 vs. max unflagged $877.81 at training | The threshold being recomputed per batch |

The fourth row is the highest-value assertion in the table. A misaligned merge produces a fully-populated, correctly-typed, entirely wrong feature matrix, and every downstream metric in this framework would look normal.

---

## 7. Monitoring Scorecard

A single weekly view. Every row is measurable without a label; the "Baseline" column says where the number came from, and rows with no measured baseline say so.

| # | Metric | Baseline | Alarm | Source |
|---:|---|---|---|---|
| 1 | PSI, Class C behavioural features (6) | Training split | >0.25 sustained ×2 → page | §2.3 |
| 2 | PSI, Class A frequency features (5) | Training split | >0.25 sustained ×2 → schedule refresh | §2.3 |
| 3 | Unseen-category rate, Class A | 0% by construction | >2% (uncalibrated starting value) | §2.3 |
| 4 | KS statistic, Class C | Training split | D > 0.10 | §2.2 |
| 5 | Class B positive rates (7) | `high_amount_transaction` 5.02%; `TransactionType_Debit` 77.4%; occupation shares 26.2/25.1/24.9/23.8% | ±5pp sustained ×2 | §2.3 |
| 6 | `high_amount_transaction` rate **never moves** | — | A pinned 5.02% is a bug alarm | §2.4 |
| 7 | Autoencoder batch mean MSE | val 0.2966 | >0.40 ×2 | §3.2 |
| 8 | Autoencoder batch P99 MSE | val 0.6339, val max 0.9633 | P99 > 0.9633 | §3.2 |
| 9 | Anchor Spearman pairs | LOF↔VAE 0.839, LOF↔K-Means 0.838, AE↔VAE 0.837 | drop >0.15 | §3.2 |
| 10 | Mean pairwise Spearman (self-excluded) | K-Means 0.670, HDBSCAN 0.665, DBSCAN 0.235 | material move | §3.2 |
| 11 | Priority-tier rate | 1.04% | outside 0.5–2.0% ×2 (**band not empirically calibrated**) | §5.1 |
| 12 | Standard-tier rate | 5.02% | outside 3.0–8.0% ×2 (**band not empirically calibrated**) | §5.1 |
| 13 | Flagged-set Jaccard, per model, run-to-run | IF 0.6021, LOF 0.5124, AE 0.3726 | below the model's measured floor | §4.2, §5.2 |
| 14 | Flagged-set Jaccard, ensemble | **none — not measured** | cannot alarm until measured | §5.2 |
| 15 | Priority-tier composition | occupation / channel / driving-model / SHAP-driver mix | material move | §5.3 |
| 16 | Ensemble member count | 9 or 11, per manifest | any deviation → **fail the run** | §6.1 |
| 17 | Canary-set score reproducibility | frozen, incl. `TX000275` @ 0.9951 | any deviation beyond FP tolerance | §6.2 |
| 18 | Feature-engineering assertions | §6.3 table | any failure → **fail the run** | §6.3 |

Rows 16, 17 and 18 are hard failures — the run stops. Everything else is a signal.

---

## 8. What to Build First

In order, on a cost-versus-value basis:

1. **The feature-engineering assertions (§6.3).** The code already exists in `src_research_v2/04_feature_verification.py`. It is a day of work to move it into the batch job, and it catches the class of failure that is otherwise invisible.
2. **The ensemble member-count assertion (§6.1).** One line. Prevents a silently different score.
3. **The canary set (§6.2), with `TX000275` in it.** Half a day. Catches every frozen-constant regression at once.
4. **Alert-volume tracking (§5.1) and the `high_amount_transaction` rate bug-detector (§2.4).** Trivial, and immediately meaningful to the operations team who own the queue.
5. **PSI/KS with the Class A/B/C triage (§2).** The real drift layer. Build it triaged from day one — an untriaged version will fire constantly on the frequency columns and be switched off within a month.
6. **The Class A paired population statistics (§2.3).** This is what makes frequency drift *actionable* rather than just visible, and it is what distinguishes this pipeline's monitoring from the in-house one's.
7. **The ensemble stability baseline (§5.2).** Not a monitor — a measurement. It closes the largest evidential gap in Phase 14 (v2)'s recommendation.
8. **Investigator-label capture (Phase 15 v2 §7.5).** Not monitoring at all, but it is the only thing that would let this framework ever measure precision instead of proxies.

---

## 9. Honest Limits of This Framework

1. **It cannot detect that the model is wrong.** Every metric here detects *change*. A model that was flagging the wrong transactions on day one, consistently, would pass every check indefinitely. Only labels fix this.
2. **The alert-volume and unseen-category bands are proposals, not measurements.** There is one batch of data in this project. §5.1 and §2.3 say so at the point of use rather than in a footnote.
3. **No ensemble-level stability baseline exists** (§5.2, Phase 14 v2 §5), so row 14 of the scorecard cannot be alarmed on today.
4. **The blend-in failure mode is undetectable by design** (§3.3). Fraud that deliberately uses common devices, common merchants and common locations is invisible to a feature set built entirely from population frequencies, and no amount of monitoring on these 18 columns will surface it. This is the honest cost of the capability gap Phase 5 (v2) §3 documented, restated here because a monitoring framework is exactly where a reader might otherwise assume it had been solved.
5. **Two of the eleven ensemble members cannot be monitored the way the others are.** DBSCAN and HDBSCAN are refit on the full history every batch (Phase 15 v2 §4), so "score drift" for them conflates a data change with a refit — a confound the other nine do not have.
6. **Retraining is the remedy for most Class A alarms, and retraining itself changes the flagged set by 40–63%** (§4.2). This framework can tell you the encodings are stale; it cannot make refreshing them free.

---

## 10. Handoff to Phase 17

Phase 17 (`research_v2/15_final_research_report.md`) consolidates this pipeline end to end and sets its results honestly against the in-house 46-feature pipeline's. The monitoring-specific inputs it carries forward: this feature set's defining fragility is **frequency-encoding population churn** (§0), its measured retrain churn is **worse** than the in-house pipeline's (§4.2), and its most valuable untaken measurement is **ensemble-level stability** (§5.2).
