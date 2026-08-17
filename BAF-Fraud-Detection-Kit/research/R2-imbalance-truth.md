# R2 — The Imbalance Truth

**Question:** the competition objective statement says "data balancing." Should you SMOTE 1M rows at 1.1% positives before feeding LightGBM?

**Answer: no — and you can prove it on their own data in about 40 minutes.** This file gives you the evidence, the mechanism, the ablation that demonstrates it, and the exact words to say when a judge asks why you ignored the brief. (You won't have ignored it. You'll have done balancing in the one place it actually changes the outcome.)

---

## 0. The verdict box

Do these in order. Everything is justified below.

| # | Do this | Why |
|---|---|---|
| **1** | **Train LightGBM `objective='binary'` on the full 1M rows, no resampling, no class weights.** This is your control and probably your final model. | Balancing does not improve ranking metrics for strong learners. Established across 73 datasets. |
| **2** | **Split temporally: train `month<6`, validate `month==6`, test `month==7`.** Never random-split BAF. | Prevalence and score distributions drift month to month. A random split leaks the future and inflates every number you report. |
| **3** | **Run the ablation in §9 and put its chart on a slide.** | This is your differentiator. It converts "we skipped the brief" into "we tested the brief." |
| **4** | **Spend the time you saved on features and on the temporal split.** Velocity ratios, missingness indicators, categorical encodings. | These move AUC. Balancing does not. |
| **5** | **Treat `scale_pos_weight` as a tuned hyperparameter over `[1, 90]` — jointly with `min_child_weight`.** Don't hardcode it to 90. | It is not a free win, and it silently destroys your regularization (§5). Expect the tuner to land near 1. |
| **6** | **Use random undersampling only as an exploration accelerator, and never below ~1:10.** Apply the prior correction in §7. Final model on full data. | 5–10× faster iteration in a hackathon. Near-zero ranking cost at mild ratios. Honest "balancing" for the rubric. |
| **7** | **Calibrate last, with isotonic on an untouched holdout — only if you need probabilities.** | Costs exactly zero AUC (monotone). Buys you an expected-loss slide. |
| **8** | **Set the operating point as the 95th percentile of *validation-negative* scores. Bootstrap it. Report realized test FPR.** | This is where balancing belongs. §8. |

**Do NOT:** SMOTE/ADASYN as a default; resample before splitting; resample the validation or test set; build EasyEnsemble/RUSBoost by hand; use `lambdarank` or pairwise objectives; submit hard 0/1 labels to an AUC leaderboard.

**The one number to remember:** across **1,736 controlled comparisons** (7 sampling methods × 8 classifiers × 31 datasets), sampling changed AUROC significantly in **only 10% of cases — and 61% of those changes were *degradations*.** (Kim & Hwang, *PLOS ONE* 2022.)

---

## 1. The spine: two families of metrics, and almost everyone conflates them

Every argument in this file reduces to one distinction. Internalize it and the entire contradictory literature snaps into focus.

**Family A — ranking / threshold-free metrics.** ROC-AUC, PR-AUC, **TPR at a fixed FPR**. These depend *only on the order* of your scores. Apply any strictly increasing function to every score — add a constant to the logit, square it, run it through a sigmoid — and **these metrics do not change by a single digit.**

**Family B — threshold-dependent metrics.** Accuracy, F1, precision, recall *at a fixed cutoff* (nearly always 0.5). These depend on the order **and** on where the scores sit relative to the cutoff.

Now the mechanism:

> **Class balancing — SMOTE, ADASYN, random oversampling, undersampling, `scale_pos_weight` — acts, to first order, as a monotone upward shift of predicted probabilities. It is a threshold move wearing a data-preprocessing costume.**

Therefore, mechanically and unavoidably:

- On **Family B** it has a **spectacular** effect. Your model went from predicting 0.011 everywhere (nothing exceeds 0.5, recall = 0, F1 = 0) to predicting around 0.5 (recall explodes). A paper reporting "recall improved from 0.04 to 0.71" is reporting a threshold move, not a modelling gain.
- On **Family A** it has **approximately no effect**, because you moved every score in the same direction and the order barely changed.

**Both of your metrics are Family A.** Kaggle ROC-AUC: Family A. The BAF paper's TPR@5%FPR: a point on the ROC curve, determined entirely by score order — Family A. So the entire class of balancing interventions is aimed at a target you are not being scored on.

This single fact explains why the literature looks like it's at war with itself. It isn't. **Pro-balancing papers report F1 and recall at 0.5. Anti-balancing papers report AUC and calibration.** They are both right about different things. Check which family a paper reports before you believe it — this is the single most useful filter you can apply while reading.

---

## 2. Q1 — Does SMOTE/ADASYN/oversampling improve ranking metrics for GBDTs?

### The strongest evidence against: Elor & Averbuch-Elor (2022)

**"To SMOTE, or not to SMOTE?"** — Yotam Elor, Hadar Averbuch-Elor, arXiv:2201.08528. Code at `github.com/aws/to-smote-or-not`.

This is the study your judges' objection dies on. It is directly on point:

| Design element | Detail |
|---|---|
| Datasets | **73** (filtered from 128) |
| Strong classifiers | **LightGBM, XGBoost, CatBoost** — exactly your candidate set |
| Weak classifiers | MLP, SVM, decision trees |
| Balancing methods | Random oversampling, **SMOTE**, SVM-SMOTE, **ADASYN**, Polynomial-fit SMOTE |
| Metrics | Family A (ROC-AUC, logloss, Brier) **and** Family B (F1, F2, Jaccard, balanced accuracy) |

Findings, and note how precisely they map onto §1:

1. **On probability/ranking metrics (AUC, logloss, Brier): balancing did not improve the strong classifiers.** Full stop.
2. **On label metrics with a fixed 0.5 threshold: balancing helped everything** — every classifier, strong and weak.
3. **On label metrics with the threshold optimized on validation data: balancing only helped the weak learners** (MLP, SVM).
4. Their framing: logloss optimization "is generally not consistent when using a fixed decision threshold" but becomes consistent once the threshold is tuned. Balancing and threshold-tuning are **substitutes**, and their empirical result is that they "yield comparable prediction performance."
5. Best overall configuration in the study: **CatBoost, no balancing.**

They also note that prior studies demonstrating SMOTE's value did so by focusing on precisely the settings where it works — weak learners, fixed thresholds, label metrics.

Point (2) versus (3) is your entire argument, run as a controlled experiment across 73 datasets. Balancing's apparent benefit *is* the threshold move; give the model a tuned threshold and the benefit evaporates for strong learners.

### The largest single sweep: Kim & Hwang (2022)

**Kim & Hwang**, *PLOS ONE* 17(7):e0271260 — **7 sampling methods × 8 classifiers (including XGBoost) × 31 datasets = 1,736 comparisons.** The result is the cleanest headline number in this entire literature:

> **Sampling significantly changed AUROC in only 10.0% of cases — and 61.3% of those significant changes were *degradations*.**

So the modal outcome of resampling is "nothing happened," and when something *did* happen, it was more likely to be bad than good. That is your one-slide answer to "but shouldn't you balance?"

### The strongest evidence against, part 2: the calibration harm

**van den Goorbergh, van Smeden, Timmerman, Van Calster (2022)**, *JAMIA* 29(9):1525 — "The harm of class imbalance corrections for risk prediction models" (arXiv:2202.09101). Tested RUS, ROS, and SMOTE on logistic and ridge regression:

- **Discrimination: no gain.** Corrections "did not result in higher areas under the ROC curve." The median AUROC of uncorrected models was *never lower* than corrected ones.
- **Calibration: severe harm.** At a 1% event fraction (your regime), calibration intercepts fell to **−4.5 or lower** for corrected models, versus **−0.05 to 0.03** for uncorrected. Calibration slopes dropped below 1 — predictions too extreme.
- Their conclusion: "outcome imbalance is not a problem in itself"; correction "may even worsen model performance."
- Crucially: **"similar results were obtained by shifting the probability threshold instead."** Same mechanism, stated by the authors.

**Carriero, Luijken, de Hond, Moons, van Calster, van Smeden (2025)**, *Statistics in Medicine* (arXiv:2404.19494) extends this to machine-learning models via Monte Carlo simulation across sample sizes, predictor counts, and event fractions. In **all** simulated scenarios, models built without correction had **equal or better calibration**. And the sting in the tail: the miscalibration introduced by correction — systematic risk over-estimation — **could not always be undone by recalibration afterwards.**

That last point matters more than it looks. The reassuring story "I'll SMOTE and then just recalibrate" is not reliably true.

### The evidence *for* — and how to read it honestly

There is a large literature reporting big SMOTE wins on fraud data, typically on the Kaggle credit-card dataset (0.17% positives), with headline figures like AUC 0.97–0.98 for SMOTE/SMOTE-ENN + XGBoost. Treat these with three specific suspicions:

1. **No control arm at parity.** The comparison is often SMOTE+tuned-XGBoost against untuned-XGBoost, or against logistic regression. That's a tuning result, not a balancing result.
2. **Family B metrics.** Read the results table. If the gains are in recall/F1 at 0.5 and AUC is flat, you have just re-derived §1.
3. **Resampling before the split.** This is the big one — see §3.

And where balancing genuinely *does* help — be honest about this, it makes you more credible, not less:

- **Weak learners.** Elor et al. confirm this directly. Logistic regression, SVMs, shallow trees, small MLPs benefit. You aren't using those.
- **Fixed 0.5 thresholds you cannot change.** If a pipeline hardcodes the cutoff, balancing is a legitimate workaround.
- **Absolute rarity** (point 4 below) — a few dozen positives, not eleven thousand.
- **Hyperparameters fixed a priori** from domain knowledge rather than tuned.

### BAF-specific evidence

**"Objective over Architecture: Fraud Detection Under Extreme Imbalance in Bank Account Opening"** (*Computation* 13(12):290, 2025) works on the **NeurIPS 2022 BAF Base benchmark at 1.10% prevalence** — your exact dataset. It compares logistic regression, RBF-SVM, random forest, LightGBM, and a GRU on all 1,000,000 accounts, treating class weighting as a tuned hyperparameter.

Its headline result is worth reading twice, because it is §1 in the wild:

> **AUCs were similar across model families**, yet the classical models converged to high-precision/low-recall solutions (1–6% fraud recall) while the GRU reached 78% recall at 5% precision (AUC = 0.8800).

Ranking performance: comparable. Recall at the default operating point: wildly different. The paper's own conclusion is that **"objective choice and operating point matter at least as much as architecture."** Which is exactly the recommendation in this file — balance at the decision layer.

**Read it with one caveat, honestly stated.** That paper tested *only class weighting* — it ran no resampling comparison — and its LightGBM arm reports AUC ≈ **0.814** at a 0.5 threshold, which is below what a properly tuned LightGBM reaches on BAF. Treat it as a directionally useful, under-tuned data point, not as a benchmark to beat. **Which leads to the opportunity: there is no published sampling ablation on BAF at all. The experiment in §9 is genuinely novel, and it costs you an afternoon.** Say that to the judges.

Also note what the dataset authors did. Jesus, Pombal, Alves, Cruz, Saleiro, Ribeiro, Gama, Bizarro, *"Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation"* (NeurIPS 2022 D&B, arXiv:2211.13358) built their baselines from **100 LightGBM models with hyperparameters from random search**, evaluated at **TPR@5%FPR** — a threshold chosen because "each false positive is a dissatisfied customer." **No resampling, undersampling, or oversampling appears in the baseline description**; imbalance is handled entirely through the choice of metric and the hyperparameter search. *(Verify this against that paper's own methods section and appendix before quoting it verbatim at a judge — it comes from PDF extraction, not a clean render.)* They put "Imbalanced" in the title and still reached for a strong learner plus a fixed-FPR operating point.

### Why SMOTE is *especially* wrong on BAF specifically

Four reasons that are about this dataset, not about SMOTE in general. These are what make your defence sound like engineering rather than contrarianism:

1. **The `−1` sentinels get interpolated into meaningless values.** `prev_address_months_count`, `current_address_months_count`, and `bank_months_count` use **−1 to encode missing**, and that missingness is one of the strongest fraud signals in the data (no history to check = thin file = synthetic identity). SMOTE interpolates between a −1 and a 24 and produces **11.5 months**, a value that asserts a history that never existed. You have destroyed your best signal and replaced it with a lie.
2. **`customer_age` is rounded to the decade** — 9 distinct values, effectively categorical. Interpolation yields ages like 37.2 that appear nowhere in the test set. The GBDT will happily split on `age < 38.5` and learn to detect *synthetic rows*, not fraud.
3. **BAF is already synthetic** — CTGAN-family generation from a real account-opening dataset. Running SMOTE on it is synthesizing from a synthesizer, compounding generator artifacts into a second-order fiction.
4. **This is relative imbalance, not absolute rarity — and only absolute rarity is a modelling problem.** You have **~11,000 positive examples**. That is not a small class; it's a mid-sized dataset in its own right. SMOTE's original motivation (Chawla, Bowyer, Hall, Kegelmeyer, JAIR 2002) was datasets where the minority class had a few hundred members and the learner (C4.5, Ripper, Naive Bayes) literally could not estimate its density. LightGBM on 11,000 positives has plenty to work with; the 1:90 ratio just means the *prior* is low, and the prior is a threshold problem, not a density problem.

> **Point 4 is the argument that wins the room. "Imbalance ratio" and "minority class size" are different quantities, and only the second one breaks learners.** 11,000 positives at 1:90 is a comfortable regime. 40 positives at 1:90 would not be.

**Verdict on Q1: No.** SMOTE/ADASYN/ROS do not reliably improve ROC-AUC or TPR@FPR for GBDTs on large imbalanced tabular data, and on BAF they actively corrupt three high-value features. Run them once in the ablation to prove it; don't ship them.

---

## 3. The leakage trap — the real reason SMOTE papers show huge gains

If you take one operational rule from this file, take this one.

**SMOTE creates each synthetic point as a convex combination of a real minority point and one of its k nearest minority neighbours.** So if you resample *before* splitting:

- A synthetic row that lands in **train** may be a blend of a **train** positive and a **test** positive.
- Your model has now literally seen a weighted average of test-set labels' feature vectors.
- AUC rockets to 0.98+, and it is entirely fake.

This is *the* most common methodological error in the applied imbalance literature and in Kaggle notebooks, and it is why the published effect sizes are so much larger than controlled studies find. On BAF it is doubly fatal, because the split is temporal: SMOTE across the month boundary blends **months 6–7 into months 0–5**. You are training on the future.

**Rules, non-negotiable:**

```python
# WRONG — the single most common bug in this field
X_res, y_res = SMOTE().fit_resample(X, y)
X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res)

# RIGHT — resample inside the fold, training data only, after the temporal split
from imblearn.pipeline import Pipeline        # imblearn's Pipeline, NOT sklearn's
pipe = Pipeline([("smote", SMOTENC(categorical_features=cat_idx)),
                 ("clf",   LGBMClassifier())])
# Only imblearn's Pipeline makes the sampler fold-aware and skips it at predict time.
```

- **Never resample the validation or test set.** Ever, for any reason. Your evaluation set must carry the true 1.1% prior or every number you compute is meaningless — and note that PR-AUC in particular is *prevalence-dependent*, so a resampled test set silently inflates it. (ROC-AUC is prevalence-independent, which is exactly why it's the safer leaderboard metric.)
- Use `SMOTENC`, not `SMOTE`, if you insist on trying it — BAF has ~5 genuine categoricals (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`). Plain `SMOTE` on one-hot columns produces fractional category memberships, which is nonsense.

**Judge-proofing bonus:** run SMOTE *both* ways once, report both numbers, and say "the published gains are reproducible only under the leaky protocol." That is a genuinely memorable slide and it demonstrates methodological maturity better than any leaderboard position.

---

## 4. Q2 — Random undersampling, and ensembled undersampling

Undersampling deserves separate treatment because on 1M rows it isn't primarily a balancing technique — **it's a compute lever**, and in a hackathon compute is time and time is your real constraint.

### The theory says it's ranking-neutral

Randomly dropping negatives changes `P(y)` but leaves both class-conditionals `p(x|y)` untouched. The Bayes-optimal ranking is the likelihood ratio `p(x|1)/p(x|0)`, which doesn't involve the prior at all. So **in the infinite-data limit, RUS costs exactly zero ranking performance** — it shifts your calibration by a known constant (§7) and nothing else.

The finite-sample cost is pure **variance**: with fewer negatives you estimate `p(x|0)` less precisely, especially in its tails, which is exactly where the rare, weird, legitimate applications live — the ones that generate your false positives at a 5% FPR cutoff.

### What the numbers say

- **Kim & Hwang (2022)**, above: across 1,736 cases, AUROC moved significantly in 10% of them, and most of those moves were downward.
- **He et al., ADKDD'14** (Facebook, boosted trees + linear, billions of rows) is the best large-scale evidence: **uniform subsampling to 10% of the data cost roughly 1% normalized entropy**, and their tuned negative-downsampling rate was **0.025**. Note the metric is NE, which is calibration-sensitive, so this is an upper bound on what a pure ranking metric would lose.
- The **Fraud Detection Handbook** (Le Borgne & Bontempi, ULB) reports on a transaction-fraud benchmark: XGBoost **0.872 ± 0.01** AUC versus *weighted* XGBoost **0.872 ± 0.01** — identical to three decimals.

### Where the loss actually starts

Nobody has published the RUS-ratio curve for BAF, so this is judgement, stated as judgement:

| Ratio | Negatives kept | Expect |
|---|---|---|
| Full (1:90) | 989,000 | Baseline |
| 1:25 | ~276,000 | AUC loss inside seed noise |
| **1:10** | **~110,000** | **AUC loss inside seed noise. ~8× faster. This is the sweet spot.** |
| 1:5 | ~55,000 | Small but measurable loss begins |
| 1:1 | 11,029 | **You have discarded 99% of your negatives.** Expect real degradation |

> **The 1:1 "fully balanced" configuration — the one the objective statement implicitly suggests — is the single worst point on this curve.** It throws away ~978,000 of 989,000 negatives to hit a cosmetic 50/50 ratio that no metric you're scored on cares about.

### Ensembled undersampling: EasyEnsemble, BalanceCascade, RUSBoost

**Liu, Wu & Zhou (2009)**, *Exploratory Undersampling for Class-Imbalance Learning*, IEEE TSMC-B 39(2):539–550 (EasyEnsemble, BalanceCascade) and **Seiffert, Khoshgoftaar, Van Hulse & Napolitano (2010)**, *RUSBoost*, IEEE TSMC-A 40(1):185–197, are both real, well-cited papers. Read them with two facts in hand:

1. **Both predate LightGBM (2016)** and benchmark against C4.5, AdaBoost/C4.5, SMOTE, and plain RUS. Those are weak baselines by 2026 standards. "Beats C4.5" is not "beats a tuned LightGBM."
2. **Structurally, EasyEnsemble cannot add information.** It bags over *disjoint* negative subsets, so the union of its members sees all the negatives — it is a parallel approximation of full-data training, not a source of anything full-data training lacks. The Fraud Detection Handbook's numbers are consistent with this: balanced bagging 0.879 ± 0.01 against XGBoost 0.872 ± 0.01, overlapping intervals, and balanced random forest gave no gain at all.

No credible head-to-head of EasyEnsemble or RUSBoost against a tuned GBDT on full data, on ranking metrics, at N ≈ 10⁶ appears to exist.

**And if you do want bagged undersampling, don't hand-roll it.** LightGBM already implements it natively and per-iteration: `bagging_fraction` with `bagging_freq`, or the class-aware pair **`pos_bagging_fraction` / `neg_bagging_fraction`**, which subsample each class at a different rate on every boosting round. That is balanced bagging done inside the algorithm, with none of the orchestration and no extra models to average.

### A warning about a paper you will find

You will search "undersampling beats oversampling" and hit **Drummond & Holte, "C4.5, Class Imbalance, and Cost Sensitivity: Why Under-Sampling beats Over-Sampling."** Do not cite it in support of RUS on LightGBM. It is an **ICML 2003 *workshop* paper**, it uses **C4.5 only**, it evaluates with **cost curves** (expected misclassification cost across cost/prior operating points), and it says **nothing about AUC or about gradient boosting**. It also concludes that a plain least-cost classifier often beats undersampling anyway. A judge who knows the paper will catch a loose citation; a judge who doesn't is not helped by one.

**Verdict on Q2: use RUS as a compute lever, not as a balancing strategy.** Explore at 1:10 with the prior correction applied, then train your final model on the full data. Skip EasyEnsemble/BalanceCascade/RUSBoost entirely — if you want the variance reduction, `neg_bagging_fraction` gives it to you for one line.

---

## 5. Q3 — `scale_pos_weight`, `class_weight`, `is_unbalance`

**What they are.** `is_unbalance=true` (LightGBM) auto-sets positive weights to `n_neg/n_pos` ≈ 90 for BAF. `scale_pos_weight` (both libraries) sets it manually. They are **mutually exclusive in LightGBM** — set one, not both.

**Do they improve ranking?** Essentially no, but the reason is more subtle than "it's just a monotone transform," and you should get this right if a technical judge probes.

Weighting is **not exactly rank-preserving.** It scales the gradients and Hessians of the positive class, which changes **split gains**, which changes **tree structure**. So the learned ranking function genuinely differs. But the change is not systematically an *improvement* — it's a different point in a noisy space. The large, systematic, predictable effect is the **level shift** in predicted probability; the effect on AUC is empirically inside seed noise.

**The gotcha nobody tells you — and this one is worth real points:**

> `min_child_weight` (XGBoost) and `min_sum_hessian_in_leaf` (LightGBM) are measured in **units of summed Hessian**. When you upweight positives by 90×, the Hessian mass in any positive-rich leaf inflates by up to 90×, so your **minimum-child-weight regularization silently loosens by a large factor.** XGBoost's `min_child_weight` defaults to **1** and is genuinely binding, so setting `scale_pos_weight=90` quietly permits far smaller, purer leaves than you intended.

A large share of "`scale_pos_weight` improved my model!" results are **"I accidentally changed my regularization."** Which means: if you tune `scale_pos_weight` *without* re-tuning `min_child_weight`, you are not measuring what you think you are measuring.

**Correct guidance:**

1. Do **not** hardcode `scale_pos_weight = n_neg/n_pos`. It is folklore, not a result.
2. Tune it as a hyperparameter over a log-ish grid — `[1, 3, 10, 30, 90]` (i.e. spanning 1 → √ratio → ratio) — **jointly with `min_child_weight`/`min_sum_hessian_in_leaf`** in the same Optuna study.
3. Select on **validation ROC-AUC**. If AUC is your metric, expect the tuner to land near **1**.
4. Know that XGBoost's own docs recommend it: *"balance the positive and negative weights via `scale_pos_weight`, use AUC for evaluation"* for overall performance, versus *"you cannot re-balance the dataset"* and *"set `max_delta_step` to a finite number (say 1)"* when you need correct probabilities. Cite it accurately, then note that this is **heuristic documentation guidance, not a controlled study** — and that Elor et al.'s 73-dataset experiment finds no ranking gain for strong learners. The docs are usefully honest about the cost, though: they explicitly frame weighting as incompatible with calibrated probability.

**Verdict on Q3: mostly a threshold shift.** Tune it, don't assume it, and always re-tune `min_child_weight` alongside it or your ablation is invalid.

---

## 6. Q4 — Custom objectives: focal loss, asymmetric losses, ranking objectives

### The theoretical argument, which is decisive

Log-loss is a **strictly proper scoring rule**. Its population minimizer is the *true* posterior `P(y=1|x)`. And the true posterior is, by construction, the **AUC-optimal ranking function** — nothing can rank better than the truth.

So: **at the population optimum, log-loss already gives you the best achievable ranking.** There is no headroom for a different objective to find a better ordering.

Focal loss is **not** a proper scoring rule. Its population minimizer is a *distorted* posterior. That distortion is monotone in `p`, so at the population optimum focal loss yields **the same ranking** as log-loss — it cannot improve AUC in the limit. Any observed gain must come from a **finite-sample or optimization effect** (an implicit regularizer, an easy-example down-weighting that changes which splits get chosen), not from optimizing a better target.

That is a real effect and it can be positive. But it is a hyperparameter-shaped lottery ticket, not a principled improvement — and you should budget for it accordingly.

### The empirical record

**Luo, Yuan & Xu (2024)**, "Improving GBDT Performance on Imbalanced Datasets: An Empirical Study of Class-Balanced Loss Functions" (arXiv:2407.14381) is the most thorough study available: **5 balanced losses** (weighted cross-entropy, focal, asymmetric loss, asymmetric cross-entropy, asymmetric weighted CE) × **3 GBDTs** (XGBoost, LightGBM, SketchBoost) × **40 datasets**. Binary results: 13 of 15 datasets improved, by 0.38% to 28.91%.

Now read the fine print, because it is the whole story:

> **Their primary evaluation metric is F1-score. ROC-AUC is not reported.**

This is §1 again, in the most recent and most careful paper on the topic. The improvements are real — in **Family B**. The paper does not establish a ranking-metric gain, because it does not measure one.

A **Journal of Cheminformatics** (2022) study on imbalanced bioassay modelling with custom GBDT losses found focal loss best on **PR-AUC, accuracy, F1, and MCC**, while on ROC-AUC, LDAM edged it out on the HIV dataset — mixed, dataset-dependent, no clean ranking win. And a widely-read practitioner implementation of focal loss for LightGBM reported that on credit-card fraud, tuning α and γ produced **no convincing ROC-AUC improvement over plain log-loss.**

### Ranking objectives (`lambdarank`, `rank:pairwise`)

Don't. Three reasons:

1. **Wrong tool.** `lambdarank` optimizes NDCG over *query groups* with graded relevance. Binary classification with one global "query" degenerates to plain pairwise ranking, and you lose the calibrated-probability output entirely.
2. **Computationally hostile.** Pairwise objectives are O(n₊ × n₋) = 11,000 × 989,000 ≈ **10¹⁰ pairs**. In a hackathon this is a non-starter.
3. **No credible evidence of gains** on large tabular fraud. And per the theory above, there is no headroom for one.

**Verdict on Q4: skip.** If you have spare time after the ablation, focal loss is the only one worth a single arm (A10), tuned α and γ. Expect ΔAUC ≈ 0. Do not put it on the critical path.

---

## 7. Q5 — Calibration, and the prior-correction formula

### What resampling does to your probabilities

Undersample negatives to 10%, and your model believes fraud is ~10% likely instead of 1.1%. Oversample to 1:1 and it believes fraud is a coin flip. Predicted probabilities are inflated by a large, systematic, *computable* factor — this is precisely van den Goorbergh et al.'s calibration intercept of −4.5.

**Why you should care even in a hackathon where the metric is AUC:** in fraud, **the probability is the product.** Expected loss per application = `p(fraud) × exposure`. A review-capacity budget, a cost-benefit slide, a "we'd save the bank £X" claim — all of them need calibrated `p`, and all of them are exactly the kind of business-grounded slide that wins hackathons. Miscalibrated probabilities make every one of those numbers wrong by ~90×.

### The prior-correction formula

This result has been derived independently in at least four literatures, which is a good sign it's right. It is **King & Zeng (2001)**, *Logistic Regression in Rare Events Data*, *Political Analysis* 9:137–163 (as an intercept correction); **Elkan (2001)**, *The Foundations of Cost-Sensitive Learning*, IJCAI-01 (as a general base-rate change); and **He et al. (2014)**, ADKDD (as Facebook's production negative-downsampling calibration). All three are algebraically the same statement of Bayes' rule under a prior shift.

**Setup.** You keep **all** N₁ positives and a random fraction **β ∈ (0,1]** of the N₀ negatives.

| Symbol | Meaning | BAF value |
|---|---|---|
| `π` | true positive rate in the target population | ≈ 0.011 |
| `π̃` | positive rate in the **resampled** training set = `N₁ / (N₁ + βN₀)` | depends on β |
| `p̃(x)` | probability predicted by the model **trained on resampled data** | — |
| `p(x)` | corrected probability, on the true population scale | what you want |
| `β` | fraction of negatives retained | e.g. 0.1 |

**Odds form:**

```
   p(x)             p̃(x)         π          1 − π̃
─────────  =  ───────────  ×  ───────  ×  ────────
 1 − p(x)       1 − p̃(x)       1 − π         π̃
```

**Logit form — use this one, it is numerically stable and one line of code:**

```
logit p(x)  =  logit p̃(x)  +  log( π / (1−π) )  −  log( π̃ / (1−π̃) )
```

**Explicit probability form:**

```
                w · p̃(x)                                π (1 − π̃)
p(x)  =  ──────────────────────────  ,     where  w  =  ───────────
          w · p̃(x)  +  (1 − p̃(x))                       (1 − π) π̃
```

**The simplification that makes this trivial.** For pure negative-undersampling — all positives kept, fraction β of negatives kept — the correction factor collapses to exactly β:

```
π̃/(1−π̃) = N₁/(βN₀) = (1/β) · π/(1−π)     ⟹     w = β
```

so:

> ### **logit p(x) = logit p̃(x) + log β**
>
> Keep 10% of negatives → add `log(0.1) = −2.303` to every logit. That's the whole correction.

**And note what it does to your leaderboard score: nothing.** Adding a constant to every logit is strictly monotone, so ROC-AUC and TPR@5%FPR are *bit-for-bit identical* before and after correction. The correction exists purely to make your probabilities mean something. Which is also the cleanest possible demonstration of §1 — you can show a judge the same model, before and after a 90× change in its predicted probabilities, scoring exactly the same AUC.

```python
import numpy as np
raw = model.predict(X, raw_score=True)     # LightGBM logits — never round-trip through p
p_corrected = 1.0 / (1.0 + np.exp(-(raw + np.log(beta))))
```

Take the raw margin directly rather than recovering the logit from a probability; you avoid a needless precision loss at small `p`, which is the entire range you live in.

**Cross-checks against the published forms**, so you can cite whichever your audience knows:

- **King & Zeng** state it as an intercept correction on a fitted logistic model: `β₀_corrected = β̂₀ − ln[ ((1−τ)/τ) · (ȳ/(1−ȳ)) ]`, where `τ` is the population positive rate and `ȳ` the sample one. **Slope coefficients need no adjustment** — which is the same statement as "the ranking is unchanged." They also advise collecting only about 2–5× more controls than cases, i.e. **stop undersampling around 1:5, not 1:1.**
- **He et al.** give the probability form directly for negative downsampling at rate `w`: `q = p / (p + (1−p)/w)` — identical to the `w = β` result above.
- **Elkan's** general form, for moving from a training base rate `b` to a target base rate `b′`: `p′ = b′(p − p·b) / (b − p·b + b′·p − b·b′)`. Expand the odds form with `w = b′(1−b)/((1−b′)b)` and you land on exactly this.
- **Saerens, Latinne & Decaestecker (2002)**, *Neural Computation* 14(1):21–41, give an EM procedure for when the target prior is *unknown*. You don't need it — your τ is known to be ~1.1%.

### When the formula is valid — read this before using it

The correction is exact **only when resampling changes `P(y)` while leaving `P(x|y)` untouched.**

| Method | Formula valid? | Why |
|---|---|---|
| Random **undersampling** of negatives | ✅ **Exact** | Pure prior shift. `P(x\|y)` unchanged. |
| Random **oversampling** of positives (with replacement) | ✅ Exact in expectation | Duplicates don't change the minority distribution. |
| `scale_pos_weight` / `is_unbalance` | ⚠️ Approximate | Weighting also changes split selection and tree structure. Verify empirically. |
| **SMOTE / ADASYN** | ❌ **Invalid** | They *invent a new* `P(x\|y=1)`. There is **no closed-form correction.** You must recalibrate empirically. |

That last row is another reason to avoid SMOTE: it puts your probabilities in a place no formula can bring them back from.

Two further caveats: (a) `π` must match the deployment prior, and BAF's **monthly prevalence varies 0.85%–1.5%**, so use the test-period prior if you know it and treat the correction as approximate otherwise; (b) Carriero et al. (2025) found resampling-induced miscalibration is **not always fully repairable** by post-hoc recalibration, so "SMOTE now, calibrate later" is not a safe plan.

### Empirical recalibration — and the fact that makes it free

If you can't use the formula (SMOTE), or you just want to be safe: fit **Platt scaling** (a 1-D logistic regression on the logits) or **isotonic regression** on a **held-out calibration set that carries the true prior and was never resampled**. Isotonic is more flexible and preferred at your sample size; Platt is safer on small calibration sets.

> **The fact that makes this a free action: monotone recalibration cannot change your ranking.** Platt scaling is strictly monotone, so it leaves ROC-AUC, PR-AUC, and TPR@5%FPR **exactly** unchanged. Isotonic is monotone non-decreasing, so it preserves them up to a negligible effect from newly created ties.

Two direct consequences: **calibration can never rescue a ranking problem** (don't reach for it to fix AUC), and **calibration can never cost you leaderboard points** (so add it at the end, for free, for the business slide).

### Do you even need to calibrate?

Probably not, if you follow the verdict box. The classic result that boosted trees produce sigmoid-distorted probabilities — **Niculescu-Mizil & Caruana, ICML 2005, "Predicting Good Probabilities With Supervised Learning"** — concerns **maximum-margin boosting with exponential loss** (AdaBoost-style), which pushes probability mass away from 0 and 1. **LightGBM/XGBoost with `objective='binary'` optimize log-loss, a proper scoring rule**, and on 1M rows they come out close to calibrated already.

So: don't cargo-cult a calibration step. **Plot a reliability curve and compute the Brier score on your control model first.** If it's already calibrated, say so on the slide — "our probabilities are calibrated out of the box because we didn't break them" is a strong line.

---

## 8. Q6 — Threshold selection at a fixed FPR

### First: check whether you need a threshold at all

**If the Kaggle metric is ROC-AUC, submit continuous scores. Never submit hard 0/1 labels.** Submitting labels collapses your ROC curve to a single point and typically costs you 0.15–0.30 AUC outright. This sounds too obvious to mention. It is a top-3 cause of inexplicably bad leaderboard scores.

### For TPR@5%FPR, the threshold is a quantile, not a search

The 5%-FPR operating point is, by definition:

```python
t = np.quantile(scores[y_val == 0], 0.95)      # 95th percentile of NEGATIVE scores
tpr_at_5 = (scores[y_val == 1] >= t).mean()
```

Note what determines it: **the negative class only.** This is the key to the overfitting question.

### The overfitting risk here is genuinely small — and you can quantify it

The standard error of a sample quantile is `√(q(1−q)/n) / f(t)`. With ~198,000 negatives in a two-month test period, the sampling error on the realized FPR is:

```
SE ≈ √(0.05 × 0.95 / 198,000) ≈ 0.00049
```

**Realized FPR ≈ 5.0% ± 0.1% at 95% confidence.** You have a million negatives; the 95th percentile of that distribution is one of the most precisely estimated quantities in your entire pipeline.

> **Contrast this sharply with an F1-maximizing threshold, which is driven by the ~11,000 positives and is *far* noisier — and with a fixed-recall threshold, which is noisier still. Fixed-FPR thresholds are cheap to estimate precisely. Fixed-recall thresholds are not.** This asymmetry is why the fraud industry specifies operating points in FPR terms, and it's worth a sentence on your slide.

### The real risk is temporal drift, not sampling noise

BAF's prevalence swings **0.85%–1.5% across months**, and the score distribution moves with it. A threshold frozen on months 0–5 **will not** realize 5% FPR on months 6–7. That is the honest threat, and addressing it is a differentiator.

### The protocol

1. **Split temporally.** Train `month < 6`, validate `month == 6`, test `month == 7`. Never random-split.
2. **Select hyperparameters on validation ROC-AUC** — a *threshold-free* criterion.
3. **Set the threshold as the empirical 95th percentile of validation-negative scores.** Don't grid-search a metric; you're estimating a quantile, and the direct estimator is unbiased and precise.
4. **Evaluate once on the untouched test period.** Report the **realized** FPR alongside TPR. The gap between target 5% and realized FPR *is* your drift measurement — put that number on a slide.
5. **Bootstrap it.** Resample validation negatives 1,000× with replacement, recompute the threshold each time, and report a CI on both the threshold and the resulting test TPR. Ten lines of code that directly and quantitatively answer "did you overfit your threshold?"
6. **Re-estimate per period in deployment.** The threshold is a quantile of the *score* distribution, which needs **no labels** — only the scores of applications you've already seen. So you can re-fit the operating point monthly in production, with zero label latency, and hold FPR steady under drift. This is the operations-grade answer and almost no competing team will have it.

Because step 2 uses a threshold-free criterion, reusing the validation period for step 3 is a much weaker form of reuse than the nested-CV literature is worried about. With 198k negatives, it is not your bottleneck. If you want to be maximally clean, hold out `month == 5` for hyperparameters and `month == 6` for the threshold.

---

## 9. The ablation table — your single most valuable artifact

Build this. It converts a defensive position ("we didn't balance") into an offensive one ("we measured balancing, here's what it does").

**Hold constant across every arm** — this is what makes it evidence rather than anecdote:
same features · same temporal split (train 0–5, val 6, test 7) · same tuning budget (e.g. 30 Optuna trials, selected on **validation ROC-AUC**) · same **5 seeds**, reported as mean ± std.

| # | Arm | ROC-AUC ↑ | TPR@5%FPR ↑ | Brier ↓ | Calib. slope (→1) | F1 @ 0.5 | Train time |
|---|---|---|---|---|---|---|---|
| A0 | **LightGBM, no balancing (control)** | | | | | | 1× |
| A1 | + `is_unbalance=true` | | | | | | |
| A2 | + `scale_pos_weight` tuned ∈[1,90], `min_child_weight` re-tuned | | | | | | |
| A3 | + Random oversampling → 1:1 | | | | | | |
| A4 | + SMOTE-NC → 1:1 (**inside fold**) | | | | | | |
| A5 | + SMOTE-NC → 1:4 (mild) | | | | | | |
| A6 | + ADASYN | | | | | | |
| A7 | + RUS to **1:10** (β≈0.11) **+ prior correction** | | | | | | ~0.12× |
| A8 | + RUS to **1:1** — the "fully balanced" arm the brief implies | | | | | | ~0.02× |
| A9 | + `neg_bagging_fraction=0.1` (LightGBM's native balanced bagging) | | | | | | |
| A10 | + Focal loss (α, γ tuned) | | | | | | |
| A11 | **A0 + isotonic calibration** | | | | | | |
| A12 | *(optional, for the leakage slide)* SMOTE applied **before** the split | | | | | | |

**Two columns do the work: `ROC-AUC` and `F1 @ 0.5`, side by side.** That adjacency is the entire argument, visible at a glance.

### Predict the results before you run it

Write these predictions down *first*, then show that you called them. Judges find a correct prediction far more persuasive than a correct result.

- **ROC-AUC and TPR@5%FPR: flat across A0–A10**, differences within ±0.003, comfortably inside the 5-seed noise band.
- **F1 @ 0.5: jumps enormously for A1–A6.** Possibly from near-zero to something respectable. This is the illusion, isolated.
- **Brier and calibration slope: markedly worse for A1–A6.** A7 is *repaired* by the prior correction — a clean demonstration that the formula works.
- **A11 has ROC-AUC identical to A0 to the last decimal** (monotone transform) with a better Brier. This is your proof that calibration is free.
- **A7 trains ~8× faster at near-identical AUC** — the honest, useful form of "data balancing."
- **A8 is the one arm that visibly *loses* ranking performance**, because it discards 99% of the negatives. The fully-balanced configuration finishes last on the metric you're scored on. That is a very good sentence to be able to say out loud.
- **A12's AUC is absurdly high** (~0.97+). That's the leak. Label it clearly as a negative control.

### The chart

Two bar charts, side by side, same arms on the x-axis:

**Left: "F1 @ 0.5" — huge bars, dramatic differences. Right: "ROC-AUC" — a flat line.**

**Caption: *"Balancing moves the threshold, not the model."***

That is your whole thesis in one image, generated from their data, in your colours.

---

## 10. How to defend this to the judges

The objective statement says "data balancing." So **do not say you skipped it.** You didn't. Say this instead:

> ### **"We did balance. We balanced at the decision layer, because that's the only layer where it changes the answer — and we ran the experiment to prove it."**

Deliver it in five beats.

**1 — Concede the premise properly.** "Balancing exists to fix one thing: the mismatch between a symmetric training loss and an asymmetric business objective. That's a real problem and we take it seriously. There are exactly two places to fix it — in the data, or at the decision. We tested both."

**2 — The mechanism, in one sentence.** "Our metric is a ranking metric, and ranking metrics are invariant to any monotone rescaling of scores. Resampling is, to first order, exactly such a rescaling. That's why it moves F1-at-0.5 dramatically and leaves AUC flat. It isn't that balancing does nothing — it's that what it does is change the threshold, and we can change the threshold directly."

**3 — Your own evidence.** *[chart]* "Eleven arms. Same features, same temporal split, same tuning budget, five seeds. F1-at-0.5 moves by [X]. ROC-AUC moves by [0.00Y] — inside our seed noise. On this dataset, resampling is a threshold move."

**4 — Name the cost.** "And it isn't free. Resampling took our calibration slope from [0.98] to [0.31] and our Brier score from [a] to [b]. In fraud, the probability *is* the product — it's how you compute expected loss per application and how you size a review team. We're not trading that away for a metric that doesn't move. Worse, SMOTE interpolates the `−1` missing-value sentinels on `prev_address_months_count` and `bank_months_count` into fractional months, which fabricates a credit history that never existed — and thin-file missingness is one of our strongest signals."

**5 — Show where you *did* balance.** "So we balanced where it acts: we set the operating point at the 95th percentile of validation-negative scores, bootstrapped it for a confidence interval, and re-estimate it per month — which needs no labels, only scores — so FPR stays at 5% under the prevalence drift this dataset has by design. That's balancing, done at the layer that changes the outcome."

### Have these ready for the follow-up questions

**"But SMOTE is standard practice / this paper got 0.98 with SMOTE."**
"Two things. First, most published SMOTE gains are reported on F1 and recall at a 0.5 threshold — Family B metrics. On AUC they're usually flat, and often the AUC just isn't reported. Second, the very large gains typically come from resampling before the train/test split. SMOTE builds each synthetic point by interpolating between real minority points, so if you resample first, synthetic rows in training are blends of test-set positives. We ran it both ways — here's the leaky number and here's the honest one."

**"Isn't 1.1% too imbalanced to learn from?"**
"That conflates two different quantities. The *ratio* is 1:90, but the *absolute* minority count is about 11,000 positives — that's a decent-sized dataset on its own. Balancing exists for absolute rarity, where the learner can't estimate the minority density at all. LightGBM has plenty of positives here. The low ratio only means the prior is low, and a prior is a threshold problem, not a density problem."

**"Why not just undersample to 50/50? That's what the brief asks for."**
"We ran it — arm A8. Undersampling to 1:1 means keeping 11,029 negatives and throwing away 978,000, which is 99% of our majority-class data. It's the one arm in our ablation that actually *loses* ranking performance, because the rare, unusual-but-legitimate applications that generate our false positives at a 5% cutoff live in exactly the tail we deleted. We do use undersampling — at 1:10, as a speed lever, with the prior correction applied so the probabilities stay honest. That's an 8× faster training loop at the same AUC, which is what let us run this ablation at all. King and Zeng, who wrote the standard reference on rare-event logistic regression, recommend about 2 to 5 controls per case — nobody who has thought about it recommends 1:1."

**"What if we're wrong about the metric?"**
"We checked. ROC-AUC is a ranking metric and TPR-at-5%-FPR is a point on the ROC curve, so both are determined purely by score order. The argument holds under either. If the metric were F1 at a fixed 0.5 cutoff we'd resample — and we'd still just be moving the threshold, only less precisely."

**"Show me you're not just being contrarian."**
"We're not — balancing genuinely works in a well-defined regime, and we can name it. Elor and Averbuch-Elor tested five balancing methods across 73 datasets. Balancing helped weak learners — MLPs, SVMs, shallow trees — and it helped everything when the threshold was pinned at 0.5. Once the threshold was tuned, the benefit survived only for the weak learners. We're using a strong learner with a threshold-free metric, which is precisely the regime where the effect goes away. Their best overall configuration was CatBoost with no balancing."

### The rubric-compliance play

If the marking scheme literally awards points for implementing a balancing technique, **do not silently omit it.** Implement RUS + prior correction (arm A7) as a first-class, documented part of your pipeline — it's genuinely useful for fast iteration, it's genuinely "data balancing," and it's mathematically principled. Then present the full ablation as your *analysis*.

You collect the rubric point **and** the differentiation. The teams around you will have SMOTE in their pipeline and no idea whether it helped.

---

## 11. Citation ledger

Verified sources. Everything attributed here, I checked exists; anything stated without attribution is a mechanism or a derivation, flagged as such.

| Work | Where | What it establishes |
|---|---|---|
| Elor & Averbuch-Elor (2022), *To SMOTE, or not to SMOTE?* | arXiv:2201.08528; code `github.com/aws/to-smote-or-not` | 73 datasets; LightGBM/XGBoost/CatBoost; balancing gives no ranking gain for strong learners; helps at fixed 0.5 threshold; helps weak learners |
| **Kim & Hwang (2022)** | *PLOS ONE* 17(7):e0271260 | **1,736 comparisons** (7 samplers × 8 classifiers × 31 datasets): sampling significantly changed AUROC in **10.0%** of cases; **61.3% of those were degradations** |
| He, Pan, Jin, Xu, Liu, Xu, Shi, Atallah, Herbrich, Bowers, Candela (2014) | ADKDD'14 (Facebook) | Production-scale negative downsampling with boosted trees; 10% subsample ≈ 1% NE cost; tuned rate 0.025; the `q = p/(p+(1−p)/w)` calibration form |
| Liu, Wu & Zhou (2009), *Exploratory Undersampling* | IEEE TSMC-B 39(2):539–550 | EasyEnsemble / BalanceCascade. **Baselines are C4.5-era**; no head-to-head vs tuned GBDT exists |
| Seiffert, Khoshgoftaar, Van Hulse, Napolitano (2010), *RUSBoost* | IEEE TSMC-A 40(1):185–197 | Same caveat — predates LightGBM, weak baselines |
| King & Zeng (2001), *Logistic Regression in Rare Events Data* | *Political Analysis* 9:137–163 | Prior correction as an intercept shift; **slopes unchanged**; advises ~2–5× controls per case |
| Elkan (2001), *The Foundations of Cost-Sensitive Learning* | IJCAI-01 | General base-rate change formula; threshold-vs-resampling equivalence theorem |
| Saerens, Latinne & Decaestecker (2002) | *Neural Computation* 14(1):21–41 | EM adjustment of outputs to *unknown* new priors. Not needed here |
| **Drummond & Holte (2003)** — *cite with care* | ICML'03 **Workshop** on Learning from Imbalanced Datasets II | "Under-sampling beats over-sampling" — but **C4.5 only, cost curves, no AUC, no GBDTs**. Do not cite as support for RUS on LightGBM |
| van den Goorbergh, van Smeden, Timmerman, Van Calster (2022) | *JAMIA* 29(9):1525; arXiv:2202.09101 | RUS/ROS/SMOTE: no AUC gain, severe miscalibration at 1% event rate; threshold shift gives the same result |
| Carriero, Luijken, de Hond, Moons, van Calster, van Smeden (2025) | *Statistics in Medicine*; arXiv:2404.19494 | ML models, Monte Carlo: uncorrected models had equal or better calibration in **all** scenarios; miscalibration not always repairable |
| Jesus, Pombal, Alves, Cruz, Saleiro, Ribeiro, Gama, Bizarro (2022), *Turning the Tables* | NeurIPS 2022 D&B; arXiv:2211.13358 | The BAF dataset itself; 1M rows, 32 cols, 1.1%; TPR@5%FPR metric and its business rationale; baselines = 100 random-search LightGBM models |
| *Objective over Architecture* (2025) | *Computation* 13(12):290 | **BAF Base, 1.10%, 1M rows.** Similar AUCs across model families; recall differences driven by operating point; "objective choice and operating point matter at least as much as architecture" |
| Luo, Yuan & Xu (2024) | arXiv:2407.14381 | 5 class-balanced losses × 3 GBDTs × 40 datasets; gains reported — **on F1; ROC-AUC not reported** |
| Chawla, Bowyer, Hall, Kegelmeyer (2002), *SMOTE* | *JAIR* 16:321–357 | The original method; motivated by small minority classes with C4.5/Ripper/Naive Bayes |
| Niculescu-Mizil & Caruana (2005) | ICML '05, *Predicting Good Probabilities With Supervised Learning* | Sigmoid distortion in **max-margin/exponential-loss** boosting; Platt vs isotonic. Does **not** transfer directly to log-loss GBDTs |
| XGBoost official docs | `xgboost.readthedocs.io` → *Notes on Parameter Tuning* | "Balance the positive and negative weights via `scale_pos_weight`, use AUC for evaluation" vs. "you cannot re-balance"/`max_delta_step` when probabilities matter. **Heuristic guidance, not a controlled study** |
| *Tuning gradient boosting for imbalanced bioassay modelling with custom loss functions* (2022) | *J. Cheminformatics* 14 | Focal loss best on PR-AUC/F1/MCC; ROC-AUC results mixed and dataset-dependent |

**Stated without attribution** (mechanisms and derivations, not citations): the monotone-invariance argument in §1; the proper-scoring-rule argument in §6; the `min_child_weight` Hessian-units interaction in §5; the SMOTE-leakage mechanism in §3; the likelihood-ratio argument for RUS in §4; the quantile standard-error calculation in §8; the `w = β` derivation in §7. These follow from definitions and are checkable, not claims about anyone's published results.
