# Algorithm Comparison — Why LightGBM, and How Honestly That Claim Holds Up

*Companion to `final_report.md`. That report states the selected model in one paragraph
(section 6). This document is the full justification behind that paragraph: the mechanics
of every candidate algorithm, the real 20-way sweep the selection was made from, and an
unflinching look at where the evidence is strong and where it is thin. Every number below
comes from `reports/metrics/model_comparison.csv`, `models/model_meta.json`, and
`reports/metrics/test_evaluation.json` — nothing here is projected or assumed.*

## 1. What was actually compared

Four model families, five imbalance-handling strategies each, 20 runs total, all trained
on the same 700,000-row stratified train fold and scored on the same 150,000-row
validation fold (1,655 positives, 1.103% prevalence):

| Family | none | class_weight | random_undersample | SMOTE | SMOTE+undersample |
|---|---|---|---|---|---|
| **Logistic Regression** | ROC .8766 / PR .1431 / TPR .5039 / 3.5s | .8775 / .1409 / .5027 / 5.5s | .8769 / .1425 / .5027 / 0.6s | .8756 / .1424 / .4949 / 4.5s | .8763 / .1411 / .4967 / 1.3s |
| **Random Forest** | .8725 / .1367 / .4900 / 249.1s | .8697 / .1222 / .4792 / 227.1s | .8788 / .1446 / .5033 / 27.8s | .8751 / .1280 / .4785 / 42.6s | .8759 / .1256 / .4864 / 51.7s |
| **LightGBM** | **.8946 / .1687 / .5366 / 47.8s** | .8874 / .1626 / .5196 / 44.8s | .8928 / .1682 / .5408 / 16.3s | .8938 / .1662 / .5402 / 67.2s | .8943 / .1658 / .5402 / 25.5s |
| **XGBoost** | .8961 / .1678 / .5486 / 97.0s | .8927 / .1685 / .5341 / 90.8s | .8943 / .1666 / .5414 / 47.8s | .8948 / .1686 / .5396 / 169.7s | .8950 / .1658 / .5378 / 60.3s |

*(cells read ROC-AUC / PR-AUC / TPR@5%FPR / train seconds)*

Selection rule, fixed in `config.yaml` before the sweep ran: `primary_metric: pr_auc`
(explicitly not accuracy, not ROC-AUC), applied mechanically in `training.py` as
`max(results, key=lambda k: results[k][primary_metric])`. That rule picks
**`lightgbm / none`**, PR-AUC 0.1687. CatBoost and classic `GradientBoostingClassifier`
were not included in this sweep — reasons for both are in section 5.

Two claims are supported by this table with very different amounts of evidence, and the
rest of this document keeps them separate rather than blurring them into one verdict.

## 2. The claim the table supports strongly: boosting beats bagging and linear models

Every one of the ten LightGBM/XGBoost cells (PR-AUC 0.1626–0.1687) sits above every one of
the ten Random Forest/Logistic Regression cells (PR-AUC 0.1222–0.1446), across all five
imbalance strategies, with zero overlap and roughly a 0.018 gap at the closest boundary.
That is complete rank separation, and it holds for a structural reason, not a tuning
accident.

**Boosting reduces bias; bagging reduces variance.** Random Forest bootstraps and averages
independently-grown, already low-bias trees — a device for reducing variance around a
fixed bias floor. LightGBM and XGBoost fit each new tree to the *residual gradient* of the
ensemble so far, which is a bias-reduction device: capacity keeps getting spent closing the
specific gap the model is still missing. At 700,000 rows, variance is cheap and bias is the
binding constraint, so boosting is attacking the error term that actually matters here.

The split-criterion mechanics make this concrete. Random Forest splits on Gini impurity
computed from raw class counts. At 1.1% prevalence, Gini impurity is 0.0218 — the impurity
reduction available from isolating a small fraud pocket is small, and loses out to splits
that shave noise out of the 98.9% majority class. Fraud structure is close to invisible to
that criterion. The diagnostic is right there in the table: `random_forest`'s *best* arm
is `random_undersample` (PR-AUC 0.1446), trained on roughly 85,000 rows — **Random Forest
got better with 8x less data**, because undersampling to a ~9% positive rate restores the
contrast Gini needs. LightGBM and XGBoost split on sums of gradients and Hessians instead
(`g = p − y`, `h = p(1−p)` for logistic loss), and under 1.1% prevalence a fraud row that
the model hasn't learned yet carries `|g| ≈ 1` while a confidently-scored legitimate row
carries `|g| ≈ 0`. Boosting is already adaptively upweighting the rare class, every round,
before any resampling is layered on — which is also why resampling barely moves it (section
4).

Logistic Regression has a second, separate ceiling: it can only use interactions it is
explicitly handed. The `risk_x_income` feature in this pipeline exists because someone had
to manually construct it — a depth-3 tree finds that same interaction, and a few hundred
boosting rounds finds thousands of others, unsupervised. That gap shows up directly:
Logistic Regression's best PR-AUC (0.1431) trails LightGBM's worst (0.1626). It is kept as
a candidate anyway, and should stay in any future sweep, for three concrete reasons that
have nothing to do with winning: its coefficients are directly auditable for adverse-action
explanations a bank is legally required to provide; it trains in 3.5 seconds, making it a
cheap canary for feature leakage (a Logistic Regression that suddenly scores 0.99 AUC means
a feature is leaking, long before a boosted model would surface the same problem); and it
sets a documented floor that any more complex model has to clear. One claim it does *not*
support is monotonicity as a unique advantage — both LightGBM and XGBoost support monotone
constraints directly (`monotone_constraints`), so if the bank needs "higher credit risk
score never decreases predicted fraud probability" as a hard guarantee, that is available
without giving up the boosted model.

**Caveat that belongs in this section, not hidden from it:** Random Forest ran under real
memory constraints (`max_depth: 15`, `max_samples: 0.5`, capped at 6 threads — see the
incident log in `final_report.md` section 9b) rather than an unconstrained configuration.
Its measured 0.024 PR-AUC gap behind LightGBM is therefore an upper bound on the true gap,
not a floor. That does not change the direction of the verdict — an unconstrained Random
Forest would also be markedly slower, reinforcing the operational case — but the comparison
was not resource-neutral, and a document that pretends otherwise is the kind of thing a
careful reviewer catches and then distrusts everything else in.

## 3. The claim the table does *not* support: "LightGBM is the best algorithm"

This is the part worth being precise about, because the table is frequently read too
confidently here.

`lightgbm / none` (PR-AUC 0.1687) beats the closest XGBoost arms (`xgboost / class_weight`
0.1685, `xgboost / smote` 0.1686) by 0.0001–0.0002. Three ways to see that this margin is
not evidence of a real ordering:

- **It is smaller than the resolution of the metric itself.** PR-AUC here is
  `average_precision_score` over 1,655 positives, so the maximum influence any single
  fraud case has on the score is bounded by 1/1655 ≈ 0.0006. The LightGBM-over-XGBoost
  margin is about 17% of the effect of moving *one* fraudulent application in the ranking.
- **It is dwarfed by each family's own noise.** Across the five imbalance strategies,
  XGBoost's PR-AUC spans 0.1658–0.1686 (range 0.0028) and LightGBM's spans
  0.1626–0.1687 (range 0.0061) — both larger, in one case 30–60x larger, than the
  0.0001–0.0002 gap being used to declare a winner between families.
- **It doesn't survive the train/test split.** `lightgbm / none` scores PR-AUC 0.1656 on
  the untouched test set versus 0.1687 on validation — a 0.0031 drop. The margin the
  selection rule acted on is about 3–6% of the amount the winning model's own score moved
  between validation and test.

The same pattern holds on the metrics XGBoost wins: `xgboost / none` leads on ROC-AUC
(0.8961 vs. 0.8946) and TPR@5%FPR (0.5486 vs. 0.5366), a TPR edge worth about 20 additional
frauds caught per 150,000 applications — a real quantity at bank volume. But that edge
(0.0120) is smaller than XGBoost's own spread across its five arms (0.0145, i.e.
`xgboost / none`'s TPR is the single best point in its own family's noise), and LightGBM's
best TPR arm (`random_undersample`, 0.5408) sits within 0.0006 of XGBoost's best undersampled
arm (0.5414). The two families interleave once every arm is on the table, not just the
"none" arms. When the gap between families is smaller than the spread within each family,
the table has not measured a difference — it has measured overlap.

**The honest claim, and the one that survives scrutiny, is a narrower one:** gradient
boosting is decisively better than bagging or linear models on this data; LightGBM and
XGBoost are statistically indistinguishable on the metric this project pre-registered as
its selection criterion; and LightGBM was chosen on cost and operational grounds once the
primary metric came back silent, not because the primary metric handed it a win.

### The steelman for choosing XGBoost instead

It deserves to be stated in full rather than dismissed. `xgboost / none` wins two of the
three headline metrics, including the one closest to a real operating point (TPR@5%FPR),
and it stays competitive at test time (test TPR 0.5586 in the deployed run, consistent with
what the sweep predicted). Its PR-AUC is also more *stable* across preprocessing choices
— range 0.0028 versus LightGBM's 0.0061 — which is itself a legitimate criterion when a
production pipeline might change its resampling step later. XGBoost's regularization
defaults (`lambda=1`, `min_child_weight=1`) are more conservative out of the box than
LightGBM's (`lambda_l1=0`, `lambda_l2=0` by default), and its sparsity-aware split finding
has a genuine, specific advantage on this dataset once the missing-value sentinels are
handled correctly (section 6) — something LightGBM's histogram binning does not do as
directly.

**The rebuttal is the same noise argument from above, applied to this specific case:**
the quantity being used to prefer XGBoost — its TPR@5%FPR lead — is the best cell in its
own family's spread, not a result that stands apart from it. Given that, the deciding
factors become legitimate secondary ones, not tie-breaking rationalizations: `lightgbm /
none` trains in 47.8 seconds against XGBoost's 97.0 — roughly 2x, which compounds directly
under any future hyperparameter search (a 200-trial tuning pass is the difference between
roughly 2.7 hours and 5.4 hours at this scale); and the winning configuration for both
families is `none` — no resampling step in the deployed pipeline — which is a stronger
production result than either family being ahead of the other, because it means training
distribution equals production distribution and predicted probabilities are calibrated to
the true 1.1% base rate (section 7). CatBoost is kept as a documented challenger rather than
run in this sweep (section 5); XGBoost should be treated the same way here — a bank running
champion/challenger governance already has the XGBoost arm as the natural challenger model,
and this document is the artifact that explains why, if a future sweep ranks it first, that
would not overturn this decision, because the decision was never resting on the margin
between the two in the first place.

## 4. Why rebalancing helps Random Forest but not either boosted model

Look again at the table: `random_forest`'s best arm is `random_undersample` — a real,
meaningful jump from PR-AUC 0.1367 to 0.1446. For LightGBM and XGBoost, no strategy beats
`none` by a meaningful margin, and `class_weight` is measurably *worse* for both
(LightGBM: 0.1626 vs. 0.1687; XGBoost: comparable ROC-AUC but noticeably lower TPR).

The general reason: ROC-AUC and PR-AUC are both *ranking* metrics, and rebalancing is, to
first order, a shift in the score distribution's intercept — a roughly monotone
transformation that changes how confident the scores are, not how they're ordered. It helps
only when the learner's capacity allocation was distorted by the base rate to begin with,
which is exactly Random Forest's Gini problem from section 2, and exactly what gradient-
based splitting already avoids.

There is also a mechanism specific to this project's configuration worth calling out,
because it explains why `class_weight` is not just neutral but actively the worst LightGBM
arm. `scale_pos_weight` reweights the loss, but it also silently distorts the
Hessian-based leaf-size regularizers both libraries rely on. At this base rate, an
individual row's Hessian is roughly `p(1−p) ≈ 0.0109`, so LightGBM's
`min_sum_hessian_in_leaf: 1.0` corresponds to needing about 92 unweighted rows' worth of
Hessian mass in a leaf. Once `scale_pos_weight ≈ 90` is applied, a *single* fraud row
contributes almost 0.98 Hessian on its own — nearly satisfying the leaf-size constraint by
itself. XGBoost's `min_child_weight: 5` (≈460 rows unweighted) collapses to roughly 5
positive rows under the same reweighting. The regularization configured is not the
regularization actually in effect once class weighting is turned on, and that predicts
exactly the table's result: `lightgbm / class_weight` is LightGBM's worst arm on every
metric.

SMOTE has its own, dataset-specific problem here. Missingness in this dataset is encoded as
negative sentinel values, not `NaN` (section 6) — `prev_address_months_count` is −1 in
71.3% of rows, `intended_balcon_amount` is negative in 74.3%. SMOTE interpolates linearly
between a minority-class row and its nearest neighbors, which means it will average real
values with sentinel values and can produce a synthesized row with, for example, "−0.4
months at previous address" — a value that means nothing, injected into exactly the
missingness pattern that legitimately helps identify synthetic-identity fraud (a thin credit
file is itself a fraud signal in this dataset, not noise to be smoothed over). That this
still costs 1.4–3.5x the training time of `none` for no PR-AUC benefit is the table
confirming a data problem, not the model failing to exploit synthetic data properly.

## 5. Algorithms considered and not run, and why that's a deliberate choice, not a gap

**CatBoost** was researched but not included in the empirical sweep. Its two headline
mechanisms both have a specific, checkable reason to expect a small effect on this dataset:

- *Ordered boosting* exists to correct "prediction shift" — the bias introduced when the
  same rows used to fit a tree are also used to compute the gradients that tree is trained
  on (Prokhorenkova et al., NeurIPS 2018). Their Theorem 1 shows this bias term scales as
  `1/(n−1)`. At `n = 700,000`, that term is on the order of 1.4×10⁻⁶ of its coefficient —
  the problem ordered boosting solves is largest on small datasets, which is exactly what
  CatBoost's own documentation says ("Ordered mode... usually provides better quality on
  small datasets"). It is also expensive: naive ordered boosting requires training up to
  `n` supporting models, and even the practical approximation is documented as
  computationally costly enough that CatBoost defaults to `Plain` boosting on CPU.
- *Ordered target statistics* solve real target leakage in categorical encoding, but the
  problem scales with cardinality. This dataset's categoricals top out at 7 distinct values
  (`employment_status`, `housing_status`); LightGBM's own partition-based categorical
  splitting (`max_cat_to_onehot: 4`, below this dataset's cardinality) already handles that
  cleanly without needing target statistics at all.

CatBoost remains a reasonable documented challenger for a future sweep — its symmetric,
oblivious trees are a real regularizer that could matter under a different feature set —
but running it here would likely have spent compute confirming a small effect the mechanism
already predicts to be small, which is why it wasn't prioritized in the version of the sweep
that shipped.

**Classic `GradientBoostingClassifier` and AdaBoost** were excluded from the start on a
complexity argument, not a hand-wave: pre-sorted split-finding costs `O(n_features · n log
n)` per node, versus `O(n_features · n)` for histogram-based `hist`/LightGBM/CatBoost. At
`n = 700,000`, the `log₂ n ≈ 20` factor alone makes classic GBM roughly an order of
magnitude slower per split, with no native categorical support to offset it. AdaBoost adds
a distinct problem for this data: it optimizes exponential loss via reweighting
misclassified rows, which is the most outlier-sensitive of the standard margin losses — at
1.1% prevalence with the label noise that real fraud data always carries, it will
disproportionately chase mislabeled positives rather than genuine fraud structure.

## 6. Corrections this analysis surfaced that the original framing got wrong

Two of the arguments commonly reached for in this kind of comparison do not survive contact
with this specific dataset, and it is worth saying so directly rather than repeating them:

**"Native categorical handling avoids a fragile one-hot pipeline" is not a LightGBM-specific
advantage here.** `src/models.py` already passes `enable_categorical=True` to XGBoost's
`DMatrix` (lines 133–135) — both libraries keep the categorical path native by default on
`none`. The real, narrower version of this argument is that choosing `strategy: none` is
what *preserves* native categorical handling for either family; the SMOTE arms are the ones
forced onto a dense, one-hot/imputed view, which is a fair reason to prefer `none` but not a
reason to prefer LightGBM over XGBoost specifically.

**"High-cardinality categoricals" was the wrong framing for this dataset.** Profiling the
raw file directly: `payment_type` (5 categories), `device_os` (5), `employment_status` (7),
`housing_status` (7), `source` (2). Maximum cardinality is 7. One-hot expansion would add
roughly 21 columns in place of 5 — a non-event, not a blowup. The categorical-handling
argument for LightGBM/XGBoost over one-hot encoding is a real, general argument (it matters
enormously on features like user IDs or merchant IDs with thousands of categories), but it
is not a material differentiator on BAF's Base variant, and any measured gap attributable to
encoding strategy on this dataset should be read as noise rather than a cardinality effect.

**Missingness is sentinel-encoded, not `NaN`, which means most of the "handles missing
values automatically" argument for LightGBM/XGBoost is currently inert.** There are zero
actual `NaN` values in `Base.csv` — missingness shows up as `-1` in
`prev_address_months_count` (71.3% of rows), negative values in `intended_balcon_amount`
(74.3%), `-1` in `bank_months_count` (25.4%), and similar sentinels in
`current_address_months_count`, `session_length_in_minutes`, and
`device_distinct_emails_8w`. The pipeline already converts these to `NaN` plus an explicit
`_is_missing` flag during feature engineering (`final_report.md` section 4), which is the
right fix — but it means any claim that a library's *built-in* missing-value handling (as
opposed to this project's explicit sentinel conversion) is doing the work would be
incorrect. XGBoost's sparsity-aware split finding, which learns an optimal default
direction for missing values per node, is the single strongest *specific* argument for
XGBoost over LightGBM once this conversion is in place, since it treats missingness as a
feature-specific asymmetric decision rather than routing it through a shared histogram bin.
This has not yet been isolated as an ablation — it's flagged in section 9 as the most
promising piece of unfinished analysis.

**GOSS and EFB — two of LightGBM's headline mechanisms in the original paper — were almost
certainly not active in the run that produced these results, and shouldn't be cited as the
reason LightGBM won.** `data_sample_strategy` defaults to ordinary bagging; GOSS is opt-in
(`data_sample_strategy: goss`) and was only split out as its own parameter in LightGBM
v4.0.0. Exclusive Feature Bundling targets sparse, high-dimensional feature spaces (its
motivating case is one-hot-encoded text); this pipeline's 30 dense numeric/low-cardinality
features on the `none` strategy give EFB nothing to bundle. The honest attribution for
LightGBM's speed and quality here is histogram-based binning and leaf-wise growth, not GOSS
or EFB — both of which remain worth trying as a follow-up (`data_sample_strategy: goss` on
this exact config) rather than being credited retroactively for a result they weren't
switched on for.

## 7. Metric choice: why PR-AUC decided this, and why the alternative isn't crazy either

`config.yaml` fixes `primary_metric: pr_auc` ahead of the sweep, and that choice deserves
its own justification, not just an assertion.

ROC-AUC is invariant to the base rate — it would report the same curve whether fraud were
1.1% or 50% of applications, because it's built from true-positive rate against
false-positive rate, and false-positive rate normalizes by the (very large) negative class.
Davis & Goadrich (ICML 2006) prove that ROC dominance and PR dominance are equivalent for
one curve strictly above another, but that this equivalence does **not** extend to area
under the curve — their own constructed example has one curve with the *lower* ROC-AUC
(0.813 vs. 0.875) but the *higher* PR-AUC (0.514 vs. 0.038). Saito & Rehmsmeier (PLOS ONE,
2015) make the base-rate sensitivity concrete: the same ROC operating point can correspond
to 160 false positives against 500 true positives in a balanced setting, or 1,600 false
positives against the same 500 true positives under 10x imbalance — an identical ROC curve
describing a tenfold difference in the false-positive burden an investigations team
actually carries.

The arithmetic on this exact dataset makes the stakes plain. At population scale (1,000,000
applications, 1.103% fraud → 11,029 positives, 988,971 negatives), operating at 5% FPR
means 49,449 false positives. At the measured TPR@5%FPR of 0.548, that's 6,044 true
positives caught against 55,493 total flags — **precision of 10.9%, or roughly 8 false
alerts for every genuine fraud case an analyst reviews.** Even at perfect recall, 5% FPR
caps precision at 18.2% by arithmetic alone, independent of model quality — a bound worth
stating explicitly so no one reads a strong ROC-AUC as implying a workable investigation
queue at that operating point. PR-AUC is sensitive to exactly this cost because precision
compares false positives against true positives directly, rather than against the (mostly
irrelevant, from an investigator's chair) pool of true negatives.

That said, PR-AUC is not beyond challenge, and the document is stronger for citing the
challenge rather than pretending the metric choice is settled science. McDermott et al.
(NeurIPS 2024, "A Closer Look at AUROC and AUPRC under Class Imbalance") argue formally
that PR-AUC is not a universally superior *model-selection* criterion under imbalance, and
specifically warn that it can favor improvements concentrated in subpopulations that already
have higher positive rates — which is directly relevant given this project's own fairness
finding (below) that the over-50 group has close to 3x the fraud prevalence of the rest of
the population. The defensible position is not "PR-AUC is simply correct" but that it is
the more informative *reporting* metric for this specific use case, because its baseline is
calibrated to the true 1.1% prevalence rather than fixed at 0.5, and it tracks the quantity
an investigations team actually experiences — while acknowledging that a per-subgroup
PR-AUC breakdown, not just an aggregate one, is the right follow-up given the fairness
result below.

## 8. Fairness holds regardless of which algorithm wins

`customer_age > 50`, the protected attribute defined in the BAF paper, shows the same
disparity on the deployed LightGBM model as the paper reports for this dataset in general:
on the untouched test split, at the threshold hitting 5% FPR overall, the age ≤50 group has
FPR 0.0445 (n=143,629) against 0.1748 for the age >50 group (n=6,371) — a predictive-equality
ratio of 0.255, meaning applicants over 50 are falsely flagged roughly 3.9x more often. This
number is not a LightGBM artifact to be fixed by picking XGBoost instead — it reproduces
across the sweep and matches the source paper's own published finding, which points to it
being a property of the *data* (a real, measurable relationship between age and the
features that predict fraud in this dataset) rather than a modeling choice. No mitigation is
applied in the current deployed model; this is reported as a finding for a bank fairness
review to act on, not something the algorithm comparison in this document resolves.

## 9. What would make this document evidence instead of argument

The reasoning above is honest about where the evidence is strong (family-level: boosting
over bagging/linear) and where it rests on a judgment call under thin margins (LightGBM over
XGBoost specifically). Converting the thin part from argument into measurement is cheap
relative to the rest of this project, and is the natural next step:

1. **Paired bootstrap over the saved validation predictions** (1,000 resamples, report
   P(LightGBM PR-AUC > XGBoost PR-AUC)) — the honest prediction going in is close to 0.5,
   and a measured tie is considerably more persuasive to a skeptical reviewer than an
   asserted one.
2. **Repeat the top four arms across 5 seeds**, report mean ± standard deviation, so the
   0.0001 gap can be read against an actual noise estimate instead of inferred from
   within-family spread as a proxy.
3. **Separate the early-stopping metric from the selection metric.** Both `train_lightgbm`
   and `train_xgboost` currently early-stop on `metric: "auc"` using the same validation
   rows the sweep later ranks by PR-AUC — each candidate's tree count was tuned to maximize
   ROC-AUC, then candidates were compared on PR-AUC computed on data they'd already
   influenced their own stopping point against. This is very likely why validation PR-AUC
   (0.1687) exceeds the test PR-AUC (0.1656): early stopping on `average_precision` directly,
   or on a held-out inner split, would close that gap and could plausibly move the
   0.0001-wide ranking.
4. **Re-run with sentinel values converted to `NaN` before training** rather than only in
   the engineered feature set, so XGBoost's sparsity-aware split finding (section 6) gets a
   fair chance to show its documented advantage on the two features that are 70%+ missing.
5. **Add a temporal holdout** (train on months 0–5, test on 6–7, `month` dropped) alongside
   the current random split. All three current splits (train/val/test) contain rows from
   all 8 months, and fraud prevalence in this dataset drifts from 0.87% in month 2 to 1.47%
   in month 7 — a random split lets future-regime rows leak into training in a way a bank's
   model-risk review will likely flag, even though it correctly mirrors the
   competition-style benchmark this project's numbers are otherwise compared against.
6. **Report calibration** (Brier score, or expected calibration error), not just ranking
   metrics — the deployed threshold is chosen by minimum expected cost
   (`cost_fp=50, cost_fn=500`), which is only a principled procedure if the underlying
   probabilities are calibrated, and `strategy: none` is the configuration most likely to
   preserve that property (section 3's steelman rebuttal), but it has not been directly
   measured here.

None of these six items are expected to overturn the family-level conclusion in section 2.
Several of them could plausibly move the LightGBM-vs-XGBoost margin in either direction —
which is exactly the point: that margin was never the load-bearing part of this decision.
