# R4 — Fairness, Explainability, and the Pitch

**What this file is for.** Everyone else in the room will submit a LightGBM model and an accuracy number. This file is the plan for winning instead. It is built on the one fact most competitors will never learn: **BAF was purpose-built for fairness research.** That is not decoration on your submission — it is the submission.

Every library claim below was **executed**, not recalled. Where something failed, it says so. Verification environment and dates are in §9.

> **Read §2.4 before you plan your time.** The most attractive-looking library in this space (FairGBM, by the same team that made the dataset) did not work when tested. Knowing that now is worth more than anything else in this file.

---

## 1. Fairness, done correctly

### 1.1 The one metric that matters here, quoted from the paper

The paper defines its fairness metric in one sentence. Quoted verbatim from *Turning the Tables* §3:

> "we want to guarantee that the probability of being wrongly classified as a fraudulent application is independent of the sensitive attribute value of the individual. Hence we measure the ratio between FPRs, i.e., **predictive equality**. The ratio is calculated by **dividing the FPR of the group with lowest observed FPR with the FPR of the group with the highest FPR**."

Three consequences that people get wrong:

1. It is **min/max**, not group-A-over-group-B. The value is always in `[0, 1]`, and **1.0 is perfectly fair**. If you compute `FPR_old / FPR_young` you will get a number above 1 and your plot will be upside-down relative to the paper's Figure 1.
2. It is a **ratio**, not a difference. `fairlearn`'s `MetricFrame.difference()` is a different quantity. Use `.ratio()`, or compute it yourself.
3. It is measured on **legitimate applicants only** (the FPR denominator is the negatives). Fraudsters' experience is not what this metric is about.

```python
def predictive_equality(y_true, y_pred, group):
    """Paper's definition: min(group FPRs) / max(group FPRs). 1.0 = parity."""
    import numpy as np
    fprs = []
    for g in np.unique(group):
        m = (group == g) & (y_true == 0)      # negatives in this group only
        fprs.append(y_pred[m].mean())
    fprs = np.array(fprs)
    return fprs.min() / fprs.max(), fprs
```

### 1.2 The protected attribute — and a discrepancy worth knowing

The paper states its three candidate protected attributes and its choice:

> "the dataset provides the age, personal income, and employment status of the applicant… we focus on customer age… we create a categorical version by separating applicants with **age>50** in one group and **≤ 50** in the other group."

And on why 50:

> "we defined a threshold for the studied protected attribute (age), at the value of 50. We selected this value as it represents a decent compromise between group size (**approximately an 80/20 split**) and prevalence (**approximately 2 times larger for the older group**)."

⚠️ **It is `age > 50`, not `age >= 50`.** You will see `>= 50` stated in secondary sources. It matters: `customer_age` is rounded to the decade (9 distinct values), so `> 50` puts the entire `50` bucket in the *younger* group. Using `>= 50` moves a whole decade across the boundary and changes both group sizes and the metric. Use `>`, and verify you land near the paper's 80/20 split:

```python
older = df["customer_age"] > 50
print(older.mean())                                    # expect ≈ 0.20
print(df.groupby(older)["fraud_bool"].mean())          # expect ≈ 2x higher for older
```

If those two checks don't reproduce, stop and find out why before you build anything on top.

### 1.3 The other three definitions, and why they are wrong here

For a binary prediction `Ŷ`, true label `Y`, protected attribute `A`:

| Criterion | Formal condition | Equalises | Right for BAF? |
|---|---|---|---|
| **Demographic parity** | `P(Ŷ=1 \| A=a)` equal ∀a | Selection rate | **No** — fraud prevalence genuinely differs ~2x by age group. Forcing equal flag rates forces you to *under*-police the higher-risk group and *over*-police the other. It ignores `Y` entirely. |
| **Equal opportunity** | `P(Ŷ=1 \| Y=1, A=a)` equal ∀a | TPR (recall) | **No** — this equalises how well you catch *fraudsters* per group. Nobody is harmed by an undetected fraudster's group membership. Wrong harm. |
| **Predictive equality** | `P(Ŷ=1 \| Y=0, A=a)` equal ∀a | **FPR** | **Yes.** Equalises how often *legitimate* applicants are wrongly denied. |
| **Equalized odds** | Both TPR and FPR equal ∀a | Both | Defensible, stricter, harder. Predictive equality is the paper's choice; matching it makes you comparable. |

**The sentence that makes this land with judges:** *this task is punitive.* A positive prediction denies someone a bank account. So the fairness question is not "does the model catch fraud equally" — it is "does the model wrongly punish equally". The paper says exactly this:

> "This metric is appropriate for our punitive setting, as a positive classification translates into denial of banking services."

### 1.4 How to compute the FPR ratio at a fixed global operating point

This is the step most people get wrong. **A fairness metric is meaningless without a stated threshold.** You must fix one global threshold, apply it to everyone, and *then* measure per-group FPR.

```python
import numpy as np

# 1. Choose the operating point on NEGATIVES ONLY -> gives exactly 5% global FPR
thr = np.quantile(scores[y_test == 0], 0.95)
pred = (scores >= thr).astype(int)

# 2. Now measure per-group FPR at that single shared threshold
pe, fprs = predictive_equality(y_test, pred, older_test)
print(f"FPR by group: {fprs}, predictive equality = {pe:.3f}")
```

Using the negative-score quantile is exact and avoids `roc_curve` grid artefacts. (In testing, `roc_curve`-derived thresholds landed at 4.62% actual FPR rather than 5.00% because of grid granularity — small, but it makes numbers irreproducible.)

### 1.5 The five mistakes that will sink a fairness slide

1. **Comparing FPRs at different thresholds per group and calling it a fairness measurement.** Per-group thresholds are a *mitigation* (§2.2). If you apply them and then report the FPR ratio, you have measured your own intervention, not the model. Report both, labelled.
2. **Reporting the ratio without the underlying rates and group sizes.** A ratio of small counts is noisy. Always show `FPR_young`, `FPR_old`, and the negative counts behind each. With ~20% of the data in the older group, its FPR is estimated from ~4x fewer samples.
3. **No confidence intervals.** Bootstrap them. Two test months is a thin basis for any claim, and month-to-month wobble will look like a trend when it is noise.
4. **Conflating base rates with bias.** The older group genuinely has ~2x the fraud prevalence. That is prevalence disparity deliberately built into the dataset — *not itself* unfairness under predictive equality, which conditions on `Y=0`. If you say "the model is biased because older people have more fraud," a judge who knows the dataset will correct you.
5. **"Fairness through unawareness."** Dropping `customer_age` does not make the model fair; age is reconstructible from correlated features. Be careful how you test this — see the honesty note in §2.1.

---

## 2. Mitigation that actually works on GBDTs

Ranked by (benefit ÷ effort ÷ risk). Measured numbers below come from controlled synthetic runs (§9) — they demonstrate *mechanism and direction*, and **you must reproduce them on real BAF**. Do not quote my numbers.

### 2.1 Tier 1 — Sample reweighting (do this first)

Reweight so each group contributes equal *negative* mass, then train normally. One `sample_weight` argument, no new dependency, no change to inference, no protected attribute needed at predict time.

```python
w = np.ones(len(y_tr))
neg = (y_tr == 0)
for g in (0, 1):
    m = neg & (grp_tr == g)
    w[m] = neg.sum() / (2 * m.sum())
clf = lgb.LGBMClassifier(...).fit(X_tr, y_tr, sample_weight=w)
```

In testing this moved predictive equality **0.172 → 0.571 with TPR@5%FPR unchanged at 0.0876**. That is a large fairness gain for zero performance cost and about four lines of code. It is the best effort-to-benefit ratio available and it is where you should start.

> **Honesty note on unawareness.** In my synthetic data, dropping the protected attribute *did* improve the ratio (0.643 → 0.960) — but only because I generated age with no proxy features. Real BAF has proxies (`credit_risk_score`, `proposed_credit_limit`, housing/employment status all correlate with age). **Run this ablation yourself on real data.** If it improves fairness there too, that is a finding; if the fairness gap survives dropping the column — which is what I expect — that is a *better* finding, and it is the cleanest possible demonstration of why unawareness is not a fairness strategy.

### 2.2 Tier 1 — Group-specific thresholds (most effective, but read the warning)

Pick a threshold per group so each group hits the same FPR. This achieves near-perfect parity essentially by construction:

```python
thr_g = {g: np.quantile(scores[(grp == g) & (y == 0)], 0.95) for g in (0, 1)}
pred  = np.where(grp == 1, scores >= thr_g[1], scores >= thr_g[0]).astype(int)
```

Measured: predictive equality **1.000**, TPR@5%FPR **0.0876 → 0.0657**. So it works, and it costs about a quarter of your recall. That is the trade-off, stated honestly.

⚠️ **The legal warning you must deliver alongside it.** This applies a *different decision rule to people because of a protected characteristic*. In US credit law that is the classic shape of **disparate treatment**, which is a stronger prohibition than disparate impact and is generally not curable by a good motive. Present per-group thresholds as **an analytical upper bound on achievable fairness** — "this is the best parity obtainable by thresholding alone, and here is what it costs" — not as a deployment recommendation. A judge with finance background will respect that distinction enormously and will be unimpressed if you miss it.

### 2.3 Tier 2 — Fairlearn `ThresholdOptimizer` (works, with a serious trap)

**Verified real and installable:** `fairlearn` 0.14.0 installed cleanly on Python 3.14. The constraint string you want is `"false_positive_rate_parity"`. Verified signature:

```python
ThresholdOptimizer(*, estimator=None, constraints='demographic_parity',
                   objective='accuracy_score', grid_size=1000, flip=False,
                   prefit=False, predict_method='auto', tol=None)
# fit(X, y, *, sensitive_features)      <- sensitive_features REQUIRED, keyword-only
# predict(X, *, sensitive_features, random_state=None)
```

⚠️ **Trap 1 — it has no concept of your FPR budget.** It equalises FPR *between* groups without constraining the *level*. Measured on 1%-prevalence data:

| `objective` | Resulting global FPR | TPR | Predictive equality |
|---|---|---|---|
| `accuracy_score` (**default**) | 0.003 | **0.0000** | 0.256 |
| `selection_rate` | **1.000** | 1.000 | 1.000 |
| `true_positive_rate` | **0.481** | 0.664 | 0.974 |
| `balanced_accuracy_score` | 0.057 | 0.088 | 0.836 |

Three of the four objectives are **degenerate** at this prevalence. The default flags nobody (accuracy is maximised by never predicting fraud on a 1% positive class). `selection_rate` flags everybody. `true_positive_rate` achieves lovely fairness at a **48% false positive rate** — you would be declining half your legitimate customers. **Only `balanced_accuracy_score` produced a usable operating point.** If you demo this without checking the resulting global FPR, you will show the judges a beautiful fairness number attached to a useless model.

⚠️ **Trap 2 — its predictions are randomised.** `predict()` takes a `random_state`, and in testing **7.7% of rows changed decision** between two random seeds. For a bank, that means the same applicant with the same data gets a different answer depending on a coin flip — indefensible for adverse-action purposes, and a great thing to *say* you noticed.

**Verdict:** worth including as a benchmark point on your trade-off curve. Set `objective="balanced_accuracy_score"`, always report the achieved global FPR next to the fairness number, and mention the randomisation. Do not make it your headline method.

### 2.4 ⚠️ Tier 4 — FairGBM: real, published, and it did not work

This is the one that looks like a guaranteed win: a fairness-constrained LightGBM from **Feedzai — the same team that built BAF** — published at **ICLR 2023**. Its case study is the *Account Opening Fraud* dataset, which is BAF's real-world precursor: 8 months, ~1% fraud, a 5% FPR business requirement, age-group FPR parity. It is, on paper, purpose-built for your exact problem.

**What is genuinely true:**
- The paper is real: *FairGBM: Gradient Boosting with Fairness Constraints*, ICLR 2023, [arXiv 2209.07850](https://arxiv.org/abs/2209.07850). Repo: [github.com/feedzai/fairgbm](https://github.com/feedzai/fairgbm).
- The method is sound: cross-entropy-based differentiable proxies for FPR/FNR, optimised as a non-zero-sum two-player game (model player descends the *proxy*-Lagrangian; multiplier player ascends the *original* Lagrangian). It's a fork of LightGBM 3.x.
- The paper reports ~2x LightGBM training time, and an order-of-magnitude speedup vs. Fairlearn's Exponentiated Gradient.

**What I found when I actually ran it:**

| Test | Result |
|---|---|
| `pip install fairgbm` on Windows | **Exits 0 — appears to succeed** |
| `import fairgbm` on Windows | ❌ **Fails.** Wheel is tagged `py3-none-any` but contains only `lib_lightgbm.so` — a Linux binary. Error names `lib_lightgbm.dll`, which is not in the package. |
| Import on Linux (`python:3.11-slim`) | ❌ Fails until you `apt-get install libgomp1` |
| Import on Linux with `libgomp1` | ✅ **Works** — fairgbm 0.9.14, with numpy 2.4.6 / sklearn 1.9.0 |
| Training | ✅ Runs, 2.3–2.5x slower than LightGBM (matches the paper) |
| **Does the fairness constraint do anything?** | ❌ **No.** Output was **bit-identical** to unconstrained LightGBM in *every* configuration tested. |
| `constraint_group_column` (the documented low-level path) | ❌ **Segmentation fault**, core dumped |

The no-op finding is the decisive one, so here is the evidence rather than the assertion. I verified that the parameters reach the booster (`objective='constrained_cross_entropy'`, `constraint_type='FPR'`, `multiplier_learning_rate` set) and that `constraint_group` is correctly stored on the `Dataset` (`get_field('constraint_group')` returns the right 0/1 array). Then:

- `multiplier_learning_rate` of **0.01, 0.1, 1.0, 5.0, and 50.0** → `np.allclose` **True** across all pairs.
- Real group vs. **all-zeros** group vs. **random** group → identical.
- `constraint_type="FPR"` and the default `"FPR,FNR"` → identical.
- With and without `global_target_fpr=0.05`, `objective_stepwise_proxy`, `constraint_stepwise_proxy` → identical.
- sklearn wrapper and the low-level `fairgbm.train()` API → identical to each other and to plain LightGBM.

If the constraint were engaging at all, `mlr=0.01` and `mlr=50` could not produce the same model. Training *is* ~2.3x slower, so the extra computation runs — it just never reaches the trees. The last PyPI release is **November 2022**, and open repo issue **#55** acknowledges unfixed numpy 2.x / sklearn 1.6+ incompatibilities.

**Verdict: do not bet your hackathon on FairGBM.** Budget zero hours. It is Linux-only, silently installs then fails on Windows/macOS, and in my testing the headline feature was inert.

**But do mention it.** This is a genuine credibility asset, and it converts a dead end into a strength:

> "The dataset authors also published FairGBM, a fairness-constrained GBDT at ICLR 2023 — in-processing would be the principled approach here. We evaluated it. The PyPI wheel is tagged as universal but ships only a Linux binary, so it installs without error and then fails on import on Windows. On Linux we got it running, and the fairness constraint produced output bit-identical to unconstrained LightGBM across every multiplier learning rate from 0.01 to 50. So we used post-processing and reweighting instead, and we're reporting that trade-off honestly rather than claiming an in-processing result we couldn't reproduce."

That is a paragraph almost nobody else in the room can deliver, and it demonstrates exactly the engineering judgment judges are trying to detect.

**If you still want to try it** (do this in the first 20 minutes or not at all): Google Colab or `docker run python:3.11-slim` + `apt-get install -y libgomp1` + `pip install fairgbm`. Verify it does something before building on it — train with `mlr=0.01` and `mlr=50` and check the predictions differ. If they don't, walk away.

### 2.5 What to skip

- **Fairlearn `ExponentiatedGradient` / `GridSearch`** — outputs binary predictions, not scores, so you cannot threshold to a 5% FPR operating point. The FairGBM paper excluded EG from its AOF comparison for exactly this reason: it "produces a randomized binary classifier that implicitly uses a 0.50 decision threshold… severely sub-optimal to maximize TPR." Structurally wrong for this problem.
- **AIF360** — heavy, awkward API, no advantage over fairlearn here.
- **SMOTE-for-fairness variants** — see `R2-imbalance-truth.md`; adds risk, unlikely to help ranking metrics.

---

## 3. The fairness–performance trade-off curve

**This is your single strongest visual.** Claiming "our model is fair" is unfalsifiable and judges have heard it. Showing the *frontier* — and where on it you chose to sit, and why — is a different class of submission. It reframes fairness from a box-tick into an engineering decision with a cost you have quantified.

### 3.1 The main plot: TPR@5%FPR (x) vs predictive equality (y)

This deliberately mirrors the paper's Figure 1 axes — `Performance (Recall)` against `Fairness (FPR Ratio)` — so a judge who knows the paper immediately recognises it, and you can say "we reproduced the paper's Figure 1 on our own models."

Generate the frontier by sweeping a per-group FPR target from "one shared threshold" to "fully equalised":

```python
rows = []
for alpha in np.linspace(0, 1, 11):
    # alpha=0 -> single global threshold ; alpha=1 -> per-group equalised FPR
    t = np.where(grp == 1,
                 (1-alpha)*thr_global + alpha*thr_grp[1],
                 (1-alpha)*thr_global + alpha*thr_grp[0])
    pred = (scores >= t).astype(int)
    pe, fprs = predictive_equality(y_te, pred, grp)
    rows.append(dict(alpha=alpha, tpr=recall_score(y_te, pred),
                     global_fpr=pred[y_te==0].mean(), pred_equality=pe))
frontier = pd.DataFrame(rows)
```

Measured shape (synthetic — reproduce it yourself):

| alpha | TPR | global FPR | predictive equality |
|---|---|---|---|
| 0.00 | 0.0876 | 0.050 | 0.172 |
| 0.25 | 0.0730 | 0.045 | 0.272 |
| 0.50 | 0.0657 | 0.043 | 0.444 |
| 0.75 | 0.0584 | 0.045 | 0.678 |
| 1.00 | 0.0657 | 0.050 | **1.000** |

Note it is **not monotone** — alpha=0.75 scored *worse* TPR than alpha=1.0. Say so if it happens on your data. Interpolating thresholds is a convenient sweep, not a true Pareto frontier. If you want a clean frontier, sweep the target FPR *within each group* directly and keep only non-dominated points.

### 3.2 Plot every method as a point on the same axes

Overlay: unconstrained LightGBM · reweighted · per-group thresholds · `ThresholdOptimizer` · (FairGBM, if you got it working). Mark the Pareto front. Then put a dot on your chosen operating point and **write one sentence on the slide explaining why that dot**.

### 3.3 The second plot that almost nobody will make

**Predictive equality as a function of the global FPR budget.** Fairness is not a property of a model; it is a property of a model *at a threshold*. Measured:

| Global FPR budget | TPR | Predictive equality |
|---|---|---|
| 1% | 0.015 | 0.139 |
| 5% | 0.088 | 0.172 |
| 10% | 0.190 | 0.185 |
| 20% | 0.314 | 0.256 |

The disparity is **worst at the tightest budget** — exactly where a real fraud system operates. That is a genuinely interesting, non-obvious finding, it is cheap to produce, and it lets you say: *"tightening the fraud filter doesn't just cost recall, it concentrates the harm on the older group."* If that reproduces on real BAF, it is your best single slide.

---

## 4. Explainability that isn't a SHAP bar chart

Every team that does explainability will show one global SHAP summary plot. Assume that slide is worth zero. Here is how to be worth something.

### 4.1 Verified SHAP mechanics (get these right)

```python
import shap
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_sample)
sv = sv[1] if isinstance(sv, list) else sv     # keep this defensive line
```

⚠️ **A confusing gotcha, verified:** on `shap` 0.52.0 with `LGBMClassifier`, `shap_values()` emits a `UserWarning` saying output "has changed to a list of ndarray" — but it actually returns a **plain `(n, n_features)` ndarray**, and `expected_value` is a **scalar float**. The warning contradicts the behaviour. The defensive line above handles both, so keep it.

The modern callable API returns an `Explanation` object and is cleaner if you're plotting: `shap.Explainer(model)(X)` → `.values` `(n, f)`, `.base_values` `(n,)`.

**Zero-dependency alternative** — LightGBM computes exact tree SHAP natively, no `shap` install:
```python
contribs = model.predict(X, pred_contrib=True)   # (n, n_features+1), last col = base
```
Each row sums to the raw log-odds. Useful if `shap` install is slow or you want to sanity-check.

### 4.2 The reason code for a declined application

This is the deliverable, not the bar chart. Regulation B §1002.9(b)(2) is worth putting on a slide verbatim, because it is startlingly on-point and it has been law for decades:

> "The statement of reasons for adverse action… must be **specific** and indicate the **principal reason(s)** for the adverse action. Statements that the adverse action was based on the creditor's internal standards or policies **or that the applicant… failed to achieve a qualifying score on the creditor's credit scoring system are insufficient**."

"You didn't pass our model's threshold" is *expressly declared insufficient in the regulation text*. That single quote justifies your entire explainability section.

Build a per-decision artefact like this:

```
APPLICATION 4471-A  ·  DECLINED  ·  score 0.0306  (threshold 0.0257)
Principal reasons (ranked):
  1. Email address bears little resemblance to applicant name
  2. Internal credit risk score below acceptance range
  3. Elevated application velocity from this source in past 6h
Decision flips if:  name_email_similarity ≥ 0.42   (currently 0.11)
Reviewer: [ Confirm ]  [ Override ]     Model version 2026-08-13-a
```

Ranked, in plain language, tied to features actually in the model, with an actionable counterfactual and a human override. That is what a bank actually ships.

### 4.3 The differentiator: audit your own reason codes

Here is the sharp point, and it is the most sophisticated thing in this file. **SHAP explains the *score*; the adverse action is the *threshold crossing*.** These come apart. A feature can have a large SHAP value while being irrelevant to whether the applicant was declined; a feature with a modest SHAP value can be the one that actually flips the decision.

So measure it. For each declined applicant, compute a **pivotality** ranking — set each feature to its median among *approved* applicants and see how far the score drops — then compare against the SHAP ranking:

```python
approved = X_test[scores < thr]
piv = {}
for c in X_test.columns:
    alt = row.copy(); alt[c] = approved[c].median()
    piv[c] = score_of(row) - model.predict_proba(alt)[:, 1][0]
```

Measured: **Kendall tau between SHAP-ranked and pivotality-ranked reasons ≈ 0.51** across declined applicants. The two rankings agree about half the time. That is a real, quantified, honest finding produced in ~15 lines, and it is exactly the kind of thing that wins a room:

> "We didn't just generate reason codes — we validated them. The features SHAP ranks highest are only moderately correlated with the features that actually flip the decision. So we rank our reason codes by decision pivotality, not by attribution magnitude."

Supporting literature if challenged: Barocas, Selbst & Raghavan, *The Hidden Assumptions Behind Counterfactual Explanations and Principal Reasons* (FAT\* 2020, [arXiv 1912.04930](https://arxiv.org/abs/1912.04930)) — directly about US credit-law principal reasons; and Kumar et al., *Problems with Shapley-value-based explanations as feature importance measures* (ICML 2020, [arXiv 2002.11097](https://arxiv.org/abs/2002.11097)).

**Also state the baseline problem:** SHAP values depend on the background distribution you choose. Change it and your top-3 reasons change. Say which baseline you used and that it was a deliberate choice.

### 4.4 Regulatory framing — get this right or don't say it

This is where overconfident teams get destroyed by a knowledgeable judge. Three corrections most sources will lead you into:

⚠️ **1. ECOA / Regulation B governs *credit*, not deposit accounts.** BAF is bank *account opening*. If there's no credit feature, the ECOA specific-reasons duty doesn't attach; **FCRA** is the operative regime (15 USC 1681m — and note FCRA requires notice and CRA identification, *not* specific reasons). Deposit denials typically route through specialty consumer reporting agencies like **ChexSystems** or **Early Warning Services**.
   → **Say:** *"ECOA is the gold standard for what explaining an ML denial means, and it's the right design target — even though a pure deposit product is FCRA-governed, and many account-opening flows bundle overdraft or credit features."* Accurate, and shows you know the difference.

⚠️ **2. The CFPB AI circulars were withdrawn.** Circulars **2022-03** and **2023-03** — the ones saying a creditor "cannot justify noncompliance… based on the mere fact that the technology it employs… is too complicated or opaque" — were **withdrawn effective 12 May 2025** along with 66 other guidance documents. The CFPB still hosts the pages without a withdrawal banner, so "the page is live" proves nothing.
   → **Say:** *"CFPB withdrew its AI adverse-action circulars in 2025 as a deregulatory step — but they never created the duty. 12 CFR 1002.9(b)(2) did, and that's unchanged. Withdrawing an interpretation doesn't repeal the rule, and ECOA carries a private right of action."* That is a stronger story than pretending they're current.

⚠️ **3. The EU AI Act probably *excludes* you — and that's your best line.** Annex III 5(b) classifies creditworthiness evaluation as high-risk **"with the exception of AI systems used for the purpose of detecting financial fraud."** A BAF-style model sits squarely in that carve-out. Also: high-risk obligations were **deferred from 2 August 2026 to 2 December 2027** by the AI Digital Omnibus. Do not cite August 2026 as a live deadline.
   → **Say:** *"The EU AI Act carves fraud detection out of its high-risk category, so there's no compliance backstop here. That's precisely why the false-positive harm has to be handled by design rather than by regulation."*

On **GDPR**: Article 22 gives the right not to be subject to solely-automated decisions with legal/significant effects, and Art 22(3) guarantees human intervention, expressing a view, and contesting — **the word "explanation" appears in Recital 71, not in the binding article.** Whether GDPR contains a "right to explanation" is genuinely contested (Wachter et al. 2017 say no; Selbst & Powles 2017 say yes via Arts 13–15). Two CJEU cases have since narrowed the gap: **C-634/21 SCHUFA** (2023) brought credit scores themselves inside Art 22, and **C-203/22 Dun & Bradstreet** (2025) held trade secrecy can't be used to unilaterally refuse Art 15(1)(h) information. Present it as contested-but-converging; don't claim a blanket right to explanation.

---

## 5. Calibration and operating points

### 5.1 Why a bank cares

Ranking answers "is A riskier than B?". Calibration answers "of the applications scored 0.03, do ~3% actually turn out fraudulent?" A bank needs the second because the score feeds a *monetary* decision: expected loss = probability × exposure. If the probability is 3x off, every downstream number is 3x wrong — and ranking metrics (AUC, average precision) are **completely invariant** to that error. That is why AUC alone cannot be your evidence.

The most persuasive calibration demo for a business audience isn't a reliability diagram — it's this: *sum the scores in your alert queue and compare to the actual number of frauds in it.* If calibrated, they match. That single comparison lands harder than any ECE number.

### 5.2 ⚠️ The sklearn API changed — most tutorials are now wrong

Verified on **scikit-learn 1.9.0**:

```python
CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=None,
                       n_jobs=None, ensemble='auto')
```

`cv="prefit"` was deprecated in 1.6 and **removed in 1.8**. I confirmed it now raises `InvalidParameterError`. The replacement is `FrozenEstimator`, and I verified it works with `LGBMClassifier`:

```python
from sklearn.frozen import FrozenEstimator          # sklearn >= 1.6
cal = CalibratedClassifierCV(FrozenEstimator(fitted_lgbm), method="isotonic")
cal.fit(X_cal, y_cal)      # ALL of this data is used for calibration
```

Fit the calibrator on a **disjoint slice** — e.g. train on months 0–4, calibrate on month 5, test on 6–7.

Also: `brier_score_loss`'s second argument was renamed `y_prob` → `y_proba` (removed 1.7). **There is no ECE function in scikit-learn** — write it yourself.

⚠️ **Calibration can make things worse.** In testing, isotonic calibration *increased* Brier score (0.00918 → 0.01136) on an already-reasonable model. Always check; don't assume calibrating helps. Sigmoid/Platt is strictly monotone so it **cannot change AUC**; isotonic can create ties and change it slightly.

### 5.3 Imbalance handling destroys calibration — quote LightGBM's own docs

For both `is_unbalance` and `scale_pos_weight`, the LightGBM documentation says verbatim:

> "while enabling this should increase the overall performance metric of your model, it will also result in **poor estimates of the individual class probabilities**"

**The library authors tell you the probabilities are broken.** Put that on a slide.

**Recommendation: don't use either in your primary model.** Handle imbalance at the *threshold*, not in the loss. Then use the weighted model as a *demonstration*: show its reliability diagram is catastrophically off, apply the correction, show it snaps back with AUC bit-identical.

Correction for a positive-class weight `w` (derive it on the slide — it's two lines and very defensible):
```
p = p_s / (p_s + w·(1 − p_s))
```
Sanity check: `w = n_neg/n_pos ≈ 99`, `p_s = 0.5` → `p ≈ 0.0101` ≈ the base rate. For *undersampling* keeping a fraction β of negatives, the analogous form is `p = βp_s / (βp_s − p_s + 1)` (Dal Pozzolo et al., SSCI 2015). Both are strictly monotone, so **neither changes AUC** — they move only the probability scale, which is the point.

### 5.4 Defending your threshold in business terms

Expected-cost minimisation (Elkan, IJCAI 2001). With `C_FP` = cost of declining a genuine customer and `C_FN` = cost of a missed fraud:

```
t* = C_FP / (C_FP + C_FN)
```

Only the **ratio** matters:

| `C_FN : C_FP` | optimal threshold |
|---|---|
| 10 : 1 | 0.091 |
| 20 : 1 | 0.048 |
| 50 : 1 | 0.020 |
| 100 : 1 | 0.010 |

At 100:1 the optimal threshold lands essentially *at the base rate* — which is why 0.5 is never right in fraud. **And this formula is only valid on a calibrated score.** That is the sentence connecting your calibration work to your business case; make it your thesis line.

**Translate 5% FPR into human terms.** With 1M applications and ~99% legitimate, 5% FPR ≈ **49,500 wrongly-flagged genuine customers**. At 3 minutes of review each that's ~2,475 analyst-hours ≈ **15 full-time reviewers**. The paper chose TPR@5%FPR because it is a *customer-friction and capacity budget*, not a statistical convenience — the paper says each false positive is "a dissatisfied customer that may wish to change the banking company." Say this out loud; it shows you understand why the benchmark is defined the way it is.

**Also note:** ROC-AUC is misleading here. With ~990,000 true negatives, going from 5,000 to 15,000 false alerts — a 3x workload increase — moves FPR only from 0.005 to 0.015. Invisible on a ROC plot, catastrophic in the review queue. Report precision-recall and TPR at fixed FPR, and draw the PR baseline at the base rate (~0.011), not 0.5.

---

## 6. Temporal robustness

### 6.1 The protocol, from the paper

> "the first six months for training, the last two for testing. This is common practice in the fraud domain… as more recent data tends to be more faithful to the data's distribution when models are put in production."

Use **three** slices, not two: train on months 0–4, fit the calibrator *and choose the threshold* on month 5, evaluate on 6–7. If you pick the threshold on the same data you report, your operating point is optimistically biased.

### 6.2 What to measure, and the trap in my own demo

⚠️ **A caution learned the hard way.** In my synthetic run, in-sample months showed TPR@5%FPR ≈ 0.95 while out-of-sample months showed ≈ 0.09. That is not decay — it is the in-sample/out-of-sample gap. **Never plot training months and test months on the same "stability" chart without labelling them**, or you will show a cliff that means nothing and a judge will catch it.

Measure four separate things per month, because they decay independently:

1. **TPR@5%FPR with the threshold re-fitted each month** → does *ranking* decay?
2. **Alert rate at the frozen threshold** → does the *operating point* drift? This is what breaks review capacity in production and it moves even when ranking is stable.
3. **Mean predicted score vs. actual base rate** → calibration drift. BAF's prevalence moves between 0.85% and 1.5% across months, so expect this to move.
4. **Predictive equality per month** → see §6.4.

Put bootstrap CIs on every point. With two test months you cannot support a trend claim without them.

### 6.3 Population Stability Index

```
PSI = Σᵢ (Aᵢ − Eᵢ) · ln(Aᵢ / Eᵢ)
```

Conventional reading: <0.1 insignificant, 0.1–0.25 moderate, ≥0.25 significant. ⚠️ **These are rules of thumb, not a test** — and say so. The PSI null distribution depends only on sample size and bin count; the 0.25 cutoff is reasonable for n≈100–200 and far too loose at ~125,000 rows/month.

**Cheap, defensible upgrade:** build an empirical null. Split the reference month in half 200 times, compute PSI between halves, use the 95th percentile as your "no real shift" line. Compare each month against *that* instead of against 0.25. That takes ten minutes and is a genuine differentiator.

Two mechanics that matter: **fix the bin edges on the reference period** and reuse them (re-quantiling each month makes PSI meaningless), and state your epsilon for empty bins.

### 6.4 The move most teams will miss: is *fairness* stable over time?

Compute predictive equality per month **two ways** — at the frozen threshold, and at a per-month recalibrated 5%-FPR threshold. The gap between those curves separates two very different stories:

- *"The disparity is intrinsic to the model"* (both curves flat and low), versus
- *"The disparity appeared because the score distribution drifted under a fixed threshold"* (frozen curve degrades, recalibrated curve doesn't).

**An unchanged model can become unfair over time without anything about it changing.** That is a genuinely strong insight, it costs one extra loop, and I have not seen it in hackathon submissions. In my synthetic run predictive equality stayed stable (~0.16–0.20) while performance collapsed out-of-sample — fairness and performance decay on *different* clocks. Whether that holds on real BAF is exactly the kind of question worth answering on stage.

---

## 7. The pitch

### 7.1 The one-sentence thesis

> **"A false positive here isn't a metric — it's a real person denied a bank account, and on this dataset that harm lands about three times harder on applicants over 50. We built the model that measures that, prices it, and lets you choose where to sit."**

Everything else supports that sentence.

### 7.2 Narrative arc

**Setup → Reveal → Consequence → Response → Proof → Honesty**

1. **Setup.** This isn't credit-card fraud. It's *account opening* — no history, no relationship, one shot at judging a stranger. A false positive denies a real person a bank account.
2. **Reveal.** The dataset was purpose-built for fairness research. We measured predictive equality across age groups on our own model, and the older group is falsely flagged substantially more often.
3. **Consequence.** Quantify it. "At our operating point, N legitimate applicants over 50 were wrongly declined per 10,000 — versus M under 50."
4. **Response.** We can fix most of that. Here's the frontier, here's what each point costs in fraud caught, here's where we chose to sit and why.
5. **Proof.** Temporal split, calibrated scores, per-decision reason codes, stability over time.
6. **Honesty.** What we tried that didn't work, and what we'd do with another week.

### 7.3 Slide-by-slide (8 slides, ~6 minutes)

| # | Slide | The one thing it must do |
|---|---|---|
| 1 | **"A false positive is a person"** | One line: what this dataset is, what a positive prediction *does* to someone. No numbers yet. |
| 2 | **Baseline done right** | LightGBM, temporal split, TPR@5%FPR. State that accuracy is meaningless (98.9% = do-nothing). Show ROC-AUC *and* TPR@5%FPR. Establish competence fast — this is the only slide most teams will have. |
| 3 | **The reveal** | Predictive equality by age group at the 5% FPR operating point. One bar chart, two bars, huge font. Say the number out loud. |
| 4 | **The frontier** | TPR@5%FPR vs predictive equality, every method as a point, Pareto front marked, your chosen operating point circled with one sentence of justification. **Your best slide.** |
| 5 | **The threshold is a business decision** | Cost ratio → threshold table; 5% FPR = ~49,500 wrongly-declined customers ≈ 15 FTE reviewers. Calibration curve as the thing that makes the arithmetic valid. |
| 6 | **A declined application** | One real declined applicant, ranked plain-language reason codes, the counterfactual, the reviewer override. Quote Reg B: "failed to achieve a qualifying score… **insufficient**." |
| 7 | **Does it hold up?** | Per-month performance *and* per-month fairness, out-of-sample only, with CIs. PSI vs empirical null. |
| 8 | **What we got wrong** | FairGBM. Unawareness ablation. Reason-code audit (tau ≈ 0.5). What you'd do next. |

Slide 8 is not a weakness — it is the slide that separates a team that *did* the work from a team that *presented* it. Judges have watched five groups claim everything worked perfectly. Be the one that didn't.

### 7.4 The demo

**Do this:** a single interactive control — a slider for the FPR budget (or the cost ratio) — and four numbers updating live: **frauds caught · legitimate customers declined · predictive equality · reviewer hours needed**. Then click one declined applicant and show their reason codes and counterfactual.

Why it wins: it makes the judge *feel* the trade-off by moving it themselves. It proves the pipeline is real, not a screenshot. It takes about 30 lines with `streamlit` or `ipywidgets`. And it survives questions — when a judge asks "what if you tightened it?", you don't answer, you *move the slider*.

**Do not** demo live model training, anything requiring a network call, or a notebook you scroll through.

### 7.5 The five hardest questions

**Q1. "Isn't the older group just riskier? Their fraud rate is genuinely ~2x higher. Aren't you forcing the model to ignore real signal?"**
> The best question, and the one that separates people who understand the metric. Predictive equality conditions on `Y=0` — it compares false positive rates among *legitimate* applicants only. It's not affected by the base rate difference. We're not asking the model to ignore that older applicants are targeted more; we're asking that a *legitimate* 60-year-old isn't wrongly declined more often than a legitimate 30-year-old. Different prevalence doesn't justify different error rates on innocent people. And note the paper's framing: fraudsters impersonate older applicants because banks extend them larger credit lines — so the disparity originates in *fraudster* behaviour, and it's the older legitimate customer who pays for it.

**Q2. "You used per-group thresholds. Isn't that just discrimination in the other direction — and isn't it illegal?"**
> Yes, that's a real concern, which is why we present it as an analytical bound rather than a recommendation. Applying different decision rules by protected characteristic is the classic shape of disparate treatment in US credit law, and a good motive generally doesn't cure it. So we report it as "the best parity thresholding can achieve, and what it costs," and our deployable recommendation is reweighting — which changes training, not the decision rule, needs no protected attribute at inference, and got us most of the fairness gain at near-zero performance cost.

**Q3. "Your accuracy is 98.9%. So is a model that predicts 'never fraud'. Why should I believe any of this?"**
> You shouldn't believe accuracy, and we don't report it as a headline for exactly that reason — it's the do-nothing baseline on a 1.1% positive class. We report TPR at 5% FPR, which is the dataset authors' own metric, chosen because each false positive is a rejected genuine customer. [Then give your number.] We also report it on a strict temporal split — train on months 0–5, test on 6–7 — because random cross-validation leaks future data and inflates this by a lot. Here's both numbers side by side; the gap is why we split temporally.

**Q4. "SHAP is post-hoc rationalisation. How do you know those reason codes are the actual reasons?"**
> We don't fully, and we measured how much we don't. SHAP explains the *score*; the adverse action is the *threshold crossing*. So we computed a second ranking by decision pivotality — which feature, if moved to the approved-population median, actually flips the decision — and compared. Kendall tau was about 0.5. So we rank reason codes by pivotality, not attribution magnitude, and we're explicit that the SHAP baseline is a policy choice, not a technical default. There's literature on exactly this gap — Barocas, Selbst and Raghavan at FAT\* 2020.

**Q5. "This is synthetic data with an artificial protected attribute. Why does any of this matter in the real world?"**
> Fair challenge. Three answers. First, it's not arbitrary — it's a GAN trained on a real anonymised account-opening dataset, and the age disparity is inherited from real fraud patterns, not injected. Second, the method transfers even if the number doesn't: the pipeline that measures predictive equality at a fixed operating point, prices the trade-off, and produces reason codes is exactly what you'd run on production data. Third — and this is the point — the same team published FairGBM using this dataset's real-world precursor with the same 5% FPR requirement and the same age-group constraint. This isn't an academic exercise; it's a benchmark version of a live problem.

### 7.6 What judges see constantly vs. what actually lands

| Seen constantly (worth ~0) | Actually lands |
|---|---|
| Global SHAP summary bar chart | A single declined applicant's ranked reason codes + counterfactual |
| "We used SMOTE to handle imbalance" | "We tested SMOTE, it didn't help ranking metrics, here's the ablation" |
| Confusion matrix at threshold 0.5 | Threshold derived from a cost ratio, with the arithmetic shown |
| ROC curve with AUC in the corner | TPR@5%FPR + why AUC misleads at 99:1 |
| "Our model is fair" | The fairness–performance frontier with your chosen point circled |
| Feature importance ranking | The unawareness ablation showing the disparity survives dropping age |
| Hyperparameter tuning table | The temporal-vs-random split gap |
| "Future work: deploy to production" | "Here's what broke and what we'd do next" |

---

## 8. Suggested time budget

If you have one working day beyond your baseline model:

| Time | Task | Why this order |
|---|---|---|
| 30 min | Verify the `>50` split reproduces 80/20 and ~2x prevalence | Everything else is invalid if this is wrong |
| 45 min | Predictive equality at the 5% FPR point; the §3.3 budget sweep | This is the reveal — get the number early |
| 45 min | Reweighting + per-group thresholds + the frontier plot | Highest benefit-to-effort; slide 4 |
| 60 min | Temporal: per-month performance *and* fairness, with CIs | Slide 7; also protects you from Q3 |
| 45 min | Calibration + cost-ratio threshold table | Slide 5; makes the threshold defensible |
| 60 min | Reason codes + pivotality audit | Slide 6 + your answer to Q4 |
| 45 min | The slider demo | The thing they'll remember |
| 30 min | Slide 8 (what didn't work) | Cheap, and it's what wins the room |
| **0 min** | **FairGBM** | See §2.4 |

---

## 9. Verification log

Everything in this file was executed on **2026-08-13**.

**Windows host:** Python 3.14.5. Installed cleanly: `lightgbm` 4.7.0, `scikit-learn` 1.9.0, `shap` 0.52.0, `fairlearn` 0.14.0, `numpy` 2.5.2, `scipy`, `interpret` (EBM imports OK). Python 3.14 was not a barrier for any of the recommended stack.

**Linux:** `docker run python:3.11-slim` — required `apt-get install libgomp1` for fairgbm.

**Executed and confirmed:** predictive-equality computation; TPR@5%FPR via both `roc_curve` and negative-quantile; `fairlearn` `MetricFrame` `.ratio()`/`.difference()`; `ThresholdOptimizer` across four objectives (three degenerate) and its randomisation across seeds; per-group thresholds; sample reweighting; `shap.TreeExplainer` return type and the misleading warning; `shap.Explainer` callable API; LightGBM `pred_contrib`; `calibration_curve`; `brier_score_loss`; `FrozenEstimator` + `LGBMClassifier` (works); `cv="prefit"` (removed, raises); LightGBM `monotone_constraints`; SHAP-vs-pivotality Kendall tau; PSI.

**FairGBM, tested in depth:** wheel contents; Windows install-then-fail; Linux import with `libgomp1`; training at 2.3–2.5x LightGBM time; bit-identical output across `multiplier_learning_rate` ∈ {0.01, 0.1, 1.0, 5.0, 50}, across real/all-zero/random constraint groups, across `constraint_type` values, with and without `global_target_fpr` and both stepwise-proxy parameters, via both the sklearn wrapper and low-level `train()`; `constraint_group` confirmed present on the `Dataset` via `get_field`; `constraint_group_column` path segfaults.

**Paper facts** are quoted verbatim from the PDF text of [arXiv 2211.13358](https://arxiv.org/abs/2211.13358) (BAF) and [arXiv 2209.07850](https://arxiv.org/abs/2209.07850) (FairGBM), extracted locally.

### ⚠️ What you must reproduce yourself

Every **number** in §2 and §3 came from **controlled synthetic data**, not from BAF. They demonstrate that the code paths work and show the direction and rough magnitude of each effect. They are **not** results about BAF and must never be presented as such. Regenerate all of them on the real dataset before they go near a slide.

Two specific cautions: my synthetic data had **no proxies for the protected attribute**, so the "unawareness" result in §2.1 is an artefact and will likely not hold on BAF; and my synthetic data had **no temporal drift**, so the PSI and stability numbers in §6 demonstrate mechanics only.

The **~0.3 FPR ratio** figure for top-performing LightGBM models on Base comes from the paper's Figure 1 discussion — "the top performing models lying around 0.3 FPR ratio… legitimate applications from individuals in the group of higher ages are more than three times more likely to be flagged as fraudulent." Treat it as approximate, and **reproduce it on your own model rather than quoting it**. Your own reproduced number is far more persuasive than a citation, and it's the number the whole pitch hangs on.

---

## 10. Sources

- *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS 2022 — [arXiv 2211.13358](https://arxiv.org/abs/2211.13358) · [repo](https://github.com/feedzai/bank-account-fraud)
- *FairGBM: Gradient Boosting with Fairness Constraints*, ICLR 2023 — [arXiv 2209.07850](https://arxiv.org/abs/2209.07850) · [repo](https://github.com/feedzai/fairgbm)
- Fairlearn `ThresholdOptimizer` — [API reference](https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html)
- Regulation B §1002.9 — [CFPB](https://www.consumerfinance.gov/rules-policy/regulations/1002/9/) · [Official Interpretations](https://www.consumerfinance.gov/rules-policy/regulations/1002/Interp-9/)
- CFPB Circular [2022-03](https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/) · [2023-03 full text](https://www.govinfo.gov/content/pkg/FR-2024-04-17/html/2024-08003.htm) · [withdrawal, 12 May 2025](https://www.govinfo.gov/content/pkg/FR-2025-05-12/html/2025-08286.htm)
- EU AI Act [Annex III](https://artificialintelligenceact.eu/annex/3/) · [Article 86](https://artificialintelligenceact.eu/article/86/) · GDPR [Art 22](https://gdpr-info.eu/art-22-gdpr/) · [Recital 71](https://gdpr-info.eu/recitals/no-71/)
- Barocas, Selbst & Raghavan, FAT\* 2020 — [arXiv 1912.04930](https://arxiv.org/abs/1912.04930) · Kumar et al., ICML 2020 — [arXiv 2002.11097](https://arxiv.org/abs/2002.11097)
- Elkan, *The Foundations of Cost-Sensitive Learning*, IJCAI 2001 · Dal Pozzolo et al., *Calibrating Probability with Undersampling*, SSCI 2015
- LightGBM [parameter docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html) (the `is_unbalance` / `scale_pos_weight` calibration warning)

---

*Companion files: `01-DATASET-BIBLE.md` (columns, traps, protocol) · `R1-sota-and-benchmarks.md` · `R2-imbalance-truth.md`*
