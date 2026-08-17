# R1 — State of the Art & Benchmarks

**What score to target, what to build, what not to waste time on.**

Every number here is traced to a source. Where a figure could not be verified it is marked **UNVERIFIED** and given as a qualitative claim instead. Nothing is invented.

---

## 0. The 60-second version

| Question | Answer |
|---|---|
| **Metric?** | **ROC-AUC.** Confirmed on the competition page. |
| **What wins?** | **0.90444** — the actual winning private score of this exact competition. |
| **Respectable?** | **0.89 – 0.90** (that was 3rd–4th). |
| **Broke something?** | **> 0.92.** Nothing verified on this data goes near it. |
| **Split?** | ⚠️ **RANDOM 70/30, not temporal.** Quoted verbatim in §1. This changes everything. |
| **`month`?** | **Keep it as a feature.** It is in train *and* test, and the split cuts across all 8 months. |
| **Model?** | **LightGBM.** No credible evidence any deep tabular model beats a tuned GBDT here. |

> ### ⚠️ The one thing most teams will get wrong
>
> Every serious BAF paper and notebook uses the **temporal** protocol — train `month` 0–5, test 6–7 — because that is Feedzai's official design. **This competition does not.** The organiser split 1,000,000 rows **randomly** 70/30 and left `month` in both files.
>
> **Consequences:** use **stratified K-fold** (random CV *is* correct here and will track the leaderboard); **keep `month`** as a feature; ignore advice to drop it. A generic "BAF best practice" guide will actively cost you points on this leaderboard.
>
> Still *report* the temporal analysis in your presentation — see §5.

---

## 1. The competition — hard numbers

Source: [kaggle.com/competitions/1056lab-bank-account-fraud-detection](https://www.kaggle.com/competitions/1056lab-bank-account-fraud-detection). These pages render client-side and are invisible to a plain fetch; retrieved via text proxy.

- **Organiser:** Tohgoroh Matsui — the **46th 1056Lab Data Analytics Competition**, [1056Lab](https://1056lab.org), Chubu University, Japan. Community competition, Kudos only, no medals.
- **Ran:** 1–2 Jun 2023 → 31 Jul 2023.
- **Scale:** 9 entrants, 8 teams, **47 submissions total.** A small course competition, not a 3,000-team Kaggle war.
- **Metric: ROC-AUC.** Page heading reads *"Area Under the ROC Curve (AUC)"*; the competition tag is *"Area Under Receiver Operating Characteristic Curve"*.
- **Leaderboard split:** public ≈ 30% of test (~90k rows), private ≈ 70% (~210k rows).
- **No public notebooks. No discussion posts.** Both tabs are empty — there is nothing to copy.

### The split — verified verbatim

> *"I have randomly chosen 700,000 accounts (70 %) as the training data and made the remaining 300,000 accounts (30 %) test data."*

`train.csv` = 700,000 rows · `test.csv` = 300,000 rows · submission ids start at 700001. The data dictionary lists **`month` — "Month where the application was made. Ranges between [0, 7]"** as an ordinary feature.

The description says only *"1 million instances … generated using a CTGAN … about 30 realistic features"* and cites the NeurIPS paper. **It never names the variant.** Almost certainly `Base.csv`, but unconfirmed — see §8.

### Final leaderboard — complete, both boards

| # | Team | Private | Public |
|---|---|---|---|
| 1 | Ochiai | **0.90444** | 0.89294 |
| 2 | kazuyoshi teshima | 0.90437 | 0.89287 |
| 3 | Misawa Mutsuki | 0.90108 | 0.88670 |
| 4 | 牧谷虹太 | 0.89143 | 0.87718 |
| 5 | 近藤 優成 | 0.87823 | 0.85875 |
| 6 | 熊谷雄月 | 0.86790 | 0.84961 |
| 7 | EP21001青木 榛臣 | 0.83518 | 0.82871 |
| 8 | 野田隆斗 | 0.50491 | 0.50601 |

**Read this table — it is the most useful thing in this document.**

- Top two separated by **0.00007**. At the top this is decided by seed and blend, not insight.
- **No shakeup.** Private tracks public with a consistent +0.010–0.020 offset and identical ordering. Your public score is trustworthy — you do not need to hedge against a private collapse.
- 3rd place got 0.90108 in **3 submissions.** A clean, competent LightGBM lands top-3.
- Last place (0.50491) is a random ranking — a broken submission file.

**Target 0.905. Floor 0.89.**

---

## 2. Published benchmarks on BAF

The literature's headline metric is **TPR at 5% FPR** on the *temporal* split. Those numbers are **not** directly comparable to your leaderboard AUC, but they tell you what the data can support.

### 2a. The best third-party benchmark — temporal split, verified stored outputs

[VectorInstitute/anomaly-detection](https://github.com/VectorInstitute/anomaly-detection) (`BAF-demo-1.ipynb`) — BAF Base, `month` 0–5 / 6–7, drops `device_fraud_count`, one-hot, **no resampling**, outputs committed to the repo:

| Model | TPR @ 5% FPR | ROC-AUC |
|---|---|---|
| **LightGBM (tuned)** | **0.5486** | **0.8942** |
| LightGBM (default) | 0.5302 | 0.8898 |
| CatBoost | 0.5101 | 0.8836 |
| XGBoost | 0.5049 | 0.8787 |
| Isolation Forest | 0.0806 | 0.5816 |

Same repo, LightGBM across variants (TPR / AUC): I `0.5038 / 0.8842` · II `0.5323 / 0.8927` · III `0.7413 / 0.9513` · IV `0.4023 / 0.8477` · V `0.3375 / 0.7824`.

> **Variant III reaches 0.9513 AUC.** If someone quotes a 0.95 on "BAF", check which variant. Base does not do that.

### 2b. The official Feedzai anchor

Feedzai's own Kaggle notebook [`sgpjesus/train-lgbm-model`](https://www.kaggle.com/code/sgpjesus/train-lgbm-model) — Base.csv, `month<6`/`>=6`, a **default `LGBMClassifier()`**, no resampling. Printed verbatim:

```
Model TPR: 0.5254
Model FPR: 0.05
Model Threshold : 0.0446
```

**Feedzai keeps `month` as a feature** — in this notebook and in `empirical_results.ipynb`, only `fraud_bool` is dropped. The "always drop month" folklore is not Feedzai's own practice.

### 2c. Other verified TPR @ 5% FPR figures

| Result | Model | Data / split | Source |
|---|---|---|---|
| 0.4969 | LogReg (balanced) | Base, temporal, month dropped | [lennart4711 notebook](https://www.kaggle.com/code/lennart4711/baselinemodels-roc), 105 votes |
| 0.4663 | XGBoost | same | same |
| 0.5481 | LightGBM | Base, **random 80/20**, month kept | [bolouki notebook](https://www.kaggle.com/code/bolouki/bank-account-fraud-detection-eda-and-model), 29 votes |
| 0.5168 | LogReg | same | same |
| 57.9% | LightGBM | FiFAR (BAF-derived, months 1–3 / 4) | [arXiv 2312.13218](https://arxiv.org/abs/2312.13218) — *"a recall of 57.9% in validation, for a threshold of t=0.051"* |
| 47.08% | 1D-conv spiking NN | BAF suite | [CIARP 2024](https://link.springer.com/chapter/10.1007/978-3-031-76604-6_4) — authors call it *"comparable to gradient-boosting machine models"* |
| ≈ 0.448 / 0.529 / 0.559 | LightGBM, 100 random-search configs (min / median / max) | Base, temporal | Paper **Figure 1**, digitized — **estimate, ±0.002, not a printed figure** |

### 2d. ⚠️ The numbers that are routinely misquoted

| Number | What it actually is |
|---|---|
| **75.4% TPR** | LightGBM on **Feedzai's private real dataset** — *"training and testing on the original data, which was of 75.4% TPR."* **Not BAF.** |
| 69.1% / 62.9% / 60.6% | RF / LogReg / DecisionTree, also on the **private** dataset (Appendix Table 4). |
| **56.0% / 62.7% / 37.2%** | Paper **Table 2** — these evaluate *CTGAN generative fidelity* (train-on-generated / test-on-original combinations) during GAN model selection. The 56.0% cell is "train and test on generated data" for the selected model, so it is *indicative* of BAF Base, but it is a generator-selection metric, not a published benchmark. **Prefer §2a/2b, which are direct measurements.** |
| feedzai/data-bias-fraud-study CSVs (LGBM 0.719–0.802) | Tagged `bias = quadruple_50` — Feedzai's **private** dataset with synthetic bias scenarios. **Not BAF.** |

**If a teammate targets 75% TPR, they are reading the wrong number. BAF Base tops out near 0.55.**

### 2e. ROC-AUC on BAF Base

**The cleanest deep-vs-GBDT comparison** is [arXiv 2604.26188](https://arxiv.org/pdf/2604.26188), which uses BAF Base explicitly (*"1,000,000 samples and 31 features, with approximately 1.1% labeled as fraudulent"*) and includes a **LightGBM baseline in its appendix Table 9**. Test-set figures:

| Model | AUROC | AUPRC | F1 | FPR |
|---|---|---|---|---|
| **FT-Transformer** | **0.8955** | 0.1607 | 0.2342 | 0.0076 |
| **LightGBM** | **0.8953** | 0.1442 | 0.2267 | 0.0112 |
| FT-Transformer + CAR | 0.8945 | 0.1602 | 0.2268 | 0.0085 |
| FCorrTransformer | 0.8925 | 0.1458 | 0.2220 | 0.0094 |
| Feed-forward net | 0.8683 | 0.1234 | 0.1148 | 0.0020 |
| TabTransformer | 0.8635 | 0.1080 | 0.1871 | 0.0149 |

**AUPRC on BAF is only 0.11 – 0.18, and FNR is ~0.77 for every model.** AUC in the high 0.8s coexists with terrible precision. A 0.90 AUC model is good at *ranking* and still misses three-quarters of fraud.

### 2f. What the source paper does NOT give you

The Feedzai paper **never publishes a numeric TPR table per variant** — Figure 1 is a scatter plot, and [`empirical_results.ipynb`](https://github.com/feedzai/bank-account-fraud/blob/main/notebooks/empirical_results.ipynb) ships **with all outputs stripped** (`execution_count: null` throughout). Any blog quoting a precise per-variant table is re-running it or inventing it. Also: **"AUC" appears zero times in the paper.**

---

## 3. Model families — what actually wins

**Blunt answer: gradient-boosted trees.**

**The head-to-head:** on BAF Base, same pipeline, same test set — **LightGBM 0.8953 vs FT-Transformer 0.8955 AUROC.** A 0.0002 gap. Six transformer variants, one LightGBM, and the LightGBM ties the best of them at a fraction of the compute. That paper's other dataset is blunter: *"Overall, LightGBM remains the most stable and best-performing model."*

Note their own admission of the deep-model handicap: *"since neural networks cannot inherently handle missing values, while LightGBM can, we apply simple imputation using the mean … this treatment is not optimal."* The −1 sentinels (§4) are a structural GBDT advantage here.

**Corroborating:** the KAN paper ([arXiv 2408.10263](https://arxiv.org/abs/2408.10263)) on a BAF subset — LightGBM F1 **0.814**, GradientBoosting 0.813, XGBoost 0.806, then **KAN 0.766** and **MLP 0.751**. Verdict: *"KAN, in general, is not suitable for fraud detection problems."* ⚠️ That paper balanced the data 7,500/7,500, so it says nothing about imbalance — use it only as model-family evidence.

### The MDPI paper — read the health warning

[MDPI Computation 13(12):290](https://www.mdpi.com/2079-3197/13/12/290) appears to show a GRU (AUC 0.8800) beating LightGBM (0.8142). **Do not cite it that way.** Its LightGBM ran `n_estimators=100, num_leaves=31` — library defaults on 800k rows — grid-searched on a **10,000-row subsample** scored by **`f1_weighted`**, a metric dominated by the 98.9% majority. LightGBM tuning: **28.6 s**; GRU tuning: **4,118.1 s**. Its LightGBM AUC lands *below plain logistic regression* (0.8484) — the signature of a broken setup. The GRU also treats 29 *static* columns as a *sequence* ordered by dataframe index, which is meaningless, and its 78% recall is at **5% precision** (fraud-class F1 0.09, *worse* than Random Forest's 0.10).

Its actual thesis is sound and quotable — *"objective choice and operating point matter at least as much as architecture"* — but its numbers are not a GBDT benchmark. ⚠️ Near-identical Research Square preprints exist (rs-8303897, rs-8136120).

### One paper to distrust outright

[arXiv 2508.16915](https://arxiv.org/html/2508.16915) claims a spiking NN at **recall 0.908 @ FPR 0.047** on BAF — ~1.7× the verified ceiling — while listing LightGBM at 0.450. It also describes its split as *"90% training (six months) / 10% testing (two months)"*, but months 6–7 are ~20% of BAF, not 10%. **Not reconcilable.**

### The general tabular literature

| Paper | Finding |
|---|---|
| **Grinsztajn et al., NeurIPS 2022** ([2207.08815](https://arxiv.org/abs/2207.08815)) | *"tree-based models remain state-of-the-art on medium-sized data."* |
| **Shwartz-Ziv & Armon** ([2106.03253](https://arxiv.org/abs/2106.03253)) | *"XGBoost outperforms these deep models across the datasets."* Also: **deep + XGBoost ensembled beats XGBoost alone.** |
| **TabReD, ICLR 2025** ([2406.19380](https://arxiv.org/abs/2406.19380)) | On industry data **with temporal splits**, GBDTs and MLP-with-embeddings win; fancy tabular DL fails to transfer. |
| **BeyondArena** ([2606.30410](https://arxiv.org/abs/2606.30410)) | Foundation models win on small IID data; **trees dominate large (100k–1M), non-IID data** — BAF's regime. |

**Honest caveat:** none of these stratifies by imbalance ratio. "GBDTs win under extreme imbalance" is a well-supported *extrapolation* from convergent evidence, not a dedicated ablation. No such study was found.

### LightGBM vs XGBoost vs CatBoost

The one clean same-protocol comparison (§2a) gives **LightGBM 0.8942 > CatBoost 0.8836 > XGBoost 0.8787** AUC. LightGBM leads, but by ~0.015 — worth having, not worth agonising over. Widely-circulated figures favouring CatBoost trace to Research Square / preprints.org with no shared protocol — **UNVERIFIED, do not cite.**

**Use LightGBM as the workhorse; blend all three at the end for the last ~0.002.**

---

## 4. The four traps — confirmed or refuted

### Trap 1 — `device_fraud_count` is constant → **PROBABLY TRUE, VERIFY IN 5 SECONDS**

The datasheet and the competition page both document the range as **`[0, 1]`**. **Neither confirms it is empirically constant** — the "all zeros" claim is community lore. **UNVERIFIED.** Multiple credible repos (VectorInstitute, lennart4711) drop it as uninformative.

`df["device_fraud_count"].nunique()` settles it. Impact either way ≈ zero — GBDTs ignore constant features. **The least important trap here.**

### Trap 2 — `month` causes leakage or shift → **REFUTED FOR THIS COMPETITION**

The temporal concern is real *for the BAF paper*, and irrelevant *for this leaderboard*:

- The official BAF protocol **is** temporal (train 0–5, test 6–7) and shift **is** real: prevalence *"varies between 0.85% and 1.5% … higher for the later months"*, volume swings 9.5%–15% per month.
- **But this competition split randomly 70/30 and kept `month` in both files.** The organiser said so verbatim (§1). There is no future period to generalise to.

**Therefore, on this leaderboard:**
- ✅ Use **stratified K-fold**. Random CV is *correct* here and will track the leaderboard.
- ✅ **Keep `month`** as a feature. Feedzai keeps it too.
- ❌ Do **not** train on months 0–5 only — you would throw away 25% of your labelled data for nothing.

Confirm in 30 seconds, because being wrong is expensive:

```python
import pandas as pd
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
print(sorted(tr["month"].unique()), sorted(te["month"].unique()))
print(len(tr), len(te))        # expect 700000 300000
```

Both spanning 0–7 confirms the random split.

### Trap 3 — `-1` sentinels treated as real numbers → **CONFIRMED, with a critical correction**

From the official datasheet, these use a **negative sentinel for missing**:

| Column | Range | Missing |
|---|---|---|
| `prev_address_months_count` | [−1, 380] | −1 |
| `current_address_months_count` | [−1, 429] | −1 |
| `bank_months_count` | [−1, 32] | −1 |
| `session_length_in_minutes` | [−1, 107] | −1 |
| `device_distinct_emails_8w` | [−1, 2] | −1 |
| `intended_balcon_amount` | [−16, 114] | **all negatives** |

**The correction that will catch most teams** — these have legitimate negatives that are **NOT missing**:

- **`credit_risk_score`** — [−191, 389]. Negative scores are real, and this is usually the strongest single feature.
- **`velocity_6h`** — [−175, 16818]. A generator artefact, but **not** documented as missing.

**A blanket `df[df < 0] = np.nan` destroys your two best features.** Handle the six by name.

Also, datasheet **Q9**: *"There is no missing information from individual instances."* **There are no NaNs in the file** — `df.isna().sum()` returns all zeros and tells you nothing.

**Best practice:** add an `_is_missing` flag per sentinel column, then set the sentinel to `NaN` and let LightGBM route it natively. Missingness is itself predictive — a synthetic identity has no previous address *because it was invented last week*. Do **not** median-impute.

### Trap 4 — Temporal distribution shift → **CONFIRMED in the data, NEUTRALISED by this competition's split**

Real and documented (prevalence rises in later months; Variants IV/V exist to model *"a feature distribution shift across time"*; and *"The top performing models on the Base dataset are not necessarily the best ones on the other variants"*).

**But with a random 70/30 split, train and test are drawn from the same 8 months.** There is no train→test shift to defend against. This is now a **presentation asset, not a modelling constraint** — see §5.

⚠️ The 0.85%–1.5% figure describes the **original** Feedzai dataset; BAF was sampled to reproduce its monthly profile, so Base inherits a similar shape, but exact per-month rates are **UNVERIFIED**. Measure: `df.groupby("month")["fraud_bool"].agg(["mean","size"])`.

---

## 5. AUC vs TPR@low-FPR — do they select different models?

**Yes, and it is well established.**

- **Davis & Goadrich, ICML 2006** — a curve dominates in ROC space *iff* it dominates in PR space, **but** *"algorithms that optimize the area under the ROC curve are not guaranteed to optimize the area under the PR curve."*
- **Narasimhan & Agarwal (SVM^pAUC)** — directly optimising **partial AUC over a chosen FPR range** beats full-AUC-optimising methods on partial AUC. The cleanest demonstration that changing the objective changes the selected model.
- **Feedzai's own practice** — they pick the operating point explicitly (*"a threshold of t=0.051, chosen in validation to obtain 5% FPR"*), never by AUC alone.
- **The MDPI paper is an accidental natural experiment** — same five models, selection metric switched to `f1_weighted`, and every classical model collapsed to 1–6% fraud recall while AUCs stayed high.

**What to do:**
1. **Optimise ROC-AUC.** It is the leaderboard metric. Do not get creative.
2. **Also report TPR@5%FPR**, and report it on a **temporal** (months 0–5 / 6–7) holdout. That is the domain-correct evaluation and it is what a judge with banking knowledge will respect.
3. **Show both in one table**, and say out loud that the leaderboard metric is not the business metric. Ten minutes of work; it is the cheapest differentiator available.

---

## 6. Public solutions — what actually holds up

**Zero notebooks and zero discussions exist on this competition.** The public work is on the general [BAF dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).

### ✅ Trustworthy

| Source | Protocol | Result |
|---|---|---|
| [VectorInstitute/anomaly-detection](https://github.com/VectorInstitute/anomaly-detection) | temporal, no resampling, outputs committed | Full table in §2a — **best third-party numbers found** |
| [Feedzai `train-lgbm-model`](https://www.kaggle.com/code/sgpjesus/train-lgbm-model) | temporal, default LGBM | TPR 0.5254 @ 5% FPR |
| [lennart4711](https://www.kaggle.com/code/lennart4711/baselinemodels-roc) (105 votes) | temporal, month dropped | LR AUC 0.87794 / TPR 0.4969; XGB 0.86800 / 0.4663 |
| [bolouki](https://www.kaggle.com/code/bolouki/bank-account-fraud-detection-eda-and-model) (29 votes) | random 80/20, month kept, GridSearch on recall@5%FPR | **LGBM AUC 0.894982, recall 0.5481** — closest published analogue to *your* setup |
| [diogoleitao](https://www.kaggle.com/code/diogoleitao/lightgbm-with-early-stopping) | **best protocol seen**: train ≤5, val 6, test 7; Optuna ×100 | Scores only in plots → **unverified**; conclusion is that early stopping halves train time at no cost |
| [dssg/aequitas](https://github.com/dssg/aequitas) (769★) | `DEFAULT_SPLIT = {train: 0..5, validation: 6, test: 7}`, `include_month=True` | Loader only, no metrics |

### 🔴 Leaky — do not copy, and know why

| Source | Fault | Inflated to |
|---|---|---|
| [matthewmcnulty](https://www.kaggle.com/code/matthewmcnulty/bank-account-fraud) **96 votes** + forks (~130 cumulative) | `NearMiss(0.1).fit_resample(X, y)` on the **entire dataset before any split**. NearMiss is a supervised k-NN selector, so which negatives survive depends on test-set labels. **And the test set itself is resampled** — its confusion matrix shows 9.09% prevalence vs the real 1.10% | AUC 0.95, recall@5%FPR 0.78 |
| [juanjosmorenogiraldo](https://www.kaggle.com/code/juanjosmorenogiraldo/bank-fraud-detection-using-gbm) **68 votes** | `smote.fit_resample(X_test, y_test)` — **SMOTE applied to the test set.** Every metric is scored against synthetic interpolations of real minority points, which are trivially separable. Also computes `roc_curve` on **hard labels** | AUC 0.9883 (GBM) |
| [sarveshrane1997](https://github.com/sarveshrane1997/Bank-Account-Fraud-Detetction) | Correct temporal split ✅ then `SMOTE.fit_resample(X_test, y_test)` | XGB recall 1.00 |
| [ParzHe](https://github.com/ParzHe/Fraud_detection_with_LightGBM_and_XGBoost) | **Concatenates all six variants** (6M rows from the same CTGAN → near-duplicates cross the split), NearMiss on the whole pool, then random split | fraud-class **precision 0.76** — impossible at 1.1% prevalence |
| [gbiamgaurav](https://www.kaggle.com/code/gbiamgaurav/base-modelling) (Bronze) | Not leakage — degenerate. `accuracy 0.9892, F1 0.0, precision 0.0`: predicts all-negative. 0.9892 is just `1 − prevalence` | — |

**The diagnostic that isolates the cause:** a notebook doing an *unbiased random* 1:1 undersample before splitting still lands at LGBM AUC **0.8813**. Undersampling barely moves AUC. The jump to 0.95 comes specifically from **label-aware selection (NearMiss) and resampled test sets.**

### The 4-question audit for any BAF claim

1. Temporal or random split — and does it match the target task?
2. Was any resampler fit **before** the split, or applied to the **test** set?
3. Were variants concatenated? (They share a CTGAN — near-duplicates cross the split by construction.)
4. **Is fraud-class precision above ~0.05?** At 1.1% prevalence and 5% FPR it cannot be. **This is the fastest tell that the test set was rebalanced.**

---

## 7. What to build

1. **Confirm the split** (§4, Trap 2). 30 seconds. Everything depends on it.
2. **Sentinel handling.** Six columns → `_is_missing` flag + `NaN`; let the GBDT route it. Do not median-impute. Do not blanket-negate. (~15 min)
3. **One tuned LightGBM with `month` kept, validated by stratified K-fold.** This alone should reach ~0.89–0.90. Start from Feedzai's own published space ([`lightgbm_hyperparameter_space.yaml`](https://github.com/feedzai/bank-account-fraud/blob/main/notebooks/lightgbm_hyperparameter_space.yaml)):

   ```yaml
   n_estimators:     [20, 10000]   log
   max_depth:        [3, 30]
   learning_rate:    [0.02, 0.1]   log
   num_leaves:       [10, 100]     log
   min_data_in_leaf: [5, 200]      log
   max_bin:          [100, 500]
   boosting_type:    [gbdt, goss]
   enable_bundle:    [true, false]
   ```

   **Tune with AUC as the selection score** — the MDPI paper is a live demo of what selecting on the wrong metric does.
4. **Velocity ratio features.** `velocity_6h / velocity_4w` is a burst detector; all three are in the same units (applications/hour). The most defensible engineering on this dataset.
5. **Blend LightGBM + XGBoost + CatBoost.** Worth ~0.002–0.005. Last, if time remains.
6. **Report TPR@5%FPR on a temporal holdout, plus the age-group FPR ratio.** Cheap, and it is what separates a presentation from a leaderboard score.

## 8. What NOT to waste time on

| Don't | Why |
|---|---|
| **Temporal-only training (months 0–5)** | The split is random. You would discard 25% of your labels for a protocol this competition doesn't use. |
| **Dropping `month`** | It is a legitimate feature here, present in test. Feedzai keeps it too. |
| **TabNet / FT-Transformer / any net as the main model** | FT-Transformer 0.8955 vs LightGBM 0.8953 on BAF Base — a tie, for a fraction of the compute. |
| **Agonising over GBDT library choice** | LightGBM leads CatBoost/XGBoost by ~0.015 AUC. Pick LightGBM, blend at the end. |
| **SMOTE / NearMiss / heavy resampling** | For a **ranking metric**, resampling barely moves AUC (0.8813 with 1:1 undersampling vs ~0.89 without) and distorts calibration. Every inflated public score traces to it. Run the ablation once, show the judges, move on. |
| **Chasing 0.95+** | The winner scored **0.90444**; best verified BAF AUC is ~0.895. A 0.95 means you leaked — check `id`, and check you didn't resample the validation set. |
| **Quoting 75.4% TPR as a target** | Feedzai's private dataset, not BAF. See §2d. |

---

## 9. What I could NOT verify

1. **`device_fraud_count` being empirically all-zero.** Both docs say `[0, 1]`. → `nunique()` settles it.
2. **Which BAF variant the competition uses.** The description never names it. Several ranges on the Kaggle page differ from the Base datasheet — some *wider* (`velocity_6h` [−211, 24763] vs [−175, 16818]; `date_of_birth_distinct_emails_4w` [0, 42] vs [0, 39]; `device_distinct_emails_8w` [0, 3] vs [−1, 2]), which a subset of Base cannot produce. → Run `df.describe().T` against the datasheet before trusting column-level advice.
3. **Exact per-variant TPR from the source paper.** Plot only; notebook outputs stripped. The 0.448/0.529/0.559 figures are digitized estimates.
4. **Per-month fraud rates in BAF Base specifically.** The 0.85%–1.5% range is documented for the *original* dataset.
5. **`diogoleitao`'s Optuna scores** — rendered in plots, never printed.
6. **Any head-to-head GBDT-library benchmark with significance testing** on imbalanced fraud.
7. **Any evaluation of TabPFN / TabICL / SAINT / NODE on BAF.** None found — "no evidence exists", not "they lose".

---

## 10. Sources

**Primary**
- Jesus et al., *Turning the Tables*, NeurIPS 2022 D&B — [arXiv 2211.13358](https://arxiv.org/abs/2211.13358) · [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/d9696563856bd350e4e7ac5e5812f23c-Paper-Datasets_and_Benchmarks.pdf)
- Official datasheet — [feedzai/bank-account-fraud/documents/datasheet.pdf](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)
- [`empirical_results.ipynb`](https://github.com/feedzai/bank-account-fraud/blob/main/notebooks/empirical_results.ipynb) · [`lightgbm_hyperparameter_space.yaml`](https://github.com/feedzai/bank-account-fraud/blob/main/notebooks/lightgbm_hyperparameter_space.yaml) · [`sgpjesus/train-lgbm-model`](https://www.kaggle.com/code/sgpjesus/train-lgbm-model)
- Competition overview, data and leaderboard — [1056lab-bank-account-fraud-detection](https://www.kaggle.com/competitions/1056lab-bank-account-fraud-detection)
- [VectorInstitute/anomaly-detection](https://github.com/VectorInstitute/anomaly-detection)

**BAF literature**
- FiFAR — [arXiv 2312.13218](https://arxiv.org/abs/2312.13218) · Counterfactual-fairness transformers — [arXiv 2604.26188](https://arxiv.org/pdf/2604.26188) · Sun et al., *Objective over Architecture* — [MDPI Computation 13(12):290](https://www.mdpi.com/2079-3197/13/12/290) ⚠️ §3 · KANs — [arXiv 2408.10263](https://arxiv.org/abs/2408.10263) · SNNs, CIARP 2024 — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-76604-6_4) · RHOSS — [arXiv 2508.16915](https://arxiv.org/html/2508.16915) ⚠️ §3

**Tabular ML background**
- Grinsztajn et al. — [2207.08815](https://arxiv.org/abs/2207.08815) · Shwartz-Ziv & Armon — [2106.03253](https://arxiv.org/abs/2106.03253) · Gorishniy et al. — [2106.11959](https://arxiv.org/abs/2106.11959) · TabReD — [2406.19380](https://arxiv.org/abs/2406.19380) · BeyondArena — [2606.30410](https://arxiv.org/abs/2606.30410)
- Davis & Goadrich, ICML 2006 — [PDF](https://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf) · Narasimhan & Agarwal, KDD 2013 — [PDF](http://chbrown.github.io/kdd-2013-usb/kdd/p167.pdf)
