# START HERE — Bank Account Fraud Detection

Everything you need to win, in the order you need it.

---

## The verified facts

Checked against the competition page, the NeurIPS paper, Feedzai's own notebooks, and independent benchmark repos. Not guesses.

| | |
|---|---|
| **Dataset** | Feedzai **Bank Account Fraud (BAF)**, NeurIPS 2022 — 1M rows, 32 cols, ~1.1% fraud |
| **Task** | Account-**opening** fraud (not transaction fraud) — synthetic identity, identity theft, mule farming |
| **Competition metric** | **ROC-AUC** |
| **Split** | **RANDOM 70/30** — organiser's own words. `month` is in both files and IS a feature |
| **Winning score** | **0.90444** · 3rd place 0.90108 · 7th place 0.83518 |
| **Your target** | **0.905.** Floor 0.89. **Above 0.92 = leakage, go find it** |
| **Field size** | 8 teams, 47 submissions. Private board preserved public ordering — no shakeup risk |

---

## The three things that give you an edge

### 1. The split trap — and you're on the right side of it

Every generic BAF guide online says *"train on months 0–5, test on 6–7, drop `month`."* That is the **NeurIPS paper's** protocol.

**Your competition split randomly.** So: stratified K-fold, **keep `month`**, train on all 700k rows. Anyone who follows the tutorials trains on 6/8 of the data and throws away a live feature.

Verify it yourself in ten seconds — `sorted(train.month.unique())` vs `sorted(test.month.unique())` — and say in your writeup that you checked. That alone is a point of difference.

### 2. Your brief says "data balancing". Test it instead of obeying it.

Both metrics that matter here — ROC-AUC, and TPR@5%FPR — are **ranking** metrics. Ranking is invariant to any monotone rescaling of scores, and **class balancing is a monotone rescaling.** So it moves F1-at-0.5 dramatically and leaves AUC essentially flat.

That one fact dissolves the entire contradictory literature: pro-balancing papers report F1 at a fixed 0.5 threshold; anti-balancing papers report AUC and calibration. Both are right about different things.

- Elor & Averbuch-Elor, 73 datasets, LightGBM/XGBoost/CatBoost: with the threshold fixed at 0.5, balancing helped everything. With the threshold **tuned**, it helped only weak learners.
- *PLOS ONE* 2022, 1,736 controlled comparisons: sampling changed AUROC significantly in **10%** of cases — and **61% of those were degradations**.
- **Specific to this dataset:** SMOTE interpolates between rows, so it will average the `−1` missing sentinels into fabricated values — inventing applicants with "−0.4 months at previous address" and corrupting the very thin-file signal that detects synthetic identities.

**No published sampling ablation on BAF exists.** Running one is genuinely novel, costs an afternoon, and gives you a slide nobody can wave away.

### 3. This dataset was built for fairness research, and nobody in your room will use it

`customer_age` is the designated protected attribute (**age > 50**, strictly — ages are decade-rounded, so `>=` moves a whole bucket). The metric is **predictive equality**: the ratio of false-positive rates between groups, min/max, where 1.0 is parity.

The paper's own finding: strong models falsely flag **older applicants substantially more often.**

A false positive here is **a real person denied a bank account.**

And the sharpest sub-finding: **the disparity gets worse as you tighten the FPR budget** — which is exactly where real banks operate. Tightening the filter concentrates the harm on the older group.

---

## Quickstart

```bash
cd code
pip install -r requirements.txt

# Debug your whole pipeline before the real data even downloads:
python make_sample.py --rows 60000 --out sample_train.csv
python run_pipeline.py --train sample_train.csv

# Real competition data (random split is the default — correct here):
python run_pipeline.py --train train.csv --test test.csv --submit submission.csv --id-col <id>
```

Verified working end to end on Python 3.14 with LightGBM 4.7, pandas 3.0.5, scikit-learn 1.9.

---

## What's in the kit

| File | Read it when |
|---|---|
| **`00-START-HERE.md`** | Now |
| **`01-DATASET-BIBLE.md`** | Before you write any code. Every column, every trap |
| **`03-EXECUTION-PLAN.md`** | Hour-by-hour plan, slide outline, the five sentences that win the room |
| `research/R1-sota-and-benchmarks.md` | Setting your target; deciding what not to build |
| `research/R2-imbalance-truth.md` | Before you touch SMOTE. The ablation design and the argument |
| `research/R3-features-and-tuning.md` | Feature catalogue + LightGBM parameters that matter |
| `research/R4-fairness-explainability-pitch.md` | Hour 10 onward. Fairness, SHAP, reason codes, the pitch |
| `code/baf.py` | The toolkit — preprocessing, FE, metrics, fairness |
| `code/run_pipeline.py` | The pipeline — ablation, evaluation, submission |
| `code/make_sample.py` | BAF-shaped synthetic data for offline debugging |

---

## Three traps that will cost you if you miss them

1. **`min_sum_hessian_in_leaf`.** LightGBM's default of `1e-3` works out to roughly **0.1 samples** at a 1% base rate — effectively no constraint at all, so leaves form on a handful of fraud cases. Set it to `1.0`. Same story for `min_data_in_leaf` (default 20 → use 200). Already set in the pipeline.
2. **Not all negatives are missing.** Six columns use `−1` as a sentinel. But `credit_risk_score` (range −191–389) and `velocity_6h` (−175–16818) have **legitimate** negatives and are two of your best features. A blanket "negatives → NaN" rule destroys them.
3. **The famous 75.4% TPR figure is not this dataset.** It's Feedzai's private internal data. On BAF, TPR@5%FPR tops out near **0.55**. Quote 75% and a knowledgeable judge will catch it.

---

## Two things that sound great and aren't

- **Deep tabular models.** On BAF Base, same test set: LightGBM **0.8953** vs FT-Transformer **0.8955**. A rounding error. Don't spend your hackathon there — and *say* you checked, with the number.
- **FairGBM.** Real library, real ICLR 2023 paper, written by the BAF authors, and it looks perfect for this. It was tested for you: `pip install` **succeeds on Windows and then fails to import** (the wheel ships a Linux `.so`, no `.dll`). On Linux it installs but the fairness constraint appears **inert** — identical output across a 5000× range of constraint strengths. Last release Nov 2022.

  Don't use it. **Do mention you evaluated it** — that's a credibility line almost nobody else can deliver.

---

## Your thesis

> *A false positive here isn't a metric — it's a real person denied a bank account. On this dataset that harm lands roughly three times harder on applicants over 50, and it gets worse as you tighten the filter. We built the model that measures that, prices it, and lets you choose where to sit.*

Open your presentation with the fact that predicting "never fraud" scores **98.9% accuracy** and catches zero criminals. Everything follows from there.

---

## An honest word

This kit gives you verified facts where most teams will have guesses, a tested pipeline where most will have a notebook with hidden state, and a fairness angle the dataset was literally designed for. That is a real edge.

It isn't a guarantee — execution and presentation on the day are yours. But you'll walk in knowing the target score, the exact split, and three arguments most of the room won't have thought about. Go and use them.
