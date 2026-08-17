# The Dataset Bible — Bank Account Fraud (BAF)

Everything in this file is from the **official Feedzai datasheet and the NeurIPS 2022 paper**, not from a blog. Ranges and missing-value encodings are quoted from the datasheet verbatim. Trust this file over any Kaggle notebook you read.

---

## 1. What you are actually working with

| | |
|---|---|
| **Real name** | Bank Account Fraud (BAF), Feedzai, NeurIPS 2022 Datasets & Benchmarks track |
| **Paper** | *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (arXiv 2211.13358) |
| **Rows** | 1,000,000 per variant |
| **Columns** | 32 (31 features + `fraud_bool`) |
| **Positive rate** | ~1.1% overall; the paper states fraud prevalence **varies between 0.85% and 1.5% across months** |
| **Origin** | Synthetic — a CTGAN-family generator trained on a real, anonymised bank **account-opening** fraud dataset |
| **Variants** | 6: `Base`, `Variant I`–`V`. Your competition is almost certainly **Base** |

### This is account-OPENING fraud, not transaction fraud

This single distinction will separate you from most competitors, who will treat it like the famous credit-card transaction dataset and apply the wrong mental model.

There is **no transaction history and no customer relationship**. A fraudster is trying to open a *new* account using a fabricated, stolen, or synthetic identity. Every feature is a signal available at *application time*. The fraud patterns you are hunting are:

- **Synthetic identity** — a plausible but fake person. Betrayed by thin history, incoherent attributes, and machine-generated email addresses.
- **Identity theft** — a real person's details used by someone else. Betrayed by device/session mismatch and contactability failures.
- **Mule account farming** — many accounts opened in bulk. Betrayed by velocity and shared-attribute counts.

When you engineer a feature, ask *"which of these three does it catch?"* If you can't answer, it's probably noise.

---

## 2. Every column, with its fraud meaning

Ranges are quoted from the official datasheet. **Bold** marks a trap.

### Identity & applicant profile

| Column | Range | What it is | Why fraud cares |
|---|---|---|---|
| `income` | 0.1 – 0.9 | Annual income, **already in decile form** | Not rupees/dollars. It's a rank. Don't log-transform it. Incoherence with `proposed_credit_limit` is the signal. |
| `customer_age` | 10 – 90 | Age **rounded to the decade** | Only 9 distinct values — behaves like a categorical. **This is the protected attribute** for fairness (paper splits at ≥50). |
| `employment_status` | 7 anonymised values | Employment status | Anonymised — you cannot interpret levels, only their fraud rates. |
| `housing_status` | 7 anonymised values | Residential status | Often one of the strongest single features. |
| `name_email_similarity` | 0 – 1 | Similarity between email address and applicant's name | **High-value.** A real person's email usually resembles their name. Synthetic identities generate emails that don't. Low value = suspicious. |

### Address & history — the "thin file" signals

| Column | Range | What it is | Why fraud cares |
|---|---|---|---|
| `prev_address_months_count` | **−1** – 380 | Months at previous address | **−1 = MISSING**, and missingness is itself meaningful — no previous address means no history to check. |
| `current_address_months_count` | **−1** – 429 | Months at current address | **−1 = MISSING.** A very short tenure plus missing previous address is a classic synthetic-identity fingerprint. |
| `bank_months_count` | **−1** – 32 | Age of previous account, if held | **−1 = MISSING.** No banking history at all is a strong thin-file signal. |
| `days_since_request` | 0 – 79 | Days since application | Low variance, usually weak. |

### Velocity & shared-attribute counts — the bulk-fraud signals

| Column | Range | What it is | Why fraud cares |
|---|---|---|---|
| `velocity_6h` | **−175** – 16,818 | Avg applications/hour over last 6h | **Negative values exist** — an artefact of the synthetic generator. Decide explicitly how to treat them. |
| `velocity_24h` | 1,297 – 9,586 | Avg applications/hour over last 24h | |
| `velocity_4w` | 2,825 – 7,020 | Avg applications/hour over last 4 weeks | |
| `zip_count_4w` | 1 – 6,830 | Applications from same ZIP in 4 weeks | Mule farms cluster geographically. |
| `bank_branch_count_8w` | 0 – 2,404 | Applications at that branch in 8 weeks | |
| `date_of_birth_distinct_emails_4w` | 0 – 39 | Distinct emails for applicants sharing a DOB | **High-value.** One date of birth with many emails is close to a definition of synthetic identity generation. |

> **The velocity trio is your best feature-engineering territory.** All three are in the *same units* (applications per hour) over three different windows. That makes their **ratios** meaningful: `velocity_6h / velocity_4w` is a burst-detector — short-term activity relative to the long-run baseline. Raw levels are far less informative than the acceleration between windows.

### Contactability & device — the "is this a real reachable human" signals

| Column | Range | What it is | Why fraud cares |
|---|---|---|---|
| `phone_home_valid` | binary | Home phone validates | Fraudsters supply unverifiable contact details. |
| `phone_mobile_valid` | binary | Mobile validates | Both invalid is a much stronger signal than either alone — build the interaction. |
| `email_is_free` | binary | Free vs paid email domain | Weak alone; **strong combined with low `name_email_similarity`**. |
| `foreign_request` | binary | Origin country ≠ bank's country | |
| `device_os` | Windows/macOS/Linux/X11/other | Device OS | Linux/X11 are rare for retail banking consumers — often disproportionately fraudulent. Check this yourself. |
| `device_distinct_emails_8w` | **−1** – 2 | Distinct emails from this device in 8 weeks | **−1 = MISSING.** Only 4 effective values — treat as categorical. One device, multiple emails = farming. |
| `session_length_in_minutes` | **−1** – 107 | Session length on the banking site | **−1 = MISSING.** Very short sessions suggest automation/scripted applications. |
| `keep_alive_session` | binary | User's logout preference | A genuine behavioural tell — bots don't fiddle with session preferences. |
| `source` | INTERNET / TELEAPP | Application channel | **Heavily skewed** — TELEAPP is a small minority. Check whether it's large enough to matter. |

### Product & risk

| Column | Range | What it is | Why fraud cares |
|---|---|---|---|
| `credit_risk_score` | **−191** – 389 | The bank's own internal risk score | Strongest single feature in most runs. Negative values are legitimate here, **not** missing. |
| `proposed_credit_limit` | 200 – 2,000 | Requested credit limit | Fraudsters maximise take. **Ratio to `income` is the real signal.** |
| `payment_type` | 5 anonymised values | Credit payment plan | |
| `has_other_cards` | binary | Holds other cards with this bank | Existing relationship ≈ lower risk. |
| `intended_balcon_amount` | **−16** – 114 | Initial transfer ("balance consolidation") amount | **NEGATIVES ARE MISSING VALUES**, per the datasheet. Most rows are missing. Do not average this blindly. |

### Structural

| Column | Range | Notes |
|---|---|---|
| `month` | 0 – 7 | **Temporal index. This is the most important column in the dataset and it is not a feature.** See §3. |
| `device_fraud_count` | 0 – 1 | Datasheet gives range [0,1], but **in the Base variant this is widely reported to be constant 0**. Verify with `df.nunique()`; if constant, drop it — it contributes nothing and wastes a split. |
| `fraud_bool` | 0/1 | **Target.** |

---

## 3. The six traps

Most teams will hit at least three of these. Each one is a place to gain ground.

### Trap 1 — The `-1` sentinels are not numbers

Six columns use a negative sentinel for missing: `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`, and `intended_balcon_amount`.

If you impute them with the median — which several published pipelines do — **you destroy the signal entirely**, because *the missingness itself is predictive*. A synthetic identity has no previous address precisely *because it was invented last week*.

**Do this:** create an explicit `_is_missing` binary column for each, then set the sentinel to `NaN` and let LightGBM/XGBoost/CatBoost handle NaN natively.

**Be honest about why, though**, because a sharp judge may push on it. A gradient-boosted tree can already isolate a −1 with a single split, so the per-column indicator is *not* adding much information by itself. The two real reasons this step matters are:

1. **It stops −1 poisoning your engineered features.** The moment you compute `velocity_6h / velocity_4w` or `prev_address + current_address`, a −1 silently corrupts the arithmetic. This is a *correctness prerequisite*, not a clever feature.
2. **The cross-column aggregate is genuinely new.** `n_missing` — how many independent history checks came back empty — would take a tree one split per column to reconstruct. That one is a real feature.

Claiming the indicators themselves are a breakthrough is the kind of overstatement that gets punctured in Q&A. Claim the two things above instead; they're true and they're more interesting.

### Trap 2 — Mirroring the WRONG split (⚠️ this competition is the exception)

> **VERIFIED, and it inverts the standard advice.** The organisers of
> `kaggle/1056lab-bank-account-fraud-detection` state on the competition page:
> *"I have randomly chosen 700,000 accounts (70%) as the training data and made
> the remaining 300,000 accounts (30%) test data."*
>
> **Your competition uses a RANDOM 70/30 split, not the paper's temporal one.**

This matters enormously, because *every generic BAF tutorial online will tell you to split temporally* — that is the protocol from the NeurIPS paper. Following it here would be a mistake: you would train on 6/8 of the data instead of all of it, and you would throw away a usable feature.

**The rule underneath both cases:** your validation must **mirror the organisers' split**, whatever it is. Otherwise your local score does not track the leaderboard, and you tune blind.

| Situation | What to do |
|---|---|
| **This competition** (random 70/30, `month` in both files) | Stratified K-fold. **Keep `month`.** Train on all rows. |
| Reproducing the BAF paper, or a time-separated test set | Train months 0–5, test 6–7. **Drop `month`.** |

`run_pipeline.py --split random` is the default for exactly this reason. Use `--split temporal` only if you are reproducing the paper.

**Confirm it yourself in 10 seconds** — don't take my word for it:
```python
train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
print(sorted(train.month.unique()), sorted(test.month.unique()))
# overlapping ranges  -> random split   -> keep month, stratified KFold
# test strictly later -> temporal split -> drop month, month-based split
```

### Trap 3 — Assuming all negatives mean "missing"

Six columns use negatives as a missing sentinel (Trap 1). **Do not generalise that rule.** Two of the most predictive columns in the dataset have *legitimate* negative values:

- `credit_risk_score`, range **[−191, 389]** — a negative score is a genuine low score, not missing.
- `velocity_6h`, range **[−175, 16818]** — negatives are a generator artefact, but the column is real and informative.

A blanket "negatives → NaN" rule destroys two of your best features. The toolkit's `SENTINEL_COLS` list is explicit and deliberately excludes both.

*(One nuance: when you build a **ratio** from `velocity_6h`, clip it at 0 first. A negative numerator flips the ratio's sign and makes a burst look calm. Keep the raw column unclipped — the model can use it directly.)*

### Trap 4 — Optimising the wrong metric

The source paper deliberately does **not** use accuracy, F1, or AUC as its headline. It uses **TPR at 5% FPR**, and it explains why: *"each false positive is a dissatisfied customer that may wish to change the banking company."*

The Kaggle wrapper most likely scores **ROC-AUC**. These are not the same objective — AUC rewards ranking everywhere, while TPR@5%FPR only cares about the very top of the ranking.

**Do this:** optimise for the leaderboard metric to place well, but *report and present* TPR@5%FPR, because that is the domain-correct metric and it is what a judge with banking knowledge will respect. Show both. Reporting only accuracy on a 1.1% positive class is how you lose — a model predicting "never fraud" scores 98.9% accuracy. Say that out loud in your presentation; it demonstrates you understand the problem.

### Trap 5 — Reflexively reaching for SMOTE

Your objective statement says "data balancing", so you will be tempted. Be careful: for **ranking metrics** on gradient-boosted trees, synthetic oversampling frequently fails to help and can distort calibration badly. See `research/R2-imbalance-truth.md` for the evidence and the exact argument to make. The winning move is not to skip balancing — it is to **run the ablation and show the judges the evidence**. Being the team that *tested* the assumption beats being the team that *followed* it.

### Trap 6 — Ignoring that this dataset exists for fairness research

BAF was built to stress-test **fairness**, with `customer_age` as the protected attribute and predictive equality (FPR ratio between age groups) as the metric. The paper's own finding on the Base variant is that strong models falsely flag the **older** age group substantially more often.

A false positive here is a real person **denied a bank account**. Almost nobody in your hackathon will address this. See `research/R4-fairness-explainability-pitch.md`.

---

## 4. First thirty minutes — the checks that decide your architecture

Run these before you write any model code. Each answers a question that changes what you build.

```python
import pandas as pd

df = pd.read_csv("train.csv")

# 1. Does the wrapper keep `month`? Is the split temporal or random?
print(df["month"].value_counts().sort_index() if "month" in df else "NO MONTH COLUMN")

# 2. Actual class balance, overall and per month
print(df["fraud_bool"].mean())
if "month" in df:
    print(df.groupby("month")["fraud_bool"].agg(["mean", "size"]))

# 3. Which columns are constant / useless?
print(df.nunique().sort_values().head(10))          # expect device_fraud_count == 1

# 4. Confirm the sentinel columns
for c in ["prev_address_months_count", "current_address_months_count",
          "bank_months_count", "session_length_in_minutes",
          "device_distinct_emails_8w", "intended_balcon_amount"]:
    if c in df:
        print(c, "| frac negative:", round((df[c] < 0).mean(), 4))

# 5. Cardinality of categoricals
print(df.select_dtypes("object").nunique())

# 6. Is `source` usable or is TELEAPP negligible?
if "source" in df:
    print(df["source"].value_counts(normalize=True))
```

**Decision rules:**
- `month` present with the same range in train and test → **random split** (this is your competition — verified). Mirror it with stratified K-fold and **keep `month`** as a feature.
- `month` absent from test, or test months strictly later → temporal split. Months 0–5 / 6–7, and **drop** `month`.
- `device_fraud_count` has 1 unique value → drop it. *(The datasheet says range [0,1] but it is widely reported constant in Base — `nunique()` settles it in one line. This was the one thing research could not confirm from documents.)*
- Any sentinel column with a meaningful fraction of negatives → build its missingness indicator. **Only for the six columns in `SENTINEL_COLS`** — see Trap 3.

**Also worth knowing:** the competition never states *which* BAF variant it uses, and some column ranges listed on Kaggle are *wider* than Base's — which a 70% subset of Base cannot produce. So it may not be Base. This does not change your approach, but do not assume published Base numbers transfer exactly.

---

## 5. What "good" looks like — verified numbers

**Competition metric: ROC-AUC** (confirmed). The full leaderboard was recovered:

| Position | AUC |
|---|---|
| 1st | **0.90444** |
| 2nd | 0.90437 |
| 3rd | 0.90108 |
| 4th | 0.89143 |
| 5th | 0.87823 |
| 6th | 0.86790 |
| 7th | 0.83518 |

Only 8 teams and 47 submissions. Private tracks public with a consistent small offset and **identical ordering — no shakeup risk.** Third place was reached in three submissions.

### Your targets

- **Target: 0.905** — that wins it.
- **Floor: 0.89** — a well-built single LightGBM should reach roughly here.
- **Above 0.92: stop and hunt for leakage.** Nothing legitimate on this data goes there.

### Independent reference points

| Model | AUC | TPR@5%FPR | Source |
|---|---|---|---|
| Tuned LightGBM (temporal split) | 0.8942 | 0.5486 | Vector Institute, outputs committed to repo |
| LightGBM (random split, `month` kept — closest analogue to your setup) | 0.8950 | — | same |
| CatBoost | 0.8836 | — | same |
| XGBoost | 0.8787 | — | same |
| Default LightGBM | — | 0.5254 | Feedzai's own `empirical_results.ipynb` |
| FT-Transformer (deep tabular) | 0.8955 | — | vs LightGBM 0.8953 on the same test set |

**Read those last two rows carefully.** A transformer beats LightGBM by 0.0002 — statistically nothing. **Do not spend your hackathon on deep tabular models.** GBDTs are the correct choice and you can say so with a citation.

The winner's 0.904 sits only ~0.01 above a solid single model. That gap is **tuning plus blending**, not a secret architecture. It is reachable.

### Two myths to avoid repeating

- **The famous "75.4% TPR" figure is not from this dataset.** It is Feedzai's private internal data. On BAF Base, TPR@5%FPR tops out near **0.55**. If you quote 75% you will be quoting the wrong number, and it is the kind of thing a knowledgeable judge catches.
- **Accuracy is meaningless here.** 98.9% is the do-nothing baseline. Say this out loud in your presentation.

A well-built 0.90 with a fairness analysis and a clear explanation beats an inflated 0.95 that collapses under one question. And judges do ask.

---

## 6. Sources

- Feedzai, *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS 2022 Datasets & Benchmarks — [arXiv 2211.13358](https://arxiv.org/abs/2211.13358), [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/d9696563856bd350e4e7ac5e5812f23c-Paper-Datasets_and_Benchmarks.pdf)
- Official datasheet (column definitions, ranges, missing-value encodings) — [github.com/feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud)
- Dataset on Kaggle — [Bank Account Fraud Dataset Suite (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
