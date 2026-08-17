# R3 — Feature Engineering & Model Tuning Playbook (BAF Base)

**Read order under time pressure:** §0 → §1 → §2 (build the Tier-1 block only) → §5 (config) → §7 (compute) → come back for §3/§4 if you have hours left.

Everything in §5 has been checked against current official docs (LightGBM 4.x `Parameters.rst`, XGBoost stable, CatBoost docs). Parameter names and defaults are quoted, not remembered.

---

## §0 — The thing that actually wins this hackathon

Most students will spend their time on features. The larger, cheaper win is **optimising the right metric on the right split**. Get these three right before you write a single feature:

### 0.1 The metric is TPR @ 5% FPR — not AUC, not F1

Feedzai's own benchmark notebook (`notebooks/empirical_results.ipynb`) computes:

```python
fprs, tprs, thresholds = metrics.roc_curve(y_test, predictions)
threshold = np.min(thresholds[fprs == max(fprs[fprs < 0.05])])
recall    = np.max(tprs[fprs == max(fprs[fprs < 0.05])])
```

This is a **single-point** metric on the low-FPR region of the ROC curve. ROC-AUC integrates the whole curve and PR-AUC weights the high-precision region differently. **A model that wins on AUC can lose on TPR@5%FPR.** Selecting checkpoints and hyperparameters on AUC is the single most common way to leave points on the table here.

Fix it with a custom eval function and turn the built-in metrics off:

```python
import numpy as np
from sklearn.metrics import roc_curve

def tpr_at_fpr(target_fpr=0.05):
    """LightGBM native-API feval. Rank-based, so raw-score vs sigmoid input is irrelevant."""
    def _feval(y_pred, dataset):
        y_true = dataset.get_label()
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return "tpr@5fpr", float(np.interp(target_fpr, fpr, tpr)), True  # True = higher is better
    return _feval
```

Then set `metric="None"` in params. Note: LightGBM requires the **string** `"None"` (aliases `"na"`, `"null"`, `"custom"`) to disable built-in metrics — passing Python `None` does not work.

> Sanity note on a rumour you may hit: there is an old LightGBM issue (#3648, filed against 3.1.1) claiming `average_precision` optimises in the wrong direction. In current LightGBM source (`src/metric/binary_metric.hpp`) `AveragePrecisionMetric::factor_to_bigger_better()` returns `1.0f`, i.e. higher-is-better, which is correct. The built-in `average_precision` is fine on 4.x. You still shouldn't early-stop on it here, because it isn't the competition metric.

### 0.2 The split is temporal — months 0–5 train, 6–7 test

From the same notebook: `df[df["month"] < 6]` / `df[df["month"] >= 6]`. The `month` column exists precisely so that you validate under temporal drift.

**Never use random K-fold on this dataset.** Random folds leak the future into the past and will flatter every feature you build, especially any count/frequency/target encoding. Your validation design:

```python
train = df[df.month <= 3]   # fit
valid = df[df.month.isin([4, 5])]  # early stopping + hyperparameter selection
test  = df[df.month >= 6]   # touch once, at the very end
```

If you want more signal for selection, use an **expanding-window** scheme (fit 0–2 → validate 3; fit 0–3 → validate 4; fit 0–4 → validate 5) and average the TPR@5%FPR. It costs 3× compute but is far more honest about drift than a single split.

For the **final** submission model, refit on all of months 0–5 using the iteration count found on the validation split (scale it up by roughly `n_train_full / n_train_partial`), since early stopping needs a holdout you no longer have.

### 0.3 Honest caveat about this dataset that changes how you should spend time

BAF is **synthetic** — generated with a CTGAN trained on a real anonymised account-opening dataset. The generator preserves marginals and much of the pairwise structure, but there is **no guarantee it preserved the higher-order interactions** that fraud domain knowledge suggests should exist.

Practical consequence: a feature with an impeccable fraud rationale (e.g. "address history longer than the applicant has been alive ⇒ fabricated identity") may simply carry **no signal in BAF**, because the GAN never encoded that constraint. Conversely, some artefact of the generator may be highly predictive and make no domain sense.

**So: every feature below is a hypothesis, not a fact.** §6 gives you the ablation loop. Do not ship 40 features on faith — engineered ratios are frequently *worse* than nothing for GBDTs, which can already approximate monotone ratios through successive splits. The features that reliably help trees are the ones expressing something a **sequence of axis-aligned splits cannot cheaply represent**: multiplicative/divisive relationships across very different scales, cross-row aggregates, and cross-column consistency checks.

---

## §1 — Load fast and correctly

```python
import numpy as np, pandas as pd

CAT_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
BIN_COLS = ["email_is_free", "phone_home_valid", "phone_mobile_valid",
            "has_other_cards", "foreign_request", "keep_alive_session"]

df = pd.read_csv("Base.csv")
for c in CAT_COLS:
    df[c] = df[c].astype("category")
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
for c in BIN_COLS + ["fraud_bool", "month"]:
    df[c] = df[c].astype("int8")
```

Drop the dead column, but verify first rather than trusting the datasheet:

```python
assert df["device_fraud_count"].nunique() == 1, "device_fraud_count is NOT constant — keep it"
df = df.drop(columns=["device_fraud_count"])
```

Polars alternative if CSV load is annoying you (~5–10× faster read):

```python
import polars as pl
df = pl.read_csv("Base.csv").to_pandas()   # then apply the dtype loop above
```

---

## §2 — Feature catalogue

Each block is copy-pasteable. **Tier labels are my honest prior on expected value, defined in §3.** Build Tier 1 first, measure, then decide whether to continue.

### 2.1 Sentinel handling and missingness — **Tier 1 (mechanism), Tier 3 (as raw indicators)**

Read this before you paste it, because the conventional advice here is half wrong.

For a GBDT, the raw `-1` sentinel is **already perfectly separable** by a split at `x <= 0`. An explicit `is_missing` indicator column is therefore close to redundant *for the raw feature*, and I would not expect it to move the needle on its own. This is the part of "always add missingness indicators" that gets over-sold.

What *does* matter, and is genuinely commonly missed:

1. **Leaving `-1` in place silently corrupts every derived feature.** `proposed_credit_limit / bank_months_count` with `bank_months_count = -1` produces a large negative number that collides with the legitimate range. Every ratio in §2.3–§2.7 is wrong unless you NaN-ify first.
2. **The row-level missingness *count* is a real, new signal.** Thin-file / fabricated applicants are missing several fields at once. No single split can express "how many fields are absent"; that's a cross-column aggregate.
3. **Missingness *co-occurrence patterns*** carry the same idea in a form trees can use directly.

```python
NEG1_SENTINEL = ["prev_address_months_count", "current_address_months_count",
                 "bank_months_count", "session_length_in_minutes",
                 "device_distinct_emails_8w"]

# 1. explicit indicators (cheap; keep them, but don't expect much alone)
for c in NEG1_SENTINEL:
    df[f"{c}_isna"] = (df[c] == -1).astype("int8")
df["intended_balcon_amount_isna"] = (df["intended_balcon_amount"] < 0).astype("int8")

MISS_FLAGS = [f"{c}_isna" for c in NEG1_SENTINEL] + ["intended_balcon_amount_isna"]

# 2. row-level missingness count — the actually-valuable one
df["n_missing"] = df[MISS_FLAGS].sum(axis=1).astype("int8")

# 3. missingness co-occurrence signature (6 flags -> one 0..63 integer)
#    NOTE: use *2**i, not the << operator — pandas Series does not implement bitwise shift.
df["missing_pattern"] = sum(df[c].astype("int16") * (2 ** i)
                            for i, c in enumerate(MISS_FLAGS)).astype("int16")

# 4. NOW NaN-ify, so derived features are not poisoned
for c in NEG1_SENTINEL:
    df[c] = df[c].mask(df[c] == -1)
df["intended_balcon_amount"] = df["intended_balcon_amount"].mask(df["intended_balcon_amount"] < 0)
```

LightGBM, XGBoost and CatBoost all handle `NaN` natively — do **not** impute. Imputation destroys the "missing" direction the tree would otherwise learn.

> `device_distinct_emails_8w` has range −1..2. After NaN-ifying, its real domain is {0, 1, 2}. A value of 0 distinct emails on a device is itself odd and worth watching.

### 2.2 Velocity ratios — **Tier 1**

The three `velocity_*` columns are application-rate aggregates over 6h / 24h / 4w windows, **on the same units**. Their ratios express *acceleration*: a burst of applications in the last 6 hours relative to the 4-week baseline is the classic bot-driven / mule-farm signature. A tree can only approximate a ratio by many splits on both columns, so this is exactly the case where hand-crafting earns its keep.

**Critical detail:** `velocity_6h` ranges −175..16818 — it goes **negative**, an artefact of the anonymisation/noise process. `velocity_24h` (min 1297) and `velocity_4w` (min 2825) are strictly positive, so they are safe denominators. Clip the numerator, never divide by `velocity_6h`.

```python
v6  = df["velocity_6h"].clip(lower=0)
df["vel_6h_over_24h"] = v6 / df["velocity_24h"]
df["vel_24h_over_4w"] = df["velocity_24h"] / df["velocity_4w"]
df["vel_6h_over_4w"]  = v6 / df["velocity_4w"]

# second-order: is the short-window burst outpacing the medium-window burst?
df["vel_acceleration"] = df["vel_6h_over_24h"] / df["vel_24h_over_4w"]

# log-space differences are better behaved than raw ratios for heavy tails
df["vel_log_6h_24h"] = np.log1p(v6) - np.log1p(df["velocity_24h"])
df["vel_log_24h_4w"] = np.log1p(df["velocity_24h"]) - np.log1p(df["velocity_4w"])

# spread/dispersion across the three windows
V = df[["velocity_6h", "velocity_24h", "velocity_4w"]].clip(lower=0)
df["vel_cv"] = V.std(axis=1) / (V.mean(axis=1) + 1e-6)
```

**Within-month normalisation — a drift-handling trick.** The velocity distributions shift across months. Normalising each row against its *own month's* median removes level shift and leaves the anomaly:

```python
for c in ["velocity_6h", "velocity_24h", "velocity_4w"]:
    med = df.groupby("month")[c].transform("median")
    iqr = df.groupby("month")[c].transform(lambda s: s.quantile(.75) - s.quantile(.25))
    df[f"{c}_mnorm"] = (df[c] - med) / (iqr + 1e-6)
```

This is **label-free** (no target leakage) and each month's statistics use only that month's rows, so there is no cross-time leakage either. It is *transductive* — it needs the test month's rows in batch. That is fine for this hackathon's batch scoring; flag it if the rules forbid transductive features.

### 2.3 Identity-coherence: `name_email_similarity` × email/device — **Tier 1**

Synthetic identities are assembled, not lived. The email is generated, so it does not match the claimed name (`name_email_similarity` low), it sits on a free provider (`email_is_free`), and the same device or the same date-of-birth appears under multiple email addresses (`device_distinct_emails_8w`, `date_of_birth_distinct_emails_4w`). Any one of these is weak; the **conjunction** is the signal, and conjunctions of a continuous variable with binary flags are cheap for you and expensive for a tree.

```python
df["nes_x_free_email"]  = df["name_email_similarity"] * df["email_is_free"]
df["nes_low_and_free"]  = ((df["name_email_similarity"] < 0.20) & (df["email_is_free"] == 1)).astype("int8")

# "identity farm": one DOB or one device fanning out across many emails
dev_emails = df["device_distinct_emails_8w"].fillna(0)
df["id_farm_score"]     = df["date_of_birth_distinct_emails_4w"] * dev_emails
df["dob_emails_x_free"] = df["date_of_birth_distinct_emails_4w"] * df["email_is_free"]

# similarity discounted by how many identities share the DOB
df["nes_per_dob_email"] = df["name_email_similarity"] / (1.0 + df["date_of_birth_distinct_emails_4w"])

# full synthetic-identity conjunction
df["synthetic_identity"] = (
    (df["name_email_similarity"] < 0.30) &
    (df["email_is_free"] == 1) &
    (df["date_of_birth_distinct_emails_4w"] >= 2) &
    (df["phone_home_valid"] == 0)
).astype("int8")
```

`date_of_birth_distinct_emails_4w` (0–39) is, on rationale alone, the strongest single identity-fabrication column in the schema: a legitimate applicant contributes one email per DOB; 39 distinct emails sharing a date of birth is an identity mill. Give its interactions priority.

### 2.4 Credit coherence: limit vs income vs risk score — **Tier 1**

`proposed_credit_limit` is applicant-influenced; `credit_risk_score` and `income` are bureau/derived. Fraudsters optimise for **payout**, so they request limits that are incoherent with the income and risk profile they present. "Incoherence between a requested amount and an assessed capacity" is a *residual*, and residuals across differently-scaled columns are the canonical case for manual features.

```python
df["limit_per_income"] = df["proposed_credit_limit"] / df["income"]          # income is 0.1..0.9
df["log_limit_per_income"] = np.log1p(df["proposed_credit_limit"]) - np.log1p(df["income"])

# credit_risk_score min is -191 -> shift to strictly positive before dividing
crs_pos = df["credit_risk_score"] + 200
df["limit_per_risk"] = df["proposed_credit_limit"] / crs_pos
df["risk_per_income"] = df["credit_risk_score"] / df["income"]

# rank-space incoherence: high requested limit but low assessed risk score (or vice versa)
r_limit  = df["proposed_credit_limit"].rank(pct=True)
r_risk   = df["credit_risk_score"].rank(pct=True)
r_income = df["income"].rank(pct=True)
df["incoh_limit_vs_risk"]   = r_limit - r_risk
df["incoh_limit_vs_income"] = r_limit - r_income
df["incoh_risk_vs_income"]  = r_risk  - r_income

# residual of limit against its own risk-bucket norm
bucket = pd.cut(df["credit_risk_score"], bins=20, labels=False)
df["limit_resid_in_risk_bucket"] = (
    df["proposed_credit_limit"] - df.groupby(bucket)["proposed_credit_limit"].transform("median")
)
```

Note `proposed_credit_limit` (200–2000) is almost certainly a small set of discrete tiers. Check `df["proposed_credit_limit"].value_counts()`; if it has ≤10 distinct values, **also treat it as a categorical** — the tier itself may matter more than its magnitude.

### 2.5 Address history / thin file — **Tier 2**

Real people leave an address trail. A fabricated identity has no previous address and a short current one. The plausibility checks are the interesting part: an address history that exceeds the applicant's adult lifetime is arithmetically impossible for a real person.

```python
# thin file: no previous address at all AND barely any current-address history
df["thin_file"] = (
    (df["prev_address_months_count_isna"] == 1) & (df["current_address_months_count"] < 12)
).astype("int8")

df["total_address_months"] = (df["prev_address_months_count"].fillna(0)
                              + df["current_address_months_count"].fillna(0))

# customer_age is rounded to the decade; guard the subtraction (min age is 10)
adult_months = ((df["customer_age"] - 18).clip(lower=1) * 12)
df["addr_months_over_adult_life"] = df["total_address_months"] / adult_months
df["addr_implausible"] = (df["total_address_months"] > adult_months).astype("int8")

# banking tenure relative to age — same fabrication logic
df["bank_months_over_adult_life"] = df["bank_months_count"] / adult_months
df["new_bank_and_thin_file"] = ((df["bank_months_count"].fillna(-1) <= 1)
                                & (df["thin_file"] == 1)).astype("int8")

# stability composite
df["stability_score"] = (df["current_address_months_count"].fillna(0)
                         + df["bank_months_count"].fillna(0)
                         + 12 * df["phone_home_valid"]
                         + 12 * df["has_other_cards"])
```

### 2.6 Device / session behaviour — **Tier 2**

Bot-driven and scripted applications differ from humans in *tempo*: near-zero deliberation, minimal session length, many identities per device. `device_os` is a proxy for the automation stack (headless Linux/X11 vs consumer Windows/macOS).

```python
dev_emails_safe = df["device_distinct_emails_8w"].fillna(1).clip(lower=1)
df["session_per_device_email"] = df["session_length_in_minutes"] / dev_emails_safe

df["bot_tempo"] = ((df["days_since_request"] < 0.01)
                   & (df["session_length_in_minutes"] < 2)).astype("int8")

df["session_x_keepalive"] = df["session_length_in_minutes"] * df["keep_alive_session"]
df["foreign_and_fast"]    = df["foreign_request"] * (df["days_since_request"] < 0.5).astype("int8")

# how many trust signals are absent at once
df["trust_signal_count"] = (df["phone_home_valid"] + df["phone_mobile_valid"]
                            + df["has_other_cards"] + (1 - df["email_is_free"])
                            + (1 - df["foreign_request"])).astype("int8")

# session length relative to the norm for that OS (label-free)
df["session_resid_by_os"] = (
    df["session_length_in_minutes"]
    - df.groupby("device_os", observed=True)["session_length_in_minutes"].transform("median")
)
```

### 2.7 Geography / branch concentration — **Tier 2**

```python
df["zip_x_branch"]        = np.log1p(df["zip_count_4w"]) * np.log1p(df["bank_branch_count_8w"])
df["zip_per_branch"]      = df["zip_count_4w"] / (df["bank_branch_count_8w"] + 1)
df["zip_count_x_velocity"] = np.log1p(df["zip_count_4w"]) + np.log1p(df["velocity_4w"])
```

### 2.8 Count / frequency and target encoding, done temporally safely — **Tier 3 on raw cats, Tier 2 on combos**

**The raw categoricals do not need encoding.** Cardinalities are 5, 7, 7, 2, 5. LightGBM handles these natively and optimally (see §4). Target-encoding a 5-level categorical is a waste of your afternoon and adds leakage risk for zero upside.

Where encoding *can* earn its place is on **high-cardinality combinations**, which express applicant-archetype rarity — mule-account applications cluster into unusual profile combinations.

```python
combo = (df["payment_type"].astype(str) + "|" + df["employment_status"].astype(str) + "|"
         + df["housing_status"].astype(str) + "|" + df["device_os"].astype(str))
df["profile_combo"] = combo.astype("category")     # up to 5*7*7*5 = 1225 levels

# --- frequency encoding: fit on TRAIN MONTHS ONLY, then map everywhere ---
freq = combo[df["month"] < 6].value_counts(normalize=True)
df["profile_combo_freq"] = combo.map(freq).fillna(0.0).astype("float32")
df["profile_combo_rare"] = (df["profile_combo_freq"] < 1e-4).astype("int8")
```

**Target encoding with an expanding temporal window.** The only safe construction: to encode month *m*, use labels strictly from earlier **labelled** months (0–5 only — never use test-month labels, which you would not have in production).

```python
PRIOR = df.loc[df.month < 6, "fraud_bool"].mean()
K = 100  # smoothing; higher = shrink harder to the prior

parts = []
for m in sorted(df["month"].unique()):
    hist = df[df["month"] < min(m, 6)]                 # labelled history only
    rows = df.index[df["month"] == m]
    if len(hist) == 0:                                  # month 0 has no history
        parts.append(pd.Series(PRIOR, index=rows, dtype="float32")); continue
    s = hist.groupby("profile_combo", observed=True)["fraud_bool"].agg(["sum", "count"])
    sm = (s["sum"] + K * PRIOR) / (s["count"] + K)
    parts.append(df.loc[rows, "profile_combo"].map(sm).fillna(PRIOR).astype("float32"))

df["profile_combo_te"] = pd.concat(parts).sort_index()
```

Month 0 gets the constant prior — that is correct and unavoidable; it is why you should validate on months 4–5, not month 0.

If you prefer a library, `sklearn.preprocessing.TargetEncoder` (added in scikit-learn **1.3**) does cross-fitted encoding — but note its cross-fitting is **random K-fold, not temporal**:

```python
from sklearn.preprocessing import TargetEncoder
# signature: TargetEncoder(categories='auto', target_type='auto', smooth='auto', cv=5, ...)
te = TargetEncoder(target_type="binary", smooth="auto", cv=5)
Xtr_enc = te.fit_transform(Xtr[["profile_combo"]], ytr)   # cross-fitted — use for TRAIN
Xte_enc = te.transform(Xte[["profile_combo"]])            # full-fit map — use for TEST
```

The single most important rule: `fit(X, y).transform(X)` is **not** equal to `fit_transform(X, y)`. Only `fit_transform` applies the internal cross-fitting scheme. Calling `fit().transform()` on your training data leaks the target and will produce a feature that dominates importance charts and collapses on test.

### 2.9 Features to skip

- **Polynomial expansions / `PolynomialFeatures`** — trees do not benefit, and you multiply your search cost.
- **PCA / SVD components** — destroys axis-alignment, which is the one thing GBDTs are good at.
- **Standardisation / scaling** — irrelevant to trees. Do not waste time.
- **SMOTE and friends** — see §5.5. On 1M rows with a rank-based metric this is almost always a net negative and always a large time cost.
- **Imputing the sentinels with mean/median** — actively harmful; you are deleting the signal.

---

## §3 — What is likely to actually matter, ranked

My honest prior, ordered by (expected effect size) × (probability it survives ablation). **This is reasoning, not measurement — see §6.**

| # | Feature block | Why I rank it here | Confidence |
|---|---|---|---|
| **1** | **Correct metric + temporal validation (§0)** | Not a feature, but the largest expected gain available. Selecting on TPR@5%FPR instead of AUC changes which model you ship. Zero risk, low cost. | High |
| **2** | **Velocity ratios & acceleration (§2.2)** | Same-unit columns, so ratios are semantically meaningful. Divisive relationships across a 3-order-of-magnitude range are genuinely expensive for axis-aligned splits. Directly encodes bot/burst behaviour. | Med-High |
| **3** | **`date_of_birth_distinct_emails_4w` interactions (§2.3)** | Strongest identity-fabrication column in the schema on pure rationale. Its products with `email_is_free` / `device_distinct_emails_8w` encode "identity farm" conjunctions. | Med-High |
| **4** | **Credit-limit ÷ income and rank-incoherence (§2.4)** | `proposed_credit_limit` is applicant-controlled and payout-linked; incoherence with income/risk is the core account-opening fraud tell. Rank-space differences are cheap and robust. | Med-High |
| **5** | **`n_missing` + `missing_pattern` (§2.1)** | A cross-column *count* that no single split can express. The per-column `_isna` flags that usually get all the attention are near-redundant; the aggregate is the real feature. | Medium |
| 6 | Within-month velocity normalisation (§2.2) | Directly attacks the train/test drift that the temporal split is designed to punish. Higher variance — may help a lot or nothing. | Medium |
| 7 | Address/age plausibility (§2.5) | Impeccable rationale, but `customer_age` is rounded to the decade, which blunts the arithmetic check badly. Most at risk from the CTGAN caveat in §0.3. | Med-Low |
| 8 | Device/session tempo (§2.6) | Sensible, but `session_length_in_minutes` and `days_since_request` are already in the model and trees split them easily. Marginal. | Med-Low |
| 9 | `profile_combo` frequency encoding (§2.8) | Adds a rarity signal trees cannot compute. Real but small. | Low-Med |
| 10 | `profile_combo` target encoding (§2.8) | Highest leakage risk per unit of expected gain in the whole document. Only attempt once everything else is done and the temporal harness is trustworthy. | Low |
| — | Per-column `_isna` indicators alone | Almost certainly redundant with the raw `-1` for a GBDT. Keep them (they cost nothing) but do not expect a lift. | Low |

**The sentinel→NaN conversion in §2.1 step 4 is not optional at any tier** — it is a correctness prerequisite for blocks 2, 4, 5, 7 and 8, not a feature in its own right.

### The one-paragraph version
Build §2.1 (all of it), §2.2, §2.3, §2.4. That is roughly 30 columns and about 20 minutes of work. Ablate as a block against baseline. If the block helps, ablate *within* it to find the sub-block carrying the gain, and drop the rest — fewer features means faster search, which means more search, which usually beats better features.

---

## §4 — Categorical handling: recommendation and exact settings

### 4.1 The situation
Cardinalities: `payment_type` 5, `employment_status` 7, `housing_status` 7, `source` 2 (heavily skewed toward INTERNET), `device_os` 5. **All are low-cardinality.** This makes most of the categorical-encoding debate moot — the methods will land within noise of each other, and you should not spend hours here.

### 4.2 Recommendation: **LightGBM native categorical, with `max_cat_to_onehot` raised to 8**

This is the specific detail worth knowing. LightGBM's `max_cat_to_onehot` defaults to **`4`**, documented as: *"when number of categories of one feature smaller than or equal to `max_cat_to_onehot`, one-vs-other split algorithm will be used."*

With the default of 4, your 5- and 7-level features fall **above** the threshold, so LightGBM uses its many-vs-many algorithm: it sorts categories by gradient statistics and finds a partition. That is powerful for high-cardinality features and a **known overfitting risk on rare categories** — which is exactly your situation with 1.1% positives, where a rare category may hold only a handful of frauds.

Setting `max_cat_to_onehot=8` puts every BAF categorical below the threshold, forcing the **exact one-vs-rest** enumeration. For ≤7 levels this is both tractable and non-overfitting. This costs nothing and removes a real failure mode.

```python
params.update(
    max_cat_to_onehot = 8,     # default 4 -> forces exact one-vs-rest for all BAF cats
    cat_smooth        = 50.0,  # default 10.0; raise it — "reduce effect of noises in categorical features"
    cat_l2            = 10.0,  # default 10.0; L2 on the categorical split
    min_data_per_group= 200,   # default 100; raise it for 1% positives
    max_cat_threshold = 32,    # default 32; inert once max_cat_to_onehot dominates
)
# and pass:
train_set = lgb.Dataset(X, y, categorical_feature=CAT_COLS, free_raw_data=False)
```

Pandas `category` dtype columns are auto-detected, but pass `categorical_feature` explicitly anyway — it is self-documenting and immune to a dtype slipping back to `object`.

### 4.3 Why not the alternatives

**One-hot encoding** — will perform essentially identically, because §4.2 makes LightGBM do one-vs-rest internally anyway. The difference is that manual one-hot adds 26 sparse columns which dilute `feature_fraction` sampling and slow histogram construction slightly. No upside. *Use it only if you also need a linear model or a scikit-learn pipeline for stacking.*

**Target encoding on the raw categoricals** — no. Five levels do not need a learned representation; you are adding leakage surface for no capacity gain. Reserve TE for `profile_combo` (§2.8) if at all.

**CatBoost ordered target statistics** — CatBoost's ordered CTRs are a genuinely principled leakage-free target encoding, and this is the setting they were built for. But their advantage is concentrated on **high-cardinality** categoricals, which you do not have. Where CatBoost *does* stay interesting on BAF is as an **ensemble member** — it is a structurally different learner (oblivious/symmetric trees), so its errors decorrelate from LightGBM's, and averaging the two ranks usually adds a little TPR. Run it if you have spare time, not as your primary. See §5.6 for the config, which has several traps.

`cat_features` accepts column names or indices (if you pass names, the frame must carry names for *all* columns). The columns must be **int or str** — a float categorical raises `CatBoostError: cat_features must be integer or string...`, and so does a `NaN` in a categorical column. Cast before fitting:

```python
for c in CAT_COLS:
    df[c] = df[c].astype(str).fillna("__MISSING__")   # CatBoost: no float, no NaN in cat cols
```

### 4.4 XGBoost
If you use XGBoost, native categoricals require `enable_categorical=True`, pandas `category` dtype, and `tree_method` in `{"hist", "approx"}` (the docs state: *"Supported tree methods are `approx` and `hist`"*). Same one-hot argument applies via `max_cat_to_onehot=8`.

---

## §5 — Hyperparameters

> **Verification status, stated honestly.** Every feature snippet in §2 and the LightGBM config/`feval`/early-stopping code in §5.2 were **executed** against a synthetic frame built to this exact schema on **LightGBM 4.7.0** — they run clean (83 columns, no infs, no NaN leaks). That run is also what caught two errors that were in an earlier draft of this document: the `<<` bug in §2.1, and a wrong claim about `goss` in §5.2.
> The **XGBoost and CatBoost configs are documentation-verified only** — those libraries were not installed on the machine used to write this, so their parameter names and defaults are quoted from current official docs and source but the configs were not run. Expect them to be correct; do not be shocked by a typo. Run them once with `n_estimators=10` before trusting a long job to them.

### 5.1 The imbalance arithmetic you need before touching anything

Train months 0–5 ≈ **750k rows** with ≈ **8,200 positives**. That number governs every capacity parameter below.

**The single most important consequence:** for logloss, the Hessian of an observation is `p(1−p)`. Because the base rate is ~1%, predicted `p` is small, so a typical Hessian is ≈ **0.01**. This means:

- LightGBM's `min_sum_hessian_in_leaf` (default **`1e-3`**) corresponds to roughly **0.1 samples**. It is completely non-binding. Raising it to `1.0`–`10.0` (≈100–1000 effective samples) turns it into a real, positives-aware regulariser — it constrains leaves by *confidence mass*, not raw row count, so it directly prevents leaves built on three lucky frauds.
- XGBoost's `min_child_weight` (default **`1`**) is the same quantity and, at ~0.01 Hessian per row, already implies ≈100 rows. Its default is therefore much stronger than LightGBM's — do not port a LightGBM value across naively.

`min_data_in_leaf` (LightGBM default **`20`**, aliases include `min_child_samples`) is far too permissive here. With 8,200 positives spread over 64 leaves you average ~128 positives/leaf; a leaf of 20 rows is pure noise. **Start at 200.** This is the parameter I would tune first and the one most likely to be mis-set in your competitors' notebooks.

### 5.2 LightGBM — starting config

```python
import os, lightgbm as lgb

params = dict(
    objective        = "binary",
    metric           = "None",        # STRING "None" — disables built-ins; we use feval
    boosting         = "gbdt",        # alias boosting_type; options gbdt / rf / dart

    learning_rate    = 0.05,          # default 0.1
    num_leaves       = 64,            # default 31
    max_depth        = -1,            # default -1 (no limit); let num_leaves govern

    min_data_in_leaf        = 200,    # default 20  <-- the critical one
    min_sum_hessian_in_leaf = 1.0,    # default 1e-3 <-- non-binding by default, see 5.1
    min_gain_to_split       = 0.0,    # default 0.0

    feature_fraction = 0.70,          # default 1.0 (alias colsample_bytree)
    bagging_fraction = 0.80,          # default 1.0 (alias subsample)
    bagging_freq     = 1,             # default 0 — MUST be >0 or bagging_fraction is ignored

    lambda_l1        = 0.0,           # default 0.0
    lambda_l2        = 5.0,           # default 0.0

    max_bin          = 255,           # default 255
    max_cat_to_onehot = 8,            # default 4  — see §4.2
    cat_smooth        = 50.0,         # default 10.0
    cat_l2            = 10.0,         # default 10.0
    min_data_per_group= 200,          # default 100

    scale_pos_weight = 1.0,           # default 1.0 — leave it, see §5.5
    num_threads      = os.cpu_count(),# default 0 (= all cores)
    force_col_wise   = True,          # skips the row/col-wise auto-benchmark at startup
    verbosity        = -1,
    seed             = 42,
    deterministic    = False,         # set True only for final reproducibility runs (slower)
)

booster = lgb.train(
    params,
    lgb.Dataset(Xtr, ytr, categorical_feature=CAT_COLS),
    num_boost_round = 3000,
    valid_sets      = [lgb.Dataset(Xva, yva, categorical_feature=CAT_COLS, reference=...)],
    feval           = tpr_at_fpr(0.05),
    callbacks       = [lgb.early_stopping(150, first_metric_only=True),
                       lgb.log_evaluation(100)],
)
```

**`bagging_freq=1` is mandatory** — `bagging_fraction` alone does nothing, because `bagging_freq` defaults to `0`, which disables bagging entirely. This is the most common silent misconfiguration in LightGBM.

**GOSS gotcha — verified by running it, because the docs alone are misleading here.**

Feedzai's own published search space (`notebooks/lightgbm_hyperparameter_space.yaml`) contains `boosting_type: ["gbdt", "goss"]`, written for LightGBM 3.x. The current docs list `boosting` as accepting only `gbdt`, `rf`, `dart`, and document GOSS under a separate parameter:

```python
params["data_sample_strategy"] = "goss"   # default "bagging"; options: bagging | goss
```

**However, `boosting="goss"` still works in LightGBM 4.7.0** — I tested it; it trains fine and is accepted as a legacy spelling despite not appearing in the documented option list. So Feedzai's YAML will not error on that account.

The trap is different and it *will* hit you, because it is triggered by the config in §5.2:

```
LightGBMError: Cannot use bagging in GOSS
```

**GOSS is incompatible with bagging, under either spelling.** Since the starting config above sets `bagging_fraction=0.8, bagging_freq=1`, adding GOSS to your search space without removing the bagging parameters makes **every GOSS trial crash**. If you search over GOSS, make it conditional:

```python
if trial_uses_goss:
    params["data_sample_strategy"] = "goss"
    params.pop("bagging_fraction", None); params.pop("bagging_freq", None)
    params["top_rate"], params["other_rate"] = 0.2, 0.1   # goss's own sampling knobs
```

Honestly: GOSS is a **speed** optimisation, not an accuracy one. On 750k rows it is not the lever you need — negative downsampling (§7.3) buys far more. Prefer to leave it out of the search entirely and keep bagging.

(Feedzai's space is still a useful sanity check on ranges: `n_estimators` 20–10000 log, `max_depth` 3–30, `learning_rate` 0.02–0.1 log, `num_leaves` 10–100 log, `min_data_in_leaf` 5–200 log, `max_bin` 100–500.)

### 5.3 LightGBM — search space

```python
space = {
  "learning_rate":            ("loguniform", 0.01, 0.15),
  "num_leaves":               ("int_log",    16,   256),
  "min_data_in_leaf":         ("int_log",    50,   2000),   # widen HIGH, not low
  "min_sum_hessian_in_leaf":  ("loguniform", 0.1,  20.0),
  "feature_fraction":         ("uniform",    0.4,  1.0),
  "bagging_fraction":         ("uniform",    0.5,  1.0),    # keep bagging_freq = 1
  "lambda_l1":                ("loguniform", 1e-3, 10.0),
  "lambda_l2":                ("loguniform", 1e-3, 50.0),
  "min_gain_to_split":        ("loguniform", 1e-4, 1.0),
  "max_bin":                  ("int",        63,   255),
  "cat_smooth":               ("loguniform", 1.0,  200.0),
  "path_smooth":              ("loguniform", 1e-3, 100.0),  # default 0; smooths small leaves
}
```

Deliberate choices in this space:

- **`min_data_in_leaf` upper bound is 2000, not 200.** Feedzai capped at 200; with 8k positives I expect the optimum to sit higher than most people search. Extending the range upward is the cheapest way to find a regularisation win.
- **`learning_rate` and `n_estimators` are not searched jointly.** Fix a low learning rate, set `num_boost_round` generously (3000), and let early stopping choose the count. Searching both is wasteful — they trade off almost exactly, so you would be burning trials to rediscover a hyperbola. Only revisit if wall-clock forces `learning_rate` up.
- **`max_depth` is omitted.** With `num_leaves` in the space, `max_depth` is largely redundant and the two interact confusingly. Add `max_depth ∈ [4, 12]` only if you see overfitting that `num_leaves` cannot control.
- **`path_smooth`** is under-used and well-suited here: it shrinks leaf outputs toward their parents in proportion to leaf size, which is exactly the right prior when positives are rare. Note the docs say it requires `min_data_in_leaf >= 2` to take effect.

### 5.4 XGBoost — starting config

```python
params = dict(
    objective        = "binary:logistic",
    eval_metric      = "aucpr",       # replace with a custom TPR@5%FPR feval if you can
    tree_method      = "hist",        # default "auto"; "hist" is what you want at 1M rows
    device           = "cpu",         # default "cpu"

    grow_policy      = "lossguide",   # default "depthwise"; lossguide ~ LightGBM's leaf-wise
    max_depth        = 0,             # 0 = no limit, required for pure lossguide
    max_leaves       = 64,            # default 0

    learning_rate    = 0.05,          # default 0.3 (alias eta) — the default is far too high
    min_child_weight = 5.0,           # default 1  (Hessian sum; ~= 500 rows here, see §5.1)
    gamma            = 0.0,           # default 0  (alias min_split_loss)
    subsample        = 0.8,           # default 1
    colsample_bytree = 0.7,           # default 1
    reg_lambda       = 5.0,           # default 1
    reg_alpha        = 0.0,           # default 0
    max_delta_step   = 1,             # default 0 — see below
    max_bin          = 256,           # default 256

    enable_categorical = True,        # requires pandas "category" dtype + hist/approx
    max_cat_to_onehot  = 8,
    scale_pos_weight   = 1.0,         # default 1
)
```

**`max_delta_step`** is the XGBoost-specific parameter worth knowing here. The docs describe it as *"helpful in logistic regression when class is extremely imbalanced"* — it caps the magnitude of each leaf's weight update, preventing a leaf of a few positives from producing an enormous score. Values of **1–10** are the documented suggestion. At a 1% base rate this is cheap insurance; include it in your space as `int ∈ [0, 10]`.

Note XGBoost's `learning_rate` default is **0.3**, versus LightGBM's 0.1. Leaving it at default is a common and costly mistake.

### 5.5 Class imbalance: what to do, and what not to

**Recommendation: leave `scale_pos_weight = 1` and `is_unbalance = False`. Do not resample.**

Reasoning:

1. **Your metric is rank-based.** TPR@5%FPR depends only on the *ordering* of scores. Class weighting mostly rescales the score distribution and shifts calibration; it does not reliably improve ordering. The intuition that "imbalance must be corrected" comes from fixed-0.5-threshold accuracy/F1 settings — which is not your setting. You pick your threshold from the ROC curve at the end regardless.
2. It does have a second-order effect on split gains, so it is not *literally* inert — which is why it belongs in your search space with a **narrow** range (`scale_pos_weight ∈ [1, 10]`), not set to `n_neg/n_pos ≈ 90` on principle.
3. `scale_pos_weight` and `is_unbalance` are **mutually exclusive in LightGBM** — setting both is a configuration error. `is_unbalance=True` computes the ratio automatically, which is the aggressive setting you probably don't want.
4. Both **destroy probability calibration**. If any part of your submission needs calibrated probabilities (expected-loss thresholding, a stacking layer), wrap the model in `CalibratedClassifierCV(method="isotonic")` fitted on a held-out slice — do not try to fix it with weights.
5. **SMOTE / ADASYN / random oversampling: skip.** On 1M rows the synthesis cost is large, the synthetic minority points are interpolations in a space with meaningful categorical and sentinel structure (interpolating between `bank_months_count = NaN` and `= 30` is meaningless), and the gains for GBDTs on a ranking metric are consistently marginal-to-negative in the literature. This is the highest time-cost / lowest-expected-value option available to you.

Random **under**sampling of the majority is different — that is a *compute* technique, not an imbalance fix, and it is covered in §7.3.

### 5.6 CatBoost — starting config, and four traps

CatBoost is your **ensemble diversifier**, not your primary. If you run it, these four things are non-obvious and each one silently costs you if you get it wrong.

**Trap 1 — `min_data_in_leaf` does nothing under the default grow policy.** The docs state it "can be used only with the Lossguide and Depthwise growing policies." The default `grow_policy` is `SymmetricTree`. Worse, it appears to be **silently ignored** rather than raising an error (see CatBoost issue #2889, showing leaves of 1–63 samples at `min_data_in_leaf=100`). If you port the "raise `min_data_in_leaf` because positives are rare" logic from §5.1 into CatBoost and leave the default grow policy, **you have configured nothing**.

**Trap 2 — and you cannot simply switch grow policy for free.** Setting `grow_policy` to `Depthwise` or `Lossguide` forces `boosting_type="Plain"`: *"Ordered boosting is not supported for nonsymmetric trees."* So leaf-size control and ordered boosting are mutually exclusive. **My recommendation: keep `SymmetricTree` and regularise with `l2_leaf_reg` + `depth` instead.** Symmetric trees are themselves a strong regulariser (every node at a level shares a split), which is well-suited to 8k positives, and it keeps CatBoost structurally different from your LightGBM — which is the entire reason it is in your ensemble.

**Trap 3 — ordered boosting is not on by default on CPU anyway.** The `boosting_type` default is `Plain` on CPU. (It is `Ordered` on *GPU* for ≤50k objects.) If you want ordered boosting's leakage guard, you must ask for it: `boosting_type="Ordered"`. It is significantly slower.

**Trap 4 — setting `l2_leaf_reg` silently disables the automatic learning rate.** CatBoost auto-selects `learning_rate` only if none of `l2_leaf_reg`, `leaf_estimation_iterations`, `leaf_estimation_method` is set. Touch any of them and it reverts to `0.03`. Set `learning_rate` explicitly so you always know what you are running.

```python
from catboost import CatBoostClassifier

# has_time relies on physical row order — sort by month FIRST
tr = tr.sort_values("month").reset_index(drop=True)

cb = CatBoostClassifier(
    iterations        = 3000,                # default 1000
    learning_rate     = 0.05,                # default 0.03 — set explicitly, see Trap 4
    depth             = 6,                   # default 6 (16 if grow_policy=Lossguide)
    l2_leaf_reg       = 10.0,                # default 3.0 — raise it; this is your main knob
    grow_policy       = "SymmetricTree",     # default; see Traps 1 & 2
    boosting_type     = "Plain",             # CPU default; "Ordered" is slower, see Trap 3
    bootstrap_type    = "MVS",               # CPU default for binary classification
    subsample         = 0.8,                 # MVS default 0.8
    rsm               = 0.8,                 # default None(=1). CPU ONLY — errors on GPU
    random_strength   = 1.0,                 # default 1
    border_count      = 254,                 # CPU default 254 (GPU 128)
    one_hot_max_size  = 10,                  # default 2 — raise so all BAF cats are one-hot
    max_ctr_complexity= 2,                   # default 4; lower = much faster, fine at low card.
    has_time          = True,                # see below
    eval_metric       = "PRAUC:type=Classic",# valid; see note
    od_type           = "Iter",              # default "IncToDec"; "Iter" = plain early stopping
    od_wait           = 150,                 # default 20 — far too twitchy for a fraud metric
    auto_class_weights= None,                # default; see §5.5 — leave it off
    thread_count      = -1,
    random_seed       = 42,
    verbose           = 200,
)
cb.fit(Xtr, ytr, cat_features=CAT_COLS, eval_set=(Xva, yva), use_best_model=True)
```

**`has_time=True`** makes CatBoost use the input row order instead of random permutations when computing categorical target statistics and choosing tree structure — and it additionally forces `permutation_count` to 1. For temporally ordered fraud data this is what stops a row's CTR being computed from future rows. **It only works if you actually sorted by `month` first.**

**Metric strings** (verified): `PRAUC` is valid and takes `type` ∈ {`Classic`, `OneVsAll`}, default `Classic` — use `"PRAUC:type=Classic"` for binary. `AUC` is valid too, but note its `type` defaults to **`Ranking`** in the binary context, with `Classic` as the alternative. `PRAUC` appears to be eval-only (not usable as `loss_function`). As with LightGBM, none of these is your competition metric — compute TPR@5%FPR yourself on `cb.predict_proba(Xva)[:, 1]` and select on that.

**`od_type` semantics differ:** `Iter` stops training N iterations after the best; `IncToDec` (the default) uses a threshold-based heuristic. You want `Iter` — it is the behaviour you expect from "early stopping rounds".

**Do not combine** `auto_class_weights`, `class_weights` and `scale_pos_weight` — the docs mark all three as mutually exclusive. (`auto_class_weights` accepts exactly `None`, `Balanced`, `SqrtBalanced`.)

### 5.7 Early stopping

- Stop on **TPR@5%FPR** via `feval`, not AUC (§0.1).
- `early_stopping_round` (aliases `early_stopping_rounds`, `early_stopping`, `n_iter_no_change`; default **`0`** = disabled) → set **150** at `learning_rate=0.05`. Fraud metrics at a fixed operating point are noisy; a short patience will stop you on noise.
- If you pass multiple eval metrics, set `first_metric_only=True` (default `false`) or LightGBM stops when *any* metric stops improving.
- Log the best iteration from every trial. If your best trials consistently hit `num_boost_round` without stopping, your ceiling is too low, not your learning rate.

---

## §6 — Ablation protocol (do not skip this)

Feature-engineering value on this dataset is a **claim you must test**, not a fact you can assume — especially given §0.3.

```python
BLOCKS = {
  "baseline":  [],
  "missing":   MISS_FLAGS + ["n_missing", "missing_pattern"],
  "velocity":  [c for c in df if c.startswith("vel_")],
  "identity":  ["nes_x_free_email","nes_low_and_free","id_farm_score",
                "dob_emails_x_free","nes_per_dob_email","synthetic_identity"],
  "credit":    [c for c in df if c.startswith(("limit_","incoh_","risk_per"))],
  "address":   ["thin_file","total_address_months","addr_months_over_adult_life",
                "addr_implausible","bank_months_over_adult_life","stability_score"],
  "device":    ["session_per_device_email","bot_tempo","session_x_keepalive",
                "foreign_and_fast","trust_signal_count","session_resid_by_os"],
}

import itertools
results = {}
for name in BLOCKS:
    cols = RAW_COLS + BLOCKS[name]
    scores = [fit_and_score(cols, seed=s) for s in (0, 1, 2)]   # 3 seeds — see below
    results[name] = (np.mean(scores), np.std(scores))
```

**Run every configuration with at least 3 seeds and report the standard deviation.** With ~8,200 positives, the seed-to-seed spread in TPR@5%FPR is large. A block that improves the mean by less than one standard deviation has not been shown to help — keeping it costs you search speed and adds overfitting surface for nothing. Be ruthless: drop it.

Add blocks **greedily** (best block first, then re-test the remainder conditional on it), rather than assuming independence — velocity and device features overlap in what they capture.

---

## §7 — Compute reality on a student laptop

### 7.1 What is fast and what is not

Assume 4–8 cores, 8–16 GB RAM, no GPU.

| Operation | Rough cost | Notes |
|---|---|---|
| `pd.read_csv` on Base.csv | 10–25 s | ~2–4 s with `polars`/`pyarrow`. Do it once, cache to Parquet. |
| Feature block from §2 | 5–20 s | Vectorised; negligible. The `for m in months` TE loop is the slow one. |
| One LightGBM fit, 750k × 60, ~1000 trees | **1–4 min** | This is your unit of currency. Budget everything in these. |
| One CatBoost fit, same data | 3–10 min | Ordered boosting is materially slower. Plain is faster. |
| One XGBoost `hist` fit | 2–6 min | Comparable to LightGBM, usually slightly slower. |
| 100-trial random search, full data | **3–7 hours** | Will not finish before your deadline. |
| SHAP `TreeExplainer` on 1M rows | 10–40 min | Sample 20k rows. `shap.TreeExplainer(m).shap_values(X.sample(20_000))`. |

**The bottleneck is the search, never the fit.** One model is fine; a hundred is not.

### 7.2 Cache aggressively

```python
df.to_parquet("baf_features.parquet")   # ~10x faster reload than CSV, preserves dtypes
```

Memory: 1M × 60 columns at float64 is ~480 MB *per copy*, and pandas makes copies liberally. Downcast to `float32` (§1) and you halve it. Build features once, write Parquet, and never re-run the FE cell.

### 7.3 Make the search finish: negative downsampling

This is the highest-leverage compute trick available. **Keep every positive, sample the negatives.**

```python
pos = tr[tr.fraud_bool == 1]
neg = tr[tr.fraud_bool == 0].sample(frac=0.10, random_state=0)
search_tr = pd.concat([pos, neg]).sample(frac=1, random_state=0)   # ~82k rows, ~10x faster
```

At 10% negatives you go from 750k to ~82k rows — roughly a **9× speedup**, turning a 5-hour search into ~35 minutes. Ranking is largely preserved under negative downsampling (it is a monotone-ish transform of the scores), which is exactly what a rank-based metric needs.

**Three caveats that matter, in order of importance:**

1. **Scale the capacity parameters back up when you refit on full data.** `min_data_in_leaf = 200` on 82k rows is *not* equivalent to 200 on 750k rows — it is ~9× more restrictive in relative terms. When you move the best config to the full dataset, multiply `min_data_in_leaf` (and `min_data_per_group`) by roughly your inverse sampling ratio. Forgetting this is the standard way this trick backfires.
2. **Always validate on the full, un-downsampled validation months.** Downsample the *training* rows only. If you downsample validation too, your FPR axis is distorted and the 5% operating point means something different.
3. Downsampling adds variance to the search. Use it to find the *region* of good hyperparameters, then do a short refinement search at full scale around the winner.

### 7.4 Other levers, in order of value-per-effort

1. **Lower `max_bin` to 63 during search.** Histogram construction cost scales with bins; 63 vs 255 is a solid speedup at a small accuracy cost. Restore 255 for final fits.
2. **Use Optuna with a pruner.** `optuna.pruners.MedianPruner` or `HyperbandPruner` kills bad trials after a few hundred rounds instead of running all 3000. Typically 2–3× effective throughput.
3. **`force_col_wise=True`.** LightGBM otherwise runs a small benchmark at the start of *every* fit to choose row-wise vs col-wise. At 60 features col-wise wins; setting it explicitly removes that overhead and the warning spam.
4. **Single validation split during search, expanding-window only for the final 5 candidates.** Do not pay 3× for cross-validation on trials you will discard.
5. **Set `n_jobs`/`num_threads` to *physical* cores, not logical.** Hyperthreading typically gives LightGBM nothing and sometimes hurts. `psutil.cpu_count(logical=False)`.
6. **Do not run other things.** A browser with 40 tabs will halve your throughput on 8 GB.

### 7.5 A realistic time budget

| Phase | Budget | Output |
|---|---|---|
| Load, dtype, cache to Parquet | 15 min | `baf_raw.parquet` |
| Baseline LightGBM, correct metric + temporal split | 20 min | **Your reference number.** Get this before any FE. |
| Tier-1 features (§2.1–2.4) | 30 min | ~30 new columns |
| Block ablation, 3 seeds (§6) | 60 min | Keep/drop decision per block |
| Downsampled random/Optuna search, 60–100 trials | 60 min | Hyperparameter region |
| Full-scale refinement, ~15 trials | 45 min | Final config (remember §7.3 caveat 1) |
| Final refit on months 0–5 + threshold selection | 20 min | Submission |
| Seed-averaged ensemble of 5 fits | 30 min | Usually +a small, reliable gain |

Total ≈ 4.5 hours. **Do the baseline in the first 35 minutes.** Without a reference number, every subsequent decision is guesswork.

### 7.6 The cheapest remaining win
Once you have a final config, fit it **5 times with different seeds and average the predicted ranks**:

```python
from scipy.stats import rankdata
preds = np.mean([rankdata(fit(seed=s).predict(Xte)) for s in range(5)], axis=0)
```

Rank-averaging is scale-free, so it is safe even if the individual models are differently calibrated. Given the seed variance noted in §6, this is a near-free reduction in variance and it converts your "lucky seed" risk into a modest, dependable gain. Add a CatBoost fit to the average (§4.3) if time remains.

---

## Appendix — Parameter names verified against current official docs

| Parameter | Library | Default | Note |
|---|---|---|---|
| `num_leaves` | LightGBM | `31` | aliases `max_leaves`, `max_leaf_nodes`; `1 < num_leaves <= 131072` |
| `min_data_in_leaf` | LightGBM | `20` | aliases `min_data_per_leaf`, `min_child_samples`, `min_samples_leaf` |
| `min_sum_hessian_in_leaf` | LightGBM | `1e-3` | aliases `min_sum_hessian`, `min_child_weight` |
| `learning_rate` | LightGBM | `0.1` | aliases `shrinkage_rate`, `eta` |
| `feature_fraction` | LightGBM | `1.0` | alias `colsample_bytree` |
| `bagging_fraction` / `bagging_freq` | LightGBM | `1.0` / `0` | **freq must be > 0** or bagging is off |
| `lambda_l1` / `lambda_l2` | LightGBM | `0.0` / `0.0` | aliases `reg_alpha` / `reg_lambda` |
| `max_cat_to_onehot` | LightGBM | `4` | one-vs-other when n_categories ≤ this |
| `cat_smooth` / `cat_l2` | LightGBM | `10.0` / `10.0` | |
| `min_data_per_group` | LightGBM | `100` | |
| `max_cat_threshold` | LightGBM | `32` | |
| `max_bin` | LightGBM | `255` | alias `max_bins` |
| `early_stopping_round` | LightGBM | `0` | aliases `early_stopping_rounds`, `n_iter_no_change` |
| `first_metric_only` | LightGBM | `false` | |
| `metric` | LightGBM | `""` | valid here: `auc`, `average_precision`, `binary_logloss`; `"None"` disables |
| `boosting` | LightGBM | `gbdt` | documented options `gbdt`/`rf`/`dart`; `goss` is undocumented but **still accepted in 4.7.0** (tested) |
| `data_sample_strategy` | LightGBM | `bagging` | options `bagging`/`goss` — the documented home for GOSS. **Either spelling errors if bagging is also set** |
| `path_smooth` | LightGBM | `0` | needs `min_data_in_leaf >= 2` to take effect |
| `scale_pos_weight` / `is_unbalance` | LightGBM | `1.0` / `false` | mutually exclusive |
| `eta` / `learning_rate` | XGBoost | `0.3` | note: 3× LightGBM's default |
| `max_depth` | XGBoost | `6` | set `0` for pure `lossguide` |
| `min_child_weight` | XGBoost | `1` | Hessian sum, ≈100 rows at 1% base rate |
| `gamma` | XGBoost | `0` | alias `min_split_loss` |
| `lambda` / `alpha` | XGBoost | `1` / `0` | aliases `reg_lambda` / `reg_alpha` |
| `max_delta_step` | XGBoost | `0` | docs recommend 1–10 for extreme imbalance |
| `grow_policy` | XGBoost | `depthwise` | `lossguide` ≈ LightGBM leaf-wise |
| `max_leaves` / `max_bin` | XGBoost | `0` / `256` | |
| `tree_method` | XGBoost | `auto` | categorical support needs `hist` or `approx` |
| `enable_categorical` | XGBoost | `False` | requires pandas `category` dtype |
| `TargetEncoder(...)` | sklearn ≥1.3 | `categories='auto', target_type='auto', smooth='auto', cv=5` | `fit_transform` cross-fits; `fit().transform()` **leaks** |

### CatBoost (verified against catboost.ai/docs and `catboost/catboost` master)

| Parameter | Default | Note |
|---|---|---|
| `iterations` | `1000` | aliases `n_estimators`, `num_boost_round`, `num_trees` |
| `learning_rate` | `0.03` | auto-selected **only** if `l2_leaf_reg` / `leaf_estimation_*` are all unset |
| `depth` | `6` | `16` if `grow_policy=Lossguide`; CPU max 16 |
| `l2_leaf_reg` | `3.0` | alias `reg_lambda`; main regularisation knob for SymmetricTree |
| `boosting_type` | **`Plain` on CPU** | `Ordered` only on GPU for ≤50k objects — ordered boosting is *not* the CPU default |
| `bootstrap_type` | **`MVS`** (subsample 0.8) | CPU + non-MultiClass. **Not** `Bayesian` — so `bagging_temperature` is inactive |
| `subsample` | `0.8` under MVS | `0.66` for Poisson/Bernoulli |
| `rsm` | `None` (=1) | alias `colsample_bylevel`. **CPU only** — errors on GPU outside pairwise ranking |
| `one_hot_max_size` | **`2`** | raise to ~10 so all BAF categoricals are one-hot rather than CTR-encoded |
| `min_data_in_leaf` | `1` | alias `min_child_samples`. **Lossguide/Depthwise only — silently ignored on SymmetricTree** |
| `max_ctr_complexity` | `4` | lower to 2 for speed; low-cardinality cats don't need depth-4 combinations |
| `random_strength` | `1` | |
| `border_count` | `254` CPU / `128` GPU | alias `max_bin` |
| `od_type` | `IncToDec` | use `Iter` for conventional early stopping |
| `od_wait` | `20` | far too small for a noisy fraud metric — raise to ~150 |
| `grow_policy` | `SymmetricTree` | non-symmetric forces `boosting_type=Plain` |
| `has_time` | `False` | uses row order for CTRs + tree structure; also forces `permutation_count=1` |
| `langevin` | `False` | CPU only |
| `model_size_reg` | `0.5` | not documented on any docs page; value from `oblivious_tree_options.cpp` |
| `auto_class_weights` | `None` | accepts `None` / `Balanced` / `SqrtBalanced`; mutually exclusive with `class_weights` and `scale_pos_weight` |
| `eval_metric` | — | `"PRAUC:type=Classic"` valid (`type` ∈ Classic/OneVsAll). `AUC` defaults to `type=Ranking` in binary context |
| `cat_features` | `None` | indices or names; columns must be int/str — float or NaN raises |

Two CatBoost claims below full confidence, flagged honestly: (a) `min_data_in_leaf` on `SymmetricTree` is documented as unsupported and no error-raising check was found in the options-validation layer, so it appears to be *silently ignored* — behaviour inferred, corroborated by CatBoost issue #2889; (b) `PRAUC` being eval-only rather than usable as `loss_function` is inferred from its placement in the docs, not explicitly stated. Neither changes the recommendation.

**Sources**
- [LightGBM Parameters (latest)](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [LightGBM `binary_metric.hpp`](https://github.com/microsoft/LightGBM/blob/master/src/metric/binary_metric.hpp)
- [XGBoost Parameters (stable)](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [XGBoost categorical data tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)
- [CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/)
- [sklearn TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
- [Feedzai BAF repository](https://github.com/feedzai/bank-account-fraud) — `notebooks/empirical_results.ipynb`, `notebooks/lightgbm_hyperparameter_space.yaml`
- [Turning the Tables (NeurIPS 2022)](https://papers.nips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html)
