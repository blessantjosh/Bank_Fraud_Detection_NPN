# Phase 5 (v2) — Feature Engineering (Teammate Feature Set)

Business context (fraud scenarios, hypothesis table, what this schema structurally cannot see) is carried over unchanged from `research/01_business_understanding.md` — the raw data (`data/bank_transactions_data_2.csv`) has not changed, only the feature-engineering approach applied to it. This report does not re-litigate Phase 1.

**Canonical input for this pipeline, per client direction:** `artifacts_research/features_teammate_merged.csv` — 2,512 rows × 28 columns: 10 ID/display columns (`TransactionID, AccountID, TransactionDate, TransactionType, Location, DeviceID, IP Address, MerchantID, Channel, CustomerOccupation`, reattached by verified exact positional alignment, per the task brief — not re-derived here) + 18 engineered feature columns. This file — not `artifacts_research/features_v2.csv` — is the authoritative feature matrix for everything from this point forward in `research_v2/`. The in-house 46-feature pipeline (`research/`, `src_research/`, `artifacts_research/features_v2.csv`) is left untouched as historical reference and is referenced here only for contrast.

---

## 1. Feature Dictionary — the Teammate's 18 Features

| Feature | Formula / meaning | Business rationale | Note |
|---|---|---|---|
| `TransactionAmount` | `StandardScaler(TransactionAmount)` | Transaction size — the single most basic amount signal | z-score, not dollars; see Section 2 |
| `CustomerAge` | `StandardScaler(CustomerAge)` | Demographic context; extreme ages combined with atypical behavior are a weak synthetic-identity proxy (Phase 1, scenario 5) | z-score |
| `TransactionDuration` | `StandardScaler(TransactionDuration)` | How long the transaction took — unusually short/long durations for a channel can indicate automation (bot/script) or friction (failed attempts before success) | z-score |
| `LoginAttempts` | `StandardScaler(LoginAttempts)` | Credential-guessing / account-takeover proxy (Phase 1, scenario 1) | z-score; near-discrete in raw units (mostly 1, rarely up to 7 per Phase 2/4 of the in-house pipeline) — scaling does not change that discreteness |
| `AccountBalance` | `StandardScaler(AccountBalance)` | Account size context — the same $500 transaction means different things at different balance levels | z-score |
| `account_frequency` | Global count of transactions for this `AccountID` across the whole dataset, standardized | Customer activity-level proxy | **Not leakage-safe for real-time scoring**: a global (whole-dataset) count includes each account's own future transactions, not just prior ones — fine for offline research/anomaly-scoring on a static dataset, but would need to become a prior-only running count (like the in-house `CustomerTxnCountSoFar`) before any live deployment |
| `device_frequency` | Global count of transactions sharing this `DeviceID`, standardized | Shared-device / mule-account proxy (Phase 1, scenario 3) | Same global-count caveat as above |
| `ip_frequency` | Global count of transactions sharing this `IP Address`, standardized | Shared-IP / mule-account proxy | Same caveat |
| `merchant_frequency` | Global count of transactions sharing this `MerchantID`, standardized | Merchant popularity — legitimately shared by many customers, weaker fraud signal by nature | Same caveat |
| `amount_to_balance_ratio` | Standardized transform of (a quantity strongly correlated with, but not exactly recoverable as) `TransactionAmount / AccountBalance` — see Section 2.3 for the verification | Scale-relative signal: a transaction's size relative to what the account actually holds | Same concept as the in-house `Amount_to_Balance_Ratio`, independently arrived at — good convergent validation (see `research/04_feature_engineering.md` Section 3) |
| `high_amount_transaction` | Binary flag, `1` if raw `TransactionAmount` is (in effect) above the dataset's ~95th percentile (~$878), else `0` — recovered empirically in Section 2.4, not stated in the source | Global outlier-amount flag — catches unusually large transactions independent of account context | A *global*, not personalized, threshold: a $900 transaction from a $50,000-balance account is flagged the same as a $900 transaction from a $200-balance account |
| `TransactionType_Debit` | One-hot dummy, `Credit` dropped as baseline | Debit is the majority class (1,944/2,512, 77.4%) — Credit transactions are the minority pattern | Single dummy since `TransactionType` is binary |
| `Channel_Branch`, `Channel_Online` | One-hot dummies, `ATM` dropped as baseline | Channel mix is a behavioral-consistency signal (Phase 1) | 2 dummies for 3 categories |
| `CustomerOccupation_Engineer`, `CustomerOccupation_Retired`, `CustomerOccupation_Student` | One-hot dummies, `Doctor` dropped as baseline | Occupation-vs-behavior mismatch is a weak synthetic-identity proxy (Phase 1, scenario 5) | 3 dummies for 4 categories |
| `Location_FE` | Frequency encoding of `Location` (proportion of rows at that location), standardized | Cheap single-number representation of a 43-category field; rare locations get a low value | Conceptually identical to the in-house `Location_Freq` — independently arrived at (see `research/04_feature_engineering.md` Section 3) |

---

## 2. Verification, With Real Numbers (`src_research_v2/04_feature_verification.py`, `artifacts_research_v2/phase5_6_feature_verification.json`)

### 2.1 Missing values and duplicates — re-checked, not assumed

0 missing cells (0/70,336), 0 duplicate full rows, 0 duplicate `TransactionID`s. Row order and `TransactionID` values are confirmed to align exactly with `data/bank_transactions_data_2.csv` (`(raw["TransactionID"].values == df["TransactionID"].values).all()` — **True**).

### 2.2 Scaling verification — genuinely all 18 columns, not just the 5 called out in the brief

The task brief flags 5 columns (`TransactionAmount, CustomerAge, TransactionDuration, LoginAttempts, AccountBalance`) as StandardScaler-scaled. Checked directly, **all 18 columns are mean-0/std-1 scaled**, including the frequency counts and the ratio feature — not just the 5:

| Feature | Mean | Std |
|---|---:|---:|
| TransactionAmount | 0.0000 | 1.0002 |
| CustomerAge | 0.0000 | 1.0002 |
| TransactionDuration | 0.0000 | 1.0002 |
| LoginAttempts | 0.0000 | 1.0002 |
| AccountBalance | 0.0000 | 1.0002 |
| account_frequency | 0.0000 | 1.0002 |
| device_frequency | 0.0000 | 1.0002 |
| ip_frequency | 0.0000 | 1.0002 |
| merchant_frequency | -0.0000 | 1.0002 |
| amount_to_balance_ratio | -0.0000 | 1.0002 |
| Location_FE | -0.0000 | 1.0002 |

(std is 1.0002, not exactly 1.0000, because `pandas.std()` uses the `ddof=1` sample-std convention while `StandardScaler` divides by the `ddof=0` population std — a real, expected, and inconsequential 1/n vs. 1/(n-1) artifact, not evidence the scaling is off.) The three binary/dummy families (`high_amount_transaction`, `TransactionType_Debit`, `Channel_*`, `CustomerOccupation_*`) are, correctly, **not** standardized — they remain 0/1 flags, as a one-hot/binary feature should.

### 2.3 `amount_to_balance_ratio` spot check

Correlation between the raw `TransactionAmount / AccountBalance` ratio (computed directly from `data/bank_transactions_data_2.csv`) and the scaled `amount_to_balance_ratio` column: **r = 0.9467**. Strong, but not a perfect 1.0 — confirms this feature is measuring materially the same underlying quantity as the in-house `Amount_to_Balance_Ratio` (independent convergence, a good cross-check) without claiming an exact, recoverable formula (the teammate's pipeline is not available to inspect directly; some difference is plausibly explained by a small amount of additional handling — e.g. capping or a small denominator floor for low-balance accounts — that isn't recoverable post-scaling). Raw ratio distribution: mean 0.200, median 0.052, max 7.896 — identical to the in-house pipeline's finding for the same raw quantity (`research/04_feature_engineering.md`, Section 2.2), as expected since both are derived from the same raw columns.

### 2.4 `high_amount_transaction` spot check

126/2,512 rows flagged (5.02%). The boundary is razor-sharp against the raw dollar amount: the minimum raw `TransactionAmount` among flagged rows is **$878.63**, the maximum among unflagged rows is **$877.81** — a gap of 82 cents. This sits almost exactly on the raw dataset's 95th percentile (**$878.18**, computed directly). **Conclusion: `high_amount_transaction` is a global top-5%-by-raw-amount threshold flag**, not a personalized one — confirmed empirically since the scaled file does not expose the literal threshold.

### 2.5 Dummy/one-hot baseline check

Recovered which category each one-hot family silently drops as its reference level, by cross-referencing the raw category counts: `TransactionType_Debit` implies **Credit** is the dropped baseline (raw counts: Debit 1,944 / Credit 568); `Channel_Branch`/`Channel_Online` implies **ATM** is dropped (Branch 868 / ATM 833 / Online 811); `CustomerOccupation_Engineer/Retired/Student` implies **Doctor** is dropped (Student 657 / Doctor 631 / Engineer 625 / Retired 599).

### 2.6 Per-account sequence length — re-verified on this file, not assumed

495 accounts, mean 5.075 transactions/account (min 1, max 12, median 5) — **identical** to the in-house pipeline's Phase 8 finding, confirmed by direct recomputation on `features_teammate_merged.csv` rather than assumed carried over (expected, since both feature sets derive from the same 2,512-row raw CSV with verified-aligned rows). 428/495 accounts (86.5%) have ≥3 transactions, covering 2,402/2,512 rows (95.6%) — this exact finding is reused for the Phase 8 (v2) LSTM-AE feasibility scoping in `06_model_development.md`.

---

## 3. Honest Comparison: 18-Feature Teammate Set vs. 46-Feature In-House Set

| Dimension | In-house (46 features, `research/04_feature_engineering.md`) | Teammate (18 features, this document) | What it means for detection |
|---|---|---|---|
| **Personal-baseline / velocity features** | `Expanding_Mean/Median/Std/Max/MinAmount`, `Rolling3_Mean/StdAmount`, `Amount_ZScore_Account`, `Amount_vs_AccountAvg`, `Velocity_1D/7D_Count`, `TimeSinceLastTxn`, `CustomerTxnCountSoFar`, `SpendCV_Account` | **None** — no per-account running statistics, no rolling windows, no time-since-last-transaction | The teammate set has no way to ask "is this large *for this specific customer*", only "is this large globally" (`high_amount_transaction`) or "how does this account's overall balance/amount z-score compare to the population" (`amount_to_balance_ratio`, both population-level). This is the single biggest capability gap: Phase 1's strongest-fit scenario, account takeover via a personally-atypical amount, is materially weaker to detect here |
| **Account-takeover / novelty proxies** | `DeviceNoveltyFlag`, `LocationNoveltyFlag` (first-time device/location for this account) | **None** — `device_frequency`/`ip_frequency` are *global* popularity counts, not *per-account* novelty flags | This is the specific gap called out explicitly per the task brief: the in-house set can directly ask "has this account ever used this device/location before?"; the teammate set can only ask "how common is this device/location across the whole dataset?" — a global-popularity proxy is a materially weaker, indirect substitute for a personal-novelty signal. A device used by exactly one account for its first-ever transaction looks identical (low `device_frequency`) whether that's a brand-new customer's only device or a stolen card's fraud-only device — the in-house `DeviceNoveltyFlag` at least distinguishes "new to this account" from "rare overall" |
| **Cyclical time encoding** | `Hour_sin/cos`, `DOW_sin/cos` | **None** | Already documented in-house as near-zero variance on this specific dataset (all transactions fall in a narrow Mon–Fri, 16:00–18:21 window) — the absence here is a real gap in principle but not a practical loss on *this* dataset |
| **Cross-account sharing (mule proxy)** | `DeviceSharedAccounts_Prior`, `IPSharedAccounts_Prior`, `MerchantSharedAccounts_Prior` (point-in-time, prior-only) | `device_frequency`, `ip_frequency`, `merchant_frequency` (global, not point-in-time) | Conceptually the same idea (shared infrastructure across accounts), computed less rigorously (global count includes future transactions, not leakage-safe for live scoring) — usable for offline research, not directly production-safe without rebuilding as a prior-only count |
| **Amount-to-balance ratio** | `Amount_to_Balance_Ratio` | `amount_to_balance_ratio` | **Same idea, independently arrived at** — good convergent validation, no capability gap |
| **Global high-amount flag** | Not built directly (uses graded `Amount_ZScore_Account`/`Amount_to_RollingMean_Ratio` instead) | `high_amount_transaction` | Teammate set adds a cheap global-threshold flag the in-house set intentionally didn't build standalone (in-house Phase 5, Section 3, flagged this as "worth adopting", not yet done) |
| **Location encoding** | `Location_enc` (label) + `Location_Freq` (frequency) | `Location_FE` (frequency) | Same idea, independently arrived at |
| **Behavioral flags** | `ElevatedLoginFlag`, `ATM_Credit_InteractionFlag` | None (raw `LoginAttempts` only, scaled) | Minor gap — a discretized login-attempts flag is easy to reconstruct from `LoginAttempts` if needed downstream |
| **Total feature count** | 46 (2 ID cols reused separately) | 18 | Simpler, faster to train/score, easier to reason about — a real, deliberate trade against detection granularity |

**Net verdict:** this is a real, honest trade-off, not a strictly worse or strictly better set. The teammate's 18 features are simpler, faster, and easier to maintain/reason about, and they independently converge on the two most obviously-useful ratio/encoding ideas (`amount_to_balance_ratio`, `Location_FE`) that the in-house pipeline also built. But the teammate set has **no personal-baseline (expanding/rolling) features and no per-account novelty flags** — the two feature families the in-house Phase 5 report identified as the primary signal for the account-takeover scenario (Phase 1's strongest-fit fraud scenario) and for personalized "is this unusual *for this customer*" detection generally. Everything this pipeline detects going forward is, structurally, closer to "is this transaction unusual in the population" than "is this transaction unusual for this specific account" — a materially different and narrower detection capability that should be stated plainly to the bank, not glossed over.

---

## 4. Handoff to Phase 6/7

- `artifacts_research/features_teammate_merged.csv` (28 columns: 10 ID/display + 18 features) is the input to Phase 6 (preprocessing/dimensionality-reduction verification) and Phase 7 (PCA/UMAP/t-SNE/autoencoder), both in `research_v2/05_feature_selection_and_preprocessing.md`.
- The 10 ID/display columns must be excluded from any model input — `src_research_v2/config_research_v2.py::FEATURE_COLS_V2` is the single source of truth for the 18-column model schema, referenced by every downstream v2 script.
- The `account_frequency`/`device_frequency`/`ip_frequency`/`merchant_frequency` leakage caveat (Section 1) is noted here for completeness but not re-derived as a prior-only feature in this phase — out of scope per the task brief (the teammate's 18 columns are treated as final).
