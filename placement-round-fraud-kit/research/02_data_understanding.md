# Phase 2 — Data Understanding

Source: `data/bank_transactions_data_2.csv` — 2,512 rows, 16 raw columns, 495 unique `AccountID`s (average 5.08 transactions/account), 2,512 unique `TransactionID`s (one row = one transaction, no duplicate keys). No fraud label exists anywhere in the file. All numbers below come from `src_research/01_data_understanding.py` run against the raw CSV; outputs are persisted in `artifacts_research/numeric_summary.csv`, `artifacts_research/categorical_summary.json`, `artifacts_research/datetime_summary.json`, `artifacts_research/dataset_facts.json`.

---

## 1. Feature Inventory

| # | Feature | Type | Description | Business meaning | Expected behavior |
|---|---|---|---|---|---|
| 1 | `TransactionID` | string | Unique transaction key | Primary key, no analytical value beyond joins | 2,512/2,512 unique — confirmed |
| 2 | `AccountID` | string (categorical) | Customer account key | Groups transactions belonging to the same customer | 495 unique values, repeated an average of 5.08 times each |
| 3 | `TransactionAmount` | float | Monetary value of the transaction | Core signal — fraud is disproportionately amount-driven (unusually large, or structured just-under-threshold amounts) | Right-skewed; most transactions small, a long tail of large ones |
| 4 | `TransactionDate` | string, `DD-MM-YYYY HH:MM` | Timestamp of the transaction | Enables recency, velocity, and time-of-day/day-of-week features | Should span a realistic operating period at varied times |
| 5 | `TransactionType` | categorical (`Debit`/`Credit`) | Direction of funds movement | Debit-heavy books are typical of a retail transaction ledger | 2 categories, imbalanced toward Debit |
| 6 | `Location` | categorical (city name) | Geographic origin of the transaction | New/rapidly-changing location for an account is a classic ATO/card-cloning signal | Moderate cardinality, uneven frequency across cities |
| 7 | `DeviceID` | categorical | Device identifier used for the transaction | A new or rarely-seen device on an established account is a fraud signal; a device shared across many unrelated accounts suggests a mule ring | High cardinality, long tail of rare/singleton devices |
| 8 | `IP Address` | categorical | Network origin (IPv4) | Same logic as `DeviceID` — novelty and cross-account reuse both matter | High cardinality, long tail of rare/singleton IPs |
| 9 | `MerchantID` | categorical | Merchant/counterparty code | Merchant concentration risk, merchant-specific fraud patterns | Moderate, bounded cardinality (closed set of merchants) |
| 10 | `Channel` | categorical (`ATM`/`Online`/`Branch`) | Channel through which the transaction was made | Channel-specific baselines differ (Online fraud patterns differ from ATM/Branch) | 3 categories, roughly comparable volumes |
| 11 | `CustomerAge` | int | Customer's age in years | Demographic segmentation; extreme values (very young/old with high-value or high-velocity activity) can be a synthetic-identity flag | Bounded, adult population range |
| 12 | `CustomerOccupation` | categorical | Customer's stated occupation | Demographic segmentation; occupation-atypical spend can be a weak synthetic-identity/ATO proxy | Small, fixed set of categories |
| 13 | `TransactionDuration` | int (seconds) | How long the transaction took to complete | Should correlate with `Channel` (Branch > Online > ATM, typically); anomalously short/long durations for a channel are suspicious | Right-skewed, channel-dependent |
| 14 | `LoginAttempts` | int | Number of login attempts before this transaction | Security signal — elevated attempts suggest credential guessing / account takeover | Heavily concentrated at 1, rare higher values |
| 15 | `AccountBalance` | float | Account balance associated with the transaction | Provides scale context for `TransactionAmount` (is this large relative to what this account normally holds?) | Wide range, right-skewed |
| 16 | `PreviousTransactionDate` | string, `DD-MM-YYYY HH:MM` | Nominally "this account's previous transaction timestamp" | **Not usable as designed** — see Section 4 | Only 7 distinct values across all 2,512 rows, all within 6 minutes of each other; this is a single bulk-export timestamp, not per-account history. Recency/velocity features must be derived from `TransactionDate` sorted per `AccountID` instead. |

---

## 2. Numerical Features

Computed directly (`scipy.stats.skew`, `scipy.stats.kurtosis`, Fisher convention — 0 = normal):

| Feature | Min | Max | Range | Mean | Median | Variance | Std | Skewness | Excess Kurtosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransactionAmount | 0.26 | 1,919.11 | 1,918.85 | 297.59 | 211.14 | 85,232.61 | 291.95 | **1.74** | **3.63** |
| CustomerAge | 18 | 80 | 62 | 44.67 | 45.00 | 316.56 | 17.79 | 0.15 | -1.22 |
| TransactionDuration | 10 | 300 | 290 | 119.64 | 112.50 | 4,894.93 | 69.96 | 0.60 | -0.26 |
| LoginAttempts | 1 | 5 | 4 | 1.12 | 1.00 | 0.36 | 0.60 | **5.17** | **26.61** |
| AccountBalance | 101.25 | 14,977.99 | 14,876.74 | 5,114.30 | 4,735.51 | 15,217,350 | 3,900.94 | 0.60 | -0.57 |

**Interpretation:**
- **TransactionAmount** is strongly right-skewed (skew 1.74, excess kurtosis 3.63 — fatter tail and sharper peak than normal). Median ($211.14) sits well below the mean ($297.59), confirming a small number of large transactions pull the average up. This is exactly the shape expected if fraud (or simply high-value legitimate activity) lives in the tail.
- **CustomerAge** is close to symmetric (skew 0.15) and slightly platykurtic (excess kurtosis -1.22, i.e. flatter than normal / closer to uniform across 18–80) — no demographic red flags from shape alone.
- **TransactionDuration** is moderately right-skewed (0.60) with a flatter-than-normal peak (-0.26); most transactions complete quickly but a real subset run long, plausibly Branch-channel transactions (confirmed in Phase 4 bivariate analysis).
- **LoginAttempts** is the standout: skew 5.17 and excess kurtosis 26.61 indicate an extreme, sharply-peaked distribution with a long thin tail. This is confirmed by the raw counts: 2,390/2,512 rows (95.14%) have exactly 1 login attempt; only 122 rows (4.86%) have 2–5 attempts. This column behaves more like a rare-event flag than a continuous variable — elevated `LoginAttempts` (2+) is itself a small, well-separated population worth treating as a near-binary security signal rather than a smooth numeric feature.
- **AccountBalance** is right-skewed (0.60) with a flatter-than-normal center (-0.57); a wide balance range ($101 – $14,978) means `TransactionAmount` must be judged relative to `AccountBalance`, not on an absolute scale, to be meaningful for fraud scoring.

---

## 3. Categorical Features

| Feature | Cardinality | Top category (share) | Singleton categories (count = 1) | Categories with count = 2 |
|---|---:|---|---:|---:|
| TransactionType | 2 | Debit — 1,944 rows (77.39%) | 0 | 0 |
| Channel | 3 | Branch — 868 rows (34.55%) | 0 | 0 |
| CustomerOccupation | 4 | Student — 657 rows (26.15%) | 0 | 0 |
| MerchantID | 100 | M026 — 45 rows (1.79%) | 0 | 0 |
| Location | 43 | Fort Worth — 70 rows (2.79%) | 0 | 0 |
| AccountID | 495 | AC00202 — 12 rows (0.48%) | 24 | 43 |
| IP Address | 592 | 200.136.146.93 — 13 rows (0.52%) | 40 | 75 |
| DeviceID | 681 | D000142 — 9 rows (0.36%) | 72 | 119 |

**Interpretation:**
- `TransactionType`, `Channel`, `CustomerOccupation`, and `MerchantID` are low-to-moderate cardinality with no singleton categories — safe to one-hot or target-encode without a rare-category collapsing step.
- `Location` (43 cities) is evenly enough distributed (top city only 2.79% of volume) that it is a reasonable candidate for frequency- or target-encoding.
- `DeviceID`, `IP Address`, and `AccountID` are the high-cardinality identifiers with genuinely long tails: 72 devices (10.6% of the 681 distinct devices) and 40 IPs (6.8% of 592) appear only once in the whole dataset. These singleton/rare identifiers are exactly the raw material for "novel device," "novel IP," and "shared-infrastructure-across-accounts" features flagged in Phase 1 — they should not be one-hot encoded directly (cardinality too high) but consumed via count/frequency features instead.
- 24 of 495 accounts (4.85%) appear in the data exactly once. A single-transaction account has no internal history to build a personal baseline against — this is a structural limitation for any account-relative feature ("deviation from this account's own mean") and should be flagged, not silently imputed with a population average that would mask the case.

---

## 4. Datetime Features

### `TransactionDate`
- Range: **2023-01-02 16:00** to **2024-01-01 18:21** — a 364-day span.
- Day-of-week distribution (all 2,512 rows fall Monday–Friday; **zero** weekend transactions):

  | Day | Count | Share |
  |---|---:|---:|
  | Monday | 1,070 | 42.60% |
  | Friday | 373 | 14.85% |
  | Thursday | 368 | 14.65% |
  | Tuesday | 360 | 14.33% |
  | Wednesday | 341 | 13.57% |
  | Saturday | 0 | 0.00% |
  | Sunday | 0 | 0.00% |

- Hour-of-day distribution — **all transactions fall in a 3-hour window**:

  | Hour | Count | Share |
  |---|---:|---:|
  | 16:00 | 1,316 | 52.39% |
  | 17:00 | 819 | 32.60% |
  | 18:00 | 377 | 15.01% |

- Monthly volume is otherwise fairly even (roughly 160–226 transactions/month across Jan–Dec 2023, plus a partial 13-row tail in Jan 2024) — no strong month-over-month trend or seasonality beyond ordinary variation.

**Finding, stated honestly:** this is not realistic 24/7 transaction timestamp data. Every transaction in the dataset was recorded on a weekday, and 100% of transactions cluster in the 16:00–18:21 window (a ~2.3 hour band), with Monday alone accounting for 42.6% of all volume. This is very likely an artifact of how the dataset was generated/exported rather than genuine customer behavior, and it caps what "time-of-day" or "day-of-week" features can contribute: there is no real overnight/weekend baseline to compare off-hours activity against, so classic "3am transaction" or "weekend anomaly" fraud signals are structurally invisible in this data. Time-based feature engineering here should focus on **inter-transaction gaps and velocity per account** (which still varies meaningfully), not absolute hour/day-of-week, since the latter carries almost no discriminative information (91% of the variance is explained by which weekday bucket a row's export batch landed in, not by customer behavior).

### `PreviousTransactionDate`
Confirms the pre-existing finding with hard numbers: only **7 unique timestamps** across all 2,512 rows, spanning **6.0 minutes** (2024-11-04 08:06:00 to 2024-11-04 08:12:00), centered at 2024-11-04 08:09:00 — roughly ten months *after* the latest `TransactionDate` (2024-01-01). The per-timestamp row counts (240, 435, 431, 430, 391, 423, 162) show no relationship to `AccountID` or transaction history — this is unambiguously a single bulk data-export moment stamped onto every row, not a real "last transaction" field. It is treated as a near-constant, low-information column for all downstream analysis; any recency/velocity feature is built from `TransactionDate` sorted per `AccountID` instead, per the standing project decision.

---

## 5. Dataset Facts Summary

| Fact | Value |
|---|---:|
| Rows | 2,512 |
| Columns | 16 |
| Unique accounts | 495 |
| Unique transaction IDs | 2,512 (100% unique) |
| Total missing cells | 0 |

*Next: Phase 3 (Data Quality Assessment) and Phase 4 (EDA) — see `03_data_quality_and_eda.md`.*
