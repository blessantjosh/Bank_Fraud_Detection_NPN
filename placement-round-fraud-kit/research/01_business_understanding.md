# Phase 1 — Business Understanding

Dataset in scope: `data/bank_transactions_data_2.csv` — 2,512 transactions, 495 accounts, 16 raw columns, **no fraud label**. All feature names below are the actual columns in this file, not generic banking-dataset boilerplate.

Columns available: `TransactionID, AccountID, TransactionAmount, TransactionDate, TransactionType (Debit/Credit), Location, DeviceID, IP Address, MerchantID, Channel (ATM/Online/Branch), CustomerAge, CustomerOccupation, TransactionDuration, LoginAttempts, AccountBalance, PreviousTransactionDate`.

One structural caveat that shapes everything below: `PreviousTransactionDate` is **not** a real per-account last-transaction timestamp — the existing v1 pipeline (`LIMITATIONS.md`) found all values cluster within minutes of a single data-export moment (2024-11-04). It is a snapshot artifact, not behavioral history. Any "time since last transaction" or velocity feature must be derived from `TransactionDate` itself (sorted per `AccountID`), not from this column. This report treats that as established, not re-litigated.

---

## 1. Banking Domain Analysis

### What constitutes normal banking behavior

For a given `AccountID`, "normal" is a *stable band* around that account's own history, not a fixed global rule — a $14 debit is normal for one customer and a red flag for another. Normal behavior in this schema looks like:

- Transaction amounts that stay within a few multiples of that account's historical mean/median for its `TransactionType`.
- A consistent `Channel` mix per customer (someone who always uses ATM+Branch suddenly transacting Online is a bigger signal than someone who already mixes channels).
- Recurring `DeviceID` / `Location` — most legitimate customers transact from a small, repeating set of devices and places.
- `LoginAttempts` of 1, occasionally 2 (mistyped PIN/password).
- `TransactionDuration` consistent with the `Channel` (a Branch transaction naturally takes longer than an ATM tap; huge variance within the same channel for the same customer is unusual).
- `AccountBalance` that moves in proportion to `TransactionAmount` in a way consistent with the account's typical cash-flow pattern.

### What constitutes suspicious behavior

| Category | Signal in this schema | Rationale |
|---|---|---|
| **Amount anomalies** | `TransactionAmount` far from the account's own historical mean/median (not the population mean) | Fraud usually deviates from *personal* baseline, not the global average — a $5,000 transaction is unremarkable for a high-balance account and alarming for one that never exceeds $200 |
| **Velocity anomalies** | Many transactions from the same `AccountID` in a short `TransactionDate` window | Card testing, account takeover cash-out, and mule accounts all produce bursts |
| **Time-based anomalies** | Transactions at hours/days atypical for that account; unusual gaps then a burst | Off-hours activity and dormant-then-active patterns are classic ATO signatures |
| **Geographic anomalies** | New or rapidly alternating `Location` for an account, especially combined with tight time windows (implies impossible/implausible travel) | Card cloning and account takeover from a new region |
| **Account/device/identity anomalies** | New `DeviceID` / new `IP Address` combined with high amount or elevated `LoginAttempts`; a `DeviceID`, `IP Address`, or `MerchantID` shared across many distinct `AccountID`s | Device/IP reuse across otherwise-unrelated accounts is a standard mule-network signature; repeated `LoginAttempts` before a successful transaction suggests credential guessing |
| **Behavioral consistency anomalies** | Sudden change in `Channel` preference, `TransactionType` mix, or `CustomerOccupation`-atypical spend pattern | Behavior drift is often the first observable sign of a compromised account, before amounts even look unusual |

### What this dataset structurally cannot see

Being explicit about this now avoids overclaiming later: there is no currency/country-crossing field, no counterparty/beneficiary account field (only a coarse `MerchantID`), and no explicit transaction network/graph. So true cross-account money-laundering *layering* chains (A→B→C→D) are not directly observable — only proxies (shared `DeviceID`/`IP Address`/`MerchantID` across accounts) are available. This is stated up front, not discovered as a surprise in the limitations section later.

---

## 2. Fraud Scenarios and Hypothesis Table

| # | Scenario | Behavioral signature | Observable in this schema? | Primary proxy features |
|---|---|---|---|---|
| 1 | **Account takeover (ATO)** | New device/location + elevated login attempts + atypical amount, often followed by a rapid drain | **Yes — strongest fit** | `DeviceNoveltyFlag`, `LocationNoveltyFlag`, `LoginAttempts`, `Amount_vs_AccountAvg` |
| 2 | **Transaction bursts / card testing** | Many small-to-medium transactions from one account in a very short window, often escalating in size | **Yes** | per-account transaction count in rolling windows (velocity features), `TimeSinceLastTxn` |
| 3 | **Mule accounts** | An account that receives/moves funds and shares infrastructure (device, IP, merchant) with other accounts rather than having organic independent behavior | **Partially** — shared `DeviceID`/`IP Address`/`MerchantID` across accounts is visible; true fund-flow-in/fund-flow-out chaining is not (no counterparty ledger) | `DeviceTxnCount`, `IPTxnCount`, `MerchantTxnCount` (cross-account reuse) |
| 4 | **Unusual spending behavior / compromised card** | A single account's spend profile shifts abruptly — different `TransactionType` mix, different `Channel`, different amount tier, sustained (not just a one-off outlier) | **Yes** | rolling mean/std deviation features, `Amount_vs_AccountAvg`, spending-variability features |
| 5 | **Synthetic identities** | An account with a thin, internally-inconsistent, or newly-fabricated behavioral history (e.g., `CustomerAge`/`CustomerOccupation` combinations that behave atypically for that demographic, very short history before high-value activity) | **Weakly** — no account-opening-date or KYC-verification field exists, so this can only be approximated via occupation/age-vs-behavior mismatches, not directly detected | `CustomerOccupation` one-hot vs. behavior, `CustomerAge` |
| 6 | **Money laundering (layering)** | Structured/threshold-avoiding amounts, rapid pass-through across multiple accounts | **No direct signal** — would require a counterparty/beneficiary account field or an explicit transaction graph, neither of which exists here | none directly; only weak proxies via shared `MerchantID` |

**How to read this table going forward:** scenarios 1, 2, and 4 are where this dataset has real signal and should drive feature engineering priority. Scenarios 3 and 5 get partial, proxy-only coverage — reported as such, not inflated. Scenario 6 is explicitly out of scope for this dataset and should not be claimed as "detected" anywhere downstream, even if an anomaly model happens to flag a transaction that a human labels "looks like laundering" — that would be a coincidental narrative, not a validated capability.

---

*Next: Phase 2 (Data Understanding) and Phase 3 (Data Quality Assessment) — in progress against the actual CSV, see `02_data_understanding.md` / `03_data_quality.md`.*
