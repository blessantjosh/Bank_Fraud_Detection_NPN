# Fraud Detection Pipeline — Technical Documentation

Dataset: `bank_transactions_data_2.csv` — 2,512 transactions, 495 accounts, 16 raw columns, no fraud label.
All numbers below are from an actual run of the pipeline in `src/` against this exact file, not illustrative estimates.

---

## Stage 1 — Feature Engineering

**Dropped:** `PreviousTransactionDate` (7 unique values, all clustered within minutes of one export timestamp on 2024-11-04 — a data-export artifact, not a real per-account "previous transaction" time). `TransactionID` (pure identifier). `AccountID` (used only as a grouping key during engineering, excluded from the model matrix — including it directly would let the model memorize individual accounts instead of learning behavior). `DeviceID`, `IP Address`, `MerchantID` (high-cardinality raw identifiers, converted into count features instead of encoded directly).

**20 engineered features** (after encoding):

| Feature | Definition |
|---|---|
| `TransactionAmount`, `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` | raw numeric columns |
| `Amount_vs_AccountAvg` | `(amount - prior_account_mean) / prior_account_mean`, where `prior_account_mean` is that account's **expanding mean of all prior transactions only** (computed via `groupby().shift().expanding()`, so no future/current-row leakage). First-ever transaction for an account falls back to the type average. |
| `Amount_vs_TypeAvg` | deviation from the global mean amount for that `TransactionType` (Debit/Credit) |
| `DeviceNoveltyFlag` / `LocationNoveltyFlag` | 1 if this Device/Location has never appeared before in this account's history (first-ever transaction counts as novel = 1) |
| `TimeSinceLastTxn` | hours since this account's previous transaction, from `TransactionDate` itself; first transaction filled with the dataset median gap |
| `DeviceTxnCount`, `IPTxnCount`, `MerchantTxnCount` | how many total transactions that Device/IP/Merchant has across the whole dataset |
| `Location_enc` | label-encoded (cardinality > 10) |
| `TransactionType_Debit`, `Channel_Branch`, `Channel_Online`, `CustomerOccupation_Engineer/Retired/Student` | one-hot, first category per group dropped as reference level (avoids the perfect multicollinearity that made the MCD detector's covariance matrix singular in Stage 2) |

Two matrices are saved: unscaled (`features.csv`, used by the tree-based models, which don't need scaling) and `StandardScaler`-scaled (`features_scaled.csv`, used by the distance-based detectors in Stage 2).

---

## Stage 2 — Unsupervised Anomaly Ensemble

No label exists, so four independent detectors vote on which transactions look statistically unusual. Contamination assumption: **5%** (documented, unverified — applied identically to all four so votes are comparable).

| Detector | Principle | Flagged |
|---|---|---|
| Isolation Forest | isolates points via random tree partitioning; anomalies need fewer splits | 126 (5.02%) |
| Local Outlier Factor | local density much lower than neighbors | 126 (5.02%) |
| One-Class SVM | outside the learned soft boundary around normal data | 120 (4.78%) |
| MCD (EllipticEnvelope) | robust Mahalanobis distance from a covariance estimate that isn't itself dragged off by outliers | 126 (5.02%) |

**Vote-count distribution** (of 2,512 transactions):

| Votes | Count |
|---|---|
| 0 | 2,230 |
| 1 | 146 |
| 2 | 79 |
| 3 | 34 |
| 4 | 23 |

Note: `DeviceNoveltyFlag` is ~99.5% constant (almost every transaction is a "new" device for its account), which makes MCD's robust covariance estimate ill-conditioned on some resamples — this produces benign convergence warnings (suppressed in the script), not invalid output.

---

## Stage 3 — Confidence-Tiered Labeling

| Tier | Rule | Count |
|---|---|---|
| **High confidence fraud** | 3–4 of 4 detectors agree | 57 |
| **Medium confidence / needs review** | exactly 2 agree | 79 |
| **Normal** | 0–1 agree | 2,376 |

Binary label for the supervised model: High + Medium → `is_fraud=1` (**136 transactions, 5.41% prevalence**); Normal → `is_fraud=0`. The 3-tier column is kept in `labeled.csv` for the demo, since "needs review" is a real operational category a binary label would discard.

---

## Stage 4 — Balancing

Stratified 80/20 split **before** any resampling (resampling before splitting would leak synthetic points derived from test-fold minority rows into training):

- Train: 2,009 rows (109 fraud, 5.43%)
- Test: 503 rows (27 fraud, 5.37%)

SMOTE fit on the training fold only, `k_neighbors=5` (109 real minority rows is enough for 5-NN interpolation to stay meaningful) → **3,800 rows, 1,900/1,900 (50/50)**.

Alternative: class-weighting via `scale_pos_weight = 17.43` (ratio of majority to minority in the training fold), trained directly on the original 2,009-row imbalanced fold with no synthetic data.

---

## Stage 5 — Model + Explainability

**Primary model:** XGBoost (`n_estimators=200, max_depth=4, learning_rate=0.05`) trained on the SMOTE-balanced training set.
**Comparison model:** identical XGBoost config, trained on the original imbalanced fold with `scale_pos_weight`.

### SMOTE vs. class-weighting — measured, not assumed

| | ROC-AUC | PR-AUC |
|---|---|---|
| SMOTE | 0.9428 | 0.5934 |
| Class-weighted | **0.9532** | **0.7398** |

**Class-weighting measures better here**, especially on PR-AUC — the metric that matters most under 5% prevalence. With only ~109 real minority rows, SMOTE's k=5 interpolation likely blurs the sharp 0/1 novelty-flag boundaries the tree would otherwise split on cleanly. The SMOTE model still ships as the primary model (the brief explicitly requires demonstrating SMOTE), but this comparison is reported honestly rather than assumed — **for a real deployment at this dataset size, class-weighting would be the better default.**

### Global SHAP importance (mean |SHAP value| on the test set)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `Amount_vs_AccountAvg` | 0.944 |
| 2 | `Channel_Branch` | 0.696 |
| 3 | `LoginAttempts` | 0.679 |
| 4 | `LocationNoveltyFlag` | 0.626 |
| 5 | `Channel_Online` | 0.603 |
| 6 | `CustomerOccupation_Retired` | 0.408 |
| 7 | `DeviceNoveltyFlag` | 0.354 |
| 8 | `TransactionAmount` | 0.322 |
| 9 | `TimeSinceLastTxn` | 0.260 |
| 10 | `MerchantTxnCount` | 0.253 |

### Individual explanation

Highest-risk transaction in the test set (row 370, true tier = fraud, predicted P(fraud) = **0.981**):

| Feature | Value | SHAP contribution |
|---|---|---|
| `DeviceNoveltyFlag` | 0 (known device) | **+5.87** |
| `Channel_Branch` | False | −0.70 |
| `Channel_Online` | True | +0.69 |
| `Amount_vs_AccountAvg` | −0.99 (well below its own average) | −0.42 |
| `LoginAttempts` | 1 | −0.39 |

Worth flagging honestly: a *known* device (novelty flag = 0) is the single largest driver of predicted risk for this transaction — the opposite of the intuitive "new device = risky" story. This reflects a real pattern the anomaly ensemble picked up in this specific dataset (it isn't wrong or a bug), but it's a reminder that SHAP explains what the model learned from the generated labels, not a verified causal fraud mechanism.

### Decision tree rules (depth=3, for the slide)

Trained with `class_weight="balanced"` — meaning the printed predicted class weighs each fraud row ~17x more heavily than a normal row, so a leaf can predict "Fraud" even when raw normal rows outnumber raw fraud rows in it. Both counts are shown below for transparency:

```
IF Amount_vs_AccountAvg <= 1.57 AND LoginAttempts <= 2.50 AND DeviceNoveltyFlag <= 0.50
   -> predict FRAUD   (n=7: 0 normal / 7 fraud — 100% fraud rate)

IF Amount_vs_AccountAvg <= 1.57 AND LoginAttempts <= 2.50 AND DeviceNoveltyFlag > 0.50
   -> predict NORMAL  (n=1600: 1571 normal / 29 fraud — 1.8% fraud rate)

IF Amount_vs_AccountAvg <= 1.57 AND LoginAttempts > 2.50 AND AccountBalance <= 7793.22
   -> predict FRAUD   (n=49: 31 normal / 18 fraud — 36.7% fraud rate, raw-minority but
                        weighted-majority due to class_weight="balanced")

IF Amount_vs_AccountAvg <= 1.57 AND LoginAttempts > 2.50 AND AccountBalance > 7793.22
   -> predict NORMAL  (n=20: 19 normal / 1 fraud — 5.0% fraud rate)

IF Amount_vs_AccountAvg > 1.57 AND Amount_vs_AccountAvg <= 8.06 AND TransactionAmount <= 823.88
   -> predict NORMAL  (n=184: 178 normal / 6 fraud — 3.3% fraud rate)

IF Amount_vs_AccountAvg > 1.57 AND Amount_vs_AccountAvg <= 8.06 AND TransactionAmount > 823.88
   -> predict FRAUD   (n=95: 74 normal / 21 fraud — 22.1% fraud rate, raw-minority but
                        weighted-majority due to class_weight="balanced")

IF Amount_vs_AccountAvg > 8.06 AND CustomerAge <= 22.50
   -> predict NORMAL  (n=5: 5 normal / 0 fraud — 0.0% fraud rate)

IF Amount_vs_AccountAvg > 8.06 AND CustomerAge > 22.50
   -> predict FRAUD   (n=49: 22 normal / 27 fraud — 55.1% fraud rate)
```

Plain-English summary for the slide: **large deviation from an account's own historical average amount is the dominant fraud signal**, with login attempts, device novelty, and customer age refining the call at the margins.

---

## Stage 6 — Robust Evaluation

**Why accuracy is misleading:** a model that predicts "Normal" for every single transaction scores **94.63% accuracy** while catching zero fraud. The actual trained model scores 93.84% accuracy — *lower* than the naive baseline — because it's spending some of its correctness budget on catching real fraud cases the naive model ignores entirely. Accuracy alone would make the naive model look better; it isn't.

| Metric | Value |
|---|---|
| Precision | 0.441 |
| Recall | 0.556 |
| F1-score | 0.492 |
| ROC-AUC | 0.943 |
| PR-AUC | 0.593 |

**Confusion matrix** (test set, threshold 0.5):

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 457 (TN) | 19 (FP) |
| **Actual Fraud** | 12 (FN) | 15 (TP) |

### Cost-based threshold framing

Illustrative costs (not real bank figures): false positive = $5 (customer friction from wrongly blocking a legitimate transaction), false negative = $250 (an uncaught fraud loss).

- Total cost at the default 0.5 threshold: **$3,095**
- Minimum cost **$900** at threshold **0.09**

Lowering the threshold catches more fraud at the cost of more false positives — with a 50:1 cost ratio between missing fraud and annoying a legitimate customer, the math strongly favors a much lower threshold than the default 0.5. This 0.09 threshold (labeled "review") and a separate high-precision 0.94 threshold (labeled "block") drive the Streamlit demo's tiering.

---

## Stage 7 — Streamlit Demo

`app_streamlit.py` takes one transaction's details, engineers its features with the same `fe_utils.transform_new()` logic used in training (so training-time and live-scoring feature engineering can't drift apart), scores it with the Stage 5 XGBoost model, and maps probability to a tier:

- `< 9%` → **Auto-approve**
- `9%–94%` → **Manual review**
- `≥ 94%` → **Block**

Shows the top 3 SHAP-attributed features for that specific prediction. Verified directly (bypassing the UI) for both an existing account with history and a brand-new account with none — both score sensibly.

---

## Stage 8 — Limitations

See `LIMITATIONS.md` for the 4 slide-ready caveats: no ground-truth label exists (the label is the anomaly ensemble's own judgment, not verified fraud); this dataset is far smaller than the 1M rows the brief describes; the label is circular by construction (same features engineer → detect → train, so strong SHAP attribution confirms internal consistency, not real-world causality); and what would need to happen before production deployment (validate against real investigator-labeled cases, re-tune the 5% contamination assumption, re-run the cost sweep with the bank's actual figures).
