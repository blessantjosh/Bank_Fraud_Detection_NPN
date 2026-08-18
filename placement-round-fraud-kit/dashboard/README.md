# Argus — Behavioral Anomaly Intelligence

A browse-first fraud-analytics console built on top of the **research_v2
pipeline** — the client-designated final pipeline, built on the teammate's
18-feature matrix (`../research_v2/`, `../src_research_v2/`,
`../artifacts_research_v2/`). It does not retrain or recompute anything: it
loads that pipeline's artifacts and adds a serving layer and a web UI.

> **Data-source note.** Argus was originally wired to the v1 pipeline
> (`../artifacts/`: `labeled.csv`, `xgb_model.json`, `reference.pkl`,
> `thresholds.json`) and showed risk tiers derived from a supervised XGBoost
> model trained to reproduce a 4-detector unsupervised ensemble. It now serves
> the research_v2 pipeline instead. The visual design, branding and page
> structure are unchanged — this was a data-source swap, not a redesign.

## What it serves

| Surface | Source |
|---|---|
| Risk score | `ensemble_percentile_average` — Percentile Aggregation over 11 models (Phase 12 v2 recommendation), from `artifacts_research_v2/ensemble_scores_v2.csv` |
| Risk tiers | Phase 13 (v2) cutoffs, read from `artifacts_research_v2/threshold_analysis_v2.json`: **≥ 0.951023 → Priority review** (26 txns, 1.04%), **≥ 0.867124 → Standard review** (100 more, 126 total at 5.02%), else Normal |
| Per-transaction explanation | Precomputed **Isolation Forest** and **Autoencoder** SHAP, shown side by side, from `shap_isolation_forest_v2.csv` / `shap_autoencoder_v2.csv` — a lookup, never a recomputation |
| Per-model detail | All 11 ensemble members' per-row percentiles and top-5% flags, from `model_scores_all.csv` |
| Model Comparison page | 12-model comparison: flagged rates, internal validity (`internal_validity_metrics_v2.csv`), bootstrap stability (`stability_bootstrap_jaccard_v2.csv`), consensus weights (`ensemble_weights_v2.json`), strategy agreement (`ensemble_pairwise_comparison_v2.csv`) |
| Explainability page | Global mean\|SHAP\| for both models (`shap_global_importance_comparison_v2.csv`), the ρ = −0.3705 divergence, three worked cases, the score distribution and the Phase 13 threshold tables |
| Raw transaction fields | `data/bank_transactions_data_2.csv`, joined on `TransactionID` |

**There is no automatic block tier**, and this is deliberate: a cost-optimal
blocking threshold requires counting false negatives, which requires a fraud
label this dataset does not have (Phase 13 v2 §1).

**Nothing on the dashboard is hand-typed from a report.** Every number is read
from an artifact at startup, so a stale artifact produces a visibly stale
dashboard rather than one that silently disagrees with the pipeline behind it.
`GET /api/meta` returns the full provenance, including a self-check that the
reloaded Isolation Forest and Autoencoder artifacts reproduce the published
per-row scores (max error 1.0×10⁻¹⁶ and 4.2×10⁻⁸ respectively).

## What's here

```
dashboard/
  backend/
    api_server.py    FastAPI app: loads the research_v2 artifacts, builds the
                      ledger, serves the API and the frontend.
    queue_state.json  created on the first Investigation Queue action; safe to delete.
  frontend/
    index.html, css/style.css, js/*.js   Plain HTML/CSS/JS, no build step,
                                          no CDN, no external fonts/icons.
```

## Requirements

Everything the research_v2 pipeline already uses (pandas, numpy,
scikit-learn, torch, shap, joblib) plus two additions used only by the
dashboard:

```
pip install fastapi uvicorn
```

(Tested against fastapi 0.136.1, uvicorn 0.46.0. No paid services, no external
network calls; the app runs fully offline once these packages are installed.)

## Running it

From the `dashboard/` directory:

```
cd dashboard
python -m uvicorn backend.api_server:app --port 8000
```

Then open **http://127.0.0.1:8000/**. The backend serves the frontend itself
via FastAPI `StaticFiles`, so there is nothing else to start and no CORS
configuration is needed — do not open `frontend/index.html` directly as a
`file://` URL, since its JavaScript calls the API using same-origin relative
paths (`/api/...`).

Startup takes roughly 20 seconds: it loads the raw CSV, the 18-feature matrix,
the ensemble scores, both SHAP matrices and the model artifacts, and runs the
score-reproduction self-check described above. There is no SHAP recomputation
and therefore no disk cache to warm.

## Pages

1. **Overview** — KPI tiles (total, priority, standard, flag rate, average
   amount), risk-tier distribution against the Phase 13 cutoffs, transaction
   volume over time, top-10 highest-risk transactions, and an "About this
   system" note stating plainly what the score is, that there is no block tier,
   and that this feature set cannot ask "is this unusual *for this customer*".
2. **Transaction Explorer** — search/filter/sort/paginate all 2,512
   transactions; click a row for a detail drawer with the raw fields, the
   ensemble score and rank, per-model percentile chips for all 11 members, and
   **both** SHAP breakdowns.
3. **Investigation Queue** — the same transactions sorted by ensemble score
   descending, with Approve/Escalate/Block actions that persist to
   `backend/queue_state.json`, and a CSV export. This queue is the only
   label-generating mechanism in the project (Phase 15 v2 §7.5).
4. **Model Comparison** — the 12-model table (flagged rate, silhouette,
   Davies-Bouldin, Calinski-Harabasz, self-excluded mean Spearman/Jaccard,
   ensemble weight), the internal-validity and retrain-stability charts, the
   consensus weights, and the four ensemble strategies compared pairwise —
   with the honest caveats attached in the UI, not omitted.
5. **Explainability** — the ρ = −0.3705 divergence between the two explained
   models, both global-importance charts, three worked cases (`TX000275`,
   `TX000615`, `TX001029`), the ensemble score distribution with both cutoffs
   marked, the percentile-threshold table with its cost ceiling, and the
   statistical-threshold table showing that mean+3σ and Q3+1.5×IQR flag **zero**
   transactions on a bounded percentile-averaged score.
6. **Account Scenario Simulator** — see below.

## The Account Scenario Simulator (formerly "What-if")

The v1 What-if Simulator scored a brand-new, free-form hypothetical
transaction. **That is not honestly possible on this feature set.** Five of the
eighteen features (`account_frequency`, `device_frequency`, `ip_frequency`,
`merchant_frequency`, `Location_FE`) are population-level statistics computed
across the whole dataset. An invented transaction has no `device_frequency`
until you decide what population to count over, and inventing one produces a
confident-looking score built on a fabricated input.

**The tab was therefore rebuilt, not repointed** (option (a) of the two
documented in Phase 15 v2 §7.3):

- It **requires an existing `AccountID`** and prefills every field from that
  account's most recent real transaction.
- Device, IP, merchant and location are **dropdowns of values that exist in the
  data**, and their true historical frequencies are used. Submitting an invented
  device returns an explicit 400 explaining why, rather than scoring it.
- You vary only the fields that genuinely belong to one transaction: amount,
  balance, type, channel, occupation, age, duration, login attempts.
- The remaining features are computed **exactly** from frozen training
  constants — `StandardScaler(log1p(amount))`, `StandardScaler(log1p(amount /
  (balance + 1)))`, and the frozen `$878.18` high-amount threshold (all three
  recovered and verified in Phase 14 v2 §5).
- It is scored by **Isolation Forest and the Autoencoder only**, not the full
  ensemble, because DBSCAN and HDBSCAN cannot score an unseen row at all. The
  UI says so, and reports a two-model percentile average against a two-model
  reference distribution rather than pretending to produce the deployed
  11-model score.

**Verified exact:** re-entering `TX000275`'s real field values returns Isolation
Forest 0.11465 and Autoencoder 1.84051 — matching the published
`score_isolation_forest` (0.1146454949) and `score_autoencoder` (1.8405135) —
and the live TreeExplainer attribution reproduces the precomputed SHAP row
(`LoginAttempts` +1.7138, `amount_to_balance_ratio` +1.60652,
`high_amount_transaction` +1.13117).

## Verification performed

Server started on a real port and every endpoint hit with `curl`:

- `/api/meta` — provenance; confirms the score source, both thresholds
  (0.951023 / 0.867124), the 11-member list, and the model-reload self-check.
- `/api/kpis` — 2,512 total, **26 priority**, **100 standard**, 2,386 normal,
  flag rate 0.0502, average amount $297.59; top-risk row 1 is **TX000275 at
  0.9951, flagged by 11 of 11 models**.
- `/api/transactions?risk_tier=priority` — 26; `?risk_tier=standard` — 100.
- `/api/transactions/TX000275` — raw $1,176.28 against a $323.69 balance
  (3.634×), 5 login attempts, rank **1 of 2,512**, all 11 models flagging; IF
  SHAP `LoginAttempts` +1.7138 / `amount_to_balance_ratio` +1.60652; AE SHAP
  `amount_to_balance_ratio` +1.56977 — all matching `research_v2/09_explainability.md` §2.
- `/api/model-comparison` — Elliptic Envelope silhouette 0.5409 / CH 592.47
  (leader), Autoencoder 0.1724 (last), DBSCAN mean ρ 0.2352 (lowest) and weight
  0.0513; stability IF 0.6021 / LOF 0.5124 / AE 0.3726; Rank vs. Percentile
  ρ 0.9999, Jaccard 0.9688; PC1 54.9%.
- `/api/explainability` — divergence ρ **−0.3705**, top-10 overlap 3;
  thresholds P95/P97/P99/P99.5 → 126/76/26/13 flagged; mean+3σ and Q3+1.5×IQR
  → **0 flagged** on the percentile score, 29 and 79 on the unbounded
  weighted average.
- `/api/simulator/options` — 495 accounts, 681 devices, 592 IPs, 100 merchants,
  43 locations; `/api/simulator/account/AC00454` — 4 transactions.
- `/api/score` — exact reproduction of TX000275 (above); a benign variation of
  the same account ($25.00, 1 login attempt) drops to a **Normal** tier at a
  two-model average of 0.4088; an invented device is rejected with an
  explanation.
- `/api/queue`, `/api/queue/action`, `/api/queue/export` — action persists to
  `queue_state.json` and appears in the filtered queue and the CSV export.
- Static assets (`/`, `js/*.js`, `css/style.css`) all return HTTP 200, and all
  five frontend `.js` files pass `node --check`.

## Known limitations carried over from the pipeline

"Risk" here is pattern-consistency with an unsupervised anomaly ensemble on an
unlabelled dataset, not a verified fraud outcome. This feature set is built from
population-level statistics and has no per-account baseline or novelty features,
so it detects "unusual in the population" rather than "unusual for this
customer" — account takeover is only partially detectable. See
`../research_v2/15_final_research_report.md` §13 for the full list, and
`../LIMITATIONS.md` for the project-level caveats. Both points are surfaced in
the UI itself (Overview "About this system" panel and the sidebar footer).
