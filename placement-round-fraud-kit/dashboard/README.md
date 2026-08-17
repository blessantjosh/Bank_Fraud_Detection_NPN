# Argus — Behavioral Anomaly Intelligence

A browse-first fraud-analytics console built on top of the existing v1 pipeline
(`../src`, `../artifacts`). It does not retrain or recompute anything from that
pipeline — it loads the same artifacts (`labeled.csv`, `anomaly_votes.csv`,
`reference.pkl`, `xgb_model.json`, `thresholds.json`, `decision_tree_rules.txt`)
and adds a scoring layer, a SHAP explanation layer, and a web UI on top.

## What's here

```
dashboard/
  backend/
    api_server.py   FastAPI app: loads artifacts, scores all 2,512 transactions,
                     precomputes SHAP once, serves the API and the frontend.
    cache/           shap_values.npy is written here on first run (instant after).
    queue_state.json created on first Investigation Queue action; safe to delete.
  frontend/
    index.html, css/style.css, js/*.js   Plain HTML/CSS/JS, no build step,
                                          no CDN, no external fonts/icons.
```

## Requirements

Everything the v1 pipeline already uses (pandas, numpy, scikit-learn, xgboost,
shap, joblib) plus two additions used only by the dashboard:

```
pip install fastapi uvicorn
```

(Tested against fastapi 0.136.1, uvicorn 0.46.0 — any reasonably recent version
of each should work. No paid services, no external network calls; the app runs
fully offline once these packages are installed.)

## Running it

From the `dashboard/` directory:

```
cd dashboard
python -m uvicorn backend.api_server:app --reload
```

Then open **http://127.0.0.1:8000/** in a browser. The backend serves the
frontend itself (via FastAPI `StaticFiles`), so there is nothing else to start
and no CORS configuration is needed — do not open `frontend/index.html`
directly as a `file://` URL, since its JavaScript calls the API using
same-origin relative paths (`/api/...`).

First startup takes a few seconds: it loads the raw CSV, re-applies the same
sort `fit_engineer()` used, scores all 2,512 transactions with the SMOTE
XGBoost model, and runs `shap.TreeExplainer` once over every row. On every
run after the first, `backend/cache/shap_values.npy` is reused, so startup is
close to instant.

To use a different port: `python -m uvicorn backend.api_server:app --port 8099`.

## Pages

1. **Overview** — KPI tiles, risk-tier distribution, transaction volume over
   time, top-10 highest-risk transactions, and a condensed "About this
   system" note (the full caveats live in `../LIMITATIONS.md`).
2. **Transaction Explorer** — search/filter/sort/paginate all 2,512
   transactions; click a row to open a detail drawer with the full raw
   fields, verdict, per-transaction SHAP breakdown, and which of the 4
   unsupervised detectors flagged it.
3. **Investigation Queue** — the same transactions sorted by risk score
   descending, with Approve/Escalate/Block actions that persist to
   `backend/queue_state.json`, and a CSV export.
4. **Model Comparison** — the 4-detector ensemble comparison, SMOTE vs.
   class-weighted XGBoost, the confusion matrix, and the naive-baseline-vs-
   actual-accuracy contrast, using the real measured numbers from the v1
   pipeline (hardcoded in `api_server.py`, not recomputed).
5. **Explainability** — global SHAP importance, the decision-tree rules
   rendered as if/then statements, and a real cost-based threshold sweep
   (recomputed at startup from `artifacts/split.pkl`, not hand-typed numbers).
6. **What-if Simulator** — secondary, visually de-emphasized tab for
   hypothesizing about one brand-new transaction via `fe_utils.transform_new`.

## Verification performed

- Backend endpoints hit directly with `curl`/Python and confirmed to return
  real data: `/api/kpis` (2,512 total, 57 high-risk, 79 review, matching the
  brief's numbers exactly), `/api/transactions` (search/filter/sort/paginate),
  `/api/transactions/{id}` (raw fields + real SHAP values + detector flags),
  `/api/queue` + `/api/queue/action` (persists to `queue_state.json`),
  `/api/queue/export` (CSV), `/api/model-comparison`, `/api/explainability`
  (real, recomputed cost-sweep curve — reproduces the $900-at-0.09 and
  $3,095-at-0.50 figures from the brief exactly), `/api/simulator/options`,
  and `/api/score`.
- Frontend rendered end-to-end with Playwright (headless Chromium): all six
  pages, the detail drawer, filtering/search, queue actions with toast
  feedback and persistence, the What-if form and its result panel, and both
  dark and light themes were screenshotted and visually inspected — no
  console errors or page errors in any of it.
- All frontend `.js` files pass `node --check` (syntax validation).

## Known limitations carried over from the v1 pipeline

"Risk" here is pattern-consistency with an unsupervised anomaly ensemble, not
a verified fraud outcome — see `../LIMITATIONS.md` for the full explanation.
This is also surfaced in the UI itself (Overview page "About this system"
panel and the sidebar footer note).
