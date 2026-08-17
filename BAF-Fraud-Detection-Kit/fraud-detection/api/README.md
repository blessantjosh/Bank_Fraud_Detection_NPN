# Fraud Detection API

A FastAPI service in front of the BAF fraud-detection model (`../src/`). It
adds authentication, role-based access control, rate limiting, audit
logging, and model-integrity checking on top of the ML pipeline the other
half of this project builds. Everything here lives under `fraud-detection/api/`
and only *imports from* `src/` -- nothing under `src/` was modified to build this.

See `SECURITY.md` for exactly what's implemented vs. what depends on a real
deployment environment this hackathon doesn't have (a real cloud account,
a real TLS certificate authority, a real WAF appliance).

## Quick start (local, SQLite, no MFA)

```bash
cd fraud-detection
pip install -r requirements.txt        # the ML pipeline's own deps (pandas, lightgbm, shap, ...)
pip install -r api/requirements.txt    # this layer's deps (fastapi, sqlalchemy, passlib, pyjwt, pyotp, slowapi, ...)

cp api/.env.example .env               # then edit .env -- at minimum set BOOTSTRAP_ADMIN_EMAIL/PASSWORD
python -m api.scripts.init_db          # creates tables + the bootstrap admin account

# If fraud-detection/models/final_model.joblib doesn't exist yet (the ML
# pipeline hasn't finished a training run), train a fast interim model so
# /predict is actually runnable:
python -m api.scripts.train_demo_model
python -m api.scripts.record_model_checksum

uvicorn api.main:app --reload --port 8000
```

Then:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"<BOOTSTRAP_ADMIN_EMAIL>","password":"<BOOTSTRAP_ADMIN_PASSWORD>"}'
```

## Why an "interim demo model" script exists

At the point this API layer was built, `src/prediction.py`, `src/training.py`
etc. were already written and working, but the concurrent agent building the
ML pipeline hadn't finished a training run yet -- `fraud-detection/models/`
was empty. Rather than leave `/predict` permanently un-exercisable while
waiting, `api/scripts/train_demo_model.py` runs the *same*
`src.preprocessing.Preprocessor` and `src.models.train_logistic_regression`
the real pipeline uses, on a 150k-row sample of the real `Base.csv`, and
writes `final_model.joblib` / `preprocessor.joblib` / `model_meta.json` in
the exact format `src/training.py` itself produces. Whenever the real
training run finishes and overwrites those three files, the API picks them
up with **zero code changes** -- same filenames, same `model_meta.json`
schema, same `feature_columns` contract. Re-run
`api/scripts/record_model_checksum.py` after either training path.

This is genuinely a lightweight LogisticRegression baseline (ROC-AUC ~0.86
on a validation sample), not the tuned LightGBM/XGBoost ablation
`src/training.py` performs -- it exists to make the security controls
(auth, RBAC, rate limiting, integrity checks, audit logging) verifiable
against a real, working prediction today, not a mock.

## Database: SQLite now, Postgres later -- on purpose

`DATABASE_URL` defaults to `sqlite:///./api/fraud_api.db`. Every query in
this codebase goes through SQLAlchemy's ORM (`api/models_db.py`), never a
raw/string-built query, specifically so that swapping the URL to Postgres
is the *entire* migration:

```
DATABASE_URL=postgresql+psycopg://fraud_api:changeme@localhost:5432/fraud_api
```

(install `psycopg[binary]`, commented out in `api/requirements.txt`, and
point `docker-compose.yml`'s `db` service credentials at the same values).

SQLite was chosen as the default -- not as an oversight, but because this is
a hackathon-timeline build with no real cloud account to provision a managed
Postgres against, and a single-file DB means the demo runs on a judge's
laptop with zero setup. The trade-off it costs: SQLite serializes writes
(one writer at a time), has no real concurrent-connection pooling, and
`api/database.py` uses `check_same_thread=False` to let FastAPI's async
worker threads share the single file, which is fine for a demo's request
volume and not something you'd want under real concurrent write load.
`docker-compose.yml` ships a real Postgres container to show the intended
production shape.

## MFA: real TOTP, off by default locally -- on purpose

`api/security.py` implements real TOTP MFA with `pyotp` --
`/auth/mfa/setup` returns a real secret + `otpauth://` URI you can scan with
any authenticator app (Google Authenticator, Authy, etc.), and
`/auth/mfa/verify` checks a real 6-digit code against it. This isn't a
stub -- see `api/tests/test_api.py` and the manual verification in the
final report for it exchanging real generated codes for real tokens.

`MFA_REQUIRED` defaults to `false`. This is a deliberate hackathon-timeline
call, not a security shortcut we're hiding: if it defaulted to `true`, the
first person who spins up this demo would create an admin account, get
logged out immediately pending an MFA code they haven't enrolled yet, and
have no way back in without touching the database directly. Production
should set `MFA_REQUIRED=true` -- when it's on, every ADMIN login is gated
on TOTP before any token is issued (`api/routers/auth.py`), and the
mfa-pending token that carries a verified password across that gap is a
separate, five-minute-lived JWT type (`mfa_pending`) that is rejected by
every other endpoint if presented as an access token.

## Roles and permissions

Five roles, enforced server-side on every route via a FastAPI dependency
(`api/rbac.py`) -- never a frontend-only check:

| Capability | VIEWER | FRAUD_ANALYST | RISK_MANAGER | AUDITOR | ADMIN |
|---|---|---|---|---|---|
| `GET /predictions` (view prediction history) | Y | Y | Y | Y | Y |
| `POST /predict`, `POST /predict/file` (run predictions) | | Y | Y | | Y |
| `GET /audit-logs` -- own actions only | | Y | | | |
| `GET /audit-logs` -- full | | | Y | Y | Y |
| `PATCH /settings/threshold` | | | Y | | Y |
| Full explainability (`top_features` in `/predict` response) | | | Y | | Y |
| `/admin/users` (create/list/update) | | | | | Y |

## Endpoints

- `GET /health` -- public, minimal.
- `POST /auth/login`, `POST /auth/refresh`, `POST /auth/mfa/verify`,
  `POST /auth/mfa/setup`, `POST /auth/mfa/enable`.
- `POST /predict` (JSON body: `{"rows": [...]}`, strict per-row schema),
  `POST /predict/file` (CSV upload).
- `GET /predictions` -- prediction metadata history.
- `PATCH /settings/threshold` -- override the model's decision threshold.
- `GET /audit-logs`.
- `GET /admin/users`, `POST /admin/users`, `PATCH /admin/users/{id}`.

## Why `/predict` doesn't call `src.prediction.predict_csv()`

`src/prediction.py`'s `predict_csv()` is gated by `src.auth.require_admin()`,
which checks a single shared `FRAUD_ADMIN_TOKEN` -- the right shape for a
trusted-operator CLI script, wrong shape for a multi-user, multi-role HTTP
service (every caller of every role would need the same secret, which
defeats RBAC entirely). This API layer calls the lower-level pieces
directly instead (`load_artifacts` + `Preprocessor.transform_*` +
`models.predict_proba`, see `api/model_service.py`) and enforces
authorization at the HTTP boundary with its own JWT+RBAC. `src.auth` and
`src.audit`'s JSONL log are untouched and still work exactly as before for
anyone still using `predict.py` from the command line.

## Running the tests

```bash
cd fraud-detection
python -m pytest api/tests/ -v
```

The suite uses a disposable SQLite file (`api/tests/test_fraud_api.db`,
recreated every run) and trains the interim demo model automatically if
`models/final_model.joblib` doesn't exist yet, so it's runnable from a clean
checkout with no manual setup steps.

## Docker

```bash
cd fraud-detection
cp api/.env.example api/.env      # fill in POSTGRES_PASSWORD, JWT_SECRET, etc.
docker compose -f api/docker-compose.yml --env-file api/.env up --build
```

See `api/Dockerfile` (non-root user, no secrets baked in, only port 8000
exposed) and `api/docker-compose.yml` (Postgres container not published to
the host network, only reachable from the `api` container).
