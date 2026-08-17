# Security posture -- this API layer

This is not a claim of OWASP/PCI/SOC2 compliance, and it isn't trying to
sound like one. It's a specific list of what this codebase actually does,
what it was checked against, and where the line sits between "this repo
handles it" and "whoever deploys this for real has to handle it." If
you're the bank engineer picking this up: read the second section first,
because that's the part a hackathon demo cannot fake or shortcut around,
and pretending otherwise would be worse than just saying so.

## Implemented in this codebase (and actually run, not just written)

**Passwords.** `passlib.CryptContext(schemes=["argon2", "bcrypt"])` in
`api/security.py`. New hashes are always Argon2id -- `argon2-cffi` 25.1.0 is
installed and working in this environment (confirmed: `pwd_context.hash()`
+ `pwd_context.verify()` round-trip tested directly, see
`api/tests/test_api.py::test_wrong_password_rejected` for the negative
case). Bcrypt is kept in the scheme list only so a hash created under a
different config still verifies; nothing in this codebase writes a bcrypt
hash on purpose.

**JWTs.** `PyJWT` 2.13.0, HS256, two token types (`access`, 15 min default;
`refresh`, 7 days default) plus a third, narrower one (`mfa_pending`, 5
min) that exists only to carry "password was correct" across the MFA gap
and is rejected by `get_current_user()` if anyone tries to use it as an
access token -- checked in `api/rbac.py` and exercised in the test suite.
Refresh tokens rotate: `/auth/refresh` immediately adds the used token's
`jti` to a `revoked_tokens` table, so a stolen refresh token that's been
rotated away from can't be replayed. `JWT_SECRET` is read from the
environment; if it isn't set, `Settings.effective_jwt_secret()` generates a
random one with `secrets.token_hex(32)` and logs a `WARNING`-level message
every time it's used, on purpose, so it's loud in server logs rather than a
silent gap -- the cost is that restarting the process invalidates every
outstanding token, which is fine for a single demo process and not fine for
anything you'd actually deploy.

**MFA.** Real TOTP via `pyotp` 2.10.0 -- `/auth/mfa/setup` returns an actual
`otpauth://` URI you can put in Google Authenticator, and `/auth/mfa/verify`
checks a real generated 6-digit code (verified manually against a live
server: setup -> scan-equivalent code generation with `pyotp.TOTP(secret).now()`
-> enable -> subsequent login correctly returns `mfa_required: true` and
withholds tokens until the code checks out, wrong code gets a 401, and the
`mfa_pending` token can't be reused as an access token). It's off by default
(`MFA_REQUIRED=false`) -- see `api/README.md`'s "MFA" section for the reasoning,
which is about not locking a hackathon judge out of their own demo, not about
the feature being fake.

**RBAC.** Five roles, one dependency function per permission
(`api/rbac.py`: `require_admin`, `require_predict`, `require_view_predictions`,
`require_audit_view`, `require_threshold_change`), applied as a FastAPI
`Depends(...)` on every route that needs it -- there is no route that checks
role in the handler body instead of the dependency layer, and no route that
skips the check because "the frontend already filters it." Every one of the
role-boundary claims below was hit with a live request during this build,
not just asserted in a docstring: Viewer -> `/admin/users` = 403, Viewer ->
`POST /predict` = 403 (but Viewer -> `GET /predictions` = 200), Fraud
Analyst -> `/settings/threshold` = 403, Fraud Analyst -> `/audit-logs` =
200 but scoped to only their own `user_id` (verified: every row returned had
`role == "FRAUD_ANALYST"` and `user_id` equal to the requesting analyst's
own id), Admin -> everything = 200.

**Rate limiting.** `slowapi`, keyed by client IP. `/auth/login` and
`/auth/mfa/verify` are capped at `LOGIN_RATE_LIMIT` (default `5/minute`);
`/predict` and `/predict/file` at `PREDICT_RATE_LIMIT` (default
`30/minute`). This was not just wired up and trusted to work -- the test
suite (`test_rate_limit_triggers_after_n_failed_logins`) sends exactly 5
failed logins, asserts all five come back 401, then sends a 6th and asserts
it comes back 429, and that test passes. `/auth/login`'s failure message is
identical (`"Invalid email or password"`, HTTP 401) whether the email
doesn't exist or the password is wrong -- confirmed side-by-side in the
test suite so the endpoint can't be used to enumerate registered emails.

**Input validation.** Every request body is a Pydantic model with
`extra="forbid"` (`api/schemas.py`) -- an unexpected field is a 422, not a
silently-ignored key. The BAF application-row schema
(`ApplicationRow`) has an explicit type and numeric range per column,
sourced from `../../01-DATASET-BIBLE.md`'s documented ranges with a
deliberately small margin beyond the observed training extremes (e.g.
`customer_age` allows up to 100 even though training data tops out at 90,
so a legitimate edge case doesn't 422; `customer_age=5000` still does).
CSV upload (`/predict/file`) is checked three separate ways before a byte
of it is trusted: the filename must end in `.csv` (not just trusted because
the browser said `Content-Type: text/csv`), the byte count is capped at
`MAX_UPLOAD_BYTES` (default 5 MB, read with a `+1`-byte peek so an
oversized file is rejected instead of partially buffered), and the header
row is checked against the exact expected BAF column set -- missing columns
and *unexpected* extra columns are both rejected, not just missing ones.

**Error handling.** One global handler per exception class in `api/main.py`:
`HTTPException` passes its own status/detail through (those are
intentional, safe-to-see messages we wrote), `RequestValidationError`
returns a sanitized field/type/message list (never the raw input value
echoed back), and a catch-all `Exception` handler logs the full traceback
server-side via `logger.error(..., exc_info=True)` and returns a fixed
generic message plus a `request_id` to the caller -- nothing about the
exception type, message, or file path crosses that boundary. Verified with
a request that sends deliberately malformed JSON: the client gets
`{"detail": "Request validation failed", ..., "request_id": "..."}` and
the string `"Traceback"` does not appear anywhere in that response body.

**Security headers & CORS.** Set unconditionally in a middleware in
`api/main.py`: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, a restrictive `Content-Security-Policy`
(`default-src 'none'`, since this is a pure JSON API with no HTML to
render), and `Strict-Transport-Security` (see the TLS caveat below for why
that header is sent even though nothing here terminates TLS itself). CORS
is `CORSMiddleware` with `allow_origins` built from `CORS_ALLOW_ORIGINS`
(comma-separated exact origins) -- never `["*"]`, and there's no code path
that falls back to a wildcard if the env var is unset (it just becomes an
empty allow-list).

**Model integrity.** `api/model_service.py` computes SHA-256 of
`final_model.joblib` and `preprocessor.joblib` on load and compares against
`models/model_checksum.json` (format and the script that writes it:
`api/scripts/record_model_checksum.py`). `ENFORCE_MODEL_CHECKSUM` is
`false` by default (so `/predict` still works before anyone has recorded a
checksum) but the check runs either way and its result
(`checksum_verified: true/false/null`) is visible at `/health` -- this was
verified live: after training the demo model and recording its checksum,
`/health` reported `model_checksum_verified: true`; deleting/mutating an
artifact would flip that to `false` and `ENFORCE_MODEL_CHECKSUM=true` would
turn that into a hard refusal to serve, not just a log line.

**MFA secret at rest.** `User.mfa_secret` is never stored as the plain
base32 TOTP secret -- `api/security.py::encrypt_mfa_secret()` encrypts it
with `Fernet` (symmetric, from the `cryptography` library; free, local, no
KMS/paid service needed) before `/auth/mfa/setup` ever writes the row, and
it is decrypted only in memory, only inside `/auth/mfa/verify` and
`/auth/mfa/enable`, only for the duration of the `pyotp.verify()` call --
never logged, never returned by any endpoint after the initial setup
response. The key comes from `MFA_ENCRYPTION_KEY`; if unset,
`Settings.effective_mfa_encryption_key()` generates a random Fernet key at
startup and logs a loud `WARNING` every time it's used, the same pattern as
`JWT_SECRET` -- the cost of not setting it is that a restart makes every
already-enrolled MFA secret undecryptable (that admin has to re-enroll via
`/auth/mfa/setup`). This was verified directly: after enrolling MFA for an
admin account, the raw `users.mfa_secret` column value was queried straight
out of the database and asserted to differ from the plaintext secret
(`api/tests/test_api.py::test_mfa_secret_encrypted_at_rest`), and a full
setup -> enable -> mfa-pending-login -> verify round trip with a real
generated TOTP code still succeeds after the change.

Being precise about what this does and doesn't buy: it protects the secret
against anyone who can read the database file or a backup of it (a leaked
SQLite file, a Postgres dump, a read-only DB credential) but *not* against
someone who can read both the database and the app's environment
variables at the same time -- that's an inherent limit of symmetric
encryption without an external key store, not something this change
papers over. Moving `MFA_ENCRYPTION_KEY` itself into a real secrets
manager/KMS is the same "real secrets manager" gap already listed below,
not a new one.

**Audit trail.** `audit_log` table, append-only by construction -- there is
no `UPDATE`/`DELETE` code path against it anywhere in `api/`, only `INSERT`
via `api/audit.py::write_audit()`. It records who (user id + role), what
(action + resource), the outcome (`SUCCESS`/`DENIED`/`ERROR`), a
correlating `request_id`, and a short non-sensitive `detail` string -- never
a password, a token, or a full feature vector. `prediction_records`
similarly stores row counts, a SHA-256 of the input (not the input itself),
and risk-level counts, matching the "metadata only" design the ML side's
own `src/audit.py` JSONL log already uses for CLI runs.

## Deployment-environment responsibility (this repo does not, and cannot, provide these)

**TLS termination and certificate management.** Nothing in this repo
obtains, renews, or terminates a TLS certificate, and nothing should --
there's no real certificate authority reachable from a hackathon sandbox,
and standing up one ourselves would just be a fake CA that provides no
actual protection while looking like it does. The `Strict-Transport-Security`
header above is sent unconditionally; it is inert and harmless over plain
HTTP (browsers only honor it after they've already seen it over HTTPS
once), and becomes meaningful the moment a real reverse proxy (nginx,
Caddy, an ALB, whatever the bank already runs) terminates TLS in front of
this service. Put this behind that proxy. Don't run it bare on the public
internet.

**A real WAF.** This API has application-level protections (RBAC, rate
limiting, strict schemas) but no signature-based attack detection, no
bot-fingerprinting, no DDoS absorption layer. That's a network-edge concern
and belongs on whatever the bank's real edge infrastructure is (Cloudflare,
an AWS WAF, an on-prem appliance) -- none of which exist in this sandbox to
integrate against, and faking one here would mean claiming a control that
isn't real.

**Real network segmentation.** `docker-compose.yml` keeps the Postgres
container off the host's published ports (no `ports:` entry on `db`), which
is a real, correct control at the container-network layer. It is not a
substitute for VPC/subnet isolation, security groups, or a bank's actual
network segmentation policy between an app tier and a data tier -- that's
infrastructure this sandbox doesn't have an account to provision.

**A real secrets manager.** `JWT_SECRET`, `BOOTSTRAP_ADMIN_PASSWORD`,
`POSTGRES_PASSWORD` all come from environment variables / `.env` files here
-- deliberately, since there's no AWS Secrets Manager, Vault cluster, or
KMS available in this environment, and none of those are free services we
were asked to avoid reaching for anyway. `.env` is gitignored
(`fraud-detection/.gitignore` and `api/.gitignore` both exclude it, along
with `*.key`, `*.pem`, `secrets/`), and `.env.example` ships with every key
present but every value blank. In production, swap the env-var reads in
`api/settings.py` for reads from whatever secrets manager the bank already
operates -- that's a small, contained change, but it's a real one this repo
doesn't make for you.

**Encrypted backups.** SQLite writes one plain file
(`api/fraud_api.db`); Postgres (via the compose file) writes to a named
Docker volume. Neither this repo nor the compose file encrypts backups of
either -- that's a platform/ops concern (disk-level encryption, a managed
Postgres provider's backup encryption, whatever the bank's existing backup
tooling does) that depends entirely on where this actually gets deployed.

**A real SIEM / log-integrity pipeline.** The `audit_log` table is
append-only *at the application layer* (no code path updates or deletes a
row) but nothing stops someone with direct database access from editing the
SQLite/Postgres file directly, and nothing here ships these logs to an
external, tamper-evident store. A real deployment should forward
`audit_log` (and the structured request logs this service already emits to
stdout) into whatever SIEM/log-aggregation system the bank runs, with
write-once storage or hash-chaining if genuine tamper-evidence is required.
That infrastructure doesn't exist in this sandbox to wire up against.
