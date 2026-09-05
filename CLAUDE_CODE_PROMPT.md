# Hospital HMIS — Hardening & Modernization

## How to use this document

Paste the "Mission briefing" section below as your first message to Claude Code inside the `hospital` repo. Work through the phases **in order** — each phase has an exit gate, and later phases assume earlier ones are done. This is multi-week work that will span many sessions, so the first task is to create a tracking file that survives context resets.

---

## Mission briefing (paste this to Claude Code)

You are hardening and modernizing a hospital management information system (HMIS): a Flask 3.0 monolith (12 department blueprints: records, billing, pharmacy, medicine, laboratory, imaging, stores, admin, nursing, hr, mortuary, api) plus a separate FastAPI microservice (`departments/nlp/`) that does clinical NLP against a local UMLS database and calls external LLM APIs (NVIDIA NIM, Gemini). Core app DB defaults to SQLite; UMLS runs on PostgreSQL; sessions use Redis.

A prior audit found the issues addressed in the phases below. Work through them **in order**, one task at a time. Do not skip ahead to later phases while earlier ones have open items.

### Ground rules (non-negotiable)

1. **Branch per phase.** Create `hardening/phase-0`, `hardening/phase-1`, etc. Never commit directly to `main`.
2. **Test before and after.** Run `pytest` before starting any task and after finishing it. If a change breaks an existing test, fix it before moving on — don't comment out or skip the test.
3. **No behavior change without a test.** Every bug fix or new feature gets a test that would have failed before the change and passes after.
4. **Commit small and often.** One logical change per commit. Message format: `[phase-N] short description` (e.g., `[phase-0] add login_required to patient search API`).
5. **Never commit secrets.** Check `.gitignore` covers any new config/credential files before committing. Never put real API keys or passwords in code, migrations, or fixtures.
6. **Flag judgment calls, don't invent them.** For anything with real clinical, legal, or financial consequence — insurance claim rules, data retention periods, encryption key management, what counts as a "duplicate patient" — implement the scaffolding and data model, mark the business-rule gaps explicitly with `# DECISION NEEDED:` comments, and ask the human before finalizing. Do not silently invent healthcare business logic.
7. **Patient data is never destructively migrated.** Any schema migration touching existing patient, billing, or clinical tables must be additive/reversible and preceded by a backup step in the migration instructions you give the human.
8. **This is pre-production hardening.** Nothing produced here should be treated as ready for real patient data until a human explicitly signs off — say so if asked to "deploy this now."

### Before you start: create the tracker

Create `IMPLEMENTATION_PROGRESS.md` at the repo root with a checklist mirroring every task below (checkboxes, one per task). Update it — check items off, add notes on decisions made or blocked items — as you go. This is the source of truth across sessions; read it first thing in every new session before touching code.

---

## PHASE 0 — Stop the bleeding (do this first, today)

Goal: close the issues that are actively dangerous if this ever touches a network or real patient data.

### 0.1 Lock down the unauthenticated patient search API
`departments/api/patients.py` — `/api/patients/search` has no `login_required` and no rate limit, and returns patient names + IDs to anyone.
- Add `@login_required` (and a `@roles_required(...)` appropriate to who actually uses the Select2 widget this feeds).
- Add rate limiting (see 0.6).
- **Done when:** a test asserts an unauthenticated request to this endpoint returns 401/403, and an authenticated request still works.

### 0.2 Fix the frozen-date bug
Three locations evaluate `datetime.utcnow()` at import time instead of per-row/per-render:
- `departments/models/records.py:75` (`ClinicBooking.created_on`)
- `departments/models/medicine.py:368` (`note_date`)
- `departments/forms.py:89` (`note_date` form default)

Fix pattern: `default=lambda: datetime.utcnow().date()` (or `default=date.today` where a plain callable works), never `default=datetime.utcnow().date()` or `default=datetime.utcnow().date` (both freeze at import time).
- **Done when:** a test creates two records with an artificial delay (or mocks the clock) between them and asserts they get different dates when the underlying date actually differs; grep the whole codebase for any other instance of `default=datetime.utcnow().` or `default=datetime.now().` and fix those too.

### 0.3 Remove unrelated non-HMIS files
Delete (after confirming with the human they're not needed for something outside this repo's scope): `convert_markets_fintech.py`, `generate_proposal.py`, `agent.py`, `test_query.py`. These are leftovers from unrelated projects (a fintech DB migration, an agri-tech grant proposal, a generic coding-agent CLI) and a throwaway DB test script. Check `git log` on each first in case anything else imports from them (it shouldn't, per the earlier audit, but verify).
- **Done when:** the files are gone, the app still boots, and the full test suite still passes.

### 0.4 Stop leaking exception details to users
`app.py`'s login route does `flash(f'Login error: {e}', 'error')`, showing raw exception text to the person at the login screen.
- Log the full exception server-side (already partially done via `logger.error`); show the user a generic message ("Something went wrong. Please try again.").
- Audit other routes for the same pattern (`grep -rn "flash(f'.*{e}" --include="*.py" .`) and fix all of them the same way.
- **Done when:** no user-facing `flash()` or rendered template contains interpolated exception objects anywhere in the app.

### 0.5 Harden `SECRET_KEY` enforcement
Currently falls back to a hardcoded dev key unless `FLASK_ENV` is exactly `'production'`. Make it fail loudly in every environment if `SECRET_KEY` isn't set via environment variable, with a clearly documented exception only for an explicit `FLASK_ENV=testing` used by the test suite.
- **Done when:** starting the app with no `SECRET_KEY` set and `FLASK_ENV` unset or set to anything other than `testing` raises `RuntimeError` immediately.

### 0.6 Add rate limiting and lockout to authentication endpoints
The main Flask app has no rate limiting anywhere (it only exists inside the separate FastAPI NLP service via `slowapi`). Add `Flask-Limiter` to `requirements.txt` and apply it to `/login` and any other unauthenticated POST endpoint. Add a `failed_login_attempts` and `locked_until` column to the `User` model; lock the account for a configurable window after N consecutive failures, reset on success.
- **Done when:** a test hits `/login` with bad credentials past the threshold and asserts the account is locked, and a separate test asserts rate limiting kicks in on rapid requests.

**Phase 0 exit gate:** full test suite passes, `IMPLEMENTATION_PROGRESS.md` shows all Phase 0 items checked, changes are merged to `main` via reviewed PR (even if you're the only reviewer — read the diff end to end before merging).

---

## PHASE 1 — Engineering foundations

Goal: make the codebase safe to change quickly, with a safety net.

### 1.1 Real database migrations
`Flask-Migrate` is a dependency but there's no `migrations/` folder — schema is managed via `db.create_all()`, which can't alter existing tables.
- Run `flask db init`, then `flask db migrate -m "baseline schema"` against a database that reflects the *current* production-equivalent schema, so the baseline migration doesn't try to recreate everything from scratch.
- Verify `flask db upgrade` works cleanly on an empty database and reproduces the current schema exactly.
- From here on, every model change must ship with a generated migration in the same commit.
- **Done when:** `migrations/` is committed, `flask db upgrade` on a fresh DB matches `db.create_all()`'s current output, and `db.create_all()` is removed from `app.py`'s startup path.

### 1.2 Containerize the stack
No `Dockerfile` or `docker-compose.yml` exists anywhere.
- `Dockerfile` for the Flask app (gunicorn, not the dev server, as the entrypoint).
- `Dockerfile` for the FastAPI NLP service (uvicorn).
- `docker-compose.yml` wiring: Flask app, FastAPI NLP service, PostgreSQL, Redis, with environment variables sourced from `.env` (never baked into the image).
- **Done when:** `docker compose up` brings up a working stack from a clean checkout, and the README installation steps are rewritten around this instead of the manual venv walkthrough.

### 1.3 Move the core database to PostgreSQL everywhere
`SQLALCHEMY_DATABASE_URI` defaults to SQLite; only the UMLS terminology DB uses Postgres. SQLite can't handle real multi-user write concurrency.
- Change the default in `.env.example` and `config.py` to point at the same Postgres instance (a separate `hospital_core` database, distinct from `hospital_umls`).
- Confirm every model/query works under Postgres (watch for SQLite-specific assumptions — e.g., autoincrement behavior, date handling).
- **Done when:** the full test suite and a manual smoke test of each department pass against Postgres, and SQLite is no longer referenced as a default anywhere.

### 1.4 Stand up CI
No `.github/workflows` exists.
- Add a GitHub Actions workflow that on every push/PR: installs dependencies, runs `pytest`, runs a linter (`ruff`), runs a dependency/security scan (`pip-audit` and/or `bandit`).
- Make it required to pass before merge (branch protection on `main`).
- **Done when:** a deliberately failing test or lint violation blocks a PR from being mergeable in GitHub's UI.

### 1.5 Break up the god-files
`departments/medicine/routes.py` is 2,417 lines / 115 route functions. `pharmacy/routes.py` (1,577 lines) and `nursing/routes.py` (1,026 lines) have the same problem at smaller scale.
- Split each into focused sub-blueprints or modules by function (e.g., for medicine: `consultations.py`, `prescriptions.py`, `chat_bot.py`, `referrals.py`, `discharge.py`), registered under the existing parent blueprint so URLs don't change.
- Extract business logic (DB queries, validation) out of route handlers into a `services.py` per department where a route currently mixes routing, validation, and persistence in one function.
- **Done when:** no single route file exceeds ~400–500 lines, all existing tests still pass, and no URL changes (verify with a route-listing diff before/after).

### 1.6 Consistent audit logging
The `Log` model is only ever written from `admin/routes.py` and `nursing/routes.py` (plus login/logout). Patient, billing, pharmacy, lab, and imaging CRUD leave no trace of who did what.
- Build a small reusable helper (decorator or SQLAlchemy event listener) that logs create/update/delete on sensitive models (`Patient`, all `*Bill` models, `DispensedDrug`, lab results, imaging records) with user id, timestamp, model, record id, and what changed.
- Apply it consistently across `records`, `billing`, `pharmacy`, `laboratory`, `imaging`.
- **Done when:** a test creates, edits, and deletes a `Patient` and asserts a corresponding `Log` entry exists for each action with the correct user attributed.

### 1.7 MFA, lockout, and password policy
No MFA exists for any role, including admin.
- Add TOTP-based MFA (`pyotp`) as mandatory for `admin` role and optional-but-encouraged for clinical roles; store the TOTP secret encrypted (see 2.5 for the encryption approach — build this after that's in place, or use a placeholder encryption function you'll swap in).
- Enforce a basic password policy on account creation/change (minimum length/complexity) — check `departments/admin/routes.py` where `generate_password_hash` is currently called.
- **Done when:** an admin account cannot complete login without a valid TOTP code, tested with a known secret in the test suite.

### 1.8 Expand the test suite
Currently 3 test functions total, all security-focused, covering 219 routes.
- Add functional tests per department: patient registration (including edge cases — missing national ID, missing blood group, once 2.1 lands), appointment/clinic booking, billing creation, drug dispensing against stock, lab order → result flow, login/logout edge cases.
- Add `pytest-cov`; set a minimum coverage threshold in CI (start realistic — e.g., 40% — and ratchet it up over time rather than demanding 90% immediately).
- **Done when:** CI reports coverage on every run and fails if it drops below the current threshold.

**Phase 1 exit gate:** app runs entirely via `docker compose up`, migrations are the only way schema changes happen, CI is green and required, no route file is a god-file, and every sensitive-model mutation is audited.

---

## PHASE 2 — Data model & compliance

Goal: make the data model match how hospitals and Kenyan health financing actually work.

### 2.1 Redesign the `Patient` model
`national_id` is `unique=True, nullable=False` and `blood_group` is required at registration — both break real intake (newborns, unconscious patients, undocumented persons, anyone who doesn't know their blood type yet).
- Make `national_id` optional; add a separate `identifier_type` + `identifier_value` structure supporting alternates (temporary/emergency ID, birth notification number, passport, refugee ID).
- Make `blood_group` optional and editable post-registration.
- Add `created_by`, `updated_by`, `created_at`, `updated_at`, `is_active`/`deleted_at` (soft delete — never hard-delete a patient record).
- Add a `merge_candidates` or duplicate-detection query (fuzzy name + DOB + phone matching) surfaced to registration staff, plus a `PatientMerge` audit table recording any merge performed and by whom.
- **DECISION NEEDED (flag, don't decide):** exact duplicate-matching threshold/algorithm, and who is authorized to perform a merge.
- **Done when:** registering a patient with no national ID and no blood group succeeds, a soft-deleted patient is excluded from normal queries but recoverable, and a test demonstrates duplicate detection flagging two similar records.

### 2.2 Unify billing
Six parallel tables (`DrugsBill`, `WardBill`, `LabBill`, `ClinicBill`, `TheatreBill`, `ImagingBill`) each duplicate a free-text `payment_method` column.
- Design a unified `Invoice` (one per billable encounter) + `InvoiceLineItem` (one per department charge, with a `source_type`/`source_id` reference back to the originating department record) + `Payment` (method, amount, reference, timestamp) model.
- Write a migration that backfills the new tables from the existing six, without deleting the old tables until the new path is verified in production use (mark them deprecated, remove in a later phase).
- **Done when:** a single patient's full financial history across drugs/ward/lab/clinic/theatre/imaging can be pulled from one query, and old and new totals reconcile exactly on the migrated data.

### 2.3 Build the SHA/SHIF insurance module
The only reference to insurance anywhere in the codebase is a placeholder string in an HTML form. Kenya's National Hospital Insurance Fund was fully replaced by the Social Health Authority (operating three funds — Primary Healthcare, Social Health Insurance, and Emergency/Chronic/Critical Illness) as of October 2024.
- Add an `InsuranceScheme` model (SHA/SHIF, private insurers like AAR, etc.) and a `PatientInsurance` model (member number, scheme, verification status, valid-from/to).
- Add a `Claim` model linked to `Invoice`, with status tracking (draft → submitted → approved/rejected/paid) and a claims list/detail view for billing staff.
- **DECISION NEEDED (flag, don't decide):** whether to integrate a real SHA eligibility/claims API (if one is available to this facility) or build a manual claims-tracking workflow first with API integration as a later phase — this affects scope significantly, ask before building.
- **Done when:** a claim can be created against an invoice, its status updated, and a report of pending/approved/rejected claims can be generated.

### 2.4 Integrate M-Pesa (Safaricom Daraja API)
"M-Pesa" currently exists only as a free-text string value in `payment_method`.
- Implement STK Push initiation from the billing/payment screen and a callback endpoint to receive payment confirmation, updating the `Payment` record automatically on success.
- Handle failure/timeout states explicitly (don't leave a `Payment` in limbo if the callback never arrives — add a reconciliation job that queries Daraja's transaction status API for anything pending past a timeout).
- **Done when:** a test (against Daraja's sandbox) completes a full STK push → callback → payment-recorded cycle, and a simulated timeout is correctly reconciled.

### 2.5 Encryption at rest for identity fields
No field-level encryption exists anywhere; `national_id` and similar fields sit in plaintext.
- Add a `TypeDecorator` (SQLAlchemy) wrapping `cryptography.fernet.Fernet`, applied to `national_id`, TOTP secrets (from 1.7), and any other direct identifiers.
- Key management: load the encryption key from environment/secrets manager, never from source; document a key-rotation procedure even if not automated yet.
- **Done when:** inspecting the raw database column shows ciphertext, not plaintext, and the application layer transparently encrypts/decrypts.

### 2.6 Data Protection Act 2019 compliance basics
- Add a `consent` model/table recording what a patient has consented to (data sharing, AI-assisted analysis of their notes, research use) with timestamps — required before 3.3's AI governance work can be considered compliant.
- Add an admin-facing data export and erasure-request workflow (subject access request handling), even if erasure is implemented as anonymization rather than deletion for records with legal retention requirements.
- Document a data retention policy (how long each record type is kept) as a markdown doc reviewed by the human — this is a policy decision, not something to invent in code.
- **Done when:** a consent record can be attached to a patient and checked before any AI-processing action (see Phase 3), and a data export produces a complete, human-readable dump of one patient's records.

**Phase 2 exit gate:** patient registration handles real-world intake scenarios, one patient has one coherent bill, SHA/SHIF and M-Pesa are real integrations (or clearly-scoped manual workflows per the decision above) rather than text fields, and identity data is encrypted at rest.

---

## PHASE 3 — Interoperability & AI governance

Goal: make the system talk to the outside world safely, and make the AI layer trustworthy rather than just impressive.

### 3.1 FHIR interoperability layer
No HL7/FHIR support exists.
- Add a FHIR-compliant API surface (start with `Patient`, `Encounter`, `Observation`, `MedicationRequest` resources) as an adapter layer over the existing models, rather than rewriting the internal schema to be FHIR-native.
- **Done when:** a `Patient` resource can be fetched via a standard FHIR client and validates against the FHIR spec.

### 3.2 DHIS2/KHIS reporting export
No export path to Kenya's national health information system exists.
- Build a scheduled job producing the standard aggregate indicators DHIS2/KHIS expects (confirm the exact indicator set and format with the human/facility — this is a **DECISION NEEDED** item, not something to guess).
- **Done when:** a sample export file validates against DHIS2's import format.

### 3.3 Govern the AI/NLP layer
The clinical NLP service (UMLS + PubMedBERT + NVIDIA/Gemini LLM calls on SOAP notes) is technically sophisticated but has no consent gate, no human-in-the-loop review, and no audit of its suggestions.
- Gate any call that sends patient clinical text to an external API (NVIDIA, Gemini) behind the consent record from 2.6 — refuse the call and degrade gracefully if consent isn't recorded.
- Add a review step: AI-suggested disease/symptom mappings or note analysis must be presented to a clinician as a *suggestion* requiring explicit acknowledgment before being saved as part of the clinical record, and both the suggestion and the clinician's action (accepted/edited/rejected) get logged.
- Add graceful degradation for when NVIDIA/Gemini APIs are unreachable (timeout, rate limit, outage) — the core clinical workflow must not block on an external AI call.
- **Done when:** a test confirms an AI call is blocked absent consent, and another confirms the core note-taking flow completes successfully even when the external API call is mocked to fail.

### 3.4 Offline-first considerations
Traditional server-rendered Bootstrap/jQuery UI, no offline capability — relevant given inconsistent connectivity/power outside major towns.
- Scope this as a discovery task first: identify which workflows (e.g., triage, basic vitals recording) most need to survive a connectivity drop, before committing to a PWA/service-worker rebuild of the whole UI.
- **Done when:** a written recommendation exists (this may be the deliverable for this task, not code) on scope and approach, reviewed by the human before implementation begins.

**Phase 3 exit gate:** the system can exchange data externally in a standard format, and the AI layer has consent, human review, and audit wrapped around it — not just model quality.

---

## PHASE 4 — Operating like a real product

### 4.1 Monitoring & backups
- Add health-check endpoints, structured logging (confirm nothing at DEBUG level logs patient-identifying data — audit `logger.debug` calls across the codebase), and basic uptime/error alerting.
- Write and actually **test** a backup/restore runbook — a backup nobody has restored from is not a backup.

### 4.2 Documentation
- Replace the generic README with real setup (Docker-based, per 1.2), architecture overview, and an OpenAPI spec for the API surface.
- Document the two-service (Flask + FastAPI) deployment topology explicitly.

### 4.3 Team/process
- This was a single-contributor project with long inactive gaps — flag to the human (this is a people/process recommendation, not a code task) that a second reviewer materially reduces risk for software handling patient data, independent of anything else in this plan.

**Phase 4 exit gate:** this is the point where "ready for a pilot with real patients, with human clinical and compliance sign-off" becomes a reasonable claim — not before.

---

## Final note on go-live

Do not treat completion of any phase, including Phase 4, as authorization to deploy against real patient data. That decision belongs to the human running this project, ideally with clinical and legal/compliance review, not to whoever is running Claude Code.
