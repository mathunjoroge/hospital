# Consolidated fixes for Claude Code

Paste everything below into Claude Code, run from inside a clone of `mathunjoroge/hospital`. This combines two things that didn't land from previous prompts: four security/compliance gaps that were reported as fixed but verified unchanged in the actual code, and Phase B's SMS channel, which is currently a sandbox stub rather than a real integration.

**Before doing anything else**, run `git status` and `git log --oneline -5` and tell me what you see. The last four items below were previously reported complete but direct inspection of `main` shows none of them were actually changed — if there's uncommitted local work that explains the discrepancy, say so before proceeding. If `main` genuinely reflects the current state, proceed with the tasks below.

**Ground rule for this pass, given the history above**: for each task, before making any change, paste the *current* exact lines you're about to change (via `grep`/`sed -n`, not from memory or a prior report). After making the change, paste the *new* exact lines. Your final report must include both, for every task — "tests pass" alone is not sufficient evidence given what happened last time.

---

## Task 1 — Fix `SECRET_KEY` enforcement (verified still broken)

Current code in `app.py` (confirmed present as of this writing):
```python
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    if os.environ.get('FLASK_ENV') == 'production':
        raise RuntimeError("SECRET_KEY environment variable must be set in production mode.")
    secret_key = 'dev-secret-key-for-local-development-only'
    logging.warning("SECRET_KEY environment variable not found; using development fallback key.")
```
This only fails when `FLASK_ENV` is exactly `'production'` — every other value (unset, `'staging'`, a typo) silently falls back to a hardcoded key that is now public in this repo's git history. Replace it with:
```python
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    if os.environ.get('FLASK_ENV') == 'testing':
        secret_key = 'test-secret-key-not-for-production'
    else:
        raise RuntimeError(
            "SECRET_KEY environment variable must be set (FLASK_ENV=testing is the only exception)."
        )
```
Update `tests/test_secret_key.py` to assert the app raises `RuntimeError` when `SECRET_KEY` is unset and `FLASK_ENV` is unset (not just when `FLASK_ENV='production'`, which is likely what it currently tests, given the gap). Run this specific test and paste its output.

## Task 2 — Remove `db.create_all()` (verified still present)

`app.py` line ~382 still calls `db.create_all()` despite `migrations/` existing. Before removing it:
1. Confirm `flask db upgrade` against a fresh empty database produces a schema matching what `db.create_all()` currently produces. If they diverge, generate a corrective migration first — don't remove the call and hope.
2. Remove `db.create_all()` from the startup path.
3. Boot the app against a freshly-migrated database (not `db.create_all()`) and run the full test suite. Paste the pass/fail summary.

## Task 3 — Apply `EncryptedString` to `national_id` (verified still plain text)

`departments/models/records.py` still has:
```python
national_id = db.Column(db.String(50), unique=True, nullable=True, index=True)
```
1. Change this to use `EncryptedString` (the type decorator already exists and is tested — use it, don't reinvent it).
2. Write a migration that encrypts existing plaintext values in place. This touches real patient data: back up the table first (note the exact backup command you ran), and make the migration idempotent — check for the `EncryptedString` module's version prefix before re-encrypting a value, so re-running the migration doesn't double-encrypt.
3. After migrating, query the raw `national_id` column directly via `psql` (not through the ORM/ SQLAlchemy) and paste what an actual row looks like — this is the proof that matters, not a passing test.
4. Confirm patient registration, patient search, and any report/export referencing `national_id` (check `departments/records/routes.py` and anywhere else it's read directly) still work with the value transparently decrypted.

## Task 4 — Wire the AI consent gate into real call sites (verified not implemented anywhere)

`has_ai_consent()` doesn't exist in the codebase at all — the `PatientConsent` model exists but nothing checks it before an AI call fires.
1. Add `has_ai_consent(patient_id)` to `departments/models/compliance.py` — `True` only if a `PatientConsent` row exists with `consent_type='ai_diagnosis'`, `is_granted=True`, and no `revoked_at`.
2. Find every call site that sends patient-specific clinical text to NVIDIA or Gemini — `departments/medicine/chat_bot.py`, `departments/nlp/src/nvidia_client.py`, `departments/medicine/oncology.py`, and anywhere else — and call the check immediately before each external API call, not just at the route entry point.
3. When consent is missing: return a clear message to the clinician (e.g., "AI-assisted summary unavailable: patient has not consented to AI processing of clinical notes"). Do not fall back to running the call anyway.
4. Log both the check and its outcome through the existing audit-logging system.
5. Add a test per call site: no consent → call is refused and the surrounding workflow (note-taking, consultation) still completes without error; consent granted → call proceeds. Run these and paste output, plus confirm with a direct `curl`/manual request that a request to one of these endpoints for a non-consenting patient does not reach the external API (check outbound request logs or mock the HTTP layer to prove the call never fires).

## Task 5 — Give Phase B a real SMS channel

`departments/notifications/dispatcher.py` currently only has `SandboxSMSChannel` — nothing reaches a real phone. **Do not pick a provider yourself.** Ask the human which of these to use before writing any provider-specific code:
- **Africa's Talking** — typical choice for Kenya-based deployments, local SMS rates.
- **Twilio** — more globally standard, higher per-SMS cost for Kenyan numbers.
- **Defer SMS entirely, ship email-only for now** — `Flask-Mail` already exists in the app; this avoids any new recurring cost until it's confirmed to be in budget.

If you reach this task and haven't received an answer, stop here and report back what's blocking — do not default to any of the three options, including the "defer" option, without an explicit answer. (This is the same instruction the original roadmap gave for this decision; it was skipped last time, so treat this repetition as the point where it actually needs to hold.)

Once you have an answer:
1. Implement a new channel class alongside `SandboxSMSChannel` implementing the same interface (e.g. `AfricasTalkingSMSChannel` or `TwilioSMSChannel`), reading credentials from environment variables — never hardcoded.
2. Make the active channel configurable via an environment variable (e.g. `SMS_CHANNEL=sandbox|africastalking|twilio`), defaulting to `sandbox` in test/dev environments so CI doesn't require real credentials or send real messages.
3. Send one real test message to a real phone number in the sandbox/trial mode of whichever provider was chosen, and paste the provider's actual delivery confirmation (message ID, status) as evidence — not just a test asserting the function was called.
4. Confirm `OutboundNotificationLog` records the real delivery attempt and outcome (including a simulated failure case — bad number, provider timeout) correctly.

## Report back

For each of the 5 tasks: the before/after code or config, the specific command you ran to verify it (not just "pytest passed"), and its actual output. If any task is blocked (especially Task 5 without an SMS-provider answer), say so explicitly rather than marking it done with a placeholder.
