# Hardening plan: verified gaps follow-up for Claude Code

Paste everything below into Claude Code, run from inside a clone of `mathunjoroge/hospital`. This is a follow-up to the "Hospital HMIS — Hardening & Modernization" mission briefing — `IMPLEMENTATION_PROGRESS.md` marks all of Phase 0–4 complete, but direct verification found four specific items that don't actually meet their own documented "Done when" criteria. Fix these four only. Don't re-touch anything else on the tracker without new evidence it's actually broken.

Before starting: run `git status` and `git log --oneline -5`. If there's uncommitted work sitting locally that isn't reflected on `origin/main`, tell me before doing anything else — some of what's below may already be fixed in an uncommitted state.

---

## Gap 1 — `SECRET_KEY` doesn't fail loudly outside `FLASK_ENV=='production'`

Current code in `app.py`:
```python
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    if <FLASK_ENV == 'production'>:
        raise RuntimeError("SECRET_KEY environment variable must be set in production mode.")
    secret_key = 'dev-secret-key-for-local-development-only'
    logging.warning("SECRET_KEY environment variable not found; using development fallback key.")
```
The original spec requires this to fail in **every** environment except an explicit `FLASK_ENV=testing`, not just `production`. Fix it to:
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
Update or add a test asserting this: starting the app with `SECRET_KEY` unset and `FLASK_ENV` unset (or set to anything other than `testing`) raises `RuntimeError`. The existing `tests/test_secret_key.py` may already partially cover this — check whether it currently only tests the `production` case, and broaden it to cover the default/unset case too, since that's the actual gap.

## Gap 2 — `db.create_all()` still runs despite real migrations existing

`app.py:378` still calls `db.create_all()` even though `migrations/` (Flask-Migrate) exists. Remove it from the startup path. Before removing:
1. Confirm `flask db upgrade` against a fresh, empty database produces a schema that matches what `db.create_all()` currently produces (diff the resulting schemas, don't just assume).
2. If they don't match, generate a corrective migration to close the gap — don't just delete the call and hope migrations already cover everything.
3. Remove the `db.create_all()` call, verify the app still boots and the full test suite still passes using the CI setup steps (Postgres/Redis, migrated fresh DB, not `db.create_all()`).

## Gap 3 — `EncryptedString` still isn't applied to any real column

The type decorator and its tests are solid, but `national_id` in `departments/models/records.py` (and the TOTP secret column added for MFA, wherever that lives) are still plain unencrypted columns. Apply it for real:
1. Change `national_id`'s column type to `EncryptedString` and do the same for the MFA/TOTP secret column.
2. Write a migration that encrypts existing plaintext values in place (read each row, encrypt, write back) — this touches real patient data, so per the original ground rules, this must be additive/reversible: back up the table first, and make the migration idempotent (safe to re-run without double-encrypting an already-encrypted value — check for the `enc_v1:` prefix before encrypting).
3. Verify directly against the raw database (not through the ORM) that these columns now contain ciphertext, not plaintext.
4. Confirm every code path that reads/writes `national_id` (patient registration, patient search/display, any report or export that includes it) still works correctly with the value transparently decrypted — check `departments/records/routes.py` and anywhere else `national_id` is referenced directly.

## Gap 4 — AI consent gate exists as a model, not as an enforcement point

`PatientConsent` (in `departments/models/compliance.py`) is a real, correctly-designed model, but nothing in `departments/medicine/chat_bot.py`, `departments/nlp/src/nvidia_client.py`, or `departments/medicine/oncology.py` actually checks it before sending patient clinical text to NVIDIA/Gemini.

1. Add a helper, e.g. `has_ai_consent(patient_id)` in `departments/models/compliance.py`, checking for a `PatientConsent` row with `consent_type='ai_diagnosis'` and `is_granted=True` and no `revoked_at`.
2. Call this check at the actual point each external AI call is made — not just at a route's entry point, since the goal is that the external API call itself never fires without consent. Identify every call site (chat_bot's summarization call, oncology's AI-assisted features, any other `nvidia_client`/Gemini call touching patient-specific data) and gate each one.
3. Define what happens when consent is missing: the spec says "refuse the call and degrade gracefully" — return a clear message to the clinician (e.g., "AI-assisted summary unavailable: patient has not consented to AI processing of clinical notes") rather than a generic error, and don't silently fall back to running the AI call anyway.
4. Log both the check and its outcome (granted/refused) via the existing audit logging system, so there's a record of every time this gate was evaluated, not just when it blocks something.
5. Add a test: mock a patient with no consent record, assert the AI call is refused and the core note-taking/consultation flow still completes without error; a second test with consent granted asserts the AI call proceeds normally.

## Minor cleanup while you're in there

- Two split route files still exceed the ~400-500 line target from the god-file task: `departments/pharmacy/dispensing.py` (739 lines) and `departments/medicine/prescriptions.py` (633 lines). If there's a natural seam (e.g., dispensing's stock-adjustment logic vs. its dispensing-workflow logic), split further — but only if a real seam exists; don't force an arbitrary split.
- `.github/workflows/ci.yml`'s `pip-audit` step has `|| true`, meaning it can never fail the build. Either remove the `|| true` and add specific `--ignore-vuln` entries for anything currently flagged that's a genuine false positive or accepted risk (with a one-line comment explaining why each is ignored), or leave it soft-passing but rename the CI step to something honest like "Dependency audit (report-only)" so it's not misread as a real gate.

## Report back

For each of the 4 gaps: confirm the fix with the specific test that proves it, and quote the actual raw-DB or raw-request evidence where relevant (e.g., the ciphertext you saw in the database for Gap 3, the exact 401/refusal response for Gap 4). Don't report "done" from the test passing alone if the underlying behavior can be checked more directly — the whole reason this follow-up exists is that "tests passing" and "the feature actually working" turned out to be different things last time.
