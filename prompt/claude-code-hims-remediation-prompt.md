# HIMS Remediation Prompt (for Claude Code)

## Context

This is the Flask-based Hospital Management System (HIMS) at
`mathunjoroge/hospital`. A code audit (done against commit
`1528ef797916a65d7e350f2c0880dbefc68635ef`, 2026-09-08) verified several
concrete, reproducible bugs in the recently-shipped "world-class" UI modules
and the unified billing sync layer. This prompt gives you the verified
findings and an ordered task list to fix them.

**First step: confirm your working tree.** Run `git log -1 --oneline` and
`git status`. If your HEAD differs from the commit above, or you have
uncommitted local changes, treat the file/line references below as
*pointers*, not exact coordinates — locate the actual code by the function
names and route paths given, not by line number.

Work on a feature branch (e.g. `fix/worldclass-ui-audit`). Make one commit
per numbered task below so each change is independently reviewable. Do not
touch the database schema or run migrations — every fix below is
application/template code only, except where explicitly noted.

---

## Priority 1 — Live Queue dashboard shows wrong data to clinical staff

**Root cause:** `GET /appointments/api/queue/<provider_id>`
(`departments/appointments/routes.py`, function `get_live_queue`) returns
`appointment_id`, `patient_id`, `checked_in_at`, `type` — but
`departments/ui_dashboard/templates/dashboard/queue.html`'s Alpine.js reads
`patient.name`, `patient.check_in_time`, `patient.wait_time_mins`, none of
which exist in the payload. Because every binding has a `||` fallback, this
fails silently: every patient shows "Unknown Patient" and a permanent
`0 mins` wait badge, so the red/yellow/green urgency color-coding never
fires. The stat cards (`total_scheduled`, `in_consultation`) are the same
problem — those keys are never in the response either.

**Fix:**
1. In `get_live_queue`, join `departments.models.records.Patient` (via
   `Appointment.patient_id` → `Patient.id`, an internal int FK — confirm by
   checking how `patient_id` is populated in `ScheduleEngine.book_appointment`)
   to add a `patient_name` field per queue item (fall back to
   `f"Patient #{a.patient_id}"` if the Patient row is missing — there's no
   DB-level FK constraint enforcing referential integrity here).
2. Compute `wait_time_mins = int((datetime.now(timezone.utc) - a.updated_at).total_seconds() / 60)`
   and a formatted `check_in_time` (`a.updated_at.strftime("%H:%M")`) per
   item.
3. Add `total_scheduled` (today's full schedule count for the provider —
   reuse `ScheduleEngine.get_provider_schedule`) and `in_consultation`
   (count of that provider's appointments with `status == "IN_PROGRESS"`)
   to the top-level response.
4. **Keep the existing keys** (`appointment_id`, `patient_id`,
   `checked_in_at`, `type`, `waiting_count`, `provider_id`) — don't rename
   or remove them. `tests/test_worldclass_wiring.py::test_book_checkin_and_live_queue`
   asserts on `body["waiting_count"]` and `body["queue"][0]["patient_id"]`
   and must keep passing.
5. Update `queue.html`'s Alpine bindings to read the new field names
   (`patient.patient_name`, `patient.wait_time_mins`, `patient.check_in_time`),
   and bind `totalScheduled`/`inConsultation` directly from the response
   instead of the `|| this.queue.length` / `|| 0` fallbacks that are
   currently masking the missing data.

## Priority 2 — "Call In" button is a non-functional stub

`queue.html`'s `callPatient()` currently does `alert(...)` with a comment
admitting it's not implemented.

**Fix:**
1. Add `Appointment.start_consultation()` to `departments/appointments/models.py`
   (sets `status = "IN_PROGRESS"`), mirroring the existing `check_in()` /
   `mark_no_show()` pattern.
2. Add `ScheduleEngine.call_in(appointment_id)` to
   `departments/appointments/engine.py`, mirroring `mark_no_show()`: look up
   the appointment, require `status == "CHECKED_IN"` (log + return `None`
   otherwise), call `start_consultation()`, commit, return the appointment.
3. Add `POST /appointments/api/call-in/<string:appointment_id>` to
   `routes.py`, decorated the same way as the neighboring routes
   (`@login_required`, `@roles_required("doctor", "nursing")`), returning
   404 if the engine call returns `None`, else 200 with the new status.
4. Update `callPatient()` in `queue.html` to `fetch` that endpoint (POST,
   with the CSRF header — see Priority 3) and, on success, call
   `this.fetchQueue()` to refresh the list instead of showing an alert.

**Do not delete `get_live_queue_rows`** (the HTMX/Tailwind rows endpoint at
the bottom of `routes.py`). It looks orphaned from the current Bootstrap
template's perspective, but `test_book_checkin_and_live_queue` calls
`/appointments/api/queue/1/rows` directly and asserts on its output —
deleting it will break that test. Optional, low-priority cleanup while
you're in this file: that endpoint's "Start Consultation" button currently
does `hx-post="/appointments/api/check-in/{appt.id}"`, which will always
fail with a 400 for anyone already `CHECKED_IN` (check-in requires
`SCHEDULED`). If you add the `/api/call-in/<id>` route in step 3 above,
point that button at it instead — but confirm the test's assertions
(`f"Patient #{patient_pk}"` must still appear, "Queue is empty" must not)
still hold before committing that change.

## Priority 3 — CSRF tokens missing on every write action in the new UI modules

`CSRFProtect` is globally enabled in `app.py`. None of these `fetch()` POST
calls send an `X-CSRFToken` header, and none of the corresponding
blueprints are `csrf.exempt`'d, so as shipped these all 400 in a real
browser session:

- `departments/ui_billing/templates/billing/claims.html` → `POST /rcm/api/claim/scrub`
- `departments/ui_clinical/templates/clinical/prescribe.html` → `POST /clinical-safety/api/check`
- `departments/ui_mch/templates/mch/workbench.html` → `POST /mch/api/immunize`, `POST /mch/api/anc-visit`
- `departments/ui_referrals/templates/referrals/handover.html` → `POST /referrals/api/initiate`, `POST /referrals/api/discharge`
- `departments/ui_dashboard/templates/dashboard/queue.html` → the new call-in POST from Priority 2

**Fix:** `templates/base.html` already emits
`<meta name="csrf-token" content="{{ csrf_token() }}">`. Add
`'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').content`
to the `headers` object of every POST/PUT/PATCH/DELETE `fetch()` call
listed above.

Before committing, do a full sweep rather than trusting this list is
exhaustive — run:
```
grep -rn "method:\s*['\"]POST\|method:\s*['\"]PUT\|method:\s*['\"]PATCH\|method:\s*['\"]DELETE" departments/*/templates/
```
and fix any other instance you find that isn't on the list above.

## Priority 4 — Invoice docstring overstates what the sync layer actually does

`departments/models/billing.py`, class `Invoice`, docstring says: *"Single
unified invoice per patient encounter."* There is no `Encounter`/`Visit`
model anywhere in the codebase (`grep -rn "class Encounter" .` returns
nothing), and `get_or_create_open_invoice()` in `departments/billing/sync.py`
keys strictly on `patient_id` — so in practice one `DRAFT` invoice
accumulates charges across *all* of a patient's visits until it's fully
paid off, not one invoice per encounter.

**Fix:** Rewrite the docstring to describe actual behavior accurately (one
open running invoice per patient, closed to `PAID`/`PARTIAL` on payment,
new charges after that go to a new invoice), and add a short note pointing
at the deferred Encounter model as the reason per-visit invoicing isn't
possible yet. Do not attempt to add an `encounter_id` column or build the
Encounter model itself as part of this pass — that's a larger, separate
design decision; just make the documentation honest.

## Priority 5 (flag, don't silently fix) — billing sync is update- and delete-blind

In `departments/billing/sync.py`, `sync_charge()`'s idempotency check
(`if existing: return existing`) means corrections to a legacy charge's
amount/quantity after the fact never propagate to the linked
`InvoiceLineItem`. And `event_listeners.py`'s `sync_billing_events` only
inspects `session.new` and `session.dirty` — `session.deleted` is never
handled, so a deleted/reversed legacy bill row leaves an orphaned
`InvoiceLineItem` on the unified invoice forever.

This is a genuine data-integrity gap, but fixing it properly needs a
product decision (e.g., should a correction retroactively adjust the
invoice, or should it require a manual credit/adjustment line for audit
purposes — hospitals often want the latter for compliance reasons). **Do
not silently rewrite this logic.** Instead:
1. Add a clear code comment directly above `sync_charge()` and in
   `sync_billing_events()` documenting this known limitation.
2. Add an entry to `DECISIONS_PENDING.md` in the same style as the existing
   entries, describing the two options above and asking which one the
   hospital wants.

## Priority 6 (quick hygiene, do last)

1. Add `.coverage` to `.gitignore` and `git rm --cached .coverage` — it's a
   binary artifact that shouldn't be tracked, and CI already uploads it as
   a build artifact (`.github/workflows/ci.yml`).
2. `grep -rn "unpkg.com/alpinejs" departments/` — every "world-class"
   template independently loads Alpine from `@3.x.x` (a floating major
   version, no Subresource Integrity hash) via its own `<script>` tag. Move
   this single `<script>` tag into `templates/base.html` (pin an exact
   version and add an `integrity` + `crossorigin` attribute — get the hash
   from unpkg or jsDelivr's SRI generator), and remove the per-template
   copies so Alpine loads once, verified, for every page.

---

## Verification (run after each numbered section, not just at the end)

```
pytest tests/test_worldclass_wiring.py -v
```
This must stay green throughout — it's your regression guard for
Priorities 1–3. In particular watch `test_book_checkin_and_live_queue`.

If you touch `sync.py` or `billing.py` for Priority 4/5:
```
pytest tests/test_billing.py tests/test_unified_billing.py -v
```

Before declaring done, run the full suite and report the pass count you
get (it should be at or above the 255 baseline — call out clearly if
anything regresses and why):
```
pytest -q
```

And confirm the CSRF sweep is complete:
```
grep -rln "fetch(" departments/*/templates/ | xargs grep -L "X-CSRFToken"
```
(any file this prints that also contains a POST/PUT/PATCH/DELETE fetch is
something you missed.)

## What to report back when done

A short summary per priority: what changed, which files, and the
before/after test result. Explicitly call out anything from Priority 5 or
6 you decided to defer or handle differently, and why — I'd rather know
your reasoning than have you guess silently on the financial-logic pieces.
