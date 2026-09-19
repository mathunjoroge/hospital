# Laboratory Department — Fix Report

**Date:** 2026-09-19
**Scope:** `departments/laboratory/` (tests.py, results.py, panic_alerts.py, lims_routes.py, lims_service.py, reagents.py), `departments/rbac.py`, sidebar navigation, templates
**Related:** `report.md` (pharmacy gap analysis, all fixes applied)

**Test status:** 88 tests passing — 40 pre-existing lab tests (unchanged), 13 new regression tests (`tests/test_lab_gap_fixes.py`), plus security, billing-listener, and pharmacy regression suites re-verified green.

---

## P0 — Patient-safety / Security fixes

### 1. Queue "Process Test" opened the wrong request (cross-patient risk)

**Problem:** The unified patient-flow queue returns **Encounter** objects, but the worklist template linked with
`url_for('laboratory.process_lab_request', request_id=request.lab_test.id if request.lab_test else request.id)`.
Encounters have no `lab_test`/`test_name` attributes, so every queue row rendered as "Lab Investigation" and the
link passed the **Encounter id** into `process_lab_request`, which resolves `RequestedLab.query.get_or_404(encounter_id)`
— opening a *different request's* lab form (or 404).

**Fix** (`templates/laboratory/index.html`):
- The worklist table now iterates only real `RequestedLab` rows (`pending_lab_requests`) and links with `request.id` (the RequestedLab PK).
- Queue encounters (AWAITING_LAB stage) render as an informational banner pointing to the Live Queue, with a note that individual test orders appear once requested.

### 2. 2-tier verification was self-verifiable

**Problem:** `/laboratory/lis/verify` allowed role `lab_tech` — the same technician who entered the result (Tier 1)
could also verify it (Tier 2). Nothing enforced `verifier_id != updated_by`, defeating the P0-06 workflow.

**Fix** (`panic_alerts.py`):
- Verify endpoint returns **403 `SELF_VERIFICATION_BLOCKED`** when `lab_res.updated_by == verifier_id`.
- Exemption only for admin **outside** role-switch (audited supervisor override).
- A forged `verifier_id` in the request body cannot bypass the check — identity is always taken from the session (regression-tested).

### 3. Panic alerts went to the wrong person

**Problem:** The critical-value notification was sent to `receiver_id=verifier_id` — the person who *just verified*
the result and therefore already knows about it. The ordering clinician never got the alert; a critical K⁺ of 6.2 mmol/L could sit unnoticed.

**Fix** (`panic_alerts.py`):
- New `_resolve_ordering_clinician_id()` resolves the clinician from the patient's ACTIVE encounter (`provider_id`).
- On `PANIC_CRITICAL`, the alert is sent to the **ordering clinician** with a copy to the verifier (documentation/closure), de-duplicated.
- API response now includes `panic_alert_recipient_id` for auditability.

### 4. LIS result entry fabricated clinical values via silent defaults

**Problem:** `/laboratory/lis/enter` defaulted `lab_test_id=1`, `parameter_name="Hemoglobin"`, `result_value=14.0`.
A malformed request that omitted `result_value` silently recorded a **normal hemoglobin of 14.0** as if measured.

**Fix** (`panic_alerts.py`):
- `patient_id`, `lab_test_id`, `parameter_name`, `result_value` are now all **required**; missing fields return 400 with an explicit "no defaults are applied to clinical data" message.
- The lab test id is validated against the catalog — no dangling clinical results.

### 5. Role split-brain between web UI and LIS API

**Problem:** Web routes required `roles_required("laboratory", "admin")`; the LIS API required
`("lab_tech", "radiology", "admin")`. A `laboratory`-role user got 403 on the LIS API; a `lab_tech` got 403 on every web page. `ROLE_ALIASES` mapped neither.

**Fix:**
- `departments/rbac.py`: added aliases `laboratory ⇄ lab_tech` (alongside the earlier pharmacy/stores aliases).
- One lab account now works across web UI, LIS API, verification queue, and LIMS routes.

---

## P1 — Workflow fixes

### 6. Web result entry bypassed the safety engine; no verification UI

**Problem:** `process_lab_request` never called `evaluate_panic_level()`, never set `status`/`panic_status`, so
web-entered results were always `panic_status="NORMAL"` — and there was no UI anywhere to verify results
(only a JSON API). The 2-tier workflow was unreachable from the web.

**Fix:**
- `results.py`: every entered value is scored through the panic engine; `status="PENDING_VERIFICATION"`,
  `panic_status`, and `panic_message` are persisted. Flash messages differentiate CRITICAL / ABNORMAL / normal saves.
- New **Verification Queue** page: `GET /laboratory/verification_queue` + `templates/laboratory/verification_queue.html`
  — lists pending results oldest-first with panic badges and one-click Verify/Reject buttons (calls the LIS verify API with CSRF token). Criticals render with danger-row highlighting.

### 7. Non-atomic double-commit and duplicate results

**Problem:** `LabResult` was committed, then `lab_request.status=1` committed separately — a failure between them
left a result attached to a still-pending request; a retry then created duplicate results. No double-click guard existed.

**Fix** (`results.py`):
- Result creation and request completion now commit **atomically**.
- Idempotency guard: if `status==1` and `result_id` already links to an existing LabResult, the route redirects to it with an info flash instead of creating a duplicate (regression-tested).

### 8. Specimen state machine dead-ended

**Problem:** LIMS routes only covered COLLECTED / RECEIVED / REJECTED. `IN_ANALYSIS`, `COMPLETED`, `DISPOSED`
were counted on the dashboard but never set; result entry never touched specimens.

**Fix:**
- `lims_routes.py`: new endpoints `POST /api/lims/specimens/start-analysis`, `/complete`, `/dispose` — all CoC-logged via `LIMSService`.
- `results.py`: new `_advance_specimens_for_request()` bridge — result entry auto-advances linked specimens
  (uncollected → `IN_ANALYSIS`, received/collected → `COMPLETED`) with a chain-of-custody note referencing the result id.

### 9. LIMS dashboard unreachable

**Problem:** `lims_dashboard` (specimen tracking + Westgard QC UI) had no sidebar link; the sidebar's
"LIMS & QC Dashboard" pointed at the old 3-counter `dashboard()`.

**Fix** (`templates/side_bars/laboratory.html`): sidebar now has both — "Specimen Tracking & QC" → `lims_dashboard`,
"Lab Statistics" → `dashboard`, plus the new "Verification Queue" entry.

### 10. `abnormal_results` full-table O(N) scan

**Problem:** The route loaded *every* LabResult ever, JSON-parsed each in Python per page view, with no date filter or pagination.

**Fix** (`results.py`):
- Primary source: SQL filter on `panic_status IN ('PANIC_CRITICAL','ABNORMAL')` over a 90-day window (limit 200).
- Legacy rows (never evaluated) covered by a bounded 500-row fallback scan in the same window; per-parameter breakdown is computed only for the bounded result set.

### 11. Template deletion corrupted historical results

**Problem:** `edit_lab_test` physically deleted a `LabResultTemplate` when its name field was emptied — but stored
results are keyed by template id, and `view_lab_results` iterates only *current* templates, so deletion silently
dropped parameters from every historical result.

**Fix** (`tests.py`): deletion is blocked with a clear error whenever any LabResult exists for the parent test —
the parameter must be retained for historical integrity (regression-tested).

### 12. `delete_lab_test` hard-delete crash

**Problem:** Deleting a test referenced by RequestedLab/LabResult rows failed on FK integrity and surfaced as a
generic "Something went wrong."

**Fix** (`tests.py`): the route now counts references and blocks with an explicit message
("referenced by N request(s) and M result(s)") — unused tests still delete cleanly (both paths regression-tested).

---

## P2 — Polish fixes

### 13. Dashboard metrics

- **Abnormal count**: replaced `result.ilike("%abnormal%")` (which only matched if the literal word "abnormal" appeared in the JSON blob) with `panic_status IN ('ABNORMAL','PANIC_CRITICAL')`; added a critical-only counter.
- **Turnaround time**: new "Avg Turnaround (30d)" card — average hours from `RequestedLab.date_requested` to `LabResult.test_date` (joined via `result_id`).

### 14. Dead SocketIO instances

Removed three orphan `SocketIO()` instantiations (results.py, tests.py, reagents.py) that were never registered with the app — dead code implying realtime updates that didn't exist.

### 15. print() debugging

All `print()` statements replaced with `logger.exception`/`logger.debug` across results.py, tests.py, reagents.py
(including one that dumped raw patient results to stdout). The seeder's CLI `print` remains intentionally.

### 16. Session result_id cross-tab race

`process_lab_request` previously stored one `result_id` in `session`, so two tabs processing different requests
clobbered each other's identifier. Each submission now generates its own per-request UUID.

### 17. Misc

- **Currency**: process page showed `$` — now `KES` (app is Kenyan).
- **Duplicate catalog entries**: `add_lab_test` rejects a test with the same name *or* LOINC code.
- **Money integrity**: cost parsed as `Decimal` (quantized), never `float`, on add and edit — consistent with the repo's Numeric-column invariant.
- **Segregation of duties**: only laboratory/admin roles may change test pricing via `edit_lab_test` (medicine role gets 403 on price changes).
- **Pagination**: `processed_lab_results` paginated (50/page) with template pagination controls.
- **QC safety**: a REJECTED Westgard run now returns an explicit `clinical_safety_warning` — "do not release patient results from this analyzer until QC passes."
- **Reagent restock**: validates the item is category 6 (lab reagent) before filing a restock order.

---

## Verification

| Suite | Result |
|---|---|
| `tests/test_lab_flow.py` | ✅ |
| `tests/test_lims_westgard_qc.py` | ✅ (incl. new state-machine endpoints unaffected) |
| `tests/test_lis_panic_alerts.py` | ✅ |
| `tests/test_lab_catalog_seeder.py` | ✅ |
| `tests/test_lab_gap_fixes.py` (new) | ✅ 13/13 |
| `tests/test_pharmacy_gap_fixes.py` (no regression) | ✅ 10/10 |
| `tests/security/` + `tests/test_billing_event_listeners.py` + `tests/test_patient_safety_registry.py` | ✅ 25/25 |

## Files changed

| File | Change |
|---|---|
| `departments/laboratory/panic_alerts.py` | Required fields, lab-test validation, self-verification block, clinician alert routing, `_resolve_ordering_clinician_id()` |
| `departments/laboratory/results.py` | Panic engine wiring, atomic idempotent entry, specimen bridge, verification queue route, abnormal_results rewrite, pagination, logging |
| `departments/laboratory/tests.py` | Delete guards (test + template), duplicate-name check, Decimal cost, price-edit SOB, logging |
| `departments/laboratory/lims_routes.py` | start-analysis / complete / dispose endpoints, QC-reject warning |
| `departments/laboratory/reagents.py` | Restock category validation, logging |
| `departments/rbac.py` | `laboratory ⇄ lab_tech` aliases |
| `templates/side_bars/laboratory.html` | LIMS dashboard, verification queue links |
| `templates/laboratory/index.html` | Correct RequestedLab links, queue banner |
| `templates/laboratory/verification_queue.html` | New — 2-tier sign-off UI |
| `templates/laboratory/processed_lab_results.html` | Pagination controls |
| `templates/laboratory/dashboard.html` | Panic + TAT cards |
| `templates/laboratory/process_lab_request.html` | KES currency |
| `tests/test_lab_gap_fixes.py` | New — 13 regression tests |
