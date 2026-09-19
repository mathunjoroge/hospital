# Medicine Department — Fix Report

**Date:** 2026-09-19
**Scope:** `departments/medicine/` (prescribe.py, inpatients.py, consultations.py, orders.py, prescriptions.py, oncology.py, cdss.py, chat_bot.py), `departments/tasks.py`
**Related:** `report.md` (pharmacy gap analysis — all fixes applied), `labfix.md` (laboratory — all fixes applied)

**Test status:** 280 tests passing across 16 suites — 261 pre-existing tests (6 updated for the new auth/consent behavior) plus 19 new regression tests (`tests/test_medicine_gap_fixes.py`). Ruff (`E,F,W,I`) clean.

---

## P0 — Security / Patient Safety (6 fixes)

### 1. E-prescribing API was completely unauthenticated
**Where:** `departments/medicine/prescribe.py`

All 5 routes on the separately-registered `prescribe_bp` blueprint had neither `@login_required` nor `@roles_required` — and `app.py` has no global auth guard. Anyone on the network could:
- `POST /medicine/prescribe/signoff` — create prescriptions **and auto-generate invoice line items**
- `POST /medicine/prescribe/soap` — write SOAP notes into any patient's chart
- `GET /icd10`, `POST /validate`, `POST /cdss/evaluate` — abuse clinical tooling

**Fix:** All five routes now require `@login_required` + `@roles_required("medicine", "admin")`.

### 2. Fuzzy patient matching wrote clinical data to the wrong patient
**Where:** `prescriptions.py` (`prescribe_drugs`), `inpatients.py` (`admit_patient`, `add_to_theatre`), `consultations.py` (`soap_notes`), `prescribe.py` (`signoff`)

Lookups used `Patient.patient_id.ilike(f"%{id}%") OR name.ilike(...)` — a param of "P1" could resolve to "P10", and the routes then saved prescriptions, admissions, surgical encounters, and theatre charges **under the matched (wrong) patient**. Oncology had already fixed this exact bug (`_find_patient` exact-match with an explanatory comment); the fix never propagated to the core OPD/IPD routes.

**Fix:** Every one of these paths now uses an exact `Patient.query.filter_by(patient_id=...)` lookup.

### 3. All 21 `inpatients.py` routes had no role restriction
Any authenticated user (HR clerk, records) could discharge patients, admit patients, book surgery + generate theatre charges, write post-op notes, write ward-round notes, transition surgical stages, and call every HL7 ADT API. Ward-round notes record `doctor_id=current_user.id`, so a non-clinician foraged physician authorship.

**Fix:** Role decorators applied to all 21 routes:
- Clinical writes (admit, discharge, theatre booking/post-op/stage transitions, ADT APIs): `medicine`/`nursing`/`admin`
- Theatre booking + post-op + stage transitions also accept `theatre`
- Bed housekeeping API accepts `theatre`
- Theatre list / admitted patients / ward rounds read access includes `nursing`

### 4. Web admit/discharge diverged from the ADT engine (two bed state machines)
**Where:** `inpatients.py` `admit_patient` / `discharge_patient`

- Web `discharge_patient` with a legacy `bed_id=None` admission fell back to freeing **any occupied bed in the ward** — discharging patient A could free patient B's bed.
- Web path never set `bed.status` (`"OCCUPIED"`/`"DIRTY"`) → released beds never entered the housekeeping turnaround queue; no `WardBedHistory` rows.
- Web admit never updated `ward.occupied_beds`, but web discharge decremented it → occupancy counters drifted with every web admission.

**Fix:** Both paths now maintain the same state machine as `ADTEngine`:
- Discharge releases **only the admission's own bed** (ward-wide fallback removed), sets `status="DIRTY"`, writes a `WardBedHistory(action="Discharge")` row, and **recalculates** ward occupancy instead of blind-decrementing.
- Admit sets `bed.status="OCCUPIED"`, writes `WardBedHistory(action="Admit")`, and recalculates occupancy.

### 5. Forged attribution fields
**Where:** `inpatients.py` `add_to_theatre`, `admit_patient`

`created_by` (theatre) and `admitted_by` (admissions) were taken from client form data.

**Fix:** Both are now derived from `current_user.id`; client values are ignored (regression test posts `created_by=999999` and asserts it is not stored).

### 6. `url_for("medicine.patients_list")` — guaranteed BuildError 500
**Where:** `oncology.py` `new_booking` (no-patients branch)

The referenced route does not exist anywhere in the codebase.

**Fix:** Redirects to `medicine.oncology` instead.

---

## P1 — Workflow (9 fixes)

### 7. Inpatients could not get labs/imaging ordered
**Where:** `orders.py` `request_lab_tests` / `request_imaging`

Patient lookup was **waiting-list only** — every admitted (IPD) patient failed with "not found in the waiting list". Ward patients had no lab-ordering path at all.

**Fix:** Both routes resolve the patient from the `Patient` master directly. Plus a **duplicate-pending guard**: tests/imaging already requested and still pending (status=0) are skipped, so double-clicks/refresh resubmits no longer create duplicate `RequestedLab`/`RequestedImage` rows.

### 8. Post-consult staging used lifetime counts
**Where:** `consultations.py` `submit_soap_notes`

`RequestedLab.query.filter_by(patient_id=..., status=0).count()` counted pending labs from **all visits ever** — one stale request from a prior visit pinned every future encounter to `AWAITING_RESULTS`, blocking normal routing to billing.

**Fix:** Pending labs/imaging/prescriptions are counted scoped to `encounter.id`. (Dashboard KPIs intentionally remain facility-wide totals.)

### 9. E-prescribe signoff invented clinical and billing values
**Where:** `prescribe.py` `handle_prescription_signoff`

Dosage defaulted to `"500mg"`, cost defaulted to `150.0`, and the **client-supplied cost was billed directly** to the invoice. Auto-created `Medicine` master rows with no duplicate check.

**Fix:**
- `patient_id` required, exact-match validated, ≥1 valid item enforced
- `dosage` + `frequency` + `num_days >= 1` required per item (no silent defaults)
- Charge derived from the pharmacy `Drug.selling_price` catalogue — **never from the client payload**
- Medicine lookup still dedups by `generic_name` (no uncontrolled master-data growth)

### 10. Blocking external call in the request path
**Where:** `consultations.py` `submit_soap_notes`

A synchronous `requests.post("http://127.0.0.1:8000/process_note", timeout=30)` made every clinician wait up to 30 seconds whenever the NLP worker was slow or down.

**Fix:** New Celery task `departments.tasks.process_soap_note_ai_analysis` (beat-compatible, matches the existing chatbot-task pattern); the view dispatches via `.delay()` and returns immediately. Failure to queue is non-fatal and flashes a warning. `reprocess_note` (see #16) reuses the same task.

### 11. Imaging keyword scan matched substrings
**Where:** `consultations.py` `submit_soap_notes`

`keyword in word` with keywords `"ct"`, `"scan"`, `"pet"` — the word "a**ct**ually" created phantom unmatched imaging requests.

**Fix:** Word-boundary matching (exact token match or `keyword-` prefix for hyphenated forms like "x-ray machine"), with punctuation stripping.

### 12. Dispensed prescription lines could be deleted
**Where:** `prescriptions.py` `delete_prescribed_medicine`

No status guard — pharmacy-dispensed (`status=1`) lines could be deleted, orphaning dispense records.

**Fix:** Deletion blocked with a clear flash message when `status == 1`.

### 13. Bed double-booking race
**Where:** `inpatients.py` `admit_patient`

Read `occupied=False` then flipped it — two concurrent admissions could take the same bed (the same pattern pharmacy got row locks for).

**Fix:** `with_for_update()` row lock on the admit-time bed query.

### 14. `available_rooms` always reported every room available
**Where:** `inpatients.py` `available_rooms`

Filtered on `WardRoom.occupied` — a column **nothing ever writes**.

**Fix:** Rooms are listed when they have ≥1 free bed (`join Bed ... occupied.is_(False)`), which is the real constraint. Regression tests cover both directions (free room listed, fully-occupied room excluded).

### 15. Ward-round and lab-request input hardening
**Where:** `inpatients.py`, `orders.py`

- `WardRound.status` was unvalidated free text (model default "Under Treatment", form offers 4 values)
- Ward rounds could be written against **discharged** admissions
- Lab/imaging requests had no duplicate-pending guard (covered in #7)

**Fix:** Shared `WARD_ROUND_STATUSES` allowlist enforced in both ward-round POST handlers (`/ward-rounds` and `/ward-rounds/add`); both reject admissions with `discharged_on` set.

---

## P2 — Polish (6 fixes)

### 16. Dead templates + no-op reprocess route
- Removed `ward_round_note.html` and `soap_notes_summary.html` (rendered by no view)
- **`reprocess_note` was a no-op** — the notes page offered a "reprocess" button whose route just redirected (and 500'd on a bad note_id). Now: 404 guard, re-queues `process_soap_note_ai_analysis.delay(note.id)`, flashes outcome, requires medicine/admin role.

### 17. `cdss.py` allergy check used the wrong data sources
`check_patient_allergies` built "allergy history" from `patient.name` and `relationship_with_next_of_kin` — not from the `PatientAllergy` registry or nursing notes. The `/cdss/evaluate` endpoint therefore exposed a weaker allergy engine than `ClinicalSafetyEngine` (which correctly uses both registries). Two divergent CDSS paths for the same question.

**Fix:** `check_patient_allergies` now aggregates the structured `PatientAllergy` registry + nursing-note free-text allergies (with graceful degradation on lookup failure), keeping the legacy narrative heuristic as a final fallback. The existing `test_patient_allergy_screening` still passes.

### 18. Chatbot consent bypass
AI consent (DPA 2019) was only checked when `patient_id` was supplied — omitting it sent pasted clinical notes to the external LLM with no consent gate and no AI-disclosure audit row.

**Fix:** When no `patient_id` is supplied but the input matches clinical-narrative heuristics (`_contains_clinical_narrative` — markers like "patient ", "complains of", "on examination", "bp ", "prescribed", "vitals"…), the request is refused with 403 and an `AI_CONSENT_REFUSED` audit row; the user is asked to select the patient so consent can be verified. General guideline queries still pass (and are now audited as `AI_CHATBOT_GENERAL_QUERY`).

### 19. `unmatched_imaging` filter crashes + wrong template path
- Raw date strings were compared directly against a DateTime column — invalid input was a **500 on Postgres**.
- `render_template("unmatched_imaging.html")` referenced a path that doesn't exist (the file lives at `medicine/unmatched_imaging.html`) — the route **always** raised `TemplateNotFound`.

**Fix:** Dates parsed with `strptime("%Y-%m-%d")` (invalid → flash, filter skipped), end-date expanded to end-of-day, template path corrected, raw strings preserved for re-filling the form inputs.

### 20. Code hygiene
- `print()` debugging → `logger.exception` in `prescriptions.py` (get_edit_form, drug_details), `orders.py` (fetch_drugs_data), `consultations.py` (soap_notes)
- `Decimal` money accumulation in the signoff charge
- **Prescription draft cross-tab race fixed**: the shared `session["prescription_id"]` is now keyed per patient (`prescription_id_{patient_id}`), so two tabs prescribing for two patients no longer clobber each other; `save_prescription` pops the matching per-patient key
- Unused imports removed (`requests`, `PatientWaitingList`, `joinedload`, `timedelta`…)

### 21. Duplicate active theatre bookings
**Where:** `inpatients.py` `add_to_theatre`

Same patient + procedure could be booked repeatedly while active (double-click, refresh resubmit), each creating a surgical encounter and charge.

**Fix:** An existing `TheatreList` row for the same patient/procedure with `status=0` short-circuits with a warning.

---

## Verification

| Suite | Result |
|---|---|
| `test_medicine_gap_fixes.py` (new regression) | 19/19 ✅ |
| `test_eprescribing.py` | 8/8 ✅ |
| `test_theatre_booking_stages.py` / `test_theatre_stages.py` | 17/17 ✅ |
| `test_adt_bed_management.py` | ✅ |
| `test_encounter_scoping.py` / `test_patient_flow_integration.py` | ✅ |
| `test_cdss.py` / `test_advanced_cdss.py` | ✅ |
| `test_ai_consent_gate.py` / `test_chatbot_async.py` | ✅ |
| `test_theatre_operations_engine.py` / `test_theatre_surgical_module.py` | ✅ |
| `test_oncology_chemotherapy.py` / `test_oncology_hardening.py` | ✅ |
| `test_medical_file_storage.py` | ✅ |
| **Total** | **280 passed** |
| `ruff check --select E,F,W,I --ignore E501` | clean |

**Pre-existing tests updated (behavior changes are intentional):**
- `test_eprescribing.py` — 3 endpoint tests now log in via `admin_user`; signoff payload updated for required `frequency`/`num_days` and catalogue-derived charge; added `test_endpoints_require_login`
- `test_cdss.py` — 2 CDSS-endpoint tests log in via `admin_user`
- `test_chatbot_async.py` — dispatch test uses a non-clinical guideline query (clinical narratives without a patient_id are now consent-gated per #18)

---

## File-by-file change manifest

| File | Changes |
|---|---|
| `departments/medicine/prescribe.py` | Auth on all 5 routes; signoff input validation, exact patient match, catalogue-priced billing, `Decimal` totals |
| `departments/medicine/inpatients.py` | RBAC on all 21 routes; ADT-aligned admit/discharge; row-locked bed; `current_user` attribution; duplicate booking guard; ward-round validation; bed-derived `available_rooms`; `WARD_ROUND_STATUSES` |
| `departments/medicine/consultations.py` | Exact patient match in `soap_notes`; per-encounter pending counts; async NLP dispatch; word-boundary imaging scan; real `reprocess_note`; print removal |
| `departments/medicine/orders.py` | IPD-capable lab/imaging requests with duplicate guards; date-filter validation; template path fix; print removal; import cleanup |
| `departments/medicine/prescriptions.py` | Exact patient match; per-patient prescription draft key; dispensed-line delete guard; print removal |
| `departments/medicine/oncology.py` | `patients_list` BuildError fix |
| `departments/medicine/cdss.py` | Allergy check sources `PatientAllergy` + nursing notes |
| `departments/medicine/chat_bot.py` | Unscoped clinical-narrative consent gate + audit rows; `_contains_clinical_narrative` helper |
| `departments/tasks.py` | New `process_soap_note_ai_analysis` Celery task |
| `tests/test_medicine_gap_fixes.py` | **New** — 19 regression tests |
| `tests/test_eprescribing.py`, `tests/test_cdss.py`, `tests/test_chatbot_async.py` | Updated for new auth/consent behavior |
| Removed | `templates/medicine/ward_round_note.html`, `templates/medicine/soap_notes_summary.html` |
