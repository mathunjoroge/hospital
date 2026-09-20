# RECORDS ↔ SPECIALTY DEPARTMENTS BOOKING WORKFLOW
## Implementation Plan & Backlog

*Renal / Dialysis Unit + Oncology — booked-from-Records visibility, day-sheet operations, notifications*

*Version 1.1 · September 20, 2026 · Owner: Records / Renal / Oncology leads — v1.1 closes backlog items A2, A3, B4, B5, B6*

---

## 1. Status at a glance

| **Capability**                                          | **Status**     | **Where** |
|----------------------------------------------------------|----------------|-----------|
| Specialty clinics seeded in Records catalog (idempotent) | **✓ Complete** | `departments/records/clinic_bridge.py`, auto-seed in `app.py` startup |
| Records booking propagates to department schedulers      | **✓ Complete** | Renal → `SCHEDULED` DialysisSession; Oncology → `Scheduled` OncologyBooking; atomic with ClinicBooking |
| Booking source provenance (`source` column)              | **✓ Complete** | `dialysis_sessions.source` & `oncology_bookings.source` (`RENAL`/`ONCOLOGY` vs `RECORDS`) |
| Source badge on department boards                        | **✓ Complete** | Renal day sheet + oncology bookings board |
| Source filter on boards                                  | **✓ Complete** | `?source=RECORDS\|RENAL\|ONCOLOGY` (unknown values ignored) |
| Renal unit-wide schedule board (was silently P001)       | **✓ Complete** | `GET /renal/sessions`, status + date-range + source filters, JSON `scope: "unit"` |
| Patient search in renal console                          | **✓ Complete** | `GET /renal/api/search-patients` (active patients only) + header autocomplete |
| Chair-time assignment                                    | **✓ Complete** | Engine `assign_chair_time` + `PATCH /renal/sessions/<id>/chair-time` + modal in console |
| Day-sheet shift grouping (Morning/Afternoon/Evening)     | **✓ Complete** | Single-day filter triggers grouped sections; chair-time ordering within shift |
| Unassigned Records bookings flagged                      | **✓ Complete** | `needs_chair_time` in JSON; amber row + "⚠ N need chair time" header count |
| Patient notification on chair-time assignment            | **✓ Complete** | `trigger_chair_time_assigned` → `appointment_confirmed` event, best-effort |
| Soft-deleted patients blocked from booking               | **✓ Complete** | `book_clinic` rejects `is_active=False` |
| SMS channel for notifications                            | **✗ Missing**  | Dispatcher has `SandboxSMSChannel`; wire AfricasTalking for reminders (item B1) |
| Chair capacity / conflict detection                      | **✓ Complete** | `count_chair_conflicts` + `get_chair_capacity` (`RenalUnitConfig.chair_count`, default 1); full slot → 422, shared slot → `warning` in response |
| Reschedule to a different day                            | **✓ Complete** | `assign_chair_time(session_date=...)` moves the day; end time-of-day re-anchored; modal has "Session Date" field |
| Recurring dialysis series (e.g. Mon/Wed/Fri)             | **✓ Complete** | `create_session_series` + `POST /renal/sessions/<pid>/series` + console form; existing dates skipped |
| Oncology slot times / chemo chair scheduling             | **✓ Complete** | `oncology_bookings.start_time` (migration `b9e4f7c2a810`); optional slot time on new-booking form; shown on board |
| Patient portal view of specialty appointments            | **✓ Complete** | Portal dashboard "Dialysis & Oncology Appointments" card (upcoming, date+time+status) |

---

## 2. How the flow works today

```
Records: "Book Clinic" modal
  └─ POST /records/book_clinic
       ├─ ClinicBooking row (canonical patient_id, is_active guard) + audit log
       ├─ walk-in Appointment + Encounter (unchanged queue bridge)
       └─ clinic_bridge.propagate_specialty_booking()
            ├─ name ~ renal/dialysis/nephro/kidney → DialysisSession
            │    status=SCHEDULED, source=RECORDS, modality=HD (TBD at chairside)
            └─ name ~ oncology/cancer/chemo       → OncologyBooking
                 purpose=Consultation, status=Scheduled, source=RECORDS
       (department row commits atomically with the ClinicBooking)

Renal console (/renal/sessions)
  ├─ Unit board: UPCOMING default (SCHEDULED+IN_PROGRESS, soonest first)
  ├─ Filters: status · date range (start/end) · source — all compose
  ├─ Single-day view = day sheet: shift groups + chair-time ordering
  ├─ 📋 Records badge; amber ⚠ "Needs chair time" flag on unassigned bookings
  └─ 🕘 Set time → PATCH chair-time → patient email → regroups into shift

Oncology board (/medicine/bookings)
  └─ 📋 Records badge + Source filter; purpose/status filters unchanged
```

**Billing safety:** dialysis billing fires only on session `COMPLETED` (billing
event listener, §23 #7); a Records booking creates no charge. OncologyBooking
rows are never billed directly.

---

## 3. Schema changes shipped (2026-09-20)

| Migration | Change |
|-----------|--------|
| `d1a4c7e9f201` | `dialysis_sessions.source` (String 20, default `RENAL`) |
| `c8d3e6a1b405` | `oncology_bookings.source` (String 20, default `ONCOLOGY`) |
| `b9e4f7c2a810` | `oncology_bookings.start_time` (DateTime, nullable — B5 slot times) |

All applied to the dev PostgreSQL database. Specialty clinic catalog rows
("Renal / Dialysis Clinic", "Oncology Clinic", fee 1000.00) seeded; the
startup seeder re-runs idempotently on every boot.

---

## 4. Backlog (prioritized)

### A. Near-term (operational value)

- **A1 — SMS notifications.** Extend `NotificationDispatcher` dispatch in
  `trigger_chair_time_assigned` to `channels=["email","sms"]` once the
  AfricasTalking driver is configured (`SMS_CHANNEL` env). Bodies are already
  plain-text and SMS-length friendly.
- **A2 — Chair conflict detection.** In `assign_chair_time`, warn (422 or
  `warning` field in response) when another SCHEDULED session already holds
  the same date+start_time. Needs a "chair" concept or nurse-count from
  `RenalUnitConfig.chair_count`.
- **A3 — Reschedule / move day.** Add `session_date` to the chair-time PATCH
  (or a dedicated `PATCH .../reschedule`) so a missed booking can move to the
  next dialysis day; must re-run the needs-attention flag.

### B. Mid-term

- **B4 — Recurring series.** Generate N future SCHEDULED sessions from
  `RenalUnitConfig.shift_pattern` (e.g. "Mon/Wed/Fri") when a chronic dialysis
  patient is enrolled; each remains independently editable/cancellable.
- **B5 — Oncology slot times.** Port the chair-time pattern: add
  `start_time` to OncologyBooking, slot chemo bookings into day/week grid,
  reuse the flag/filter approach for unassigned slots.
- **B6 — Patient portal.** Surface upcoming dialysis sessions and oncology
  bookings (with source + chair time) on the portal appointment page; reuse
  `needs_chair_time`-free summaries only.

### C. Deferred / requires clinical sign-off

- **C1 — Kt/V adequacy alerting.** spKt/V is calculated and stored (§23 #3);
  alerting thresholds (target ≥ 1.2) and nephrologist notification still need
  sign-off per DECISIONS_PENDING §23.
- **C2 — Prescription → session prefill.** Pre-fill BFR/DFR/heparin from the
  active `DialysisPrescription` when logging a session (§23 #5).
- **C3 — Slot-based scheduling mode.** `RenalUnitConfig.scheduling_mode`
  supports `SLOT_BASED`; only build after A2 + B4 land.

---

## 5. Verification & test map

| Area | File | Notes |
|------|------|-------|
| Clinic seeder + booking propagation | `tests/test_records_specialty_booking.py` | 14 tests incl. oncology source parity |
| Renal board, filters, shifts, chair time, notifications | `tests/test_renal_module.py` | 65 tests (board/search/shift/chair-time/notify classes) |
| Outbound notification pipeline | `tests/test_outbound_notifications.py` | existing suite, untouched, passing |
| P0 security regression | `tests/security/test_p0_security.py` | passing (nurse_id still derived from auth context only) |

Ruff (`E,F,W,I`, ignore `E501`): clean across `departments/`.

**Outstanding:** full-suite run (1252 tests) exceeds the 10-minute shell
window in a single invocation; needs a chunked or `pytest-xdist` run.
Coverage gate is 45% (`pyproject.toml` addopts).

---

## 6. Deployment notes

1. `flask db upgrade` (applies the two `source` migrations).
2. Specialty clinics auto-seed on next `python app.py` start (or run
   `python -m departments.records.clinic_bridge` manually).
3. Records fees default to 1000.00 — adjust via Records → Clinics.
4. No frontend build step; templates changed in place.
