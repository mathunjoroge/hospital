# Prompt 1 of 9 — Fix active regressions (Phase 0 cleanup)

Priority: Immediate. Nothing else in this plan should be built on top of a broken baseline.

## Context

Repo: `mathunjoroge/hospital` (Flask HMIS). Two issues were confirmed by direct
investigation of the current `main` branch and must be fixed before any other phase starts.

## Branch and safety rules

- Create branch `fix/phase0-billing-regression` from `main`. Do not commit to `main` directly.
- Do not force-push, on any branch, at any point.
- Run the full test suite first and record the baseline pass/fail count before changing anything.
- Touch only the files named below. No incidental refactors.

## Task 1 — Restore billing event-listener coverage

`departments/billing/event_listeners.py` currently only wires sync for `RequestedLab`,
`RequestedImage`, and `PrescribedMedicine`. A prior version (still reachable in git history —
check `git log --all -- departments/billing/event_listeners.py` for the commit before the most
recent force-push touched this file) also wired: `DispensedDrug`, `ClinicBooking`,
`TheatreList`, `AdmittedPatient`, `PaidBill`, `DrugsBill`, `Billing`, and the department bills
(`LabBill`, `ClinicBill`, `TheatreBill`, `ImagingBill`, `WardBill`). That coverage is gone from
`main` with no test catching the loss.

Acceptance criteria:
- Every source type listed above is wired back into unified invoice sync.
- Do **not** reintroduce the module-level global list pattern (`_pending_charges = []` shared
  across requests) — that's a race condition under concurrent requests, not just a style
  choice. Prefer the mapper-level `before_insert`/`before_update` approach (raw
  `connection.execute()` query, not a session query) that was previously used for
  `InvoiceLineItem` encounter-scoping, or an equivalent pattern that doesn't rely on shared
  mutable state across requests. If you're unsure which pattern to use, stop and ask rather
  than guessing.
- Add tests in `tests/test_billing_event_listeners.py` (or a new file) covering sync for each
  of the restored source types — this gap had zero test coverage before, which is exactly
  how it went unnoticed. Don't close this task without tests that would have caught the
  regression.
- Full test suite passes; ruff clean.

## Task 2 — Fix broken test fixture

`tests/test_theatre_booking_stages.py`, function `_procedure()`, constructs
`TheatreProcedure(name=name, description="Standard procedure")`. The model has no
`description` column, and `cost` (which the model requires — `nullable=False`) isn't passed.

Fix: `TheatreProcedure(name=name, cost=100.00)` (or another valid placeholder cost — check
the model definition in `departments/models/medicine.py` for the actual constraint before
picking a value).

Acceptance criteria:
- `tests/test_theatre_booking_stages.py` passes in full.
- No other test in the suite regresses.

## Task 3 — Lint

Run `ruff check . --select E,F,W,I --ignore E501` across the whole repo (not just files you
touched) and fix anything it flags with `--fix` where safe. If a flagged import or line looks
load-bearing rather than dead, verify with a repo search before removing it — don't blindly
apply `--fix` to something you haven't checked.

## Done when

- Full test suite passes, count is ≥ baseline (should be higher — new tests added).
- `ruff check . --select E,F,W,I --ignore E501` is clean.
- Branch is pushed (not force-pushed), ready for PR, with a summary of what was restored and
  why, and the before/after test count.
