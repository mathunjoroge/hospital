# Pharmacy Department — Gap Analysis

**Date:** 2026-09-19
**Scope:** `departments/pharmacy/` and its integration points (billing, models, templates, tests, sidebar nav)

**Module inventory** (all healthy at code level): `dispensing.py`, `stock_ops.py`, `fefo.py`, `inventory.py`, `po_routes.py`, `reports.py`, `controlled_drugs_{routes,service}.py`, `ai_discovery.py`, `cheminformatics.py`, `moh_647.py` — plus strong test coverage (FEFO, ledger, PO, integrity invariants).

**Test status:** All 50 pharmacy tests pass (`test_pharmacy_fefo.py`, `test_pharmacy_dispensing.py`, `test_pharmacy_dispensing_coverage.py`, `test_pharmacy_ledger.py`, `test_pharmacy_po.py`, `invariants/test_pharmacy_integrity_invariants.py`). The 45% coverage gate failure is a repo-wide config issue, not pharmacy-specific.

---

## P0 — Correctness / Security gaps

### 1. Missing `@roles_required` on 4 sensitive routes

Any *authenticated* user (HR clerk, records, etc.) can access:

| Route | File:Line | Risk |
|---|---|---|
| `POST /save_dispensed_drugs` | `dispensing.py:300` | Mutates stock and marks prescriptions dispensed |
| `GET /patient_history` | `stock_ops.py:152` | Returns **any patient's** full clinical history: labs, imaging, vitals, SOAP notes, billing, admissions |
| `GET /get_all_batches` | `inventory.py:640` | Full stock & pricing data |
| `GET /analytics` + `/analytics/export` | `reports.py:78,148` | Sales/consumption data |

### 2. Broken drug-request lifecycle (`inventory.py:430`)

POST looks for the user's latest **"Submitted"** request and appends items to it; the cart view and `save_order` only look at **"Pending"**. After the first order is submitted, every subsequent drug request is silently attached to the already-sent order and can never be submitted again.

### 3. `record_purchase` bypasses the stock ledger (`inventory.py:334`)

Increments `batch.quantity_in_stock` with no `record_movement()` call and no drug-level stock sync — while `po_routes.receive_po_shipment` and `record_direct_receipt` correctly write `RECEIVED` rows. This path silently breaks the ledger-vs-cache `reconcile_stock_balance` invariant.

### 4. Status-field chaos corrupts reporting

`DispensedDrug.status` is written as `1` (fefo), `"Pending"` (process_dispense), `"0"`/`"COMPLETED"`/`"VOIDED"` (elsewhere). Consequences:

- MOH-647 "issued" aggregation (`moh_647.py:200`) excludes `status=1`/`"Pending"` → under-reported consumption
- `analytics()` and `patient_history` include `VOIDED` rows → over-reported sales/usage

### 5. Voiding doesn't reverse billing

`delete_dispensed_drug` / `remove_dispensed` restore stock but never void/cancel the linked `DrugsBill` or invoice line — voided dispenses remain billed.

---

## P1 — Workflow gaps

### 6. Destructive actions on GET

`remove_batch`, `remove_all_expiries`, `save_order` mutate state via GET links (prefetchable, no confirmation, no CSRF protection on GET).

### 7. FEFO API bypasses dispensing rules

`/pharmacy/fefo/dispense` enforces neither the payment gate (`has_unpaid_charges`) nor the open-encounter check that the web path enforces — two dispense paths with different rule sets. Its docstring claims "2-Step Dispensing Verification" but no verification step exists.

### 8. No scheduled alerting

`check_pharmacy_inventory_alerts` is only an on-demand endpoint; no celery beat task, no wiring to `departments/notifications`.

### 9. No drug/batch master-data CRUD

There is no UI route anywhere to create drugs/categories or edit a batch. Worse, `record_purchase` creates batches with `expiry_date=None` ("added later") but no route ever sets it — those batches sit in "normal stock" forever and are invisible to expiry alerts.

### 10. Concurrency

Only `fefo.py` uses `with_for_update()`. The `save_prescription`/`process_dispense` paths rely on the DB check constraint to fail loudly as a generic "Something went wrong" flash.

### 11. PO permission inconsistency

`submit_po_order` allows only lowercase roles, while `create_manual_po`/`record_direct_receipt` also allow `"Storekeeper"`, `"Pharmacist"` — a Storekeeper can create/receive POs but gets 403 ordering them. `ROLE_ALIASES` doesn't cover pharmacy roles at all.

### 12. Ordering-of-operations in `save_dispensed_drugs`

Stock is deducted before the encounter-open check, and all `PrescribedMedicine` rows are marked status=1 even after a partial-stock shortfall flash.

### 13. Duplicated void logic

`dispensing.py:delete_dispensed_drug` and `stock_ops.py:remove_dispensed` duplicate the void/reversal flow — drift risk.

---

## P2 — Polish

### 14. Dead/misplaced templates

`orders.html`, `lab_results.html` have no routes; `controlled_drugs.html` lives in root `templates/pharmacy/` instead of the department folder.

### 15. Sidebar gaps

No links for Expiries Report, suppliers/POs, FEFO console — and PO/supplier/RTV/smart-reorder/OTIF features are **JSON-API only with no UI at all**.

### 16. MOH-647 has no facility-facing report route

Only consumed by the DHIS2 exporter (`departments/api/dhis2_exporter.py:522`).

### 17. Minor code-quality issues

`analytics()` has no error handling (missing form dates → 500); `record_purchase` uses `float()` for money despite the repo's own Numeric invariant; `print()` debug statements in `inventory.py`/`reports.py`; excessive `logger.debug` in `dispensing.py`.

---

## Recommended quick wins (highest impact first)

1. **#1** — Add role decorators to the four unprotected routes
2. **#2** — Fix the Submitted/Pending filter in the drug-request lifecycle
3. **#3** — Add ledger rows + drug-level stock sync to `record_purchase`
