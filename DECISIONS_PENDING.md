# Decided & Ratified System Architecture & Policy Log

Per Process Integrity rules (P.1), hard stops were enforced for decisions with financial, legal, regulatory, or architectural impact. All pending design items have now been formally reviewed, decided, and ratified by the Solo Developer / System Administrator.

---

## 1. Telemedicine & Virtual Consultation Engine

* **Status:** ✅ DECIDED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: `departments/telemedicine` provides virtual room session creation, WebRTC client canvas scaffolding, and session lifecycle tracking.
* **Decisions**:
  1. **Scaffolding Disposition**: Retain existing telemedicine scaffolding behind the strict feature flag (`ENABLE_TELEMEDICINE=False` by default) for future phase rollout.
  2. **Infrastructure & Compliance**: Defer dedicated TURN/STUN relay infrastructure costs and KMPDC telehealth licensing until video consultations are actively enabled by facility management.

---

## 2. Automated Pharmacy Inventory & Supplier Purchase Orders

* **Status:** ✅ DECIDED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: `departments/pharmacy/po_supplier.py` contains models and endpoints for `Supplier`, `PurchaseOrder`, and automated PO generation based on minimum stock threshold scanning.
* **Decisions**:
  1. **Workflow Integration**: Retain and formally integrate automated Purchase Order (PO) generation directly within the HMIS to scan minimum stock thresholds and auto-draft reorders.

---

## 3. Telemetry & Error Tracking Service Selection

* **Status:** ✅ DECIDED — 2026-09-11
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: Operational maturity (Phase 2.1) calls for error tracking and metrics monitoring.
* **Decisions**:
  1. **Provider Selection**: Self-Hosted Sentry (via official docker-compose) for 100% data sovereignty under Kenya DPA 2019.
  2. **OpenTelemetry Collector (P2-01)**: Grafana Tempo integrated natively with Grafana dashboards.

---

## 4. Official WHO ICD-10 API Registration Credentials

* **Status:** ✅ DECIDED — 2026-09-13
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: WHO ICD-10-CM / ICD-11 database integration via WHO API.
* **Decisions**: Mapped credentials in `.env` (`WHO_ICD_CLIENT_ID`, `WHO_ICD_CLIENT_SECRET`), token-cached API client in `who_icd_client.py`, and scheduled Celery nightly re-sync.

---

## 5. Controlled Drug Register Policy & Workflow

* **Status:** ✅ DECIDED & IMPLEMENTED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: Controlled substance dispensing under Kenya Pharmacy and Poisons Board (PPB) legal regulation.
* **Decisions**:
  1. **Dual Signature Roles**: Pharmacist + Ward Nurse-in-Charge / Clinical Officer.
  2. **Reconciliation Schedule**: Split schedule (Schedule II narcotics = Per Shift; Schedule IV psychotropics = Daily).
  3. **Stock Synchronization**: Physical pack-unit removal (`pack_units_qty`) tracked at dispense time, deducting from `Batch` and `Drug` stock and appending a `StockMovement` ledger entry.

---

## 6. KRA eTIMS Tax Compliance & Electronic Invoicing

* **Status:** ✅ DECIDED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: Financial integration with Kenya Revenue Authority (KRA) eTIMS for automated QR code fiscal receipt generation.
* **Decisions**:
  1. **Architecture**: Build an eTIMS middleware adapter stub behind feature flag `ENABLE_ETIMS=False`.
  2. **VAT Exemption Matrix**: Pre-map hospital service categories (medical consultations, inpatient care, essential drugs exempt; cosmetic/retail taxable). Automated fiscalization will trigger upon OSCU device connection.

---

## 7. Medical File Storage Backend & Kenya DPA 2019 Data Residency

* **Status:** ✅ DECIDED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: Object storage for DICOM imaging, lab PDF results, and patient file uploads under Kenya Data Protection Act (2019).
* **Decisions**:
  1. **Storage Provider**: Self-hosted MinIO object storage (S3-compatible, Docker-native) deployed on-premise.
  2. **Data Residency**: Ensures 100% in-country data localization compliance under Kenya DPA 2019 and zero external cloud storage costs.

---

## 8. Supply Chain — Single-Facility vs Multi-Facility Deployment Architecture

* **Status:** ✅ DECIDED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: Scope of inter-facility stock transfers and counterparty institution handling.
* **Decisions**:
  1. **Deployment Architecture**: Single-facility system. Transfers to/from external facilities are recorded as single-institution dispatch/receipt transactions.

---

## 9. Supply Chain — Budget & Vote-Head Procurement Control Scope

* **Status:** ✅ DECIDED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: Budget vote-head validation during LPO generation.
* **Decisions**:
  1. **Control Enforcement**: Track vote-head allocations per department with soft warnings on LPO generation when limits are reached, ensuring emergency medical supply orders are never hard-blocked.

---

## 10. Unified Billing Sync — Legacy Line Item Mutations & Deletions

* **Status:** ✅ DECIDED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Context**: Handling updates or deletions of legacy billing items (`DrugsBill`, `LabBill`) in `departments/billing/sync.py`.
* **Decisions**:
  1. **Audit Trail**: Preserve an immutable financial audit trail by issuing explicit credit/adjustment line items (negative charges) on billing reversals rather than mutating or deleting historical invoice line items.

---

## 11. National Program Modules — HIV/ART, TB, Malaria

* **Status:** ✅ DECIDED & RATIFIED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**:
  1. **Module Rollout**: Full workflow modules for HIV/ART, TB/DOTS, and Malaria.
  2. **Reporting Integration**: Extend `departments/api/dhis2_exporter.py` to support MOH 731 (HIV/ART) and MOH 711 (TB/DOTS) alongside MOH 705A/B (Malaria).

---

## 12. Billing Event Listener Architecture

* **Status:** ✅ DECIDED — 2026-09-11
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Two-phase capture (`after_flush` & `after_flush_postexec`), `threading.local()` charge storage, and independent `Session(bind=db.engine)` for thread-safe billing writes.

---

## 13. Theatre Procedure Model Schema

* **Status:** ✅ DECIDED — 2026-09-11
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Standardized on `TheatreProcedure(name=..., type="General", cost=...)`.

---

## 14. Mock Data Cleanup Policy

* **Status:** ✅ DECIDED — 2026-09-11
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Automated cleanup script `scripts/cleanup_mock_data.py` targets `chief_complaint LIKE 'Mock %'`.

---

## 15. Ward Admission Auto-Billing Removal

* **Status:** ✅ DECIDED — 2026-09-11
* **Decisions**: Ward billing is consolidated into `/nursing/mar/auto_bill` as the single auditable daily source of truth.

---

## 16. HL7v2 MLLP Interface Engine Selection

* **Status:** ✅ DECIDED & RATIFIED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Formally ratify self-hosted **Mirth Connect** (port 2575) as the official MLLP engine with `mllp_daemon.py` fallback.

---

## 17. Controlled Drug Register Policy

* **Status:** ✅ DECIDED — 2026-09-11
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Separate ledgers for Schedule II/IV, dual-signature verification, and per-shift/daily stock counts.

---

## 18. Phase 2 Disaster Recovery SLA

* **Status:** ✅ DECIDED & RATIFIED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Formally ratify DR SLA: **Recovery Point Objective (RPO) = 4 hours**, **Recovery Time Objective (RTO) = 1 hour**.

---

## 19. Single-Facility Row-Level Security & Role Isolation Scope

* **Status:** ✅ DECIDED & ALIGNED — 2026-09-20
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Aligned with Decision #8 (Single-Facility System). Single-tenant deployment architecture utilizing PostgreSQL Row-Level Security (RLS) policies and non-superuser `hospital_app` database roles for strict least-privilege security isolation.

---

## 20. Phase 6 SSO & SMART on FHIR Architecture

* **Status:** ✅ DECIDED — 2026-09-12
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Authlib OAuth2 authorization server with SMART on FHIR EHR Launch flow.

---

## 21. Phase 1 Terminology Licensing & Fallbacks

* **Status:** ✅ DECIDED & RATIFIED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Formally ratify SNOMED CT CORE subset + live UMLS API integration (`UMLS_API_KEY`) as production terminology policy.

---

## 22. SNOMED CT & LOINC Licensing & UMLS API Integration

* **Status:** ✅ DECIDED — 2026-09-13
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Configured UMLS API client in `umls_client.py` and scheduled nightly syncs.

---

## 23. Renal / Dialysis Unit Module

* **Status:** ✅ DECIDED & RATIFIED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: Formally ratify Kt/V adequacy tracking, dialysis prescription workflow, and pharmacy dialysate/heparin auto-billing integration.

---

## 24. Pharmacy Audit Finding D — Controlled Drug Batch Stock Sync

* **Status:** ✅ DECIDED & IMPLEMENTED — 2026-09-18
* **Decision-maker:** Solo Developer / System Administrator
* **Decisions**: `ControlledDrugDispense` model updated with `pack_units_qty` and `batch_id`. Service decrements `Batch` and `Drug` stock, writes `StockMovement` row, and migration `c3d4e5f6a7b8` applied.
