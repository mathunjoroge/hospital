# Decisions Pending Human Review & Approval

Per Process Integrity rules (P.1), hard stops are enforced for decisions with financial, legal, regulatory, or architectural impact. The following items require explicit decision-making from a human stakeholder before proceeding with implementation or further feature work.

---

## 1. Telemedicine & Virtual Consultation Engine

* **Context**: `departments/telemedicine` was previously added with virtual room session creation, WebRTC client canvas scaffolding, and session lifecycle tracking. It has now been placed behind a strict feature flag (`ENABLE_TELEMEDICINE=False` by default) at the blueprint registration level.
* **Questions / Decisons Required**:
  1. **Infrastructure Costs**: Real-time video/audio streaming via WebRTC requires dedicated TURN/STUN relay infrastructure (e.g. Coturn or third-party providers like Twilio/Agora) for NAT traversal across cellular/hospital networks. Does the facility budget support hosting or subscribing to a TURN/STUN relay service?
  2. **Regulatory Compliance**: Does the facility have regulatory sign-off under Kenya Medical Practitioners and Dentists Council (KMPDC) Telehealth Guidelines and Data Protection Act (2019) requirements for recording/transmitting clinical sessions?
  3. **Scaffolding Disposition**: Should the existing telemedicine scaffolding be retained behind the feature flag for future development, or removed completely from the codebase?

---

## 2. Automated Pharmacy Inventory & Supplier Purchase Orders

* **Context**: `departments/pharmacy/po_supplier.py` contains models and endpoints for `Supplier`, `PurchaseOrder`, and automated PO generation based on minimum stock threshold scanning.
* **Questions / Decisions Required**:
  1. **Workflow Alignment**: Does the hospital procurement workflow require automated PO generation directly within the HMIS, or is procurement handled via an external ERP / financial system?
  2. **Feature Retention**: Should this PO/Supplier module be retained and formally integrated into the procurement workflow, or removed to avoid overlap with external ERP systems?

---

## 3. Telemetry & Error Tracking Service Selection

* **Context**: Operational maturity (Phase 2.1) calls for error tracking and metrics monitoring (e.g. Sentry integration).
* **Questions / Decisions Required**:
  1. **Provider Selection**: Is Sentry SaaS (free tier/paid) acceptable for error tracking, or does the hospital require self-hosted error tracking (e.g. Sentry On-Premise, GlitchTip, or OpenTelemetry/Jaeger) due to data sovereignty rules under the Data Protection Act 2019?
  2. **Volume & Budget**: Does the anticipated log/error volume fit within free tier limits, or is budget allocated for telemetry ingestion?

---

## 4. Official WHO ICD-10 API Registration Credentials (Phase A.3)

* **Context**: The system currently utilizes an expanded 50+ item common clinical diagnosis catalog in [`departments/medicine/prescribe.py`](file:///home/mathu/projects/hospital/departments/medicine/prescribe.py#L35). Importing the complete 14,000+ code WHO ICD-10-CM / ICD-11 database via WHO's API requires organizational registration credentials (`Client ID` & `Client Secret`).
* **Questions / Decisions Required**:
  1. **Credentials**: Can the facility management provide WHO ICD-API client credentials for automated FTS5 table ingestion?
  2. **Catalog Scope**: Is the 50+ item curated stopgap diagnosis catalog adequate for initial deployment while official credentials are obtained?

---

## 5. Controlled Drug Register Policy & Workflow (Phase B.1 — HARD STOP)

* **Context**: Schedule IV/V controlled substance dispensing is subject to Pharmacy and Poisons Board (PPB) legal regulation. The dual-signature and balance reconciliation rules vary by facility licensing tier.
* **Questions / Decisions Required**:
  1. **Dual Signature Roles**: Who qualifies as the mandatory second signatory for controlled drug dispensing? (e.g., two licensed pharmacists, or a pharmacist plus the ward nurse-in-charge?)
  2. **Stock Reconciliation Schedule**: Does physical inventory reconciliation occur per shift, daily, or weekly?
  3. **Schedule Differentiation**: Do Schedule II (narcotics) and Schedule IV (psychotropics) require distinct register ledgers or a single unified controlled log?

---

## 6. KRA eTIMS Tax Compliance & Electronic Invoicing (Phase B.2 — HARD STOP)

* **Context**: Financial integration with Kenya Revenue Authority (KRA) eTIMS for automated QR code fiscal receipt generation requires an active KRA PIN and VSCU/OSCU software middleware certification.
* **Questions / Decisions Required**:
  1. **Registration Status**: Does the facility currently possess an active KRA eTIMS registration and VSCU/OSCU ESD device/software license?
  2. **VAT Exemption Matrix**: Which hospital service categories are VAT-exempt under Kenyan tax law (e.g. medical consultations and essential drugs) vs. taxable (e.g. cosmetic procedures or retail supplies)?
  3. **Implementation Timing**: Should eTIMS fiscalization be built as an automated middleware bridge once registration details are provided?

---

## 7. Medical File Storage Backend & Kenya DPA 2019 Data Residency (Phase B.3 — HARD STOP)

* **Context**: DICOM imaging, lab PDF results, and patient record uploads currently sit on local Docker volume storage. Scalable production deployment requires object storage.
* **Questions / Decisions Required**:
  1. **Provider Selection**: Should storage use AWS S3, self-hosted MinIO, or Cloudflare R2?
  2. **Data Residency Compliance**: Under the Kenya Data Protection Act (2019), health data must comply with strict data localization guidelines. Is hosting on a local cloud provider or self-hosted MinIO within Kenya mandatory, or is an AWS/R2 regional bucket acceptable?

---

## 8. Supply Chain — Single-Facility vs Multi-Facility Deployment Architecture (Phase B.2 — HARD STOP)

* **Context**: The `Facility` model provides a foundation for identifying the home institution and counterparty institutions.
* **Questions / Decisions Required**:
  1. **Deployment Architecture**: Is this HMIS deployed as a single-facility system for one hospital (where transfers to/from external facilities are recorded as one-sided dispatch/receipt transactions), or a multi-facility shared system across a health network (where transfers operate as two-sided transactions in a single shared database)?
  2. **Phase E Dependency**: Phase E (Inter-Facility Transfer) implementation depends on this architecture decision.

---

## 9. Supply Chain — Budget & Vote-Head Procurement Control Scope (Phase D.1 — HARD STOP)

* **Context**: Government institutional LPO generation typically requires budget vote-head validation before LPO approval.
* **Questions / Decisions Required**:
  1. **Vote-Head Structure**: Does the facility track formal vote-head/budget allocations per department or item category within the HMIS, or is budget management handled externally?
  2. **Control Enforcement**: Should LPO approval block if requested line items exceed an allocated vote-head budget cap?

---

## 10. Unified Billing Sync — Legacy Line Item Mutations & Deletions (HARD STOP)

* **Context**: `departments/billing/sync.py` uses insertion-level idempotency (`if existing: return existing`). If a legacy bill (e.g. `DrugsBill`, `LabBill`) is updated or deleted in the legacy system after initial sync, `InvoiceLineItem` is not updated or deleted, creating potential discrepancies between legacy tables and the unified invoice.
* **Questions / Decisions Required**:
  1. **Retroactive Adjustment vs Credit Entry**: Should a legacy charge update/deletion directly modify or delete the corresponding `InvoiceLineItem`, or should it preserve an immutable audit trail by issuing an explicit credit/adjustment line item?
  2. **Session Event Scope**: Should SQLAlchemy session listeners be expanded to handle `session.deleted` events for legacy billing models?



