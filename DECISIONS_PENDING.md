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
