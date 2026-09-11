# HEALTH INFORMATION MANAGEMENT SYSTEM
## World-Class Implementation Plan

*Full gap closure roadmap — from strong regional prototype to Tier 1 hospital deployment*

*Version 1.1 · Updated September 10, 2026 (verified against the live `mathunjoroge/hospital` repo) · supersedes Version 1.0, 2025*

**Current implementation status at a glance**

| **Capability**                | **Status**     | **Gap / Next step**                                                                                  |
|-------------------------------|----------------|--------------------------------------------------------------------------------------------------------|
| MFA + RBAC + account lockout  | **✓ Complete** | Periodic review; add adaptive MFA for high-risk roles                                                  |
| AES-128 encryption at rest    | **✓ Complete** | Extend to lab results, imaging metadata, notes fields                                                  |
| HL7 FHIR R4 API               | **✓ Complete** | Add SMART on FHIR auth layer (Phase 6)                                                                 |
| Append-only audit trail       | **✓ Complete** | Wire to SIEM platform; define retention schedule                                                       |
| CDSS drug-drug interactions   | **✓ Complete** | Tune tiered suppression; P&T review workflow (Phase 1)                                                 |
| Allergy safety checks         | **✓ Complete** | Cross-sensitivity rules; free-text allergy parsing                                                     |
| Encounter state machine       | **~ Partial**  | Theatre handover UX wired (T3.2, Sep 2026) but has a known test/fixture bug — see Known Regressions; ANC continuity-of-care gaps still open |
| Break-glass emergency access  | **✓ Complete** | Expiry audit; quarterly review cadence                                                                 |
| PostgreSQL + Alembic + Docker | **✓ Complete** | Automated backup script now in place (Sep 2026) — HA setup, read replicas, and restore-drill testing still open |
| Celery async tasks + Redis    | **✓ Complete** | Dead-letter queue; task retry policy formalisation                                                     |
| TLS/HSTS baseline             | **✓ Complete** | HSTS header shipped (Sep 2026); CSP headers and full XSS/SQLi suite still open (P2-13)                 |
| Consent enforcement (DPA 2019)| **✓ Complete** | Not in v1.0 scope — built since; periodic review against Phase 1 terminology work                      |
| Referrals & appointments      | **✓ Complete** | Not in v1.0 scope — built since; wired to consent checks                                               |
| ICD-10 (50 codes)             | **~ Partial**  | Ingest full WHO ICD-10-CM/PCS 14,000+ codes (Phase 1)                                                  |
| LOINC mapping                 | **~ Partial**  | Expand to full lab orders, not just vitals (Phase 1)                                                   |
| Lab instrument interfacing    | **~ Partial**  | Build HL7 v2 MLLP engine (Phase 4)                                                                      |
| DICOM imaging                 | **~ Partial**  | Integrate Orthanc PACS + DICOMweb viewer (Phase 5)                                                     |
| SMART on FHIR / OAuth2 / SSO  | **✗ Missing**  | Critical blocker — Phase 6                                                                              |
| LDAP / Active Directory       | **✗ Missing**  | Phase 6, alongside SSO implementation                                                                  |
| SNOMED CT coding              | **✗ Missing**  | Phase 1 — problem list & procedure coding                                                              |
| HL7 v2 MLLP interface engine  | **✗ Missing**  | Phase 4 — connect physical lab analyzers                                                               |
| Automated DR failover         | **✗ Missing**  | Phase 2 — RPO <4h, RTO <1h targets                                                                      |
| Observability stack           | **✗ Missing**  | Phase 2 — OpenTelemetry + Sentry (still only an inert config flag)                                     |
| HIV/ART, TB, Malaria modules  | **✗ Missing**  | Phase 7 — after clinical review board sign-off (DECISIONS_PENDING.md #11, HARD STOP)                   |
| Controlled drug register      | **✗ Missing**  | Phase 3 — legal blocker; policy decision still open (DECISIONS_PENDING.md #5, HARD STOP)                |
| HIMSS EMRAM Stage 6/7         | **✗ Missing**  | Phase 8 — after Phases 1–7 complete                                                                    |
| Third-party penetration test  | **✗ Missing**  | Phase 8 — engage firm immediately for scoping                                                          |

**Changes since v1.0**

- Encounter state machine's theatre-handover gap was worked (T3.2) but landed with a broken test fixture (`TheatreProcedure(description=...)` — no such column) — fix is written but not yet merged as of this update.
- Two Phase 2 security items shipped early: HSTS header (P2-11) and automated DB backups (part of P2-07).
- Consent enforcement and referrals/appointments — not called out in v1.0 at all — are now built and wired together.
- No progress on Phases 1, 3, 4, 5, 6, 7. Phase 8 cannot start until they do.

**Known regression — read before resuming Phase 0/T3.8 work**

`departments/billing/event_listeners.py` was force-pushed back to an earlier draft that (a) only syncs charges for `RequestedLab`/`RequestedImage`/`PrescribedMedicine` — drug dispensing, theatre, ward admission, clinic booking, and *all* payment sync (`PaidBill`, `DrugsBill`, `Billing`) are silently unwired — and (b) reintroduced a module-level global list (`_pending_charges`) shared across requests, which is a race condition under any concurrent load. This has no test coverage to catch it. Restoring the pre-force-push version should be treated as a Phase 0 blocker, not a lint fix.

**2. Roadmap overview — replanned from today**

The original Gantt assumed a multi-role team (DevOps, Backend, Frontend, QA, Security, Clinical Lead, Data Eng, Platform Eng) running phases in parallel from Q1 2025, finishing Q4 2026. Actual progress by Sep 2026 shows a much smaller team; the schedule below replans from today against that reality — phases mostly sequential, two short overlaps where a decision (§5, §11) can be worked while the prior phase's code finishes.

| **Phase**                              | **Q3 26** | **Q4 26** | **Q1 27** | **Q2 27** | **Q3 27** | **Q4 27** | **Q1 28** | **Q2 28** |
|----------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Phase 0 – Consolidate ✓ items (finish) | ███       |           |           |           |           |           |           |           |
| Phase 1 – Terminology & CDSS hardening | ███       | ███       |           |           |           |           |           |           |
| Phase 2 – Observability & DR           |           | ███       | ███       |           |           |           |           |           |
| Phase 3 – Controlled Drug Register     |           |           | ███       | ███       |           |           |           |           |
| Phase 4 – HL7 v2 MLLP Interface Engine |           |           |           | ███       | ███       |           |           |           |
| Phase 5 – PACS / DICOMweb              |           |           |           |           | ███       | ███       |           |           |
| Phase 6 – SMART on FHIR / SSO          |           |           |           |           |           | ███       | ███       | ███       |
| Phase 7 – National Disease Programs    |           |           |           |           |           |           | ███       | ███       |
| Phase 8 – Pen Test + HIMSS readiness   |           |           |           |           |           |           |           | ███       |

Roughly 21 months from today (Q3 2026 → Q2 2028), versus ~18 months remaining on the original schedule if it hadn't slipped. Phase 3 and Phase 7 can each start their decision process (DECISIONS_PENDING.md #5, #11) now, in parallel with Phase 1 — only the code has to wait.

## Phase 0 Consolidate existing ✓ items

*Weeks 1–6 · Priority: Immediate*


**Objective**

The ten fully-implemented capabilities are production-ready in code but not yet fully operationalised. Phase 0 converts them from "passing tests" to "running in production with documented runbooks."

**Work items**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                      | **Effort** | **Owner**         |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------|
| P0-01    | Deploy MFA enforcement org-wide. Document TOTP provisioning for new staff. Verify lockout resets require admin action.                                | 3 days     | Platform Eng      |
| P0-02    | Schedule automated PostgreSQL backup test monthly. Verify restore from backup produces a consistent DB. Update docs/backup_restore_runbook.md.        | 2 days     | DevOps            |
| P0-03    | Define Celery dead-letter queue and retry policy. Wire failed task alerts to on-call channel. Document worker scaling rules.                          | 3 days     | Platform Eng      |
| P0-04    | Wire existing AuditLog SIEM export endpoint to a log aggregator (Elastic/Loki). Agree retention period with DPO (minimum 7 years per Kenya DPA 2019). | 4 days     | Security / DevOps |
| P0-05    | Conduct break-glass access quarterly review. Confirm supervisor alert delivery. Set expiry to 4 hours maximum.                                        | 1 day      | Clinical Lead     |
| P0-06    | Extend AES-128 Fernet encryption from national ID only to: lab result values, imaging report text, SOAP note body. Add migration.                     | 5 days     | Backend Eng       |
| P0-07    | Write integration test covering the full encounter lifecycle: register → triage → consult → prescribe → dispense → bill → discharge.                  | 4 days     | QA / Backend      |
| P0-08    | Document all existing FHIR R4 endpoints with OpenAPI 3.1 annotations. Validate against the official FHIR R4 CapabilityStatement schema.               | 3 days     | Backend Eng       |
| P0-09    | Configure read replica on PostgreSQL for analytics queries. Set connection pool limits per service.                                                   | 2 days     | DevOps            |
| P0-10    | Establish on-call rotation. Define P0/P1/P2 incident severity matrix. Create runbook template.                                                        | 1 day      | Eng Lead          |

**Definition of done**

- All 10 items have a Jira/Linear ticket with acceptance criteria, a reviewer, and a deploy date.

- Backup restore test passes and the result is signed off by the DevOps lead.

- Audit export is live and ingested by the SIEM within 5 minutes of event.

## Phase 1 Terminology completion & CDSS hardening

*Weeks 1–12 (parallel with Ph 0) · Priority: High*


**Objective**

Replace the 50-code ICD-10 stub and vitals-only LOINC mapping with full clinical terminology. Harden the CDSS alert engine to prevent fatigue while meeting clinical safety standards.

**1A — Full ICD-10-CM/PCS ingestion**

The WHO ICD-10-CM dataset contains 14,400+ diagnosis codes and ICD-10-PCS contains 87,000+ procedure codes. These must be ingested into a searchable PostgreSQL FTS5 table, not held in a Python list.

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                  | **Effort** | **Owner**   |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------|
| P1-01    | Obtain WHO ICD-API credentials (Client ID + Secret). Register at https://icd.who.int/icdapi. Document in DECISIONS_PENDING.md item 4.                             | 1 day      | Admin / PM  |
| P1-02    | Write icd10_importer.py: paginate the WHO API, store code + description + chapter + block in an icd10_codes table with GIN full-text index.                       | 3 days     | Backend Eng |
| P1-03    | Replace ICD10_DATABASE list in departments/medicine/prescribe.py with a DB query against icd10_codes. Maintain backward compatibility for existing coded records. | 2 days     | Backend Eng |
| P1-04    | Add Alembic migration for icd10_codes table. Seed migration runs the importer. Add to docker-compose entrypoint.                                                  | 1 day      | Backend Eng |
| P1-05    | Update FHIR R4 Condition endpoint to return WHO system URL (http://hl7.org/fhir/sid/icd-10) in coding.system.                                                     | 1 day      | Backend Eng |

**1B — SNOMED CT for problem list and procedures**

SNOMED CT is the required terminology for clinical problem lists and procedures under SMART on FHIR and Epic interoperability requirements. Obtain a NRC/SNOMED International licence (free for low-income countries under SNOMED affiliate licensing).

| **Item** | **Detail / Acceptance Criteria**                                                                                                                        | **Effort** | **Owner**          |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------------|
| P1-06    | Apply for SNOMED CT affiliate licence at snomed.org. Add to DECISIONS_PENDING.md.                                                                       | 1 day      | PM / Admin         |
| P1-07    | Download SNOMED CT International Release RF2 snapshot. Write snomed_importer.py to load concept + description + relationship tables (~350k rows).       | 5 days     | Data Eng           |
| P1-08    | Add PatientProblem.snomed_code column. Update problem list UI to typeahead-search SNOMED. Validate concept is a Clinical Finding (hierarchy 404684003). | 4 days     | Backend + Frontend |
| P1-09    | Map procedure orders to SNOMED Procedure hierarchy (71388002). Expose via FHIR R4 Procedure resource.                                                   | 3 days     | Backend Eng        |

**1C — Full LOINC for laboratory**

Current LOINC mapping covers only 6 vitals. Every orderable lab test must carry a LOINC code for interoperability with reference labs and the FHIR DiagnosticReport resource.

| **Item** | **Detail / Acceptance Criteria**                                                                                                                    | **Effort** | **Owner**   |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------|
| P1-10    | Download LOINC Table Core CSV (free registration at loinc.org). Load into loinc_codes table with LOINC_NUM, LONG_COMMON_NAME, COMPONENT, SYSTEM.    | 2 days     | Data Eng    |
| P1-11    | Add loinc_code column to LabTest model. Seed from loinc_codes for the existing test catalog (~200 tests). Require loinc_code for new test creation. | 3 days     | Backend Eng |
| P1-12    | Update FHIR R4 DiagnosticReport and Observation endpoints to include LOINC codes in coding\[\].                                                     | 2 days     | Backend Eng |

**1D — CDSS alert fatigue tuning**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                                                                                             | **Effort** | **Owner**               |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------------|
| P1-13    | Introduce alert severity tiers: CRITICAL (absolute contraindication — cannot override), HIGH (pharmacist approval required), MODERATE (warn + document), LOW (informational only, not surfaced in prescribing UI). Update SafetyAlert model. | 3 days     | Clinical Lead + Backend |
| P1-14    | Build pharmacist override workflow for HIGH-tier alerts: prescriber flags, pharmacist reviews within 2 hours, approval logged to AuditLog.                                                                                                   | 4 days     | Backend + Frontend      |
| P1-15    | Create P&T Committee alert review dashboard: shows override frequency by drug pair, alert accept/dismiss ratios, time-to-review. Monthly export to PDF.                                                                                      | 5 days     | Backend + Frontend      |
| P1-16    | Add cross-sensitivity rules: Penicillin → Cephalosporins (10% cross-react), Sulfonamides → Furosemide, NSAIDs → Aspirin. Wire to existing allergy engine.                                                                                    | 3 days     | Clinical Lead + Backend |

**Risks — Phase 1**

| **Risk**                                              | **Mitigation**                                                                                                              | **Owner**                                    |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| WHO ICD API credentials delayed                       | Use the free NLM UMLS ICD-10 flat file as a drop-in while awaiting WHO credentials. Same schema.                            | PM to escalate within 2 weeks if no response |
| SNOMED licence processing time (can take 4–6 weeks)   | Begin with SNOMED CT CORE subset (~10,000 most-used concepts) which is freely available without licence.                    | PM                                           |
| CDSS tier reclassification may remove existing alerts | Clinical lead must sign off tier mapping before deployment. Run in shadow mode (log but do not suppress) for 2 weeks first. | Clinical Lead                                |

## Phase 2 Observability, disaster recovery & security hardening

*Weeks 6–18 · Priority: High*


**Objective**

A world-class hospital cannot run on a /healthz endpoint. This phase builds the full observability, alerting, and disaster recovery capability required for a 24/7 operations centre.

**2A — Distributed tracing & error tracking**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                                      | **Effort** | **Owner**                |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------------------|
| P2-01    | Select and deploy OpenTelemetry collector (self-hosted or managed). Decision: Jaeger (self-hosted) vs Grafana Tempo (managed). Add to DECISIONS_PENDING.md item 3.                    | 2 days     | DevOps                   |
| P2-02    | Instrument Flask app with opentelemetry-instrumentation-flask. Add trace context propagation to Celery tasks. Emit spans for: DB queries \>50ms, external API calls, FHIR endpoints.  | 5 days     | Platform Eng             |
| P2-03    | Deploy Sentry (self-hosted on-premises per DPA 2019 data residency, or Sentry EU region). Configure DSN in all services. Set alert thresholds: \>5 new errors/min triggers PagerDuty. | 3 days     | DevOps                   |
| P2-04    | Build uptime dashboard (Grafana or Metabase): request rate, p50/p95/p99 latency, error rate, DB connection pool, Celery queue depth, Redis memory. 30-day retention.                  | 4 days     | DevOps + Data Eng        |
| P2-05    | Define and document SLOs: API availability ≥99.5%, prescription sign-off p95 <500ms, lab result delivery <2min from instrument receipt. Wire SLO burn-rate alerts.                  | 2 days     | Eng Lead + Clinical Lead |

**2B — Disaster recovery**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                 | **Effort** | **Owner**                 |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|---------------------------|
| P2-06    | Define RPO (Recovery Point Objective) = 4 hours, RTO (Recovery Time Objective) = 1 hour. Document in an SLA agreement signed by hospital management.             | 1 day      | Eng Lead + Hospital Admin |
| P2-07    | Implement continuous WAL archiving from PostgreSQL primary to S3/MinIO (or local NAS). Verify point-in-time recovery to within 5 minutes.                        | 4 days     | DevOps                    |
| P2-08    | Script automated DR failover: promote read replica to primary, reconfigure Flask DATABASE_URL, restart Celery workers. Run quarterly DR drill. Document results. | 5 days     | DevOps                    |
| P2-09    | Deploy hot standby PostgreSQL read replica in a separate availability zone or physical server. Configure streaming replication lag alert if lag \>30 seconds.    | 3 days     | DevOps                    |
| P2-10    | Automate DICOM backup to object storage. Define separate retention for imaging (10 years per Kenya Medical Records Act) vs transactional data (7 years).         | 2 days     | DevOps                    |

**2C — Security hardening (pre-pen-test remediation)**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                   | **Effort** | **Owner**     |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------|------------|---------------|
| P2-11    | Enforce TLS 1.2+ everywhere. Redirect all HTTP to HTTPS. Set HSTS header with max-age=31536000; includeSubDomains.                                 | 1 day      | Platform Eng  |
| P2-12    | Rotate all secrets to a secrets manager (HashiCorp Vault or AWS Secrets Manager). Remove secrets from environment variables in docker-compose.yml. | 3 days     | DevOps        |
| P2-13    | Implement Content Security Policy headers. Complete the XSS and SQL injection test suite (expand test_xss.py to cover all form inputs).            | 4 days     | Security + QA |
| P2-14    | Enable PostgreSQL row-level security (RLS) for multi-tenant data isolation between facilities (prerequisite for Phase 6 multi-facility SSO).       | 5 days     | Backend Eng   |
| P2-15    | Schedule automated dependency audit (pip-audit, npm audit) weekly in CI. Block merge if CVSS ≥7.0 advisories are unresolved.                       | 1 day      | DevOps        |

**Risks — Phase 2**

| **Risk**                                   | **Mitigation**                                                                                                                             | **Owner**    |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| Data residency conflict (OTel/Sentry SaaS) | Use self-hosted Sentry (sentry.io/self-hosted) or GlitchTip. OTel collector is always self-hosted.                                         | DPO + DevOps |
| DR drill disrupts live services            | Run first DR drill on a staging environment that mirrors production. Move to full production drill only after 2 successful staging drills. | DevOps Lead  |
| Secrets rotation breaks running containers | Use Vault dynamic secrets with lease renewal. Roll out service-by-service with rollback plan for each.                                     | Platform Eng |

## Phase 3 Controlled drug register

*Weeks 8–20 · Priority: Legal blocker*


**Objective**

Schedule II narcotics (morphine, pethidine, fentanyl) and Schedule IV psychotropics (diazepam, lorazepam, midazolam) cannot be legally dispensed without a dual-signature register under Kenya Pharmacy and Poisons Board (PPB) regulation. This phase builds that register.

**Prerequisites (from DECISIONS_PENDING.md item 5)**

- **Decision required:** Which roles qualify as the mandatory second signatory? (Two licensed pharmacists, or pharmacist + ward nurse-in-charge?)

- **Decision required:** Stock reconciliation schedule: per shift, daily, or weekly?

- **Decision required:** Separate ledgers for Schedule II vs Schedule IV, or unified controlled log?

**Work items**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                                               | **Effort** | **Owner**               |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------------|
| P3-01    | Add ControlledDrug boolean + schedule_class (II/IV/V) column to Drug model. Seed classification for all current formulary drugs. Alembic migration required.                                   | 2 days     | Backend Eng             |
| P3-02    | Build ControlledDrugDispense model: patient_id, drug_id, dose_mg, dispense_datetime, primary_pharmacist_id, second_signatory_id, second_signatory_role, balance_after, witness_signature_hash. | 3 days     | Backend Eng             |
| P3-03    | Enforce dual-signature workflow in dispensing UI: after primary pharmacist confirms, system locks dispense and requires a second named signatory. Neither can sign for both.                   | 5 days     | Backend + Frontend      |
| P3-04    | Build running balance ledger: each dispense and receipt updates a ControlledDrugBalance row. Any negative balance triggers a CRITICAL alert to pharmacy supervisor.                            | 3 days     | Backend Eng             |
| P3-05    | Build shift reconciliation screen: pharmacist enters physical count, system computes variance vs ledger balance, flags discrepancies \>0 for immediate investigation.                          | 4 days     | Backend + Frontend      |
| P3-06    | Build PPB-format printed register: exportable PDF showing date, patient, dose, balance, both signatures. Must match the official PPB Controlled Drugs Register format.                         | 4 days     | Backend + Clinical Lead |
| P3-07    | Write tests: dispense without second signatory returns 403; negative balance alert fires; shift reconciliation variance \>0 blocks close.                                                      | 3 days     | QA                      |
| P3-08    | Clinical Lead to review and sign off the workflow against PPB Pharmacy Act Cap 244 requirements before any live dispense.                                                                      | 1 day      | Clinical Lead           |

**Risks — Phase 3**

| **Risk**                                           | **Mitigation**                                                                                                                                                                                                  | **Owner**                  |
|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| PPB format changes since last published regulation | Obtain the current PPB Controlled Drug Register prescribed format directly from the PPB compliance officer. Do not rely on historic documents.                                                                  | Clinical Lead / Compliance |
| Second signatory not available on night shift      | The system must not allow dispense to proceed without a signatory. If no second signatory is available, the on-call clinical officer must be documented as the second signatory. Workflow must prompt for this. | Clinical Lead              |

## Phase 4 HL7 v2 MLLP interface engine

*Weeks 14–28 · Priority: High*


**Objective**

Physical lab analyzers (Roche Cobas, Abbott i-STAT, Sysmex, Mindray) and registration ADT feeds communicate over HL7 v2.x via MLLP (Minimal Lower Layer Protocol) on TCP port 2575. Without this interface engine, all lab results and ADT events require manual data entry — unacceptable for a world-class hospital.

**Architecture decision**

Deploy a dedicated HL7 interface engine as a separate service alongside the Flask application. Recommended: open-source Mirth Connect (NextGen Connect) or commercial Rhapsody. Both expose a web admin console and support bidirectional HL7 v2 ↔ FHIR R4 transformation. The HMIS acts as the "system of record" receiving transformed FHIR resources from the interface engine.

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                                                     | **Effort** | **Owner**                  |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|----------------------------|
| P4-01    | Select interface engine: Mirth Connect (self-hosted, free) vs Rhapsody (licensed). Document decision, estimated licensing cost, support model. Add to DECISIONS_PENDING.md.                          | 3 days     | Arch + PM                  |
| P4-02    | Add mirth-connect container to docker-compose.yml. Configure channels for: ORM^O01 (lab order outbound), ORU^R01 (result inbound), ADT^A01/A03/A08 (admit/discharge/update).                         | 5 days     | DevOps + Integration Eng   |
| P4-03    | Build HMIS HL7 result receiver: POST /api/hl7/oru endpoint accepts transformed FHIR DiagnosticReport from Mirth. Updates LabResult, triggers LIS panic alert if critical value.                      | 6 days     | Backend Eng                |
| P4-04    | Build HMIS ADT sender: on Patient.create/update/admit/discharge, publish ADT FHIR message to Mirth outbound channel for downstream subscriber systems.                                               | 4 days     | Backend Eng                |
| P4-05    | Instrument with lab analyzer vendor. Each analyzer requires a separate Mirth channel with vendor-specific HL7 dialect. Test with: (a) Roche Cobas HL7 simulator, (b) Sysmex XN-series HL7 simulator. | 8 days     | Integration Eng + Lab Team |
| P4-06    | Build LIS order workflow: clinician creates lab order in HMIS → ORM^O01 sent to analyzer → analyzer runs test → ORU^R01 received → result appears in HMIS within 2 minutes.                          | 5 days     | Backend + Frontend         |
| P4-07    | Extend panic alert workflow: if result value triggers LIS panic threshold (already implemented), fire immediate SMS + in-app push to the ordering clinician.                                         | 3 days     | Backend Eng                |
| P4-08    | Load test the MLLP connection at 50 concurrent ORU messages (realistic peak for a busy lab). Verify no message loss with Mirth persistence enabled.                                                  | 2 days     | QA + DevOps                |

**Risks — Phase 4**

| **Risk**                                    | **Mitigation**                                                                                                                                                                                                    | **Owner**       |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| Analyzer vendor HL7 dialect is non-standard | Every major analyzer vendor ships a slightly different HL7 v2 dialect. Budget 3–5 days per analyzer model for Mirth channel customisation. Collect analyzer HL7 conformance statements before development starts. | Integration Eng |
| MLLP connection stability on hospital LAN   | Hospital networks frequently drop long-lived TCP connections. Configure Mirth keepalive and reconnect-on-failure. Monitor with the observability stack from Phase 2.                                              | DevOps          |
| Result delivery delay \>2 min SLA           | MLLP is synchronous; the bottleneck is usually Mirth transform. Profile with real message volume before setting the SLA.                                                                                          | Eng Lead        |

## Phase 5 PACS / DICOMweb integration

*Weeks 20–32 · Priority: Medium*


**Objective**

DICOM metadata capture exists. Radiologists need a full PACS (Picture Archiving and Communication System) that stores, retrieves, and displays DICOM images. Deploy Orthanc (open-source PACS) wired to the HMIS imaging workflow and a DICOMweb viewer.

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                                        | **Effort** | **Owner**                   |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------------------------|
| P5-01    | Deploy Orthanc PACS as a Docker container with a persistent volume. Configure DICOMweb (WADO-RS, STOW-RS, QIDO-RS) endpoints. Add to docker-compose.yml.                                | 4 days     | DevOps                      |
| P5-02    | Integrate OHIF Viewer v3 (open-source) as an iframe or React component in the imaging department UI. OHIF loads studies from Orthanc via DICOMweb WADO-RS.                              | 6 days     | Frontend + DevOps           |
| P5-03    | Wire imaging order to PACS: when radiographer marks a study complete, Orthanc study UID is written to RequestedImage.orthanc_uid. HMIS displays a "View images" link that opens OHIF.   | 4 days     | Backend + Frontend          |
| P5-04    | Configure DICOM send from modality to PACS: set up DICOM C-STORE from the hospital CT/X-ray machines to the Orthanc AE title. Test with DICOM conformance statement from each modality. | 5 days     | Integration Eng + Radiology |
| P5-05    | Implement FHIR R4 ImagingStudy resource: maps Orthanc study UID, series, and instance references to the FHIR ImagingStudy structure. Expose via /api/fhir/R4/ImagingStudy.              | 3 days     | Backend Eng                 |
| P5-06    | Migrate existing DICOM metadata rows to include orthanc_uid. Backfill for any historical studies loaded into Orthanc.                                                                   | 2 days     | Backend Eng                 |
| P5-07    | Configure Orthanc storage to write to the object storage backend resolved in DECISIONS_PENDING.md item 7 (MinIO/S3). Set 10-year retention policy.                                      | 2 days     | DevOps                      |
| P5-08    | Radiologist acceptance test: display a CT chest, MRI brain, and chest X-ray series in OHIF with measurements, windowing, and multi-planar reconstruction (MPR).                         | 3 days     | Radiology + QA              |

**Risks — Phase 5**

| **Risk**                                     | **Mitigation**                                                                                                                                                                          | **Owner**                   |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| Modality DICOM conformance differs from spec | Older CR/DR machines may only support DICOM C-STORE, not DICOMweb. Orthanc bridges both; confirm per-modality conformance before assuming WADO-RS is available.                         | Integration Eng + Radiology |
| OHIF viewer performance on slow hospital LAN | DICOM studies can be 500MB+. Deploy Orthanc with JPEG-LS lossless transfer syntax for lossy-acceptable views. Radiologist workflows requiring lossless must use dedicated workstations. | DevOps + Radiology Lead     |

## Phase 6 SMART on FHIR, OAuth2 & enterprise identity (SSO)

*Weeks 24–40 · Priority: Critical*


**Objective**

This is the single most important gap for Tier 1 hospital interoperability. SMART on FHIR is the authentication and authorisation framework used by Epic, Cerner, Meditech, and every major US/EU health system for third-party app launch and patient data access. Without it, the HMIS cannot co-exist with or exchange data from any major EHR.

**6A — OAuth2 authorisation server**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                                              | **Effort** | **Owner**    |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------|
| P6-01    | Deploy an OIDC-compliant authorisation server. Recommended: Keycloak (open-source, self-hosted). Add to docker-compose.yml. Configure realms for staff and patient portal.                    | 5 days     | Platform Eng |
| P6-02    | Migrate existing Flask session auth to OAuth2 Bearer token validation. Flask routes check JWT access token from Keycloak. Maintain backward compat for 4 weeks via dual auth.                 | 8 days     | Backend Eng  |
| P6-03    | Implement SMART on FHIR launch sequence: EHR launch (iss parameter), standalone launch, and patient context selection. FHIR SMART configuration endpoint at /.well-known/smart-configuration. | 6 days     | Backend Eng  |
| P6-04    | Implement SMART scopes: patient/\*.read, user/\*.read, offline_access, launch/patient, launch/encounter. Scope enforcement must be checked on every FHIR resource request.                    | 4 days     | Backend Eng  |
| P6-05    | Add FHIR R4 CapabilityStatement with security.service = "SMART-on-FHIR" and .rest.security.extension SMART capabilities array.                                                                | 2 days     | Backend Eng  |

**6B — LDAP / Active Directory federation**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                             | **Effort** | **Owner**         |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------|
| P6-06    | Configure Keycloak LDAP user federation to the hospital Active Directory (or OpenLDAP). Map AD groups to HMIS roles (doctor, nurse, pharmacist, admin, lab-tech, radiology). | 4 days     | Platform Eng + IT |
| P6-07    | Implement SCIM 2.0 provisioning endpoint for automated user creation/deactivation when staff join or leave via HR system. Map to existing User model.                        | 5 days     | Backend Eng       |
| P6-08    | Configure Keycloak session revocation: deactivating a user in AD propagates to Keycloak within 60 seconds and invalidates all active sessions.                               | 2 days     | Platform Eng      |

**6C — Patient portal SSO**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                            | **Effort** | **Owner**          |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------------|
| P6-09    | Migrate PatientUser authentication to Keycloak with separate realm. Support: username/password, TOTP MFA, and optional social login (Google/Apple) for patient convenience. | 5 days     | Backend + Frontend |
| P6-10    | Implement patient-facing SMART standalone launch: patient logs in, consents to data scope, receives access token scoped to their own records only.                          | 4 days     | Backend Eng        |

**Risks — Phase 6**

| **Risk**                                                    | **Mitigation**                                                                                                                                            | **Owner**         |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Keycloak adds operational complexity                        | Keycloak requires its own PostgreSQL database, HA deployment, and monitoring. Size this as a separate project — it is infrastructure, not just a library. | Platform Eng      |
| Legacy Flask session migration breaks existing integrations | Run dual auth (session + Bearer token) for 4 weeks minimum. Monitor error rates. Deprecate sessions only after zero errors for 2 weeks.                   | Backend Eng Lead  |
| AD schema differs from expected LDAP schema                 | Map AD attributes explicitly in Keycloak LDAP config. Test with a read-only service account before enabling write-back.                                   | IT + Platform Eng |

## Phase 7 National disease program modules (HIV/ART, TB, Malaria)

*Weeks 32–52 · Priority: Medium*


**Objective**

The MCH/ANC module demonstrates the depth required for a national program implementation. HIV/ART, TB, and Malaria need the same treatment: dedicated data models, regimen-line tracking, adherence monitoring, and MOH KHIS/DHIS2 reporting integration.

**Prerequisites (from DECISIONS_PENDING.md item 11)**

- **Clinical reviewer:** Each program requires a named clinical reviewer (HIV clinician, TB clinician, malaria specialist) who signs off the data model and reporting fields.

- **Scope decision:** Confirm whether all three programs are in scope for initial deployment, or only the ones relevant to the patient population at the target facility.

- **Reporting integration:** Decide whether each program ties into the existing DHIS2 exporter from day one, or is built standalone first.

**7A — HIV / ART module**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                                             | **Effort** | **Owner**               |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------------|
| P7-01    | Define data model with HIV clinician: ARTEnrollment, ARTRegimen (line 1/2/3), AdherenceVisit, ViralLoad, CD4Count, WHOStage. Review against MOH HTS/ART register format.                     | 5 days     | Clinical Lead + Backend |
| P7-02    | Build ARTEnrollment workflow: link to patient, capture enrolment date, unique ART number, baseline CD4/WHO stage. Prevent duplicate enrolments.                                              | 4 days     | Backend + Frontend      |
| P7-03    | Build regimen tracking: prescribe first-line regimen from MOH Kenya ART formulary. Flag if patient is on unsupported combination. Switch to second-line requires clinical reason + approval. | 6 days     | Backend + Clinical Lead |
| P7-04    | Build adherence visit: date, pills dispensed, pills returned, calculated adherence %, adherence category (good/fair/poor). Flag missed visits after 7 days via Celery task.                  | 4 days     | Backend + Frontend      |
| P7-05    | Wire to DHIS2 exporter: MOH 731 ART cohort report (new enrolments, active on ART, viral load suppression). Monthly automated export.                                                         | 4 days     | Backend Eng             |

**7B — TB / DOTS module**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                         | **Effort** | **Owner**               |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------------|
| P7-06    | Data model with TB clinician: TBEnrollment, DOTSPhase (intensive/continuation), SputumResult, XpertResult, DrugSensitivity, TreatmentOutcome.                            | 4 days     | Clinical Lead + Backend |
| P7-07    | Build DOTS adherence tracker: daily observed therapy record. Community health worker (CHW) can mark doses via patient portal or mobile. Flag missed 2+ consecutive days. | 5 days     | Backend + Frontend      |
| P7-08    | Wire to DHIS2 exporter: MOH TB register (new smear-positive, treatment success rate, default rate). Quarterly export.                                                    | 3 days     | Backend Eng             |

**7C — Malaria module**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                            | **Effort** | **Owner**               |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------------|
| P7-09    | Data model: MalariaTest (RDT/microscopy), species (P. falciparum/P. vivax/P. malariae), parasite density, treatment prescribed, treatment outcome.          | 3 days     | Clinical Lead + Backend |
| P7-10    | Wire to DHIS2 exporter: MOH 705A/B malaria cases. Aggregate by facility and age group. Monthly automated export (already partially in dhis2_exporter.py).   | 2 days     | Backend Eng             |
| P7-11    | Outbreak signal: if malaria cases in any 7-day window exceed 2× the 4-week rolling average, generate a public health alert to the facility medical officer. | 3 days     | Backend Eng             |

**Risks — Phase 7**

| **Risk**                                       | **Mitigation**                                                                                                                                                     | **Owner**     |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| Clinical reviewer availability                 | Clinicians are busy. Timebox the data model design workshop to 2 sessions of 2 hours each. Use the existing MCH module as a reference template.                    | PM            |
| MOH ART formulary changes faster than the HMIS | Store regimens in a database table (not code constants) so a non-developer can update the formulary. Add a formulary change log.                                   | Clinical Lead |
| CHW mobile access for DOTS                     | CHWs may not have smartphones. Design the DOTS daily tick as a feature of the patient portal SMS confirmation flow first; smartphone app is a Phase 2 enhancement. | Product Lead  |

## Phase 8 Penetration test + HIMSS EMRAM validation

*Weeks 44–56 · Priority: Gate*


**Objective**

Phase 8 is a validation gate, not a feature sprint. It confirms that everything built in Phases 0–7 is production-secure and meets the HIMSS Electronic Medical Record Adoption Model (EMRAM) criteria for Stage 6 (closed-loop medication administration) and Stage 7 (complete EMR, external interoperability).

**8A — Third-party penetration test**

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                        | **Effort**           | **Owner**          |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| P8-01    | Engage a CREST-accredited (or equivalent) penetration testing firm. Provide the scope document from docs/security_audit_readiness.md. Budget 2–4 weeks of testing.      | 1 week (procurement) | CISO / Admin       |
| P8-02    | Scope: web application (OWASP Top 10), API (FHIR endpoints, SMART on FHIR auth), infrastructure (Docker host, PostgreSQL port exposure), break-glass escalation abuse.  | External             | Pen Test Firm      |
| P8-03    | Remediate all Critical and High findings before go-live. Medium findings must have accepted risk or remediation plan. Document in a remediation tracker signed by CISO. | Post-test            | Security + Backend |
| P8-04    | Re-test after remediation ("clean pass"). Retain pen test report for regulator inspection.                                                                              | 1 week               | Pen Test Firm      |

**8B — HIMSS EMRAM Stage 6 requirements**

Stage 6 requires closed-loop medication administration. The following must be complete and verifiable:

- Full CDSS (drug-drug interactions, allergy alerts, dose checking) — Phase 1

- Bar-coded medication administration (BCMA) — clinician scans patient wristband + drug barcode before administration. Requires hardware procurement.

- Full physician documentation (SOAP notes) in EMR — existing

- Full nursing documentation and care plan — existing

- Controlled drug register — Phase 3

- PACS / DICOMweb for radiology — Phase 5

| **Item** | **Detail / Acceptance Criteria**                                                                                                                          | **Effort**        | **Owner**                 |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|---------------------------|
| P8-05    | Procure and deploy BCMA scanner hardware (Zebra DS2208 or equivalent) in all ward medication stations. Build scan-to-MAR workflow in HMIS.                | 10 days           | IT + Backend + Nursing    |
| P8-06    | Engage a HIMSS Analytics consulting partner for an EMRAM Stage 6 gap assessment. The formal assessment requires on-site visits and structured interviews. | 3 weeks (ext)     | Hospital Admin + Eng Lead |
| P8-07    | Complete Stage 6 application, evidence package, and site visit. Expected cycle time: 3–6 months from application submission.                              | External timeline | Hospital Admin            |

**8C — HIMSS EMRAM Stage 7 requirements**

Stage 7 adds external data sharing and analytics. Requirements beyond Stage 6:

- Continuity of care documents (CCDs) shared with external providers — requires SMART on FHIR (Phase 6)

- Business intelligence and analytics warehouse — requires dedicated data pipeline beyond the current analytics dashboard

- Data warehouse for population health analytics — out of scope for this plan; estimate 6 months additional work

| **Item** | **Detail / Acceptance Criteria**                                                                                                                                              | **Effort** | **Owner**   |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------|
| P8-08    | Implement CCD (C-CDA) generation from FHIR resources: Patient, Condition, Medication, AllergyIntolerance, Immunization, Result. Test against the Cerner/Epic C-CDA validator. | 8 days     | Backend Eng |
| P8-09    | Deploy a read-optimised analytics database (PostgreSQL read replica with pg_analytics or ClickHouse). Build ETL pipeline from transactional DB to analytics DB.               | 10 days    | Data Eng    |

**3. Team and resource model**

The following roles are required to execute this plan. Roles marked \* are currently filled by the existing team based on the HMIS codebase evidence. New hires or contractors are needed for the remainder.

| **Role**                        | **Key responsibilities**                                            | **FTE** | **Status**                 |
|---------------------------------|---------------------------------------------------------------------|---------|----------------------------|
| Engineering Lead \*             | Architecture decisions, technical reviews, sprint planning          | 1.0     | **Existing**               |
| Backend Engineer ×2 \*          | Python/Flask, SQLAlchemy, FHIR API, CDSS, billing                   | 2.0     | **Existing**               |
| Frontend Engineer \*            | Jinja templates, clinical UI, patient portal                        | 1.0     | **Existing**               |
| DevOps / Platform Engineer      | Docker, PostgreSQL HA, CI/CD, observability, Keycloak               | 1.0     | **New hire**               |
| Data Engineer                   | ICD-10/SNOMED/LOINC ingestion, DHIS2 pipeline, analytics DB         | 0.5     | **New hire / contractor**  |
| Integration Engineer            | Mirth Connect, HL7 v2, DICOM, Orthanc                               | 1.0     | **New hire / contractor**  |
| Security Engineer               | Pen test prep, secrets management, RLS, SIEM                        | 0.5     | **Contractor for Phase 8** |
| Clinical Lead \*                | CDSS sign-off, ICD-10 validation, CDSS alert tuning, PPB workflow   | 0.25    | **Existing (part-time)**   |
| Clinical Informatics Specialist | SNOMED/LOINC mapping, HIMSS EMRAM assessment, CCDs                  | 0.5     | **New hire**               |
| QA Engineer                     | Integration tests, load tests, BCMA workflow testing                | 1.0     | **New hire**               |
| Project Manager                 | Sprint planning, DECISIONS_PENDING.md governance, stakeholder comms | 0.5     | **Existing or new**        |

**4. Budget estimates**

All figures are estimates. Software costs assume open-source tooling is chosen wherever available. Licensing costs depend on final vendor decisions made in DECISIONS_PENDING.md.

| **Item**                               | **One-time (USD)** | **Annual (USD)** | **Notes**                                               |
|----------------------------------------|--------------------|------------------|---------------------------------------------------------|
| Keycloak (self-hosted)                 | \$0                | \$0              | Open-source; compute cost only                          |
| Mirth Connect (open-source)            | \$0                | \$0              | NextGen Connect Community Edition                       |
| Orthanc PACS                           | \$0                | \$0              | Open-source; storage cost separate                      |
| OHIF Viewer v3                         | \$0                | \$0              | Open-source (MIT)                                       |
| OpenTelemetry collector                | \$0                | \$0              | Open-source                                             |
| Sentry (self-hosted)                   | \$0                | \$500            | Server hosting cost estimate                            |
| ICD-10-CM/PCS WHO credentials          | \$0                | \$0              | Free for LMICs via WHO affiliate                        |
| SNOMED CT licence                      | \$0                | \$0              | Free for LMICs via SNOMED affiliate                     |
| LOINC licence                          | \$0                | \$0              | Free for all uses                                       |
| PostgreSQL HA infrastructure           | \$2,000            | \$4,800          | Primary + replica + backup storage                      |
| BCMA scanners (10 units)               | \$3,500            | \$0              | Zebra DS2208 ~\$350/unit                                |
| Penetration test (external firm)       | \$12,000           | \$6,000          | Initial + annual re-test                                |
| HIMSS EMRAM assessment                 | \$8,000            | \$0              | One-time advisory engagement                            |
| HIMSS EMRAM certification fee          | \$4,500            | \$0              | Per HIMSS published fee schedule                        |
| DevOps + Integration contractor (6 mo) | \$45,000           | \$0              | Contract; convert to FTE post-implementation            |
| Clinical Informatics Specialist        | \$0                | \$28,000         | Annual salary estimate (Kenya market)                   |
| **TOTAL (estimate)**                   | **~\$75,000**      | **~\$39,300**    | **Excludes new engineer salaries already in headcount** |

**5. Governance and decision-making**

**DECISIONS_PENDING.md protocol**

The existing DECISIONS_PENDING.md file is the single source of truth for items that require human sign-off. No code that implements a hard-stop item is merged to main until that item has a resolution recorded with: (a) the decision made, (b) the name of the decision-maker, and (c) the date.

**Architecture Review Board (ARB)**

- Meets fortnightly.

- Members: Engineering Lead, Clinical Lead, DevOps Lead, Hospital CIO / IT Director.

- Agenda: review open DECISIONS_PENDING items, approve schema changes, review security posture.

**Clinical Safety Board (CSB)**

- Meets monthly.

- Members: Medical Officer, Clinical Pharmacist, Clinical Lead (HMIS), Nursing Lead.

- Agenda: CDSS alert fatigue report, controlled drug register reconciliation exceptions, break-glass access log review.

**Change management**

- All schema migrations require a peer review from a second engineer before merge.

- All changes to CDSS alert rules require sign-off from Clinical Lead and Clinical Pharmacist.

- No change to encryption configuration without CISO sign-off and a data migration plan.

- Production deployments occur during the maintenance window (Sunday 02:00–04:00 local time) unless a P0 incident requires an emergency push.

**Document information**

Version 1.0 · Architecture Review Board · Generated from HMIS gap analysis · 2025

*This document is confidential. Internal use only. Do not distribute outside the project team.*
