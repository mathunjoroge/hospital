# Health Information Management System (HIMS)

An enterprise-grade, privacy-first Health Information Management System (HIMS) designed for hospitals and clinical health facilities. Built to align with the **Kenya Data Protection Act 2019**, and standards-interoperable with **HL7 FHIR R4**, **HL7 v2 (MLLP)**, **DICOM/DICOMweb**, and the **Kenya Health Information System (KHIS / DHIS2)**.

> **Status:** Actively developed, internally tested (795/796 automated tests passing). Standards support is implemented and self-verified; it has **not** undergone formal third-party certification or audit. See [Compliance & Standards Status](#-compliance--standards-status) below before any production or clinical deployment decision.

---

## 🌟 Key Features

### 🏥 Clinical Departments & Operations
- **Patient Records & Registration**: UUID-based patient identification, duplicate detection & record merge engine, Subject Access Request (SAR) DPA 2019 compliance.
- **Triage & Nursing**: ESI Emergency Severity Index triage scoring, vitals tracking, Medication Administration Records (MAR), and nursing care plans.
- **Consultation & Medicine**: Digital SOAP notes, ICD-10 clinical diagnosis coding, clinical NLP chatbot integration, and inpatient bed management.
- **Pharmacy & Stock Control**: e-Prescribing validation, FEFO (First-Expired, First-Out) inventory control, drug interaction safety checks, and cheminformatics.
- **Laboratory Information System (LIS)**: Test ordering, result verification, and critical panic value alerts.
- **Imaging & Radiology**: Diagnostic imaging requests and DICOM metadata integration.
- **Unified Billing & Insurance**: Automated invoice generation, partial payment allocation, M-Pesa (Safaricom Daraja API) mobile money integration, and SHA / SHIF statutory insurance claim adjudication.

### 🌐 Interoperability & AI Governance
- **HL7 FHIR R4 API**: Standardized JSON endpoints for `Patient`, `Observation`, `Condition`, `DiagnosticReport`, and `MedicationRequest` resources.
- **DHIS2 / KHIS Monthly Exporter**: Automated aggregation of Kenya MOH 705A/B, MOH 711, and MOH 731 monthly reports with downloadable `dataValueSets` JSON and CSV.
- **AI Governance & Audit**: Structured SIEM audit logging, input validation guardrails, and explicit mode disclosure (Live LLM vs Offline Fallback vs RDKit).

### 🔒 Security, Compliance & Resilience
- **Persistent Database Audit Trail**: Append-only DB audit log tracking clinical views, billing payments, and administrative actions.
- **Data Encryption at Rest**: Field-level AES-128-CBC + HMAC (Fernet) encryption for sensitive patient identity attributes.
- **Offline-First PWA Support**: Service Worker caching, Web App Manifest, and automatic real-time network connectivity monitoring banner.
- **Authentication & RBAC**: Role-based access control, TOTP multi-factor authentication (MFA), 5-attempt account lockout, and password complexity enforcement.

---

## ✅ Compliance & Standards Status

Standards support is built and covered by automated tests, but **"interoperable with" and "aligned with" are not the same as formally certified**. Treat this table as the source of truth over marketing language elsewhere in this document.

| Area | Implementation status | Formal certification / sign-off |
|---|---|---|
| HL7 FHIR R4, HL7 v2 (MLLP), DICOM/DICOMweb | ✅ Implemented, covered by automated tests | Not independently conformance-tested (e.g. no Touchstone/IHE Connectathon results) |
| ICD-10 (WHO ICD-API), SNOMED CT, LOINC (NLM UMLS) | ✅ Implemented, live + cached lookups | SNOMED CT/LOINC production licensing is **interim** — formal affiliate licensing sign-off from facility management is still pending (see `DECISIONS_PENDING.md` §21–22) |
| Kenya Data Protection Act 2019 (consent, SAR export, erasure/anonymization) | ✅ Implemented, tested | Data-residency architecture decision is an open **HARD STOP** (see `DECISIONS_PENDING.md` §7) |
| HIPAA §164.312-style audit controls / HITRUST CSF domains | ✅ Implemented (hash-chained audit log, compliance engine), tested | Self-assessed only — not HITRUST-assessed or externally audited |
| WCAG 2.1 AA | ✅ Automated baseline fixes applied | Manual audit by an accredited accessibility specialist not yet performed (see `docs/accessibility_audit_report.md`) |
| Penetration testing | Internal control mapping only | No third-party penetration test has been performed (see `docs/security_audit_readiness.md`) |
| National program modules (HIV/ART, TB, Malaria), controlled-drug register | ✅ Implemented & tested (dedicated HIV/ART, TB/DOTS, Malaria modules & Controlled-Drug Register UI) | Production deployment pending clinical/legal sign-off (see `DECISIONS_PENDING.md` §5, 11) |
| KRA eTIMS Tax Compliance | ✅ Admin Console UI Implemented ([/admin/etims](file:///home/mathu/projects/hospital/templates/admin/etims.html)) | Admin enters facility KRA PIN & VSCU device serials; live production push pending active KRA PIN sign-off (see `DECISIONS_PENDING.md` §6) |

For the full list of open items, see [`DECISIONS_PENDING.md`](DECISIONS_PENDING.md).

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.12+
- PostgreSQL 16+ (or SQLite for development)
- Redis 7+

### Local Installation

```bash
# 1. Clone repository
git clone https://github.com/mathunjoroge/hospital.git
cd hospital

# 2. Set up virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export FLASK_APP=app.py
export SECRET_KEY="your-production-secret-key"
export ENCRYPTION_KEY="your-base64-32byte-fernet-key"

# 5. Run database migrations
flask db upgrade

# 6. Start application server
python app.py
```

---

## 🐳 Docker Deployment

Run the complete multi-container stack (PostgreSQL, Redis, Flask application, and NLP service) using Docker Compose:

```bash
# Start all services in detached mode
docker-compose up -d --build

# View container logs
docker-compose logs -f flask

# Check stack status
docker-compose ps
```

---

## 🧪 Testing

Execute automated unit and integration tests with coverage:

```bash
# Run all unit test suites
PYTHONPATH=. venv/bin/pytest -v

# Run with coverage report
PYTHONPATH=. venv/bin/pytest --cov=. --cov-report=term-missing
```

**Latest full-suite run:** 795 passed, 1 skipped, 0 failed (796 tests total), executed against an isolated in-memory SQLite instance. Static analysis with `bandit -r departments app.py` reports 0 medium/high-severity findings. `ruff check .` reports ~371 findings, nearly all low-severity style/modernization items (no security-relevant issues).

---

## 📡 Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | `GET` | System health check (DB connectivity & disk space) |
| `/api/fhir/R4/Patient/<id>` | `GET` | FHIR R4 Patient resource JSON |
| `/api/fhir/R4/Observation?patient=<id>` | `GET` | FHIR R4 Observation bundle (Vitals & Labs) |
| `/api/khis/export/dhis2_json` | `GET` | KHIS / DHIS2 monthly `dataValueSets` JSON |
| `/api/khis/export/csv` | `GET` | KHIS / DHIS2 monthly CSV report export |
| `/admin/audit-trail/export` | `GET` | SIEM JSON audit log export |

---

## 📚 Operational Documentation & Technical Runbooks

- **Disaster Recovery**: [Backup & Emergency Restore Runbook](docs/backup_restore_runbook.md)
- **Performance Benchmarks**: [Load & Concurrency Baseline Results](docs/load_test_results.md)
- **Interfacing Research**: [HL7 v2 Lab-Instrument & DICOM Web Architecture](docs/pacs_hl7_interfacing_research.md)
- **Security Readiness**: [Third-Party Security Audit Scope](docs/security_audit_readiness.md)
- **Clinical Governance**: [Clinical Safety Review Packaging](docs/clinical_safety_review_packaging.md)
- **Accessibility**: [WCAG 2.1 AA Compliance Baseline Report](docs/accessibility_audit_report.md)
- **Pending Human Decisions**: [Decisions Pending Stakeholder Sign-Off](DECISIONS_PENDING.md)

---

## 📜 License
Internal Enterprise & Clinical Use Only. Confidential.
