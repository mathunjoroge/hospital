# Health Information Management System (HIMS)

An enterprise-grade, privacy-first Health Information Management System (HIMS) designed for hospitals and clinical health facilities. Fully compliant with the **Kenya Data Protection Act 2019** and standards-interoperable with **HL7 FHIR R4** and **Kenya Health Information System (KHIS / DHIS2)**.

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

## 📜 License
Internal Enterprise & Clinical Use Only. Confidential.
