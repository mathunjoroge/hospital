# Health Information Management System (HIMS)

**An open-source, privacy-first hospital information system built for Kenyan health facilities: from patient registration to M-Pesa billing and monthly KHIS/DHIS2 reporting.**

<!-- Replace OWNER/REPO and the workflow filename with your real CI workflow (see .github/workflows) -->
[![CI](https://github.com/mathunjoroge/hospital/actions/workflows/ci.yml/badge.svg)](https://github.com/mathunjoroge/hospital/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![FHIR R4](https://img.shields.io/badge/HL7%20FHIR-R4-orange)
![DHIS2](https://img.shields.io/badge/KHIS-DHIS2-green)

Designed around the Kenya Data Protection Act 2019, with standards-based interoperability through **HL7 FHIR R4** and **Kenya Health Information System (KHIS / DHIS2)**.

<!-- Add a hero screenshot or short GIF here, e.g.:
![Dashboard](docs/images/dashboard.png)
-->

---

## Contents

- [Why HIMS](#why-hims)
- [How it compares](#how-it-compares)
- [Features](#features)
- [Quickstart](#quickstart)
- [Docker deployment](#docker-deployment)
- [Configuration](#configuration)
- [Testing](#testing)
- [API reference](#api-reference)
- [Security and privacy](#security-and-privacy)
- [Project status](#project-status)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Why HIMS

Most facilities need the same handful of things working together: records, clinical workflows, pharmacy, lab, billing, and the monthly reports the Ministry of Health expects. HIMS puts them in one deployable stack with Kenya-specific requirements built in rather than bolted on:

- **National reporting out of the box:** MOH 705A/B, 711 and 731 monthly aggregates, exported as DHIS2 `dataValueSets` JSON or CSV.
- **Local payments and insurance:** M-Pesa (Safaricom Daraja API) and SHA / SHIF claim adjudication alongside partial-payment invoicing.
- **Standards-based data exchange:** FHIR R4 endpoints for core clinical resources.
- **Privacy by design:** field-level encryption, an append-only audit trail, and subject access request support aligned with the Data Protection Act 2019.
- **Works with unreliable connectivity:** an offline-first PWA with automatic network status monitoring.

---

## How it compares

Established open-source health systems such as [OpenMRS](https://openmrs.org), KenyaEMR (built on OpenMRS) and [Bahmni](https://www.bahmni.org) have years of field deployment behind them, and they are a strong choice for many facilities. HIMS takes a different approach:

| | HIMS | Established OpenMRS-based systems |
| --- | --- | --- |
| **Stack** | Python / Flask, PostgreSQL, Redis, Celery | Java-based platform with a modular ecosystem |
| **Scope** | One codebase covering clinical, pharmacy, lab, imaging and billing | Modular; capabilities depend on the distribution and modules you install |
| **Kenya-specific workflows** | M-Pesa, SHA / SHIF claims and MOH 705/711/731 to KHIS exports are built into the core | Available through implementations and add-ons, varying by distribution |
| **Deployment** | Docker Compose stack, offline-first PWA | Varies by distribution |
| **Maturity** | Newer, with less field deployment | Long track record and large communities |

**HIMS may suit you if** you want a Python codebase that's easy to read and extend, or a single integrated system with Kenyan billing and reporting already wired in.

**A more established system may suit you if** you need a proven, widely supported platform today, or a large community and module ecosystem.

The right choice depends on your facility, your team's skills and your regulatory context, so please evaluate carefully. HIMS is also designed to exchange data over FHIR R4, so it doesn't have to be all-or-nothing.

---

## Features

### Clinical departments and operations

| Area | Capabilities |
| --- | --- |
| **Patient records** | UUID-based identification, duplicate detection and record merge, Subject Access Request (SAR) support |
| **Triage and nursing** | ESI (Emergency Severity Index) triage scoring, vitals tracking, Medication Administration Records (MAR), nursing care plans |
| **Consultation** | Digital SOAP notes, ICD-10 diagnosis coding, clinical NLP chatbot integration, inpatient bed management |
| **Pharmacy** | e-Prescribing validation, FEFO (first-expired, first-out) stock control, DrugCentral & OpenFDA clinical drug reference console, target protein activity, live autocomplete search |
| **Laboratory (LIS)** | Test ordering, result verification, critical panic value alerts |
| **Imaging and radiology** | Imaging requests and DICOM metadata integration |
| **Billing and insurance** | Automated invoicing, partial payment allocation, M-Pesa mobile money, SHA / SHIF claim adjudication |

### Interoperability and AI governance

- **HL7 FHIR R4 API:** JSON endpoints for `Patient`, `Observation`, `Condition`, `DiagnosticReport` and `MedicationRequest`.
- **KHIS / DHIS2 exporter:** automated monthly aggregation of MOH 705A/B, 711 and 731 reports.
- **AI governance and audit:** structured SIEM-style audit logging, input validation guardrails, and explicit mode disclosure (live LLM vs. offline fallback vs. RDKit).

### Security and resilience

- **Persistent audit trail:** append-only database log of clinical record views, billing payments and administrative actions.
- **Encryption at rest:** field-level AES-128-CBC + HMAC (Fernet) for sensitive patient identity attributes.
- **Authentication and RBAC:** role-based access control, TOTP multi-factor authentication, 5-attempt account lockout, password complexity enforcement.
- **Offline-first PWA:** service worker caching, web app manifest, and a real-time connectivity banner.

---

## Quickstart

### Prerequisites

- Python 3.12+
- PostgreSQL 16+ (SQLite works for development)
- Redis 7+

### Local installation

```bash
# 1. Clone the repository
git clone https://github.com/mathunjoroge/hospital.git
cd hospital

# 2. Create a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (see "Configuration" below)
cp .env.example .env
export FLASK_APP=app.py

# 5. Run database migrations
flask db upgrade

# 6. Start the application
python app.py
```

---

## Docker deployment

Run the full stack (PostgreSQL, Redis, Flask application and NLP service) with Docker Compose:

```bash
# Build and start all services in the background
docker-compose up -d --build

# Follow application logs
docker-compose logs -f flask

# Check service status
docker-compose ps
```

---

## Configuration

Copy `.env.example` to `.env` and set at minimum:

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Flask session signing key. Use a long random value. |
| `ENCRYPTION_KEY` | Fernet key for field-level encryption of patient identity data. |

Generate a Fernet key with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

> **Never commit `.env` or real credentials, and never use sample keys in production.** Losing the `ENCRYPTION_KEY` makes encrypted fields unrecoverable, so back it up securely.

---

## Testing

```bash
# Run the full test suite
PYTHONPATH=. venv/bin/pytest -v

# Run with coverage
PYTHONPATH=. venv/bin/pytest --cov=. --cov-report=term-missing
```

---

## API reference

| Endpoint | Method | Description |
| --- | --- | --- |
| `/healthz` | `GET` | Health check (database connectivity and disk space) |
| `/api/fhir/R4/Patient/<id>` | `GET` | FHIR R4 Patient resource |
| `/api/fhir/R4/Observation?patient=<id>` | `GET` | FHIR R4 Observation bundle (vitals and labs) |
| `/api/khis/export/dhis2_json` | `GET` | KHIS / DHIS2 monthly `dataValueSets` JSON |
| `/api/khis/export/csv` | `GET` | KHIS / DHIS2 monthly CSV export |
| `/medicine/drugs-ref/search` | `GET` | Clinical Drug Reference Console (DrugCentral & OpenFDA fallback) |
| `/medicine/drugs-ref/api/autocomplete` | `GET` | Live drug autocomplete search suggestions API |
| `/admin/audit-trail/export` | `GET` | SIEM-compatible JSON audit log export |

---

## Security and privacy

HIMS handles sensitive health data, so security is treated as a core feature. See [`SECURITY.md`](SECURITY.md) for how to report vulnerabilities responsibly, and [`DATA_RETENTION_POLICY.md`](DATA_RETENTION_POLICY.md) for retention guidance.

If you deploy HIMS with real patient data, you are responsible for your own legal and regulatory compliance (including registration and obligations under the Kenya Data Protection Act 2019), infrastructure hardening, backups and access control.

---

## Project status

HIMS is under active development. Supporting material for independent review is available (security audit scope, clinical safety review packaging, and a WCAG 2.1 AA accessibility baseline; see below), but the system should be evaluated and validated by your own clinical, security and legal stakeholders before any production use. Open questions awaiting stakeholder sign-off are tracked in [`docs/DECISIONS_PENDING.md`](docs/DECISIONS_PENDING.md).

---

## Documentation

- **Disaster recovery:** [Backup and emergency restore runbook](docs/backup_restore_runbook.md)
- **Performance:** [Load and concurrency baseline results](docs/load_test_results.md)
- **Interfacing research:** [HL7 v2 lab-instrument and DICOM web architecture](docs/pacs_hl7_interfacing_research.md)
- **Security readiness:** [Third-party security audit scope](docs/security_audit_readiness.md)
- **Clinical governance:** [Clinical safety review packaging](docs/clinical_safety_review_packaging.md)
- **Accessibility:** [WCAG 2.1 AA compliance baseline report](docs/accessibility_audit_report.md)

---

## Contributing

Contributions are welcome, from bug reports to new departments and integrations. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) first. Look for issues labelled `good first issue` or `help wanted` if you're new to the project.

---

## License

Released under the [MIT License](LICENSE).

<!--
GitHub "About" settings (repo page > gear icon next to About). Copy these, then delete this comment.

Description:
Open-source, privacy-first hospital information system for Kenyan facilities: EMR, pharmacy, lab, M-Pesa & SHA/SHIF billing, HL7 FHIR R4 API and KHIS/DHIS2 reporting.

Website: (your demo or docs URL)

Topics:
hims, emr, ehr, health-information-system, hospital-management-system, healthcare, digital-health, fhir, hl7-fhir, dhis2, khis, kenya, mpesa, flask, python, postgresql, celery, docker, pwa
-->

