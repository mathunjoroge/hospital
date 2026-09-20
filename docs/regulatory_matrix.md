# Kenya Regulatory Compliance & Health Data Governance Matrix
**System:** Hospital Management Information System (HMIS)
**Primary Regulations:** Kenya Digital Health Act (No. 15 of 2023), Kenya Data Protection Act (No. 24 of 2019), Kenya Health Act (No. 21 of 2017), & Tax Procedures Act (No. 29 of 2015).

---

## 1. Statutory Regulatory Alignment Matrix

| Regulation & Section | Statutory Requirement | System Implementation & Technical Controls | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Digital Health Act 2023 s.27** | Establishment of Kenya National Health Data Bank & Interoperability Standards | Native FHIR R4 REST API (`/api/fhir/*`) with Patient, Encounter, Observation, Condition, AllergyIntolerance, MedicationAdministration, Immunization, Procedure resources. | **COMPLIANT** |
| **Digital Health Act 2023 s.48** | Health Data Governance, Confidentiality, & Immutable Auditability | Append-only SHA-256 HMAC chained audit log (`departments/audit.py`), recording all PHI reads, updates, authentication events, and data export operations. | **COMPLIANT** |
| **Data Protection Act 2019 s.26** | Data Subject Access Right (SAR) — Right of Access to Personal Records | Automated SAR Data Export API (`export_patient_sar_data`) delivering structured JSON exports within 7-day statutory SLA. | **COMPLIANT** |
| **Data Protection Act 2019 s.30 & s.32** | Lawful Basis for Processing & Explicit Consent Management | Patient Consent Registry (`patient_consents` table) with `has_ai_consent()` gating external AI endpoints; auditable opt-in/opt-out. | **COMPLIANT** |
| **Data Protection Act 2019 s.40** | Right to Erasure / Anonymization | Automated PII Anonymizer (`anonymize_patient_data`) scrubbing direct identifiers and alternate IDs within 14-day SLA while retaining statutory clinical stats. | **COMPLIANT** |
| **Health Act 2017 s.13** | Patient Records Confidentiality & Mandatory Security Standards | Fernet AES-128-CBC column encryption (`EncryptedString`) for sensitive national IDs and secrets; Role-Based Access Control (RBAC) & TOTP MFA. | **COMPLIANT** |
| **Tax Procedures Act 2015 s.23** | 7-Year Statutory Financial & Billing Record Retention | Billing ledger retaining invoices (`invoices`), M-Pesa receipts, and insurance claims with key-rotated backup archives (`backup_db.py`). | **COMPLIANT** |

---

## 2. Governance Role Assignments & Accountability Matrix

| Role Title | Designation | Responsibilities | Assigned Lead / Contact |
| :--- | :--- | :--- | :--- |
| **Data Protection Officer (DPO)** | Chief Privacy Officer | Oversees DPA 2019 compliance, handles Data Subject Requests (SAR/Erasure), liaises with Office of Data Protection Commissioner (ODPC). | `dpo@hospital.co.ke` |
| **Chief Medical Information Officer (CMIO)** | Clinical Governance Lead | Validates CDSS drug safety rules, clinical workflow integrity, and FHIR clinical resource mappings. | `cmio@hospital.co.ke` |
| **Chief Information Security Officer (CISO)** | Security Lead | Manages TLS proxy, Fernet encryption key rotation, MFA enforcement, and penetration audit review. | `ciso@hospital.co.ke` |
| **System Administrator Lead** | Infrastructure Lead | Oversees encrypted database backup engine, PostgreSQL RLS policies, and container security isolation. | `sysadmin@hospital.co.ke` |

---

## 3. Compliance Evidence & Verification Checklist

- [x] **MFA Enforcement:** Mandatory TOTP verification on web login and API token issuance (`POST /api/auth/token`).
- [x] **Data at Rest Encryption:** Column encryption for National ID, TOTP secrets, and user tokens via `EncryptedString`.
- [x] **Data in Transit Encryption:** Caddy reverse proxy providing TLS 1.3 termination and strict security headers (`Caddyfile`).
- [x] **PHI Audit Logging:** Every read/write on clinical records emits HMAC-chained audit log records (`AuditLog`).
- [x] **External AI Protection:** Client-side PII redactor (`sanitize_pii`) scrubs names, phones, IDs, DOBs before NVIDIA NIM / Gemini API dispatch.
- [x] **Encrypted Backups:** Database backup utility (`scripts/backup_db.py`) generates Fernet-encrypted archives with SHA-256 integrity manifests.
- [x] **Interoperability Standard:** Full FHIR R4 capability statement and expanded resource coverage for national reporting.
