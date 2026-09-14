# Security & Data Protection Policy

Security and patient data privacy are paramount in HIMS. This document outlines our security architecture, compliance standards, and instructions for reporting vulnerabilities.

---

## 🛡️ Security Architecture & Controls

### 1. Data Encryption at Rest (Task 2.5)
- Sensitive identity attributes (National ID, contact details) are encrypted at rest using AES-128-CBC with HMAC (Fernet).
- Encryption key managed via `ENCRYPTION_KEY` environment variable.

### 2. Kenya Data Protection Act 2019 Compliance (Task 2.6)
- **Explicit Consent**: Consent tracking per patient for AI processing and third-party data sharing (`PatientConsent`).
- **Subject Access Request (SAR)**: Instant automated JSON data export under Section 26 (`export_patient_sar_data`).
- **Right to Erasure / Anonymization**: Anonymization tool masking personal identifiers while retaining clinical structures for statutory audits (`anonymize_patient_data`).

### 3. Persistent SIEM Audit Trail (Task 1.6 & P0-20)
- Append-only database audit log (`AuditLog`) tracking patient record views, updates, billing payments, DICOM downloads, and login activity.
- Cryptographically chained audit log for tamper evidence.
- Structured SIEM JSON export endpoint at `/admin/audit-trail/export`.

### 4. Authentication & Access Control (Task 1.7 & P0-01)
- Rate limiting and 15-minute account lockout after 5 consecutive failed login attempts.
- Multi-factor authentication (TOTP MFA) for administrative and clinical accounts.
- Enforced password complexity rules.
- **Server-Side Identity Enforcement**: All identity fallbacks (`user_id=1`, `nurse_id=1`, `verifier_id=1`, `radiologist_id=1`) eliminated. Unauthenticated access fails closed (401/403).
- **Object-Level Authorization (IDOR Protection)**: Server-side validation of patient, encounter, and facility ownership across DICOM, ICU, LIS, Pharmacy, Renal, and Oncology.

### 5. Clinical Record Integrity & Voiding Policy (P0-05 & P0-09 & P0-16)
- **Immutable Clinical History**: Physical deletion of clinical records (Oncology Notes, Dispensed Drugs) is strictly prohibited.
- **Controlled Reversal/Voiding**: Clinical and financial reversals use structured voiding (`is_voided`, `void_reason`, `voided_by`, `voided_at`) with transactional inventory restoration.

### 6. File Upload Hardening (P0-15)
- File uploads enforce `secure_filename()`, MIME type validation, extension allowlists, size limits (5MB max), and UUID-based server-side storage names outside web root.

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability or potential data exposure within HIMS, please do NOT create a public GitHub issue.

Instead, please report it directly to the security team:
- **Email**: `security@hospital.local` / `mathu@hospital.local`
- **Response SLA**: Initial response within 24 hours; status updates every 48 hours until resolution.

Please include:
- Description of the vulnerability and potential impact.
- Step-by-step proof-of-concept (PoC) to reproduce the issue.
- Affected endpoints or source code files.
