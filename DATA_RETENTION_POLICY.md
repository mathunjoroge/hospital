# Hospital Data Retention & Privacy Policy
**Compliance Standard:** Kenya Data Protection Act (DPA) 2019 & Medical Practitioners and Dentists Council Guidelines

---

## 1. Overview
This policy defines the data retention schedules, Subject Access Request (SAR) procedures, consent management rules, and Right to Erasure / Anonymization protocols for all electronic medical records (EMR) and hospital operational data.

---

## 2. Retention Schedules

| Category | Data Type | Mandatory Retention Period | Post-Retention Action |
| :--- | :--- | :--- | :--- |
| **Clinical Records** | Doctor Notes, Prescriptions, Lab Results, Imaging | 10 Years from last visit | Anonymize / Archival Storage |
| **Pediatric Records** | Children Medical History | Until patient reaches 25 years old | Anonymize / Archival Storage |
| **Financial & Billing** | Invoices, M-Pesa Transactions, Claims, Receipts | 7 Years (Kenya Tax Law / KRA) | Permanent Encryption / Archive |
| **Patient Demographics** | Name, Phone, National ID, Residence | Duration of active relationship + 2 Years | Anonymization upon request |
| **System Audit Logs** | Authentication, Merges, Access Logs | 3 Years | Purge / Cold Storage |

---

## 3. Data Protection Act (2019) Workflows

### 3.1 Consent Management (DPA Section 32)
- Explicit patient consent is required prior to processing data for AI diagnosis, medical research, or third-party sharing.
- Patients may grant or revoke consent at any time via the patient portal or records desk.

### 3.2 Subject Access Request (SAR - DPA Section 26)
- Patients have the right to request a complete machine-readable copy of their personal and financial records.
- Export requests must be fulfilled within **30 days** using the automated SAR Data Export API (`export_patient_sar_data`).

### 3.3 Right to Erasure & Anonymization (DPA Section 40)
- Patients may request erasure of personal identification data.
- Medical and tax compliance regulations prohibit complete deletion of clinical histories. Therefore, the EMR executes **Anonymization** (`anonymize_patient_data`):
  - Personal Identifiers (Name, Phone, National ID, Kin contact) are scrubbed and replaced with anonymous tokens.
  - Clinical and financial totals are retained for statutory reporting without personally identifiable information (PII).

---

## 4. Encryption & Security Controls
- **Data at Rest:** Identity fields (`national_id`, authentication secrets) are encrypted using AES-128-CBC with HMAC validation via `EncryptedString`.
- **Data in Transit:** All external communication (Daraja M-Pesa, SHA API, Telemedicine) uses TLS 1.3 encryption.
