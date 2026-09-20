# Hospital Data Retention & Privacy Policy
**Compliance Standard:** Kenya Data Protection Act (DPA) 2019, Kenya Digital Health Act 2023 & Statutory Medical Retention Guidelines

---

## 1. Overview
This policy defines the data retention schedules, Subject Access Request (SAR) procedures, consent management rules, and Right to Erasure / Anonymization protocols for all electronic medical records (EMR) and hospital operational data in strict compliance with Kenyan legislation.

---

## 2. Retention Schedules & Statutory Legal Basis

| Category | Data Type | Mandatory Retention Period | Statutory Legal Citation | Post-Retention Action |
| :--- | :--- | :--- | :--- | :--- |
| **Clinical Records** | Doctor Notes, Prescriptions, Lab Results, Imaging | 10 Years from last visit | Health Act No. 21 of 2017 s.13; MPDB Code of Ethics | Anonymize / Archival Storage |
| **Pediatric Records** | Children Medical History | Until patient reaches 25 years old | Civil Procedure Act (Cap 21) & Health Act 2017 | Anonymize / Archival Storage |
| **Financial & Billing** | Invoices, M-Pesa Transactions, Claims, Receipts | 7 Years | Tax Procedures Act 2015 s.23 & Companies Act 2015 | Permanent Encryption / Archive |
| **Patient Demographics** | Name, Phone, National ID, Residence | Duration of active relationship + 2 Years | Data Protection Act 2019 s.39 | Anonymization upon request |
| **System Audit Logs** | Authentication, Merges, Access Logs | 3 Years | Digital Health Act 2023 s.48 & Computer Misuse Act 2018 | Purge / Cold Storage |

---

## 3. Data Protection Act (2019) Workflows

### 3.1 Consent Management (DPA Section 32)
- Explicit patient consent is required prior to processing data for AI diagnosis, medical research, or third-party sharing.
- Patients may grant or revoke consent at any time via the patient portal or records desk.

### 3.2 Subject Access Request (SAR - DPA Section 26)
- Patients have the right to request a complete machine-readable copy of their personal and financial records.
- **SLA:** Export requests must be fulfilled within **7 days** (Kenya Data Protection General Regulations 2021, Regulation 9(1)) using the automated SAR Data Export API (`export_patient_sar_data`).

### 3.3 Right to Erasure & Anonymization (DPA Section 40)
- **SLA:** Right to Erasure / Anonymization requests must be processed within **14 days** (Kenya Data Protection General Regulations 2021, Regulation 14(1)).
- Medical and tax compliance regulations prohibit complete deletion of clinical histories. Therefore, the EMR executes **Anonymization** (`anonymize_patient_data`):
  - Personal Identifiers (Name, Phone, National ID, Kin contact, Alternate IDs) are scrubbed and replaced with anonymous tokens.
  - Clinical and financial totals are retained for statutory reporting without personally identifiable information (PII).

---

## 4. Encryption & Security Controls
- **Data at Rest:** Identity fields (`national_id`, authentication secrets) are encrypted using AES-128-CBC with HMAC validation via `EncryptedString`.
- **Data in Transit:** All external communication (Daraja M-Pesa, SHA API, Telemedicine) uses TLS 1.3 encryption.

---

## 5. Statutory Legal Framework
1. **Kenya Data Protection Act No. 24 of 2019** & General Regulations 2021 (Sections 26, 30, 32, 39, 40).
2. **Kenya Digital Health Act No. 15 of 2023** (Sections 27, 48 - Health Data Bank & Interoperability Standards).
3. **Health Act No. 21 of 2017** (Section 13 - Patient Records Confidentiality and Security).
4. **Medical Practitioners and Dentists Act (Cap 253)** & Code of Professional Conduct.
5. **Tax Procedures Act No. 29 of 2015** (Section 23 - Statutory Record Keeping Duration).
