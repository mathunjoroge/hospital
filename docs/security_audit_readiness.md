# Security Audit Readiness & Penetration Testing Scope

## 1. Overview & Disclaimer

Per Process Integrity rules (P.3), this document summarizes the current technical security posture implemented within the Hospital Management Information System (HMIS). **This document does not constitute or replace a formal penetration test or third-party cybersecurity audit.** It prepares third-party security assessors by mapping existing controls and identifying targeted audit vectors.

---

## 2. Implemented Defense Controls Summary

| Security Layer | Control Description | Verification Status |
|----------------|---------------------|---------------------|
| **Authentication** | Password strength validation (min length 10, uppercase, digit, special character), 5-attempt lockout, TOTP 2FA. | Tested (`test_health_password.py`) |
| **Authorization** | Role-Based Access Control (`@roles_required`) across blueprints + `@break_glass_required` auditing. | Tested (`test_break_glass.py`) |
| **Data at Rest Encryption** | AES-128-CBC + HMAC-SHA256 (`EncryptedString`) for patient national ID numbers (`records.py`). | Tested (`test_patient_redesign.py`) |
| **Audit Logging** | Centralized `AuditLog` table capturing user ID, IP address, timestamp, action, target model, and outcome. | Tested (`test_audit_logging.py`) |
| **Rate Limiting** | `Flask-Limiter` active on login/auth routes to prevent automated brute-force attacks. | Tested (`load_test_baseline.py`) |
| **CSRF Protection** | `Flask-WTF` CSRF token validation on all POST/PUT/DELETE forms. | Tested |
| **Privacy Compliance** | Data Protection Act 2019 consent gate (`has_ai_consent`) for clinical data outbound processing. | Tested (`test_ai_consent_gate.py`) |
| **Staff Credentialing** | `StaffCredential` model tracking professional licensing (KMPDC, NCK, PPB) with automated expiry alerts. | Tested (`test_staff_credentials.py`) |
| **Async Task Isolation** | Celery + Redis worker decoupling long-running jobs (AI, PDF, notifications) from web request threads. | Tested (`celery_app.py`) |


---

## 3. Recommended External Penetration Testing Scope

A qualified external penetration testing entity should focus on the following targeted scope:

### A. Break-Glass Emergency Escalation Engine
* Target: `/emergency/break-glass` and `@break_glass_required`.
* Risk Vectors: Privilege escalation, improper expiration handling, justification tampering, notification bypass.

### B. FHIR Interoperability & REST API Endpoints
* Target: `/fhir/Patient/<uuid>`, `/fhir/Observation`, `/dhis2/export`.
* Risk Vectors: Insecure Direct Object References (IDOR), unauthorized cross-tenant data extraction, schema injection.

### C. File Upload Interfaces & Path Traversal
* Target: Imaging upload routes (`departments/imaging/routes.py`) and document attachments.
* Risk Vectors: Arbitrary file upload, MIME-type spoofing, path traversal (`../`), server-side execution.

### D. Session Management & Redis Token Lifecycle
* Target: Server-side Redis sessions (`Flask-Session`).
* Risk Vectors: Session fixation, insufficient session invalidation on logout, cookie attributes (`HttpOnly`, `Secure`, `SameSite`).
