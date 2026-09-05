# HMIS Implementation Progress

## PHASE 0 — Stop the bleeding

- [x] **0.1 Lock down the unauthenticated patient search API** ✅ (already done: @login_required + rate limit)
- [x] **0.2 Fix the frozen-date bug** ✅ (`Purchase.purchase_date` default fixed in pharmacy.py)
- [x] **0.3 Remove unrelated non-HMIS files** ✅ (files were never present)
- [x] **0.4 Stop leaking exception details to users** ✅ (billing.pay_bills + pharmacy.inventory)
- [x] **0.5 Harden `SECRET_KEY` enforcement** ✅ (already done: raises RuntimeError in production)
- [x] **0.6 Add rate limiting and lockout to authentication endpoints** ✅ (already done: failed_login_attempts + locked_until)

## PHASE 1 — Engineering foundations

- [x] **1.1 Real database migrations** ✅ (Flask-Migrate / Alembic `migrations/` directory configured)
- [x] **1.2 Containerize the stack** ✅ (`Dockerfile`, `Dockerfile.nlp`, `docker-compose.yml` multi-container setup)
- [x] **1.3 Move the core database to PostgreSQL everywhere** ✅ (PostgreSQL 16 support via `psycopg2-binary` & docker-compose)
- [x] **1.4 Stand up CI** ✅ (`.github/workflows/ci.yml` running linting, bandit security audit, pip-audit, & pytest)
- [x] **1.5 Break up the god-files** ✅ (Refactored `medicine`, `pharmacy`, `nursing`, `laboratory` blueprints into focused modules)
- [x] **1.6 Consistent audit logging** ✅ (`AuditLog` DB model, `log_audit_event`, `@audited` decorator, `/admin/audit-trail` & SIEM export)
- [x] **1.7 MFA, lockout, and password policy** ✅ (TOTP MFA, 5-attempt lockout, `validate_password_strength` complexity checks)
- [x] **1.8 Expand the test suite** ✅ (30 test modules covering patient redesign, billing, insurance, FHIR, DHIS2, AI, PWA, healthz)

## PHASE 2 — Data model & compliance

- [x] **2.1 Redesign the `Patient` model** ✅ (UUID generation, age_years, soft delete, merge logic - 8/8 tests passing)
- [x] **2.2 Unify billing** ✅ (`Invoice`, `Payment` allocation, aggregate unbilled charges - 9/9 tests passing)
- [x] **2.3 Build the SHA/SHIF insurance module** ✅ (`InsuranceScheme`, `PatientInsurance`, `Claim` adjudication - 11/11 tests passing)
- [x] **2.4 Integrate M-Pesa (Safaricom Daraja API)** ✅ (STK Push, C2B callback handlers - 6/6 tests passing)
- [x] **2.5 Encryption at rest for identity fields** ✅ (`EncryptedString` Fernet AES-128-CBC + HMAC - 5/5 tests passing)
- [x] **2.6 Data Protection Act 2019 compliance basics** ✅ (`PatientConsent`, SAR JSON export, anonymize patient - 4/4 tests passing)

## PHASE 3 — Interoperability & AI governance

- [x] **3.1 FHIR interoperability layer** ✅ (5/5 tests passing)
- [x] **3.2 DHIS2/KHIS reporting export** ✅ (5/5 tests passing)
- [x] **3.3 Govern the AI/NLP layer** ✅ (ai_audit.py: structured audit logging, input validation, mode disclosure, AITimer)
- [x] **3.4 Offline-first considerations** ✅ (Service Worker `sw.js`, PWA `manifest.json`, `offline.html`, auto-network detection banner)


## PHASE 4 — Operating like a real product

- [x] **4.1 Monitoring & backups** ✅ (`GET /healthz` endpoint with DB ping & disk checks, `scripts/backup_db.py` automated backup utility)
- [x] **4.2 Documentation** ✅ (`README.md` system architecture guide, API docs, quickstart & Docker instructions)
- [x] **4.3 Team/process** ✅ (`CONTRIBUTING.md` developer workflow, `SECURITY.md` vulnerability reporting & DPA compliance details)

## NEW EXTENSION — Feature Roadmap

- [x] **Phase A — Patient Self-Service Portal** ✅ (PatientUser auth, login/register, dashboard, appointment booking, lab results release gating, billing history & STK push pay, profile management with audit logging - 5/5 tests passing)
- [x] **Phase B — Outbound Patient Communication** ✅ (Flask-Mail email driver, SMS sandbox abstraction, OutboundNotificationLog delivery auditing, 5 event triggers for appointments/labs/billing/payments/claims, 24h appointment reminder scheduler - 6/6 tests passing)




