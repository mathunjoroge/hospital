# HMIS Implementation Progress

## PHASE 0 — Stop the bleeding

- [ ] **0.1 Lock down the unauthenticated patient search API**
  - [ ] Add `@login_required` to `/api/patients/search`
  - [ ] Add rate limiting
  - [ ] Write tests asserting 401/403 for unauthenticated and 200 for authenticated requests.
- [ ] **0.2 Fix the frozen-date bug**
  - [ ] Fix `ClinicBooking.created_on`
  - [ ] Fix `note_date` in `departments/models/medicine.py`
  - [ ] Fix `note_date` form default in `departments/forms.py`
  - [ ] Write tests ensuring dates are not frozen at import time.
- [ ] **0.3 Remove unrelated non-HMIS files**
  - [ ] Delete `convert_markets_fintech.py`, `generate_proposal.py`, `agent.py`, `test_query.py` (if they exist).
  - [ ] Verify tests pass.
- [ ] **0.4 Stop leaking exception details to users**
  - [ ] Fix `flash(f'... {e}')` in `app.py`
  - [ ] Fix `flash` calls across all department routes.
  - [ ] Log full exceptions server-side.
- [ ] **0.5 Harden `SECRET_KEY` enforcement**
  - [ ] Fail loudly if `SECRET_KEY` is missing and `FLASK_ENV` is not `testing`.
  - [ ] Write tests to verify this behavior.
- [ ] **0.6 Add rate limiting and lockout to authentication endpoints**
  - [ ] Add `Flask-Limiter` to `requirements.txt`.
  - [ ] Apply rate limits to `/login`.
  - [ ] Add `failed_login_attempts` and `locked_until` columns to `User` model.
  - [ ] Write tests for rate limiting and lockout.

## PHASE 1 — Engineering foundations

- [ ] **1.1 Real database migrations**
- [ ] **1.2 Containerize the stack**
- [ ] **1.3 Move the core database to PostgreSQL everywhere**
- [ ] **1.4 Stand up CI**
- [ ] **1.5 Break up the god-files**
- [ ] **1.6 Consistent audit logging**
- [ ] **1.7 MFA, lockout, and password policy**
- [ ] **1.8 Expand the test suite**

## PHASE 2 — Data model & compliance

- [ ] **2.1 Redesign the `Patient` model**
- [ ] **2.2 Unify billing**
- [ ] **2.3 Build the SHA/SHIF insurance module**
- [ ] **2.4 Integrate M-Pesa (Safaricom Daraja API)**
- [ ] **2.5 Encryption at rest for identity fields**
- [ ] **2.6 Data Protection Act 2019 compliance basics**

## PHASE 3 — Interoperability & AI governance

- [ ] **3.1 FHIR interoperability layer**
- [ ] **3.2 DHIS2/KHIS reporting export**
- [ ] **3.3 Govern the AI/NLP layer**
- [ ] **3.4 Offline-first considerations**

## PHASE 4 — Operating like a real product

- [ ] **4.1 Monitoring & backups**
- [ ] **4.2 Documentation**
- [ ] **4.3 Team/process**
