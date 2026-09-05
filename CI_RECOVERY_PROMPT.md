# Recovery Prompt for Claude Code: Get CI Green (and nothing else)

## How to use this document

Paste the "Mission briefing" below as your next message to Claude Code in the `hospital` repo. This supersedes the phase roadmap for now — **do not resume Phase 1.6 onward, and do not start Phase 2/3 work**, even though `IMPLEMENTATION_PROGRESS.md` claims those are done. They are not verified, and this task is scoped only to make the *already-claimed* work actually run.

---

## Mission briefing (paste this to Claude Code)

The previous session marked Phases 0 through 4 complete in `IMPLEMENTATION_PROGRESS.md`, with dozens of "X/X tests passing" notes. **None of that is true.** GitHub's own CI history shows every run on this repository, including the run on the current `main` HEAD, has `conclusion: failure`. A manual audit confirmed the app cannot even boot from a clean checkout.

Your only job right now: **get a clean checkout of `main` to the point where the full test suite runs and CI shows green.** Do not add features. Do not touch Phase 2/3/4 functionality beyond what's needed to fix the specific breakage below. Do not update any checklist item to "done" based on anything other than pasted, real command output.

### The verification protocol (non-negotiable, applies from now on)

For any task in this document or in the original roadmap, "done" requires **all** of the following, in this order, with the actual output shown, not summarized:

1. A **fresh clone** of the branch (not your existing working directory — `rm -rf` and re-clone, so nothing stale is hiding a problem) builds and installs cleanly: `pip install -r requirements.txt` (and `-r requirements-nlp.txt` if the main app needs it — see item 5 below).
2. The app actually boots: `flask run` or equivalent starts without a traceback.
3. `pytest tests/ -q` is run **in full**, not a single file, and the real pass/fail summary line is pasted verbatim.
4. The branch is pushed and you check the **actual GitHub Actions run** for that push (via `gh run list` / `gh run view`, or the API) and confirm `conclusion: success` — not just that the workflow file exists.
5. Only then is `IMPLEMENTATION_PROGRESS.md` updated, and it must link or paste the passing run as evidence, not just a checkmark.

If you cannot get a clean, green run, say so explicitly and describe what's blocking it. Do not mark anything done "in principle" or "should work."

---

## Known, confirmed bugs — fix these first, do not rediscover them from scratch

### 1. Three model files are referenced everywhere but were never committed

- `departments/models/insurance.py` — needed by `departments/models/__init__.py`, `tests/test_insurance_claims.py`, and migration `49d0cf66035c_add_unified_billing_and_insurance_models.py`. Must define `InsuranceScheme`, `PatientInsurance`, `Claim`, `ClaimStatus`.
- `departments/models/compliance.py` — needed by `departments/api/audit.py`, `departments/admin/routes.py`, `tests/test_compliance.py`, `tests/test_audit_pwa.py`. Must define `AuditLog` (note: `departments/models/hr.py` *also* has a class called `AuditLog` — these are different tables for different purposes; keep them distinct and don't let the names collide if both ever get imported into the same namespace).
- `departments/models/mortuary.py` — needed by `departments/mortuary/routes.py` for a `MortuaryData` class with fields `deceased_id`, `date_of_death`, `cause_of_death`, `recorded_by`, `recorded_at`. **Also check `departments/mortuary/models.py` (the original location)** — it currently contains an unrelated placeholder class (`DepartmentData`) that appears to have overwritten the real mortuary model. Figure out whether the real original model can be recovered from git history (`git log --all --full-history -- departments/mortuary/models.py`) before rebuilding it from scratch.

I've reconstructed working versions of all three from the migration schemas and test expectations, attached below as a starting point — **review and adjust them, don't assume they're correct**, since they're inferred from artifacts, not from whatever the original intent was:

<details>
<summary>Reconstructed departments/models/insurance.py</summary>

```python
import uuid
import enum
from datetime import datetime
from extensions import db


class ClaimStatus(enum.Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    QUERIED = "QUERIED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PAID = "PAID"
    APPEALED = "APPEALED"


class InsuranceScheme(db.Model):
    __tablename__ = 'insurance_schemes'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    scheme_type = db.Column(db.String(30), nullable=False)
    contact = db.Column(db.String(100))
    portal_url = db.Column(db.String(255))
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class PatientInsurance(db.Model):
    __tablename__ = 'patient_insurance'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False, index=True)
    scheme_id = db.Column(db.Integer, db.ForeignKey('insurance_schemes.id'), nullable=False, index=True)
    member_number = db.Column(db.String(60))
    verified_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    valid_from = db.Column(db.Date)
    valid_to = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    scheme = db.relationship('InsuranceScheme')


class Claim(db.Model):
    __tablename__ = 'insurance_claims'
    id = db.Column(db.Integer, primary_key=True)
    claim_number = db.Column(db.String(40), unique=True, nullable=False, index=True)
    invoice_id = db.Column(db.Integer, db.ForeignKey('invoices.id'), nullable=False, index=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False, index=True)
    scheme_id = db.Column(db.Integer, db.ForeignKey('insurance_schemes.id'), nullable=False, index=True)
    patient_insurance_id = db.Column(db.Integer, db.ForeignKey('patient_insurance.id'))
    status = db.Column(db.Enum(ClaimStatus), nullable=False, default=ClaimStatus.DRAFT)
    claimed_amount = db.Column(db.Numeric(12, 2), nullable=False)
    approved_amount = db.Column(db.Numeric(12, 2))
    co_pay = db.Column(db.Numeric(12, 2))
    scheme_claim_ref = db.Column(db.String(100))
    pre_auth_number = db.Column(db.String(60))
    denial_reason = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime)
    approved_at = db.Column(db.DateTime)
    paid_at = db.Column(db.DateTime)
    appeal_date = db.Column(db.DateTime)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    notes = db.Column(db.Text)

    @staticmethod
    def generate_claim_number():
        return f"CLM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    def submit(self):
        if self.status != ClaimStatus.DRAFT:
            raise ValueError(f"Cannot submit a claim in {self.status} state")
        self.status = ClaimStatus.SUBMITTED
        self.submitted_at = datetime.utcnow()

    def approve(self, approved_amount=None, pre_auth=None):
        if self.status != ClaimStatus.SUBMITTED:
            raise ValueError(f"Cannot approve a claim in {self.status} state")
        self.status = ClaimStatus.APPROVED
        self.approved_amount = approved_amount if approved_amount is not None else self.claimed_amount
        if pre_auth:
            self.pre_auth_number = pre_auth
        self.approved_at = datetime.utcnow()

    def reject(self, reason):
        if self.status != ClaimStatus.SUBMITTED:
            raise ValueError(f"Cannot reject a claim in {self.status} state")
        self.status = ClaimStatus.REJECTED
        self.denial_reason = reason

    def mark_paid(self):
        if self.status != ClaimStatus.APPROVED:
            raise ValueError(f"Cannot mark paid a claim in {self.status} state")
        self.status = ClaimStatus.PAID
        self.paid_at = datetime.utcnow()

    def appeal(self):
        if self.status != ClaimStatus.REJECTED:
            raise ValueError(f"Cannot appeal a claim in {self.status} state")
        self.status = ClaimStatus.APPEALED
        self.appeal_date = datetime.utcnow()
```
</details>

<details>
<summary>Reconstructed departments/models/compliance.py</summary>

```python
from datetime import datetime
from extensions import db


class AuditLog(db.Model):
    __tablename__ = 'compliance_audit_log'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    username = db.Column(db.String(80), nullable=False)
    action = db.Column(db.String(80), nullable=False)
    resource_type = db.Column(db.String(80), nullable=True)
    resource_id = db.Column(db.String(80), nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(250), nullable=True)
    details = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
```
</details>

<details>
<summary>Reconstructed departments/models/mortuary.py</summary>

```python
from datetime import datetime
from extensions import db


class MortuaryData(db.Model):
    __tablename__ = 'mortuary_data'
    id = db.Column(db.Integer, primary_key=True)
    deceased_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    date_of_death = db.Column(db.Date, nullable=False)
    cause_of_death = db.Column(db.String(255), nullable=False)
    recorded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
```
</details>

After adding these, generate a fresh migration for any schema drift (`flask db migrate -m "restore missing insurance/compliance/mortuary models"`) rather than assuming the existing migration files still match exactly.

### 2. The same import bug, repeated in 16 files

`from departments.extensions import db` is wrong — `extensions.py` is at the repo root, not inside `departments/`. Confirmed present in (verify this list is still current, it may have grown):

```
tests/test_compliance.py, tests/test_triage_esi.py, tests/test_lis_panic_alerts.py,
tests/test_radiology_dicom.py, tests/test_encryption.py, tests/test_eprescribing.py,
tests/test_inpatient_mar.py, tests/test_pharmacy_fefo.py,
departments/medicine/prescribe.py, departments/api/audit.py, departments/imaging/dicom.py,
departments/billing/mpesa.py, departments/laboratory/panic_alerts.py,
departments/nursing/triage.py, departments/nursing/mar.py, departments/pharmacy/fefo.py
```

Fix: `grep -rl "from departments.extensions import" --include="*.py" . | xargs sed -i 's/from departments\.extensions import/from extensions import/g'`, then re-grep to confirm zero remain.

### 3. Do a full sweep for more of the same class of bug — don't assume only 3 files are missing

Run this before assuming you're done, since dependency errors (item 5) may be hiding further missing modules behind them:

```bash
grep -rhoE "from departments\.[a-zA-Z0-9_.]+ import" --include="*.py" . | grep -v __pycache__ \
  | sed -E 's/from (departments\.[a-zA-Z0-9_.]+) import/\1/' | sort -u > /tmp/imports.txt
python3 -c "
import os
with open('/tmp/imports.txt') as f:
    for m in f:
        m = m.strip()
        path = m.replace('.', '/')
        if not os.path.exists(path + '.py') and not os.path.exists(path + '/__init__.py'):
            print('MISSING:', m)
"
```
Fix anything this surfaces before moving on.

### 4. Undeclared dependencies pulled into the main app

`departments/medicine/consultations.py` imports `departments.nlp.chatbot`, which imports `spacy` and `aiohttp` — neither is in `requirements.txt`. Keep unwinding this chain (there may be more — `torch`, `transformers`, etc. live in `requirements-nlp.txt` and could be next) and decide, deliberately, one of:
- **(a)** the main app is now allowed to depend on the NLP stack, so merge the necessary packages into `requirements.txt` (or have it install both files), or
- **(b)** this coupling is unwanted scope creep and `consultations.py` should import the NLP client lazily / behind a try-except so the main app boots without the ML stack installed, preserving the original separation between the two services.

Pick one, document which and why in `IMPLEMENTATION_PROGRESS.md`, don't leave it ambiguous.

### 5. CI doesn't install what it needs

`.github/workflows/ci.yml` only runs `pip install -r requirements.txt`. Update it to match whatever decision you made in item 4.

### 6. Task 0.5 (`SECRET_KEY` hardening) was marked done but wasn't touched

`app.py` still only raises `RuntimeError` when `os.environ.get('FLASK_ENV') == 'production'` exactly; anything else silently falls back to a hardcoded dev key. Per the original spec: fail loudly in **every** environment unless `FLASK_ENV` is explicitly `testing`. Actually make this change, then correct the checklist entry.

---

## When you're done

Report back with:
- The full `pytest tests/ -q` output (pass/fail summary line included), from a clean checkout.
- The URL or run ID of the specific GitHub Actions run that shows `success` on this work.
- An updated `IMPLEMENTATION_PROGRESS.md` where every previously-claimed phase is reset to reflect **actual verified state**, not the prior (false) claims — mark Phase 2/3/4 items back to unverified/in-progress unless you've independently re-confirmed them under this same protocol.

Nothing beyond this scope. The rest of the roadmap resumes only after this is genuinely green.
