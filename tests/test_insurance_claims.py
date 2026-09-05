"""
tests/test_insurance_claims.py
───────────────────────────────
Unit tests for Task 2.3: InsuranceScheme, PatientInsurance, Claim models
and the Claim lifecycle state machine.
"""
from datetime import date
from decimal import Decimal

import pytest

from departments.models.billing import Invoice, InvoiceStatus
from departments.models.insurance import (
    Claim,
    ClaimStatus,
    InsuranceScheme,
    PatientInsurance,
)
from departments.models.records import Patient
from extensions import db

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_patient(pid):
    p = Patient(
        patient_id=pid,
        name=f"Insured Patient {pid}",
        place_of_residence="Nairobi",
        sex="Female",
        date_of_birth=date(1985, 6, 15),
        marital_status="Married",
        contact=f"07000{pid[-4:]}",
        next_of_kin="Kin",
        relationship_with_next_of_kin="Spouse",
        next_of_kin_contact="0711000000",
        national_id=None,
        emergency_contact="0722000000",
    )
    db.session.add(p)
    db.session.commit()
    return p


def _make_scheme(code, name="Test Scheme"):
    s = InsuranceScheme(code=code, name=name, scheme_type="public")
    db.session.add(s)
    db.session.commit()
    return s


def _make_invoice(patient_id):
    inv = Invoice(
        invoice_number=Invoice.generate_invoice_number(),
        patient_id=patient_id,
        status=InvoiceStatus.ISSUED,
        subtotal=Decimal("5000"),
        discount=Decimal("0"),
        grand_total=Decimal("5000"),
        amount_paid=Decimal("0"),
        balance=Decimal("5000"),
    )
    db.session.add(inv)
    db.session.commit()
    return inv


def _make_claim(invoice, patient_id, scheme_id, amount=5000):
    claim = Claim(
        claim_number=Claim.generate_claim_number(),
        invoice_id=invoice.id,
        patient_id=patient_id,
        scheme_id=scheme_id,
        status=ClaimStatus.DRAFT,
        claimed_amount=Decimal(str(amount)),
    )
    db.session.add(claim)
    db.session.commit()
    return claim


# ─────────────────────────────────────────────
# 1. InsuranceScheme
# ─────────────────────────────────────────────

class TestInsuranceScheme:
    def test_create_sha_scheme(self, app):
        with app.app_context():
            _make_scheme("SHA", "Social Health Authority")
            saved = InsuranceScheme.query.filter_by(code="SHA").first()
            assert saved is not None
            assert saved.scheme_type == "public"

    def test_unique_scheme_code(self, app):
        """Duplicate scheme codes should fail."""
        with app.app_context():
            _make_scheme("NHIF01")
            s2 = InsuranceScheme(code="NHIF01", name="Duplicate", scheme_type="public")
            db.session.add(s2)
            with pytest.raises(Exception):
                db.session.commit()
            db.session.rollback()

    def test_inactive_scheme(self, app):
        with app.app_context():
            s = InsuranceScheme(code="OLD", name="Old Scheme", scheme_type="private", is_active=False)
            db.session.add(s)
            db.session.commit()
            saved = InsuranceScheme.query.filter_by(code="OLD").first()
            assert saved.is_active is False


# ─────────────────────────────────────────────
# 2. PatientInsurance
# ─────────────────────────────────────────────

class TestPatientInsurance:
    def test_enroll_patient_in_scheme(self, app):
        with app.app_context():
            _make_patient("INS001")
            s = _make_scheme("SHA-TST")
            pi = PatientInsurance(
                patient_id="INS001",
                scheme_id=s.id,
                member_number="SHA-9999-001",
                relationship="self",
                is_active=True,
            )
            db.session.add(pi)
            db.session.commit()
            saved = PatientInsurance.query.filter_by(
                patient_id="INS001", scheme_id=s.id
            ).first()
            assert saved is not None
            assert saved.member_number == "SHA-9999-001"

    def test_patient_multiple_schemes(self, app):
        with app.app_context():
            _make_patient("INS002")
            s1 = _make_scheme("SHA-A")
            s2 = _make_scheme("AAR-X")
            for s, mn in [(s1, "SHA-1001"), (s2, "AAR-2002")]:
                db.session.add(PatientInsurance(
                    patient_id="INS002", scheme_id=s.id,
                    member_number=mn, relationship="self",
                ))
            db.session.commit()
            memberships = PatientInsurance.query.filter_by(patient_id="INS002").all()
            assert len(memberships) == 2


# ─────────────────────────────────────────────
# 3. Claim creation & number generation
# ─────────────────────────────────────────────

class TestClaimCreation:
    def test_create_draft_claim(self, app):
        with app.app_context():
            _make_patient("CLM001")
            s = _make_scheme("SHA-D1")
            inv = _make_invoice("CLM001")
            claim = _make_claim(inv, "CLM001", s.id)
            assert claim.status == ClaimStatus.DRAFT
            assert claim.claim_number.startswith("CLM-")

    def test_claim_number_sequential(self, app):
        with app.app_context():
            _make_patient("CLM002")
            s = _make_scheme("SHA-D2")
            inv1 = _make_invoice("CLM002")
            inv2 = _make_invoice("CLM002")
            c1 = _make_claim(inv1, "CLM002", s.id)
            c2 = _make_claim(inv2, "CLM002", s.id)
            assert c1.claim_number != c2.claim_number


# ─────────────────────────────────────────────
# 4. Claim lifecycle state machine
# ─────────────────────────────────────────────

class TestClaimLifecycle:
    def test_draft_to_submitted(self, app):
        with app.app_context():
            _make_patient("LC001")
            s = _make_scheme("SHA-LC1")
            inv = _make_invoice("LC001")
            claim = _make_claim(inv, "LC001", s.id)
            claim.submit()
            db.session.commit()
            assert claim.status == ClaimStatus.SUBMITTED
            assert claim.submitted_at is not None

    def test_submitted_to_approved(self, app):
        with app.app_context():
            _make_patient("LC002")
            s = _make_scheme("SHA-LC2")
            inv = _make_invoice("LC002")
            claim = _make_claim(inv, "LC002", s.id)
            claim.submit()
            claim.approve(approved_amount=Decimal("4500"), pre_auth="PA-12345")
            db.session.commit()
            assert claim.status == ClaimStatus.APPROVED
            assert claim.approved_amount == Decimal("4500")
            assert claim.pre_auth_number == "PA-12345"

    def test_submitted_to_rejected(self, app):
        with app.app_context():
            _make_patient("LC003")
            s = _make_scheme("SHA-LC3")
            inv = _make_invoice("LC003")
            claim = _make_claim(inv, "LC003", s.id)
            claim.submit()
            claim.reject("Service not covered under this scheme.")
            db.session.commit()
            assert claim.status == ClaimStatus.REJECTED
            assert "not covered" in claim.denial_reason

    def test_approved_to_paid(self, app):
        with app.app_context():
            _make_patient("LC004")
            s = _make_scheme("SHA-LC4")
            inv = _make_invoice("LC004")
            claim = _make_claim(inv, "LC004", s.id)
            claim.submit()
            claim.approve(approved_amount=Decimal("5000"))
            claim.mark_paid()
            db.session.commit()
            assert claim.status == ClaimStatus.PAID
            assert claim.paid_at is not None

    def test_rejected_to_appealed(self, app):
        with app.app_context():
            _make_patient("LC005")
            s = _make_scheme("SHA-LC5")
            inv = _make_invoice("LC005")
            claim = _make_claim(inv, "LC005", s.id)
            claim.submit()
            claim.reject("Insufficient documentation.")
            claim.appeal()
            db.session.commit()
            assert claim.status == ClaimStatus.APPEALED
            assert claim.appeal_date is not None

    def test_cannot_submit_already_submitted_claim(self, app):
        with app.app_context():
            _make_patient("LC006")
            s = _make_scheme("SHA-LC6")
            inv = _make_invoice("LC006")
            claim = _make_claim(inv, "LC006", s.id)
            claim.submit()
            with pytest.raises(ValueError, match="Cannot submit"):
                claim.submit()

    def test_cannot_mark_paid_without_approval(self, app):
        with app.app_context():
            _make_patient("LC007")
            s = _make_scheme("SHA-LC7")
            inv = _make_invoice("LC007")
            claim = _make_claim(inv, "LC007", s.id)
            claim.submit()
            with pytest.raises(ValueError, match="Cannot mark paid"):
                claim.mark_paid()

    def test_cannot_appeal_non_rejected_claim(self, app):
        with app.app_context():
            _make_patient("LC008")
            s = _make_scheme("SHA-LC8")
            inv = _make_invoice("LC008")
            claim = _make_claim(inv, "LC008", s.id)
            claim.submit()
            with pytest.raises(ValueError, match="Cannot appeal"):
                claim.appeal()
