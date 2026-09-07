"""
departments/models/insurance.py
────────────────────────────────
Task 2.3 — SHA/SHIF Insurance Module

Models:
  InsuranceScheme  – payer master (SHA, NHIF, private schemes)
  PatientInsurance – patient ↔ scheme membership
  Claim            – per-invoice insurance claim with lifecycle state-machine
"""

import enum
from datetime import datetime

from extensions import db


class ClaimStatus(str, enum.Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    QUERIED = "queried"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAID = "paid"
    APPEALED = "appealed"


class InsuranceScheme(db.Model):
    """
    Master list of insurance payers.
    Pre-seeded with SHA (Social Health Authority) and legacy NHIF.
    """

    __tablename__ = "insurance_schemes"

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(
        db.String(20), unique=True, nullable=False
    )  # e.g. 'SHA', 'NHIF', 'AAR'
    name = db.Column(db.String(120), nullable=False)
    scheme_type = db.Column(
        db.String(30), nullable=False, default="public"
    )  # public / private / capitation
    contact = db.Column(db.String(100), nullable=True)
    portal_url = db.Column(db.String(255), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    members = db.relationship(
        "PatientInsurance",
        back_populates="scheme",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )
    claims = db.relationship("Claim", back_populates="scheme", lazy="dynamic")

    def __repr__(self):
        return f"<InsuranceScheme {self.code}: {self.name}>"


class PatientInsurance(db.Model):
    """Links a patient to an insurance scheme with their membership details."""

    __tablename__ = "patient_insurance"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )
    scheme_id = db.Column(
        db.Integer, db.ForeignKey("insurance_schemes.id"), nullable=False, index=True
    )

    member_number = db.Column(db.String(60), nullable=False)
    principal_member = db.Column(
        db.String(100), nullable=True
    )  # Head of household (if dependant)
    relationship = db.Column(
        db.String(30), nullable=True
    )  # self / spouse / child / parent

    start_date = db.Column(db.Date, nullable=True)
    end_date = db.Column(db.Date, nullable=True)  # Null = still active
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    verified_at = db.Column(
        db.DateTime, nullable=True
    )  # When eligibility was last confirmed
    verified_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    patient = db.relationship(
        "Patient", backref=db.backref("insurance_memberships", lazy="dynamic")
    )
    scheme = db.relationship("InsuranceScheme", back_populates="members")

    def __repr__(self):
        return f"<PatientInsurance patient={self.patient_id} scheme={self.scheme_id}>"


class Claim(db.Model):
    """
    Insurance claim for a single invoice.
    Lifecycle: DRAFT → SUBMITTED → APPROVED/QUERIED/REJECTED → PAID
    Appeals supported via APPEALED state.
    """

    __tablename__ = "insurance_claims"

    id = db.Column(db.Integer, primary_key=True)
    claim_number = db.Column(db.String(40), unique=True, nullable=False, index=True)

    invoice_id = db.Column(
        db.Integer, db.ForeignKey("invoices.id"), nullable=False, index=True
    )
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )
    scheme_id = db.Column(
        db.Integer, db.ForeignKey("insurance_schemes.id"), nullable=False, index=True
    )
    patient_insurance_id = db.Column(
        db.Integer, db.ForeignKey("patient_insurance.id"), nullable=True
    )

    status = db.Column(db.Enum(ClaimStatus), nullable=False, default=ClaimStatus.DRAFT)

    # Financials
    claimed_amount = db.Column(db.Numeric(12, 2), nullable=False)
    approved_amount = db.Column(db.Numeric(12, 2), nullable=True)
    co_pay = db.Column(db.Numeric(12, 2), nullable=True, default=0)

    # Scheme references
    scheme_claim_ref = db.Column(
        db.String(100), nullable=True
    )  # Reference returned by payer
    pre_auth_number = db.Column(db.String(60), nullable=True)
    denial_reason = db.Column(db.Text, nullable=True)

    # Dates
    submitted_at = db.Column(db.DateTime, nullable=True)
    approved_at = db.Column(db.DateTime, nullable=True)
    paid_at = db.Column(db.DateTime, nullable=True)
    appeal_date = db.Column(db.DateTime, nullable=True)

    # Audit
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    notes = db.Column(db.Text, nullable=True)

    invoice = db.relationship("Invoice", backref=db.backref("claims", lazy="dynamic"))
    patient = db.relationship(
        "Patient", backref=db.backref("insurance_claims", lazy="dynamic")
    )
    scheme = db.relationship("InsuranceScheme", back_populates="claims")

    @staticmethod
    def generate_claim_number():
        """Generate a sequential claim number CLM-YYYYMM-NNNN."""
        now = datetime.utcnow()
        prefix = f"CLM-{now.strftime('%Y%m')}"
        last = (
            Claim.query.filter(Claim.claim_number.like(f"{prefix}-%"))
            .order_by(Claim.id.desc())
            .first()
        )
        seq = 1
        if last:
            try:
                seq = int(last.claim_number.rsplit("-", 1)[-1]) + 1
            except ValueError:
                pass
        return f"{prefix}-{seq:04d}"

    def submit(self, submitted_by=None):
        """Transition claim to SUBMITTED state."""
        if self.status != ClaimStatus.DRAFT:
            raise ValueError(f"Cannot submit claim in state {self.status}.")
        self.status = ClaimStatus.SUBMITTED
        self.submitted_at = datetime.utcnow()

    def approve(self, approved_amount, pre_auth=None):
        """Approve claim with a given amount."""
        if self.status not in (ClaimStatus.SUBMITTED, ClaimStatus.APPEALED):
            raise ValueError(f"Cannot approve claim in state {self.status}.")
        self.status = ClaimStatus.APPROVED
        self.approved_amount = approved_amount
        self.approved_at = datetime.utcnow()
        if pre_auth:
            self.pre_auth_number = pre_auth

    def reject(self, reason):
        """Reject claim with an explanation."""
        if self.status not in (ClaimStatus.SUBMITTED, ClaimStatus.QUERIED):
            raise ValueError(f"Cannot reject claim in state {self.status}.")
        self.status = ClaimStatus.REJECTED
        self.denial_reason = reason

    def mark_paid(self):
        """Mark claim as paid by the insurer."""
        if self.status != ClaimStatus.APPROVED:
            raise ValueError(f"Cannot mark paid in state {self.status}.")
        self.status = ClaimStatus.PAID
        self.paid_at = datetime.utcnow()

    def appeal(self):
        """Appeal a rejected claim."""
        if self.status != ClaimStatus.REJECTED:
            raise ValueError(f"Cannot appeal claim in state {self.status}.")
        self.status = ClaimStatus.APPEALED
        self.appeal_date = datetime.utcnow()

    def __repr__(self):
        return f"<Claim {self.claim_number} [{self.status}]>"
