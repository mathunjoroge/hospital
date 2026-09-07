import uuid
from datetime import datetime, timezone

from extensions import db


class PreAuthorization(db.Model):
    """
    Tracks pre-authorization requests for SHA/SHIF and private insurance.
    Required before elective procedures, admissions, and high-cost services.
    """

    __tablename__ = "pre_authorizations"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, nullable=False, index=True)
    insurance_scheme_id = db.Column(db.Integer, nullable=False)

    # Procedure or service being requested
    procedure_code = db.Column(db.String(50), nullable=False)
    procedure_description = db.Column(db.Text, nullable=True)

    estimated_amount = db.Column(db.Numeric(12, 2), nullable=False)

    # PENDING, APPROVED, DENIED, EXPIRED, CANCELLED
    status = db.Column(db.String(20), nullable=False, default="PENDING")

    # Insurance reference number once approved
    authorization_number = db.Column(db.String(100), nullable=True)

    # Clinical justification for the procedure
    clinical_justification = db.Column(db.Text, nullable=True)
    denial_reason = db.Column(db.Text, nullable=True)

    requested_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    resolved_at = db.Column(db.DateTime(timezone=True), nullable=True)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True)


class ClaimSubmission(db.Model):
    """
    Tracks the full lifecycle of an insurance or SHA claim.
    """

    __tablename__ = "claim_submissions"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, nullable=False, index=True)
    insurance_scheme_id = db.Column(db.Integer, nullable=True)

    # Total billed amount and approved amount
    billed_amount = db.Column(db.Numeric(12, 2), nullable=False)
    approved_amount = db.Column(db.Numeric(12, 2), nullable=True)
    paid_amount = db.Column(db.Numeric(12, 2), nullable=True)

    # DRAFT, SCRUBBING, SUBMITTED, UNDER_REVIEW, APPROVED, PAID, DENIED, APPEALED
    status = db.Column(db.String(20), nullable=False, default="DRAFT")

    # External reference from the payer (SHA, insurance company)
    payer_reference = db.Column(db.String(100), nullable=True)

    # Diagnosis codes for claim justification
    primary_diagnosis_icd10 = db.Column(db.String(10), nullable=True)
    secondary_diagnosis_icd10 = db.Column(db.String(10), nullable=True)

    # Treatment dates for the claim period
    service_start_date = db.Column(db.Date, nullable=False)
    service_end_date = db.Column(db.Date, nullable=False)

    submitted_at = db.Column(db.DateTime(timezone=True), nullable=True)
    paid_at = db.Column(db.DateTime(timezone=True), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ClaimDenial(db.Model):
    """
    Tracks denied claims and the appeal workflow.
    Critical for revenue recovery and identifying systemic billing issues.
    """

    __tablename__ = "claim_denials"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_id = db.Column(
        db.String(36), db.ForeignKey("claim_submissions.id"), nullable=False, index=True
    )

    # Reason code and description from the payer
    denial_code = db.Column(db.String(50), nullable=False)
    denial_reason = db.Column(db.Text, nullable=False)

    # NOT_APPEALED, APPEAL_IN_PROGRESS, APPEAL_APPROVED, APPEAL_DENIED, WRITTEN_OFF
    appeal_status = db.Column(db.String(30), nullable=False, default="NOT_APPEALED")

    # Clinical or administrative justification for the appeal
    appeal_justification = db.Column(db.Text, nullable=True)
    appeal_submitted_at = db.Column(db.DateTime(timezone=True), nullable=True)
    appeal_resolved_at = db.Column(db.DateTime(timezone=True), nullable=True)

    denied_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class PaymentPlan(db.Model):
    """
    Manages patient debt through structured installment payment plans.
    Reduces bad debt and improves cash flow for self-paying patients.
    """

    __tablename__ = "payment_plans"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, nullable=False, index=True)

    total_debt_amount = db.Column(db.Numeric(12, 2), nullable=False)
    installment_amount = db.Column(db.Numeric(12, 2), nullable=False)
    number_of_installments = db.Column(db.Integer, nullable=False)

    # ACTIVE, COMPLETED, DEFAULTED, CANCELLED
    status = db.Column(db.String(20), nullable=False, default="ACTIVE")

    # M-PESA, CASH, BANK_TRANSFER, CHEQUE
    preferred_payment_method = db.Column(db.String(30), nullable=True)

    # Date of next expected payment
    next_payment_due_date = db.Column(db.Date, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)
