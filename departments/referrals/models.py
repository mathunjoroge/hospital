import uuid
from datetime import datetime, timezone

from extensions import db


class Referral(db.Model):
    __tablename__ = "referrals"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, index=True, nullable=False)

    referring_facility = db.Column(db.String(150), nullable=False)
    receiving_facility = db.Column(db.String(150), nullable=False)

    clinical_summary = db.Column(db.Text, nullable=False)
    reason_for_referral = db.Column(db.String(255), nullable=False)

    status = db.Column(db.String(20), nullable=False, default="PENDING")

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class DischargeSummary(db.Model):
    __tablename__ = "discharge_summaries"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, index=True, nullable=False)
    appointment_id = db.Column(db.String(36), nullable=True)

    admission_date = db.Column(db.DateTime(timezone=True), nullable=False)
    discharge_date = db.Column(db.DateTime(timezone=True), nullable=False)

    primary_diagnosis = db.Column(db.String(255), nullable=False)
    secondary_diagnoses = db.Column(db.Text, nullable=True)

    discharge_medications = db.Column(db.Text, nullable=True)
    follow_up_instructions = db.Column(db.Text, nullable=True)
    referred_to = db.Column(db.String(150), nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
