import uuid
from datetime import datetime, timezone

from extensions import db


class Appointment(db.Model):
    """
    Core scheduling model for patient visits, follow-ups, and procedures.
    """

    __tablename__ = "appointments"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)
    provider_id = db.Column(
        db.String(50), index=True, nullable=False
    )  # Staff/Clinician ID

    scheduled_start = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    scheduled_end = db.Column(db.DateTime(timezone=True), nullable=False)

    # SCHEDULED, CHECKED_IN, READY, IN_PROGRESS, COMPLETED, NO_SHOW, CANCELLED
    status = db.Column(db.String(20), nullable=False, default="SCHEDULED")

    appointment_type = db.Column(
        db.String(50), nullable=False
    )  # e.g., CONSULTATION, TELEHEALTH, PROCEDURE
    reason_for_visit = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def check_in(self):
        self.status = "CHECKED_IN"

    def mark_ready(self):
        """Triage/vitals complete: patient is ready for the clinician."""
        self.status = "READY"

    def start_consultation(self):
        self.status = "IN_PROGRESS"

    def mark_no_show(self):
        self.status = "NO_SHOW"



class ClinicBooking(db.Model):
    """
    Clinic booking/appointment model.
    Links patients to scheduled clinic visits with consultation fees.
    
    This is separate from Appointment (which is the core scheduling model).
    ClinicBooking represents a booked clinic visit with billing information.
    """

    __tablename__ = "clinic_bookings"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )
    clinic_id = db.Column(db.String(36), nullable=True, index=True)
    appointment_date = db.Column(
        db.DateTime(timezone=True), nullable=False, index=True
    )
    status = db.Column(db.String(20), nullable=False, default="SCHEDULED")
    consultation_fee = db.Column(db.Numeric(10, 2), nullable=True, default=0)
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    patient = db.relationship("Patient", backref="clinic_bookings")
    encounter = db.relationship("Encounter", backref="clinic_booking")

    def __repr__(self):
        return f"<ClinicBooking {self.id} - Patient {self.patient_id} - {self.status}>"
