"""
departments/models/telemedicine.py
───────────────────────────────────
Phase E — Telemedicine & Virtual Consultation Engine

TelemedicineSession model tracks virtual video/audio doctor-patient consultations.
"""

import uuid
from datetime import datetime, timezone

from extensions import db


class TelemedicineSession(db.Model):
    """
    Tracks a virtual telemedicine session between a doctor and patient.
    Status transitions: SCHEDULED -> ACTIVE -> COMPLETED (or CANCELLED).
    """

    __tablename__ = "telemedicine_sessions"

    id = db.Column(db.Integer, primary_key=True)
    session_uuid = db.Column(
        db.String(36),
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True,
    )

    doctor_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False, index=True
    )
    patient_id = db.Column(db.String(50), nullable=False, index=True)
    appointment_id = db.Column(db.Integer, nullable=True)
    # FK to the Encounter created when the session starts (nullable until started)
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    status = db.Column(
        db.String(20),
        default="SCHEDULED",
        nullable=False,
        index=True,
    )  # SCHEDULED, ACTIVE, COMPLETED, CANCELLED

    room_token = db.Column(db.String(128), nullable=False)
    scheduled_start = db.Column(db.DateTime(timezone=True), nullable=True)
    actual_start = db.Column(db.DateTime(timezone=True), nullable=True)
    ended_at = db.Column(db.DateTime(timezone=True), nullable=True)

    clinical_notes = db.Column(db.Text, nullable=True)
    prescriptions_json = db.Column(
        db.Text, nullable=True
    )  # JSON array of prescribed meds

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    doctor = db.relationship("User", foreign_keys=[doctor_id])

    def __repr__(self):
        return f"<TelemedicineSession uuid={self.session_uuid} status={self.status} doc={self.doctor_id} pat={self.patient_id}>"

    def is_participant(self, user) -> bool:
        """Return True if user is the assigned doctor or admin."""
        if not user:
            return False
        if user.role == "admin":
            return True
        return user.id == self.doctor_id

    def start_session(self):
        """Transition session to ACTIVE status."""
        self.status = "ACTIVE"
        if not self.actual_start:
            self.actual_start = datetime.now(timezone.utc)

    def end_session(self, notes: str = None):
        """Complete the virtual consultation."""
        self.status = "COMPLETED"
        self.ended_at = datetime.now(timezone.utc)
        if notes:
            self.clinical_notes = notes

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_uuid": self.session_uuid,
            "doctor_id": self.doctor_id,
            "doctor_name": self.doctor.username if self.doctor else None,
            "patient_id": self.patient_id,
            "appointment_id": self.appointment_id,
            "status": self.status,
            "room_token": self.room_token,
            "scheduled_start": self.scheduled_start.isoformat()
            if self.scheduled_start
            else None,
            "actual_start": self.actual_start.isoformat()
            if self.actual_start
            else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "clinical_notes": self.clinical_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
