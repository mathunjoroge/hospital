import uuid
from datetime import datetime, timezone

from extensions import db


class Appointment(db.Model):
    __tablename__ = "appointments"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, index=True, nullable=False)
    provider_id = db.Column(db.Integer, index=True, nullable=False)

    scheduled_start = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    scheduled_end = db.Column(db.DateTime(timezone=True), nullable=False)

    status = db.Column(db.String(20), nullable=False, default="SCHEDULED")
    appointment_type = db.Column(db.String(50), nullable=False)
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

    def mark_no_show(self):
        self.status = "NO_SHOW"
