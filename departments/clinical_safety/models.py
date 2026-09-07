import uuid
from datetime import datetime, timezone

from extensions import db


class SafetyAlertOverride(db.Model):
    """
    Audit trail for when a clinician bypasses a critical safety alert
    (e.g., drug allergy, severe interaction, dose limit).
    Essential for clinical governance and liability protection.
    """

    __tablename__ = "safety_alert_overrides"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, index=True, nullable=False)
    clinician_id = db.Column(db.Integer, nullable=False)

    # ALLERGY, DRUG_INTERACTION, DOSE_LIMIT, CONTRAINDICATION
    alert_type = db.Column(db.String(50), nullable=False)
    alert_message = db.Column(db.Text, nullable=False)

    # Clinician must provide a reason for bypassing the safety check
    justification = db.Column(db.Text, nullable=False)

    overridden_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
