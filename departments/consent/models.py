import uuid
from datetime import datetime, timezone

from extensions import db


class Consent(db.Model):
    """
    DPA 2019 Compliance Model: Tracks patient consent for treatment,
    data sharing, AI chatbot usage, and telemedicine.
    """

    __tablename__ = "consents"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, index=True, nullable=False)

    # e.g., TREATMENT, DATA_SHARING, AI_CHATBOT, RESEARCH, TELEMEDICINE
    consent_type = db.Column(db.String(50), nullable=False)

    # ACTIVE, REVOKED, EXPIRED
    status = db.Column(db.String(20), nullable=False, default="ACTIVE")

    granted_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True)
    revoked_at = db.Column(db.DateTime(timezone=True), nullable=True)
    revoked_by = db.Column(db.String(100), nullable=True)
    revocation_reason = db.Column(db.Text, nullable=True)

    # Link to signed PDF or digital signature record
    document_reference = db.Column(db.String(255), nullable=True)

    def revoke(self, revoked_by, reason=None):
        self.status = "REVOKED"
        self.revoked_at = datetime.now(timezone.utc)
        self.revoked_by = revoked_by
        self.revocation_reason = reason
