"""
departments/models/fhir_subscription.py
────────────────────────────────────────
Data model for FHIR R4 Subscription webhooks (REST-hook channels).
Allows external HIEs and health systems to subscribe to real-time clinical events.
"""

import uuid
from datetime import datetime

from extensions import db


class FHIRSubscription(db.Model):
    """
    FHIR R4 Subscription resource entity.
    Defines real-time event webhooks for HIE interoperability.
    """
    __tablename__ = "fhir_subscriptions"

    id = db.Column(db.Integer, primary_key=True)
    subscription_id = db.Column(db.String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid4()), index=True)

    status = db.Column(db.String(20), nullable=False, default="active")  # active, off, error
    reason = db.Column(db.String(255), nullable=True, default="HIE Real-time Clinical Synchronization")

    # Criteria e.g. "Observation", "Encounter", "MedicationRequest", "Condition"
    criteria = db.Column(db.String(100), nullable=False, index=True)

    # Channel configuration
    channel_type = db.Column(db.String(20), nullable=False, default="rest-hook")
    endpoint_url = db.Column(db.String(500), nullable=False)

    # Security & Verification
    secret_token = db.Column(db.String(128), nullable=True)  # Secret key for HMAC-SHA256 signature
    headers_json = db.Column(db.Text, nullable=True)  # JSON representation of custom headers

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_triggered_at = db.Column(db.DateTime, nullable=True)
    failure_count = db.Column(db.Integer, nullable=False, default=0)

    def to_fhir(self) -> dict:
        """Serialize model to FHIR R4 Subscription resource format."""
        headers = []
        if self.secret_token:
            headers.append(f"X-FHIR-Token: {self.secret_token}")

        return {
            "resourceType": "Subscription",
            "id": self.subscription_id,
            "status": self.status,
            "reason": self.reason,
            "criteria": self.criteria,
            "channel": {
                "type": self.channel_type,
                "endpoint": self.endpoint_url,
                "header": headers,
            },
        }
