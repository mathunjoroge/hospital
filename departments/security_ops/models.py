import uuid
from datetime import datetime, timezone

from extensions import db


class TokenRevocation(db.Model):
    """
    Tracks JWT tokens or session IDs that have been revoked before expiration.
    Critical for immediately terminating access when staff leave or are suspended.
    """

    __tablename__ = "token_revocations"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # The unique JWT identifier (jti) or session ID being revoked
    token_identifier = db.Column(db.String(255), nullable=False, index=True)

    # The user whose access is being terminated
    user_id = db.Column(db.Integer, nullable=False, index=True)

    # The administrator performing the revocation
    revoked_by = db.Column(db.Integer, nullable=False)

    reason = db.Column(db.Text, nullable=True)

    revoked_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class AccessRequest(db.Model):
    """
    Tracks requests for access to restricted or highly sensitive patient records.
    Ensures all privileged access is justified, approved, and audited.
    """

    __tablename__ = "access_requests"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    user_id = db.Column(db.Integer, nullable=False, index=True)
    patient_id = db.Column(db.Integer, nullable=False, index=True)

    # Clinical or administrative justification for accessing the restricted record
    justification = db.Column(db.Text, nullable=False)

    # PENDING, APPROVED, DENIED, EXPIRED
    status = db.Column(db.String(20), nullable=False, default="PENDING")

    approved_by = db.Column(db.Integer, nullable=True)
    denial_reason = db.Column(db.Text, nullable=True)

    requested_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    resolved_at = db.Column(db.DateTime(timezone=True), nullable=True)
