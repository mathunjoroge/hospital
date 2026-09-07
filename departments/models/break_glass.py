"""
departments/models/break_glass.py
──────────────────────────────────
Phase D — Break-glass Emergency Access

BreakGlassAccessLog tracks each emergency override event with:
  - user identity, patient target, stated clinical reason
  - time-boxed validity window (default 4 hours)
  - unique invite token for audit traceability
  - supervisor notification status
"""

from datetime import datetime, timezone

from extensions import db


class BreakGlassAccessLog(db.Model):
    """
    Persistent audit record of every break-glass emergency override invocation.
    Append-only: rows are never deleted, only 'revoked' via is_active flag.
    """

    __tablename__ = "break_glass_access_logs"

    id = db.Column(db.Integer, primary_key=True)

    # Who invoked the override
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False, index=True
    )
    username = db.Column(db.String(128), nullable=False)
    user_role = db.Column(db.String(50), nullable=False)

    # What they needed access to
    patient_id = db.Column(
        db.String(50), nullable=True, index=True
    )  # Target patient (if applicable)
    resource_type = db.Column(
        db.String(64), nullable=True
    )  # e.g. 'Patient', 'LabResult'
    resource_id = db.Column(db.String(64), nullable=True)

    # Why — mandatory
    reason = db.Column(db.Text, nullable=False)

    # Time-box
    invoked_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    expires_at = db.Column(
        db.DateTime(timezone=True), nullable=False
    )  # invoked_at + duration
    is_active = db.Column(
        db.Boolean, default=True, nullable=False
    )  # False once expired/revoked

    # Supervisor notification
    supervisor_notified = db.Column(db.Boolean, default=False, nullable=False)
    supervisor_notified_at = db.Column(db.DateTime(timezone=True), nullable=True)

    # Request metadata
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)

    # Relationships
    user = db.relationship(
        "User", backref=db.backref("break_glass_logs", lazy="dynamic")
    )

    def __repr__(self):
        return (
            f"<BreakGlassAccessLog id={self.id} user={self.username} "
            f"patient={self.patient_id} active={self.is_active}>"
        )

    def is_valid(self) -> bool:
        """Return True if this break-glass grant is still within its validity window."""
        if not self.is_active or not self.expires_at:
            return False
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < expires

    def revoke(self):
        """Explicitly deactivate this override before its natural expiry."""
        self.is_active = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "user_role": self.user_role,
            "patient_id": self.patient_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "reason": self.reason,
            "invoked_at": self.invoked_at.isoformat() if self.invoked_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "is_valid": self.is_valid(),
            "supervisor_notified": self.supervisor_notified,
        }
