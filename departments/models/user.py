from datetime import datetime, timezone

from flask_login import UserMixin

from departments.crypto import EncryptedString
from extensions import db  # Import db from extensions


class User(UserMixin, db.Model):
    __tablename__ = "users"
    facility_id = db.Column(
        db.Integer, db.ForeignKey("facilities.id"), nullable=True, index=True
    )
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # e.g., 'records', 'nursing', etc.
    failed_login_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)
    totp_secret = db.Column(EncryptedString(255), nullable=True)
    mfa_enabled = db.Column(db.Boolean, default=False, nullable=False)

    # ── Fields added by fix/admin-department-missing-features ──────────────
    email = db.Column(db.String(120), unique=True, nullable=True, index=True)
    full_name = db.Column(db.String(120), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=True,
    )
    last_login = db.Column(db.DateTime(timezone=True), nullable=True)

    def is_locked(self) -> bool:
        if self.locked_until:
            now = datetime.now(timezone.utc)
            locked = self.locked_until
            if locked.tzinfo is None:
                locked = locked.replace(tzinfo=timezone.utc)
            return locked > now
        return False
