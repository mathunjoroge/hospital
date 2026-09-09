from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db


class PatientUser(UserMixin, db.Model):
    """
    Dedicated portal authentication model for Patients.
    Kept separate from clinical staff User accounts to ensure identity boundary.
    """

    __tablename__ = "patient_users"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.Integer, db.ForeignKey("patients.id"), unique=True, nullable=False
    )
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    failed_login_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime(timezone=True), nullable=True)
    reset_token = db.Column(db.String(255), nullable=True, index=True)
    reset_token_expiry = db.Column(db.DateTime(timezone=True), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_login = db.Column(db.DateTime(timezone=True), nullable=True)

    patient = db.relationship(
        "Patient", backref=db.backref("portal_user", uselist=False)
    )

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password, method="pbkdf2:sha256")

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def is_locked(self) -> bool:
        if self.locked_until:
            # SQLite strips tzinfo on roundtrip; treat all stored datetimes as UTC
            locked = self.locked_until
            if locked.tzinfo is None:
                locked = locked.replace(tzinfo=timezone.utc)
            if locked > datetime.now(timezone.utc):
                return True
        return False
