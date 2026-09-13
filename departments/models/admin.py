from datetime import datetime

from extensions import db


class Log(db.Model):
    __tablename__ = "logs"
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    level = db.Column(db.String(20), nullable=False)  # e.g., INFO, ERROR, WARNING
    message = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    source = db.Column(db.String(50))  # e.g., 'auth', 'pharmacy'

    # Cryptographic Tamper-Evident SHA-256 Hash Chaining (HIPAA § 164.312(b))
    previous_hash = db.Column(db.String(64), nullable=True)
    entry_hash = db.Column(db.String(64), nullable=True, index=True)

