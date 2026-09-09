import uuid
from datetime import datetime, timezone
from extensions import db

class AncVisit(db.Model):
    __tablename__ = "anc_visits"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FIXED: was Integer
    visit_number = db.Column(db.Integer, nullable=False)
    gestation_weeks = db.Column(db.Integer, nullable=False)
    high_risk_factors = db.Column(db.Text, nullable=True)
    next_appointment_date = db.Column(db.Date, nullable=True)
    visit_date = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ImmunizationRecord(db.Model):
    __tablename__ = "immunization_records"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    child_patient_id = db.Column(db.String(20), index=True, nullable=False)  # FIXED: was Integer
    vaccine_name = db.Column(db.String(100), nullable=False)
    dose_number = db.Column(db.Integer, nullable=False)
    batch_number = db.Column(db.String(50), nullable=True)
    administered_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
