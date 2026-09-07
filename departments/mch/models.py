import uuid
from datetime import datetime, timezone

from extensions import db


class AncVisit(db.Model):
    """
    Tracks Antenatal Care (ANC) visits.
    Aligns with MoH 711 register requirements for KHIS/DHIS2 reporting.
    """

    __tablename__ = "anc_visits"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, index=True, nullable=False)

    # ANC 1, 2, 3, 4+
    visit_number = db.Column(db.Integer, nullable=False)
    gestation_weeks = db.Column(db.Integer, nullable=False)

    # JSON or comma-separated string of risk factors (e.g., "Hypertension, HIV")
    high_risk_factors = db.Column(db.Text, nullable=True)
    next_appointment_date = db.Column(db.Date, nullable=True)

    visit_date = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ImmunizationRecord(db.Model):
    """
    Tracks child immunizations.
    Essential for cold chain tracking, batch recalls, and MoH 710 reporting.
    """

    __tablename__ = "immunization_records"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    child_patient_id = db.Column(db.Integer, index=True, nullable=False)

    # e.g., OPV, Pentavalent, Measles-Rubella, BCG
    vaccine_name = db.Column(db.String(100), nullable=False)
    dose_number = db.Column(db.Integer, nullable=False)

    # Crucial for adverse event tracking and batch recalls
    batch_number = db.Column(db.String(50), nullable=True)

    administered_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
