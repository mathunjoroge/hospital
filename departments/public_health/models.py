import uuid
from datetime import datetime, timezone

from extensions import db


class NotifiableDisease(db.Model):
    """
    Case-based reporting for diseases that must be reported to the
    Kenya Ministry of Health and WHO under the International Health
    Regulations (IHR 2005).

    Examples: Cholera, Measles, Yellow Fever, Viral Haemorrhagic Fevers,
    Acute Flaccid Paralysis (AFP), Anthrax, Plague.
    """

    __tablename__ = "notifiable_diseases"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, index=True, nullable=False)
    clinician_id = db.Column(db.Integer, nullable=False)

    # ICD-10 code for the notifiable condition
    icd10_code = db.Column(db.String(10), nullable=False)
    disease_name = db.Column(db.String(150), nullable=False)

    # SUSPECTED, CONFIRMED, DISCARDED
    case_classification = db.Column(db.String(20), nullable=False, default="SUSPECTED")

    # PENDING, SUBMITTED, ACKNOWLEDGED
    report_status = db.Column(db.String(20), nullable=False, default="PENDING")

    # County and sub-county for geographic surveillance
    county = db.Column(db.String(100), nullable=True)
    sub_county = db.Column(db.String(100), nullable=True)

    onset_date = db.Column(db.Date, nullable=True)
    diagnosis_date = db.Column(db.Date, nullable=False)
    reported_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    notes = db.Column(db.Text, nullable=True)


class MortalityReport(db.Model):
    """
    Tracks cause of death for facility mortality reporting.
    Aligns with MoH 736 (Death Notification) and DHIS2 mortality indicators.
    """

    __tablename__ = "mortality_reports"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.Integer, index=True, nullable=False)
    clinician_id = db.Column(db.Integer, nullable=False)

    # Primary cause of death (ICD-10)
    primary_cause_icd10 = db.Column(db.String(10), nullable=False)
    primary_cause_text = db.Column(db.String(255), nullable=False)

    # Contributing causes
    secondary_cause_icd10 = db.Column(db.String(10), nullable=True)
    secondary_cause_text = db.Column(db.String(255), nullable=True)

    # INPATIENT, OUTPATIENT, EMERGENCY, BROUGHT_IN_DEAD
    death_context = db.Column(db.String(30), nullable=False)

    # NEONATAL, INFANT, CHILD_UNDER_5, ADULT, MATERNAL
    age_category = db.Column(db.String(20), nullable=False)

    date_of_death = db.Column(db.DateTime(timezone=True), nullable=False)
    reported_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class OutbreakSignal(db.Model):
    """
    Captures early warning signals for potential disease outbreaks.
    Used for event-based surveillance and rapid response coordination.
    """

    __tablename__ = "outbreak_signals"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # e.g., "Cluster of 5+ acute watery diarrhoea cases in 48 hours"
    signal_description = db.Column(db.Text, nullable=False)
    disease_suspected = db.Column(db.String(150), nullable=True)

    # DETECTED, UNDER_INVESTIGATION, CONFIRMED_OUTBREAK, CLOSED
    status = db.Column(db.String(30), nullable=False, default="DETECTED")

    case_count = db.Column(db.Integer, nullable=False, default=1)
    county = db.Column(db.String(100), nullable=True)
    sub_county = db.Column(db.String(100), nullable=True)

    detected_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    resolved_at = db.Column(db.DateTime(timezone=True), nullable=True)
