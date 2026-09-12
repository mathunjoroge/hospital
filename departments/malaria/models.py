"""
Malaria Module Data Models
"""
import uuid
from datetime import datetime, timezone

from extensions import db


class MalariaCase(db.Model):
    """Malaria case tracking and treatment."""
    __tablename__ = "malaria_cases"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    case_number = db.Column(db.String(50), unique=True, nullable=False, index=True)  # Unique malaria case identifier
    diagnosis_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    malaria_species = db.Column(db.String(20), nullable=True)  # falciparum, vivax, ovale, malariae, knowlesi, mixed
    parasite_density = db.Column(db.Integer, nullable=True)  # parasites/µL (if microscopy) or percentage (if RDT)
    diagnosis_method = db.Column(db.String(20), nullable=True)  # microscopy, RDT, PCR
    severity = db.Column(db.String(20), nullable=True)  # uncomplicated, severe
    pregnancy_status = db.Column(db.String(20), nullable=True)  # not_pregnant, pregnant_first_trimester, etc.
    treatment_start_date = db.Column(db.DateTime(timezone=True), nullable=True)
    current_regimen_id = db.Column(db.String(36), db.ForeignKey('malaria_regimens.id'), nullable=True, index=True)
    facility_diagnosed_at = db.Column(db.String(100), nullable=True)
    # FK to the Encounter opened when this malaria case is diagnosed
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationships
    current_regimen = db.relationship('MalariaRegimen', foreign_keys=[current_regimen_id])
    treatments = db.relationship('MalariaTreatment', back_populates='malaria_case', lazy='dynamic')
    lab_results = db.relationship('MalariaLabResult', back_populates='malaria_case', lazy='dynamic')

    def __repr__(self):
        return f"<MalariaCase {self.case_number}: {self.patient_id}>"


class MalariaRegimen(db.Model):
    """Malaria treatment regimen definitions."""
    __tablename__ = "malaria_regimens"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    regimen_code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    regimen_name = db.Column(db.String(100), nullable=False)
    line_of_therapy = db.Column(db.Integer, nullable=False)  # 1, 2, 3+
    drugs = db.Column(db.Text, nullable=False)  # JSON or comma-separated list of antimalarials
    duration_days = db.Column(db.Integer, nullable=False)  # Standard duration for this regimen
    is_preferred = db.Column(db.Boolean, default=False)
    is_alternative = db.Column(db.Boolean, default=False)
    restriction_notes = db.Column(db.Text, nullable=True)  # Contraindications, etc. (e.g., pregnancy, G6PD deficiency)
    effective_from = db.Column(db.Date, nullable=False)
    effective_to = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<MalariaRegimen {self.regimen_code}: {self.regimen_name}>"


class MalariaTreatment(db.Model):
    """Malaria treatment administration records."""
    __tablename__ = "malaria_treatments"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    date_administered = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    malaria_case_id = db.Column(db.String(36), db.ForeignKey('malaria_cases.id'), nullable=False, index=True)
    dose_number = db.Column(db.Integer, nullable=True)  # Which dose in the sequence
    administered_as_directly_observed = db.Column(db.Boolean, default=False)  # DOT or self-administered
    # FK to the Encounter opened when this treatment was administered (if DOT)
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    malaria_case = db.relationship('MalariaCase', foreign_keys=[malaria_case_id], back_populates='treatments')

    def __repr__(self):
        return f"<MalariaTreatment {self.patient_id} on {self.date_administered}>"


class MalariaLabResult(db.Model):
    """Malaria laboratory results (follow-up tests)."""
    __tablename__ = "malaria_lab_results"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    test_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    malaria_case_id = db.Column(db.String(36), db.ForeignKey('malaria_cases.id'), nullable=False, index=True)
    test_type = db.Column(db.String(20), nullable=True)  # microscopy, RDT, PCR, hemoglobin, etc.
    result_value = db.Column(db.String(50), nullable=True)  # Result value (e.g., parasite density, hemoglobin level)
    result_interpretation = db.Column(db.String(20), nullable=True)  # positive, negative, etc.
    # FK to the Encounter when this test was ordered
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    malaria_case = db.relationship('MalariaCase', foreign_keys=[malaria_case_id], back_populates='lab_results')

    def __repr__(self):
        return f"<MalariaLabResult {self.patient_id}: {self.test_type} {self.result_value}>"
