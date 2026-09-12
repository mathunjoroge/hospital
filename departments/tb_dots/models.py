"""
TB/DOTS Module Data Models
"""
import uuid
from datetime import datetime, timezone

from extensions import db


class TBEnrollment(db.Model):
    """TB enrollment and tracking."""
    __tablename__ = "tb_enrollments"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    enrollment_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    tb_number = db.Column(db.String(50), unique=True, nullable=False, index=True)  # Unique TB identifier
    hiv_status = db.Column(db.String(20), nullable=True)  # positive, negative, unknown, refused_test
    art_enrollment_id = db.Column(db.String(36), db.ForeignKey('art_enrollments.id'), nullable=True)  # Link to ART if co-infected
    tb_classification = db.Column(db.String(20), nullable=True)  # pulmonary, extrapulmonary
    site_of_disease = db.Column(db.String(100), nullable=True)  # for extrapulmonary
    bacteriological_status = db.Column(db.String(20), nullable=True)  # confirmed, clinical, etc.
    treatment_start_date = db.Column(db.DateTime(timezone=True), nullable=True)
    current_regimen_id = db.Column(db.String(36), db.ForeignKey('tb_regimens.id'), nullable=True, index=True)
    facility_enrolled_at = db.Column(db.String(100), nullable=True)
    # FK to the Encounter opened when this TB enrollment starts
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationships
    current_regimen = db.relationship('TBRegimen', foreign_keys=[current_regimen_id])
    doses_taken = db.relationship('DoseTaken', back_populates='tb_enrollment', lazy='dynamic')
    sputum_results = db.relationship('SputumResult', back_populates='tb_enrollment', lazy='dynamic')
    chest_xrays = db.relationship('ChestXRay', back_populates='tb_enrollment', lazy='dynamic')
    hiv_status_records = db.relationship('HIVStatus', back_populates='tb_enrollment', lazy='dynamic')

    def __repr__(self):
        return f"<TBEnrollment {self.tb_number}: {self.patient_id}>"


class TBRegimen(db.Model):
    """TB treatment regimen definitions."""
    __tablename__ = "tb_regimens"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    regimen_code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    regimen_name = db.Column(db.String(100), nullable=False)
    line_of_therapy = db.Column(db.Integer, nullable=False)  # 1, 2, 3+
    drugs = db.Column(db.Text, nullable=False)  # JSON or comma-separated list of TB drugs
    duration_months = db.Column(db.Integer, nullable=False)  # Standard duration for this regimen
    is_preferred = db.Column(db.Boolean, default=False)
    is_alternative = db.Column(db.Boolean, default=False)
    restriction_notes = db.Column(db.Text, nullable=True)  # Contraindications, etc.
    effective_from = db.Column(db.Date, nullable=False)
    effective_to = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<TBRegimen {self.regimen_code}: {self.regimen_name}>"


class DoseTaken(db.Model):
    """TB DOTS dose taken monitoring."""
    __tablename__ = "doses_taken"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    date_taken = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    tb_enrollment_id = db.Column(db.String(36), db.ForeignKey('tb_enrollments.id'), nullable=False, index=True)
    dose_number = db.Column(db.Integer, nullable=True)  # Which dose in the sequence
    taken_as_directly_observed = db.Column(db.Boolean, default=False)  # DOT or self-administered
    # FK to the Encounter opened when this dose was observed (if DOT)
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    tb_enrollment = db.relationship('TBEnrollment', foreign_keys=[tb_enrollment_id], back_populates='doses_taken')

    def __repr__(self):
        return f"<DoseTaken {self.patient_id} on {self.date_taken}>"


class SputumResult(db.Model):
    """TB sputum smear and culture results."""
    __tablename__ = "sputum_results"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    test_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    tb_enrollment_id = db.Column(db.String(36), db.ForeignKey('tb_enrollments.id'), nullable=False, index=True)
    specimen_type = db.Column(db.String(20), nullable=True)  # sputum, gastric, etc.
    specimen_number = db.Column(db.Integer, nullable=True)  # 1, 2, 3 for series
    smear_result = db.Column(db.String(20), nullable=True)  # negative, scanty, 1+, 2+, 3+
    culture_result = db.Column(db.String(20), nullable=True)  # negative, positive, contaminated
    culture_species = db.Column(db.String(50), nullable=True)  # M. tuberculosis, etc.
    drug_susceptibility = db.Column(db.Text, nullable=True)  # JSON or text for DST results
    # FK to the Encounter when this test was ordered
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    tb_enrollment = db.relationship('TBEnrollment', foreign_keys=[tb_enrollment_id], back_populates='sputum_results')

    def __repr__(self):
        return f"<SputumResult {self.patient_id}: smear {self.smear_result}, culture {self.culture_result}>"


class ChestXRay(db.Model):
    """TB chest X-ray results."""
    __tablename__ = "chest_xrays"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    test_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    tb_enrollment_id = db.Column(db.String(36), db.ForeignKey('tb_enrollments.id'), nullable=False, index=True)
    finding = db.Column(db.String(100), nullable=True)  # normal, abnormal, cavitary, etc.
    severity = db.Column(db.String(20), nullable=True)  # minimal, moderate, advanced
    progression = db.Column(db.String(20), nullable=True)  # improved, worsened, unchanged
    # FK to the Encounter when this test was ordered
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    tb_enrollment = db.relationship('TBEnrollment', foreign_keys=[tb_enrollment_id], back_populates='chest_xrays')

    def __repr__(self):
        return f"<ChestXRay {self.patient_id}: {self.finding}>"


class HIVStatus(db.Model):
    """HIV status testing for TB patients (since TB/HIV comorbidity is high)."""
    __tablename__ = "hiv_status"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    test_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    tb_enrollment_id = db.Column(db.String(36), db.ForeignKey('tb_enrollments.id'), nullable=False, index=True)
    test_type = db.Column(db.String(20), nullable=True)  # rapid, ELISA, Western blot
    result = db.Column(db.String(20), nullable=True)  # positive, negative, indeterminate
    cd4_count = db.Column(db.Integer, nullable=True)  # cells/µL if available
    # FK to the Encounter when this test was ordered
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    tb_enrollment = db.relationship('TBEnrollment', foreign_keys=[tb_enrollment_id], back_populates='hiv_status_records')

    def __repr__(self):
        return f"<HIVStatus {self.patient_id}: {self.result}>"
