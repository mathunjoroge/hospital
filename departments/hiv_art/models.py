"""
HIV/ART Module Data Models
"""
import uuid
from datetime import datetime, timezone

from extensions import db


class ARTEnrollment(db.Model):
    """HIV Antiretroviral Therapy enrollment and tracking."""
    __tablename__ = "art_enrollments"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    enrollment_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    art_number = db.Column(db.String(50), unique=True, nullable=False, index=True)  # Unique ART identifier
    baseline_cd4 = db.Column(db.Integer, nullable=True)  # cells/µL at enrollment
    baseline_who_stage = db.Column(db.Integer, nullable=True)  # WHO clinical stage 1-4
    art_start_date = db.Column(db.DateTime(timezone=True), nullable=True)
    current_regimen_id = db.Column(db.String(36), db.ForeignKey('art_regimens.id'), nullable=True, index=True)
    facility_enrolled_at = db.Column(db.String(100), nullable=True)
    # PMTCT & HEI EID fields
    is_pregnant = db.Column(db.Boolean, default=False)
    is_breastfeeding = db.Column(db.Boolean, default=False)
    hei_infant_prophylaxis = db.Column(db.String(50), nullable=True)  # NVP_SYRUP, AZT_SYRUP, DUAL_PROPHYLAXIS
    eid_dna_pcr_6wk_result = db.Column(db.String(20), nullable=True)  # POSITIVE, NEGATIVE, PENDING
    eid_dna_pcr_12mo_result = db.Column(db.String(20), nullable=True)
    eid_antibody_18mo_result = db.Column(db.String(20), nullable=True)

    # FK to the Encounter opened when this ART enrollment starts
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationships
    current_regimen = db.relationship('ARTRegimen', foreign_keys=[current_regimen_id])
    adherence_visits = db.relationship('AdherenceVisit', back_populates='enrollment', lazy='dynamic')
    viral_loads = db.relationship('ViralLoad', back_populates='enrollment', lazy='dynamic')
    cd4_counts = db.relationship('CD4Count', back_populates='enrollment', lazy='dynamic')
    who_stages = db.relationship('WHOStage', back_populates='enrollment', lazy='dynamic')

    def __repr__(self):
        return f"<ARTEnrollment {self.art_number}: {self.patient_id}>"


class ARTRegimen(db.Model):
    """Antiretroviral therapy regimen definitions."""
    __tablename__ = "art_regimens"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    regimen_code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    regimen_name = db.Column(db.String(100), nullable=False)
    line_of_therapy = db.Column(db.Integer, nullable=False)  # 1, 2, 3+
    arv_drugs = db.Column(db.Text, nullable=False)  # JSON or comma-separated list of ARVs
    is_preferred = db.Column(db.Boolean, default=False)
    is_alternative = db.Column(db.Boolean, default=False)
    restriction_notes = db.Column(db.Text, nullable=True)  # Contraindications, etc.
    effective_from = db.Column(db.Date, nullable=False)
    effective_to = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<ARTRegimen {self.regimen_code}: {self.regimen_name}>"


class AdherenceVisit(db.Model):
    """HIV ART adherence monitoring visits."""
    __tablename__ = "adherence_visits"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    visit_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    art_enrollment_id = db.Column(db.String(36), db.ForeignKey('art_enrollments.id'), nullable=False, index=True)
    pills_dispensed = db.Column(db.Integer, nullable=True)
    pills_returned = db.Column(db.Integer, nullable=True)
    days_since_last_visit = db.Column(db.Integer, nullable=True)
    adherence_percentage = db.Column(db.Float, nullable=True)  # Calculated: (pills_taken/pills_dispensed)*100
    adherence_category = db.Column(db.String(20), nullable=True)  # good/fair/poor
    viral_load_ordered = db.Column(db.Boolean, default=False)
    cd4_ordered = db.Column(db.Boolean, default=False)
    next_visit_date = db.Column(db.Date, nullable=True)
    # FK to the Encounter opened when this adherence visit occurs
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    enrollment = db.relationship('ARTEnrollment', foreign_keys=[art_enrollment_id], back_populates='adherence_visits')

    def __repr__(self):
        return f"<AdherenceVisit {self.patient_id} on {self.visit_date}>"


class ViralLoad(db.Model):
    """HIV viral load test results."""
    __tablename__ = "viral_loads"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    visit_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    art_enrollment_id = db.Column(db.String(36), db.ForeignKey('art_enrollments.id'), nullable=False, index=True)
    test_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    viral_load_copies = db.Column(db.Integer, nullable=True)  # HIV RNA copies/mL
    test_type = db.Column(db.String(50), nullable=True)  # routine, diagnostic, confirmation
    result_date = db.Column(db.Date, nullable=True)
    # FK to the Encounter when this test was ordered
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    enrollment = db.relationship('ARTEnrollment', foreign_keys=[art_enrollment_id], back_populates='viral_loads')

    def __repr__(self):
        return f"<ViralLoad {self.patient_id}: {self.viral_load_copies} copies/mL>"


class CD4Count(db.Model):
    """HIV CD4+ T-lymphocyte count results."""
    __tablename__ = "cd4_counts"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    visit_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    art_enrollment_id = db.Column(db.String(36), db.ForeignKey('art_enrollments.id'), nullable=False, index=True)
    test_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    cd4_count = db.Column(db.Integer, nullable=True)  # cells/µL
    cd4_percent = db.Column(db.Float, nullable=True)  # Percentage of total lymphocytes
    result_date = db.Column(db.Date, nullable=True)
    # FK to the Encounter when this test was ordered
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    enrollment = db.relationship('ARTEnrollment', foreign_keys=[art_enrollment_id], back_populates='cd4_counts')

    def __repr__(self):
        return f"<CD4Count {self.patient_id}: {self.cd4_count} cells/µL>"


class WHOStage(db.Model):
    """HIV WHO clinical staging assessments."""
    __tablename__ = "who_stages"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)  # FK to patients
    visit_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    art_enrollment_id = db.Column(db.String(36), db.ForeignKey('art_enrollments.id'), nullable=False, index=True)
    assessment_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    who_stage = db.Column(db.Integer, nullable=False)  # WHO clinical stage 1-4
    defining_conditions = db.Column(db.Text, nullable=True)  # Conditions that define the stage
    result_date = db.Column(db.Date, nullable=True)
    # FK to the Encounter when this assessment was made
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # Relationship
    enrollment = db.relationship('ARTEnrollment', foreign_keys=[art_enrollment_id], back_populates='who_stages')

    def __repr__(self):
        return f"<WHOStage {self.patient_id}: Stage {self.who_stage}>"


class MOH731ARVRegimenPatientMonthly(db.Model):
    """MOH 731 HIV/AIDS Monthly Summary — ARV Regimen Patient Counts."""
    __tablename__ = "moh_731_arv_regimen_patients_monthly"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    facility_id = db.Column(db.String(50), nullable=False, default="KE_MOH_HOSPITAL_001")
    period = db.Column(db.String(6), nullable=False, index=True)  # YYYYMM
    regimen_code = db.Column(db.String(20), nullable=False, index=True)  # e.g., AF1A, AF1B, PF1A
    regimen_line = db.Column(db.String(20), nullable=False)  # Adult_1st, Adult_2nd, Ped_1st, Ped_2nd, 3rd_Line
    active_patients_male = db.Column(db.Integer, default=0, nullable=False)
    active_patients_female = db.Column(db.Integer, default=0, nullable=False)
    total_active_patients = db.Column(db.Integer, default=0, nullable=False)
    new_patients_started = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<MOH731ARVRegimen {self.period} - {self.regimen_code}: {self.total_active_patients} active>"


class MOH729BARVFCDRRMonthly(db.Model):
    """MOH 729B ARV FCDRR Monthly — ARV Commodity & Patient Load Report."""
    __tablename__ = "moh_729b_arv_fcdrr_monthly"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    facility_id = db.Column(db.String(50), nullable=False, default="KE_MOH_HOSPITAL_001")
    period = db.Column(db.String(6), nullable=False, index=True)  # YYYYMM
    arv_drug_code = db.Column(db.String(40), nullable=False, index=True)  # e.g., ARV_TLD_300_300_50
    unit_pack_size = db.Column(db.String(20), nullable=False)  # Bottle_30s, Bottle_90s, Bottle_180s
    patients_on_regimen = db.Column(db.Integer, default=0, nullable=False)
    beginning_balance = db.Column(db.Integer, default=0, nullable=False)
    quantity_received = db.Column(db.Integer, default=0, nullable=False)
    quantity_dispensed = db.Column(db.Integer, default=0, nullable=False)
    losses_adjustments = db.Column(db.Integer, default=0, nullable=False)
    ending_balance = db.Column(db.Integer, default=0, nullable=False)
    days_stocked_out = db.Column(db.Integer, default=0, nullable=False)
    months_of_stock = db.Column(db.Numeric(4, 2), default=0.00, nullable=False)
    quantity_requested = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<MOH729BARVFCDRR {self.period} - {self.arv_drug_code}: {self.patients_on_regimen} patients, stock={self.ending_balance}>"

