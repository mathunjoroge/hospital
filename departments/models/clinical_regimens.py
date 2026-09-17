"""
Master Clinical Regimens Model (NASCOP ARV, NTLD-P TB, and TPT Regimens)
"""
from datetime import datetime, timezone
from extensions import db


class MasterClinicalRegimen(db.Model):
    """
    Master Clinical Regimen identifier model supporting Kenya NASCOP (HIV),
    NTLD-P (DS-TB / DR-TB), and TPT (TB Preventive Therapy) national regimens.
    """
    __tablename__ = "master_clinical_regimens"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    program_domain = db.Column(db.String(20), nullable=False, index=True)  # HIV, TB, TPT
    nascop_ntldp_code = db.Column(db.String(30), unique=True, nullable=False, index=True)  # e.g., AF1A, DR-BPaLM, TPT-3HP
    regimen_acronym = db.Column(db.String(50), nullable=False)  # e.g., TLD, BPaLM, 3HP
    line_tier = db.Column(db.String(30), nullable=False)  # 1st_Line, 2nd_Line, 3rd_Line_Salvage, TPT
    target_population = db.Column(db.String(100), nullable=False)  # e.g., Adults ≥30kg, Children <15yrs
    drug_components = db.Column(db.Text, nullable=False)  # Composition & Dosage details
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return f"<MasterClinicalRegimen {self.nascop_ntldp_code}: {self.regimen_acronym} ({self.program_domain})>"
