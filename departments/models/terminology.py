"""
Terminology models for Phase 1 (ICD-10, SNOMED, LOINC).
"""
from extensions import db


class ICD10Code(db.Model):
    """Standardized ICD-10 diagnosis codes."""
    __tablename__ = "icd10_codes"

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)
    chapter = db.Column(db.String(100))  # e.g., "Respiratory"
    block = db.Column(db.String(100))    # e.g., "Acute upper respiratory infections"

    def __repr__(self):
        return f"<ICD10 {self.code}: {self.description[:30]}>"


class SnomedCode(db.Model):
    """Standardized SNOMED CT codes."""
    __tablename__ = "snomed_codes"

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), unique=True, nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<SNOMED {self.code}: {self.description[:30]}>"


class LoincCode(db.Model):
    """Standardized LOINC codes."""
    __tablename__ = "loinc_codes"

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), unique=True, nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<LOINC {self.code}: {self.description[:30]}>"

