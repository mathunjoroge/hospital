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

    def __repr__(self):
        return f"<ICD10 {self.code}: {self.description[:30]}>"
