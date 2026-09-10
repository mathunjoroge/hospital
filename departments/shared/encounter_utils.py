from departments.models.encounter import Encounter

def active_encounter(patient_id: str):
    """
    Returns the active (non-discharged) encounter for a given patient.
    Falls back to None if no active encounter exists.
    """
    enc = Encounter.query.filter_by(patient_id=patient_id).order_by(Encounter.id.desc()).first()
    if enc and getattr(enc, 'stage', None) != 'DISCHARGED':
        return enc
    return None
