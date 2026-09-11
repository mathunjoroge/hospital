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


def is_encounter_open_for_dispensing(encounter_id: int) -> bool:
    """Checks if an encounter is still open for dispensing. Returns False if DISCHARGED/CLOSED."""
    if not encounter_id:
        return True  # Fallback for legacy records
    enc = Encounter.query.get(encounter_id)
    if not enc:
        return True
    terminal_stages = ['DISCHARGED', 'CLOSED']
    return getattr(enc, 'stage', None) not in terminal_stages
