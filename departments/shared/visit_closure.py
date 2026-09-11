"""Phase 2: single authority for closing a visit (Encounter + legacy queue)."""
import logging

from sqlalchemy import or_

from departments.models.billing import Billing, DrugsBill, Invoice
from departments.models.medicine import PrescribedMedicine, RequestedImage, RequestedLab
from departments.shared.encounter_utils import active_encounter
from extensions import db

logger = logging.getLogger(__name__)

def _pending_count(model, patient_id: str, enc) -> int:
    q = model.query.filter(model.patient_id == patient_id, model.status == 0)
    if enc is not None:
        q = q.filter(or_(model.encounter_id == enc.id, model.encounter_id.is_(None)))
    return q.count()

def has_pending_work(patient_id: str) -> bool:
    enc = active_encounter(patient_id)
    if (
        _pending_count(RequestedLab, patient_id, enc)
        or _pending_count(RequestedImage, patient_id, enc)
        or _pending_count(PrescribedMedicine, patient_id, enc)
    ):
        return True

    if Billing.query.filter_by(patient_id=patient_id, status=0).count():
        return True
    if DrugsBill.query.filter_by(patient_id=patient_id, status=0).count():
        return True
    return bool(Invoice.query.filter_by(patient_id=patient_id, status=0).count())


def maybe_close_encounter(patient_id: str) -> bool:
    """Close the visit only when all clinical work and all bills are settled."""
    enc = active_encounter(patient_id)
    if not enc or has_pending_work(patient_id):
        return False
    enc.close()
    db.session.commit()
    logger.info("VISIT CLOSED: encounter %s for patient %s", enc.id, patient_id)
    return True


def force_close_visit(patient_id: str, reason: str = "") -> bool:
    """Staff-initiated discharge regardless of pending work (AMA, transfer...)."""
    enc = active_encounter(patient_id)
    if not enc:
        return False
    enc.close()
    db.session.commit()
    logger.info("VISIT FORCE-CLOSED (%s): encounter %s", reason or "manual", enc.id)
    return True


def determine_next_stage(patient_id: str, enc) -> str:
    pending_labs = _pending_count(RequestedLab, patient_id, enc)
    pending_imaging = _pending_count(RequestedImage, patient_id, enc)
    pending_rx = _pending_count(PrescribedMedicine, patient_id, enc)

    if pending_labs or pending_imaging:
        return "AWAITING_RESULTS"
    elif pending_rx:
        return "AWAITING_PHARMACY"
    else:
        return "AWAITING_BILLING"

def advance_after_completion(patient_id: str) -> str | None:
    enc = active_encounter(patient_id)
    if not enc:
        return None

    new_stage = determine_next_stage(patient_id, enc)
    current = getattr(enc, "stage", None)
    if current != new_stage:
        enc.set_stage(new_stage)
        db.session.commit()
        logger.info(f"Encounter {enc.id} advanced to {new_stage}")
        return new_stage
    return None
