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
    pending_billing = Billing.query.filter_by(patient_id=patient_id, status=0).count()

    current_stage = getattr(enc, "stage", None)

    # Priority 1: Tests still running → keep them in the lab/imaging queue
    if pending_labs and pending_imaging:
        # Multiple types pending, pick the most prominent
        return current_stage if current_stage in ("AWAITING_LAB", "AWAITING_IMAGING") else "AWAITING_LAB"
    if pending_labs:
        return "AWAITING_LAB"
    if pending_imaging:
        return "AWAITING_IMAGING"

    # Priority 2: Tests were requested and are now done → Doctor needs to review results
    # (enc.stage was AWAITING_LAB or AWAITING_IMAGING and tests are now complete)
    if current_stage in ("AWAITING_LAB", "AWAITING_IMAGING", "AWAITING_RESULTS"):
        # Tests done → send patient back to Doctor's "Results Review" queue
        return "WAITING_DOCTOR_RESULTS"

    # Priority 3: Prescriptions still pending dispensing
    if pending_rx:
        return "AWAITING_PHARMACY"
    # Priority 4: Billing pending → final billing stage
    if pending_billing > 0:
        return "AWAITING_FINAL_BILLING"
    # No pending work → discharge
    return "DISCHARGED"

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


def cleanup_stale_encounters(max_hours: int = 24) -> int:
    """Finds ACTIVE encounters older than max_hours in initial/unprocessed stages and auto-cancels them."""
    from datetime import datetime, timedelta, timezone

    from departments.models.encounter import Encounter

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_hours)
    stale_encounters = Encounter.query.filter(
        Encounter.status == "ACTIVE",
        Encounter.started_at <= cutoff,
        Encounter.stage.in_(["REGISTERED", "REGISTERED_UNPAID", "WAITING_TRIAGE"]),
    ).all()

    cancelled_count = 0
    for enc in stale_encounters:
        if not has_pending_work(enc.patient_id):
            enc.status = "CANCELLED"
            enc.stage = "CANCELLED"
            enc.ended_at = datetime.now(timezone.utc)
            cancelled_count += 1

    if cancelled_count > 0:
        db.session.commit()
        logger.info(
            "STALE ENCOUNTER CLEANUP: Cancelled %d inactive encounters older than %d hours",
            cancelled_count,
            max_hours,
        )

    return cancelled_count
