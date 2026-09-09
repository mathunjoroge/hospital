"""Phase 2: single authority for closing a visit (Encounter + legacy queue)."""
import logging

from departments.models.billing import Billing, DrugsBill, Invoice, InvoiceStatus
from departments.models.encounter import Encounter
from departments.models.medicine import PrescribedMedicine, RequestedImage, RequestedLab
from extensions import db

logger = logging.getLogger(__name__)


def has_pending_work(patient_id: str) -> bool:
    if RequestedLab.query.filter_by(patient_id=patient_id, status=0).count():
        return True
    if RequestedImage.query.filter_by(patient_id=patient_id, status=0).count():
        return True
    if PrescribedMedicine.query.filter_by(patient_id=patient_id, status="0").count():
        return True
    if Billing.query.filter_by(patient_id=patient_id, status=0).count():
        return True
    if DrugsBill.query.filter_by(patient_id=patient_id, status=0).count():
        return True
    open_inv = Invoice.query.filter(
        Invoice.patient_id == patient_id,
        Invoice.status.in_(
            [InvoiceStatus.DRAFT, InvoiceStatus.ISSUED, InvoiceStatus.PARTIAL]
        ),
        Invoice.balance > 0,
    ).count()
    return bool(open_inv)


def active_encounter(patient_id: str):
    return (
        Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")
        .order_by(Encounter.started_at.desc())
        .first()
    )


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
