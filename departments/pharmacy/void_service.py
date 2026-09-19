"""
departments/pharmacy/void_service.py
─────────────────────────────────────
Shared transactional void/reversal logic for dispensed drugs.

Both entry points (dispensing.delete_dispensed_drug and
stock_ops.remove_dispensed) must call this service so the two flows
cannot drift apart.

What it does, in one transaction (caller commits):
  1. Guards against double-void.
  2. Restores batch AND drug-level stock.
  3. Writes a VOID_RETURN row to the immutable stock-movement ledger.
  4. Marks the DispensedDrug VOIDED with reason/actor/timestamp.
     The unified billing event listener (Phase 1b) then auto-generates
     the invoice credit line for the reversal.
  5. Refunds any legacy DrugsBill row that was already PAID (status=1)
     for the same patient/drug/qty, so finance sees the credit.

Returns (dispensed_drug, notes) or raises VoidError.
"""

import logging
from datetime import datetime, timezone

from departments.models.billing import DrugsBill
from departments.models.pharmacy import Batch, DispensedDrug, Drug
from departments.models.stock_movement import record_movement
from extensions import db

logger = logging.getLogger(__name__)


class VoidError(Exception):
    """Raised when a dispense cannot be voided."""


def void_dispensed_drug(dispensed_drug_id: int, void_reason: str, user_id: int):
    """
    Void a dispensed drug entry and reverse stock + billing transactionally.

    The original DispensedDrug record is PRESERVED (status='VOIDED') —
    never physically deleted — for a complete audit trail.
    """
    if not void_reason or not void_reason.strip():
        raise VoidError("A void reason is required to reverse a dispensing record.")

    dispensed_drug = db.session.get(DispensedDrug, dispensed_drug_id)
    if not dispensed_drug:
        raise VoidError(f"Dispensed drug #{dispensed_drug_id} not found.")

    # Guard against double-void
    if str(dispensed_drug.status).upper() == "VOIDED":
        raise VoidError("This dispensing record has already been voided.")

    qty_returned = dispensed_drug.quantity_dispensed

    # 1. Restore batch and drug-level stock (Finding A+B)
    batch = (
        db.session.get(Batch, dispensed_drug.batch_id)
        if dispensed_drug.batch_id
        else None
    )
    drug = db.session.get(Drug, dispensed_drug.drug_id) if dispensed_drug.drug_id else None

    if batch:
        batch.quantity_in_stock += qty_returned
        db.session.add(batch)
    if drug:
        drug.quantity_in_stock += qty_returned
        db.session.add(drug)
        record_movement(
            item_type="DRUG",
            item_id=drug.id,
            movement_type="VOID_RETURN",
            quantity_delta=qty_returned,
            balance_after=drug.quantity_in_stock,
            reference_type="VOID_DISPENSE",
            reference_id=str(dispensed_drug_id),
            user_id=user_id,
            batch_id=dispensed_drug.batch_id,
            notes=f"Void by user {user_id}: {void_reason.strip()}",
        )

    # 2. Void — preserve the original clinical record
    dispensed_drug.status = "VOIDED"
    dispensed_drug.voided_by = user_id
    dispensed_drug.voided_at = datetime.now(timezone.utc)
    dispensed_drug.void_reason = void_reason.strip()
    db.session.add(dispensed_drug)

    # 3. Reverse legacy billing: if an already-PAID DrugsBill row covers this
    # exact dispense (patient + drug + qty), flip it to a negative credit row
    # so the patient's balance reflects the reversal. Unbilled dispenses
    # (receipt_number=None) were never charged, so nothing to reverse.
    if drug and dispensed_drug.receipt_number:
        legacy_bill = (
            DrugsBill.query.filter_by(
                patient_id=dispensed_drug.patient_id,
                drug_id=dispensed_drug.drug_id,
                quantity=qty_returned,
                status=1,
                receipt_number=dispensed_drug.receipt_number,
            )
            .first()
        )
        if legacy_bill:
            credit_bill = DrugsBill(
                patient_id=dispensed_drug.patient_id,
                drug_id=dispensed_drug.drug_id,
                quantity=-qty_returned,
                total_cost=-legacy_bill.total_cost,
                status=1,
                receipt_number=dispensed_drug.receipt_number,
                billed_at=datetime.now(timezone.utc),
                payment_method="VOID_REVERSAL",
                payment_reference=f"Void of dispense #{dispensed_drug_id}: {void_reason.strip()}",
            )
            db.session.add(credit_bill)
            logger.info(
                "Legacy DrugsBill credit created for voided dispense #%s "
                "(patient=%s, amount=-%s)",
                dispensed_drug_id,
                dispensed_drug.patient_id,
                legacy_bill.total_cost,
            )

    logger.info(
        "Dispensed drug VOIDED: id=%s drug=%s patient=%s actor=%s reason=%s",
        dispensed_drug_id,
        dispensed_drug.drug_id,
        dispensed_drug.patient_id,
        user_id,
        void_reason.strip(),
    )
    return dispensed_drug
