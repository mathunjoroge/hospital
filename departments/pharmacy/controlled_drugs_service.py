from departments.models.controlled_drugs import (
    ControlledDrugBalance,
    ControlledDrugDispense,
    ShiftReconciliation,
)
from departments.models.pharmacy import Batch, Drug
from departments.models.stock_movement import record_movement
from extensions import db


class ControlledDrugError(Exception):
    pass


def dispense_controlled_drug(
    patient_id,
    drug_id,
    dose_mg,
    primary_pharmacist_id,
    second_signatory_id,
    second_signatory_role,
    user_id,
    pack_units_qty=None,
    batch_id=None,
):
    """
    Enforces dual-signature workflow and updates the running balance ledger.

    Finding D fix: when pack_units_qty and batch_id are provided, also
    deducts from Batch.quantity_in_stock + Drug.quantity_in_stock and
    writes a StockMovement ledger row — all in the same transaction.
    """
    drug = db.session.get(Drug, drug_id)
    if not drug or not drug.is_controlled:
        raise ControlledDrugError("Drug is not a controlled substance.")

    if primary_pharmacist_id == second_signatory_id:
        raise ControlledDrugError(
            "Primary and second signatory cannot be the same person."
        )

    last_balance = (
        ControlledDrugBalance.query.filter_by(drug_id=drug_id)
        .order_by(ControlledDrugBalance.recorded_at.desc())
        .first()
    )

    current_balance = last_balance.balance_after if last_balance else 0.0
    new_balance = current_balance - dose_mg

    if new_balance < 0:
        raise ControlledDrugError(
            f"CRITICAL: Dispensing {dose_mg}mg would result in "
            f"negative balance ({new_balance}mg). "
            f"Physical count verification required."
        )

    # --- Finding D: validate and deduct general inventory stock ---
    batch = None
    if pack_units_qty is not None and pack_units_qty > 0:
        if batch_id is None:
            raise ControlledDrugError(
                "batch_id is required when pack_units_qty is provided."
            )
        batch = db.session.get(Batch, batch_id)
        if not batch or batch.drug_id != drug_id:
            raise ControlledDrugError(
                f"Batch {batch_id} not found or does not belong "
                f"to drug {drug_id}."
            )
        if batch.quantity_in_stock < pack_units_qty:
            raise ControlledDrugError(
                f"Insufficient batch stock: "
                f"{batch.quantity_in_stock} units available, "
                f"{pack_units_qty} requested."
            )

    dispense = ControlledDrugDispense(
        patient_id=patient_id,
        drug_id=drug_id,
        dose_mg=dose_mg,
        pack_units_qty=pack_units_qty,
        batch_id=batch_id,
        primary_pharmacist_id=primary_pharmacist_id,
        second_signatory_id=second_signatory_id,
        second_signatory_role=second_signatory_role,
        balance_after=new_balance,
    )
    db.session.add(dispense)
    db.session.flush()

    ledger_entry = ControlledDrugBalance(
        drug_id=drug_id,
        transaction_type="DISPENSE",
        quantity_change=-dose_mg,
        balance_after=new_balance,
        recorded_by=user_id,
        reference_id=dispense.id,
    )
    db.session.add(ledger_entry)

    # --- Finding D: sync general inventory + write StockMovement ---
    if pack_units_qty is not None and pack_units_qty > 0 and batch:
        batch.quantity_in_stock -= pack_units_qty
        drug.quantity_in_stock = max(
            0, drug.quantity_in_stock - pack_units_qty
        )
        db.session.add(batch)
        db.session.add(drug)

        record_movement(
            item_type="DRUG",
            item_id=drug_id,
            movement_type="DISPENSED",
            quantity_delta=-pack_units_qty,
            balance_after=drug.quantity_in_stock,
            reference_type="CONTROLLED_DISPENSE",
            reference_id=str(dispense.id),
            user_id=user_id,
            batch_id=batch.id,
            notes=(
                f"Controlled drug dispense: {dose_mg}mg "
                f"= {pack_units_qty} pack units to patient {patient_id}"
            ),
        )

    return dispense


def submit_shift_reconciliation(
    drug_id, shift_date, shift_type, physical_count, user_id
):
    """
    Computes variance between physical count and system ledger.
    """
    last_balance = (
        ControlledDrugBalance.query.filter_by(drug_id=drug_id)
        .order_by(ControlledDrugBalance.recorded_at.desc())
        .first()
    )

    system_balance = last_balance.balance_after if last_balance else 0.0
    variance = physical_count - system_balance

    status = "COMPLETED"
    if variance != 0:
        status = "DISCREPANCY"

    recon = ShiftReconciliation(
        drug_id=drug_id,
        shift_date=shift_date,
        shift_type=shift_type,
        system_balance=system_balance,
        physical_count=physical_count,
        variance=variance,
        reconciled_by=user_id,
        status=status,
    )
    db.session.add(recon)

    if variance != 0:
        raise ControlledDrugError(
            f"DISCREPANCY DETECTED: Variance of {variance}mg. "
            f"Shift cannot be closed. "
            f"Supervisor investigation required."
        )

    return recon
