from departments.models.controlled_drugs import (
    ControlledDrugBalance,
    ControlledDrugDispense,
    ShiftReconciliation,
)
from departments.models.pharmacy import Drug
from extensions import db


class ControlledDrugError(Exception):
    pass

def dispense_controlled_drug(
    patient_id, drug_id, dose_mg, primary_pharmacist_id,
    second_signatory_id, second_signatory_role, user_id
):
    """
    Enforces dual-signature workflow and updates the running balance ledger.
    """
    drug = db.session.get(Drug, drug_id)
    if not drug or not drug.is_controlled:
        raise ControlledDrugError("Drug is not a controlled substance.")

    if primary_pharmacist_id == second_signatory_id:
        raise ControlledDrugError("Primary and second signatory cannot be the same person.")

    last_balance = ControlledDrugBalance.query.filter_by(drug_id=drug_id).order_by(
        ControlledDrugBalance.recorded_at.desc()
    ).first()

    current_balance = last_balance.balance_after if last_balance else 0.0
    new_balance = current_balance - dose_mg

    if new_balance < 0:
        raise ControlledDrugError(f"CRITICAL: Dispensing {dose_mg}mg would result in negative balance ({new_balance}mg). Physical count verification required.")

    dispense = ControlledDrugDispense(
        patient_id=patient_id, drug_id=drug_id, dose_mg=dose_mg,
        primary_pharmacist_id=primary_pharmacist_id,
        second_signatory_id=second_signatory_id,
        second_signatory_role=second_signatory_role,
        balance_after=new_balance
    )
    db.session.add(dispense)
    db.session.flush()

    ledger_entry = ControlledDrugBalance(
        drug_id=drug_id, transaction_type='DISPENSE',
        quantity_change=-dose_mg, balance_after=new_balance,
        recorded_by=user_id, reference_id=dispense.id
    )
    db.session.add(ledger_entry)

    return dispense

def submit_shift_reconciliation(drug_id, shift_date, shift_type, physical_count, user_id):
    """
    Computes variance between physical count and system ledger.
    """
    last_balance = ControlledDrugBalance.query.filter_by(drug_id=drug_id).order_by(
        ControlledDrugBalance.recorded_at.desc()
    ).first()

    system_balance = last_balance.balance_after if last_balance else 0.0
    variance = physical_count - system_balance

    status = 'COMPLETED'
    if variance != 0:
        status = 'DISCREPANCY'

    recon = ShiftReconciliation(
        drug_id=drug_id, shift_date=shift_date, shift_type=shift_type,
        system_balance=system_balance, physical_count=physical_count,
        variance=variance, reconciled_by=user_id, status=status
    )
    db.session.add(recon)

    if variance != 0:
        raise ControlledDrugError(f"DISCREPANCY DETECTED: Variance of {variance}mg. Shift cannot be closed. Supervisor investigation required.")

    return recon
