from datetime import datetime, timezone

import pytest

from departments.models.controlled_drugs import ControlledDrugBalance
from departments.models.pharmacy import Drug
from departments.pharmacy.controlled_drugs_service import (
    ControlledDrugError,
    dispense_controlled_drug,
    submit_shift_reconciliation,
)
from extensions import db


@pytest.fixture
def controlled_drug(app):
    """Create a valid controlled drug with all required FK and non-null fields."""
    from departments.models.pharmacy import DrugCategory

    with app.app_context():
        # DrugCategory is required because drugs.category_id is NOT NULL in the DB
        category = DrugCategory.query.first()
        if not category:
            category = DrugCategory(name="Controlled Substances")
            db.session.add(category)
            db.session.flush()

        drug = Drug(
            generic_name="Morphine Sulfate",
            brand_name="MS Contin",
            category_id=category.id,
            dosage_form="Tablet",
            strength="10mg",
            buying_price=50.0,
            selling_price=100.0,
            quantity_in_stock=100,
            is_controlled=True,
            schedule_class="II",
        )
        db.session.add(drug)
        db.session.commit()

        # Seed initial balance in the controlled drug ledger
        balance = ControlledDrugBalance(
            drug_id=drug.id,
            transaction_type='RECEIPT',
            quantity_change=100.0,
            balance_after=100.0,
            recorded_by=1,
        )
        db.session.add(balance)
        db.session.commit()
        yield drug

def test_dispense_without_second_signatory_fails(app, controlled_drug):
    """P3-07: Dispense without valid second signatory should fail."""
    with app.app_context():  # noqa: SIM117
        with pytest.raises(ControlledDrugError, match="cannot be the same person"):
            dispense_controlled_drug(
                patient_id="P-001", drug_id=controlled_drug.id, dose_mg=10.0,
                primary_pharmacist_id=1, second_signatory_id=1,
                second_signatory_role="Pharmacist", user_id=1
            )

def test_negative_balance_alert_fires(app, controlled_drug):
    """P3-07: Dispensing more than available balance must trigger CRITICAL alert."""
    with app.app_context():  # noqa: SIM117
        with pytest.raises(ControlledDrugError, match="CRITICAL.*negative balance"):
            dispense_controlled_drug(
                patient_id="P-001", drug_id=controlled_drug.id, dose_mg=150.0,
                primary_pharmacist_id=1, second_signatory_id=2,
                second_signatory_role="NurseInCharge", user_id=1
            )

def test_shift_reconciliation_variance_blocks_close(app, controlled_drug):
    """P3-07: Shift reconciliation with variance > 0 must block close."""
    with app.app_context():  # noqa: SIM117
        with pytest.raises(ControlledDrugError, match="DISCREPANCY DETECTED"):
            submit_shift_reconciliation(
                drug_id=controlled_drug.id, shift_date=datetime.now(timezone.utc).date(),
                shift_type="NIGHT", physical_count=95.0,
                user_id=1
            )

def test_successful_dispense_updates_ledger(app, controlled_drug):
    """Valid dispense should update balance correctly."""
    with app.app_context():
        dispense = dispense_controlled_drug(
            patient_id="P-001", drug_id=controlled_drug.id, dose_mg=10.0,
            primary_pharmacist_id=1, second_signatory_id=2,
            second_signatory_role="NurseInCharge", user_id=1
        )
        db.session.commit()

        assert dispense.balance_after == 90.0
        last_balance = ControlledDrugBalance.query.filter_by(drug_id=controlled_drug.id).order_by(
            ControlledDrugBalance.recorded_at.desc()
        ).first()
        assert last_balance.balance_after == 90.0
        assert last_balance.transaction_type == 'DISPENSE'
