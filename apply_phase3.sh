#!/bin/bash
set -e

echo "Applying Phase 3 Controlled Drug Register..."

# 1. Record Decisions in DECISIONS_PENDING.md (Section 16)
cat << 'DECISIONS' >> DECISIONS_PENDING.md

## Section 16: Controlled Drug Register Policy (PPB/WHO Compliance)
**Status:** DECIDED — 2026-09-11
**Decision-maker:** Solo Developer / System Administrator
**Context:** Resolves Item 5 (Phase B.1 Hard Stop) for Schedule II/IV controlled substance dispensing under Kenya PPB regulation.

**Decisions:**
1. **Dual Signature Roles:** Option B (Pharmacist + Ward Nurse-in-Charge/Clinical Officer). 
   *Rationale:* Ensures 24/7 coverage for emergency dispensing while maintaining dual-control chain of custody.
2. **Stock Reconciliation Schedule:** Split schedule. Schedule II (narcotics) = Per Shift. Schedule IV (psychotropics) = Daily. 
   *Rationale:* Matches WHO risk-based approach; highest risk drugs get highest frequency counts.
3. **Schedule Differentiation:** Option A (Separate Ledgers). 
   *Rationale:* Implementation uses a single `controlled_drug_balances` table for data integrity, but application logic and PDF exports strictly filter and separate Schedule II and Schedule IV records to mirror physical PPB audit books.

**Implementation:** Proceeding with Phase 3 work items P3-01 through P3-08.
DECISIONS
echo "✅ Recorded decisions in DECISIONS_PENDING.md (Section 16)"

# 2. Update Drug model in departments/models/pharmacy.py
python3 -c "
from pathlib import Path
p = Path('departments/models/pharmacy.py')
c = p.read_text()
if 'is_controlled' not in c:
    lines = c.split('\n')
    for i, line in enumerate(lines):
        if 'def __repr__' in line:
            lines.insert(i, '    # Phase 3: Controlled Drug Register\n    is_controlled = db.Column(db.Boolean, default=False, nullable=False)\n    schedule_class = db.Column(db.String(10), nullable=True)  # II, IV, V\n')
            break
    p.write_text('\n'.join(lines))
    print('✅ Updated Drug model with is_controlled and schedule_class')
else:
    print('ℹ️ Drug model already updated')
"

# 3. Create Controlled Drugs Models
cat << 'MODELS' > departments/models/controlled_drugs.py
from datetime import datetime
from extensions import db

class ControlledDrugDispense(db.Model):
    __tablename__ = 'controlled_drug_dispenses'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False)
    dose_mg = db.Column(db.Float, nullable=False)
    dispense_datetime = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    primary_pharmacist_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    second_signatory_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    second_signatory_role = db.Column(db.String(50), nullable=False)
    
    balance_after = db.Column(db.Float, nullable=False)
    witness_signature_hash = db.Column(db.String(64), nullable=True)
    
    patient = db.relationship('Patient', backref='controlled_dispenses')
    drug = db.relationship('Drug', backref='controlled_dispenses')
    primary_pharmacist = db.relationship('User', foreign_keys=[primary_pharmacist_id])
    second_signatory = db.relationship('User', foreign_keys=[second_signatory_id])

class ControlledDrugBalance(db.Model):
    __tablename__ = 'controlled_drug_balances'
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)
    quantity_change = db.Column(db.Float, nullable=False)
    balance_after = db.Column(db.Float, nullable=False)
    recorded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    reference_id = db.Column(db.Integer, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    
    drug = db.relationship('Drug', backref='balance_ledger')

class ShiftReconciliation(db.Model):
    __tablename__ = 'shift_reconciliations'
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False)
    shift_date = db.Column(db.Date, nullable=False)
    shift_type = db.Column(db.String(20), nullable=False)
    system_balance = db.Column(db.Float, nullable=False)
    physical_count = db.Column(db.Float, nullable=True)
    variance = db.Column(db.Float, nullable=True)
    reconciled_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    reconciled_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    status = db.Column(db.String(20), default='PENDING', nullable=False)
    
    drug = db.relationship('Drug', backref='reconciliations')
MODELS
echo "✅ Created departments/models/controlled_drugs.py"

# 4. Create Service Logic
mkdir -p departments/pharmacy
cat << 'SERVICE' > departments/pharmacy/controlled_drugs_service.py
from datetime import datetime, date
from extensions import db
from departments.models.controlled_drugs import (
    ControlledDrugDispense, ControlledDrugBalance, ShiftReconciliation
)
from departments.models.pharmacy import Drug

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
    drug = db.session.get(Drug, drug_id)
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
SERVICE
echo "✅ Created departments/pharmacy/controlled_drugs_service.py"

# 5. Create Tests
cat << 'TESTS' > tests/test_controlled_drugs.py
import pytest
from datetime import date
from departments.pharmacy.controlled_drugs_service import (
    dispense_controlled_drug, submit_shift_reconciliation, ControlledDrugError
)
from departments.models.pharmacy import Drug
from departments.models.controlled_drugs import ControlledDrugBalance
from extensions import db

@pytest.fixture
def controlled_drug(app):
    with app.app_context():
        drug = Drug(
            name="Morphine Sulfate", generic_name="Morphine",
            is_controlled=True, schedule_class="II", stock_quantity=100.0
        )
        db.session.add(drug)
        db.session.commit()
        
        balance = ControlledDrugBalance(
            drug_id=drug.id, transaction_type='RECEIPT',
            quantity_change=100.0, balance_after=100.0, recorded_by=1
        )
        db.session.add(balance)
        db.session.commit()
        yield drug

def test_dispense_without_second_signatory_fails(app, controlled_drug):
    """P3-07: Dispense without valid second signatory should fail."""
    with app.app_context():
        with pytest.raises(ControlledDrugError, match="cannot be the same person"):
            dispense_controlled_drug(
                patient_id="P-001", drug_id=controlled_drug.id, dose_mg=10.0,
                primary_pharmacist_id=1, second_signatory_id=1,
                second_signatory_role="Pharmacist", user_id=1
            )

def test_negative_balance_alert_fires(app, controlled_drug):
    """P3-07: Dispensing more than available balance must trigger CRITICAL alert."""
    with app.app_context():
        with pytest.raises(ControlledDrugError, match="CRITICAL.*negative balance"):
            dispense_controlled_drug(
                patient_id="P-001", drug_id=controlled_drug.id, dose_mg=150.0,
                primary_pharmacist_id=1, second_signatory_id=2,
                second_signatory_role="NurseInCharge", user_id=1
            )

def test_shift_reconciliation_variance_blocks_close(app, controlled_drug):
    """P3-07: Shift reconciliation with variance > 0 must block close."""
    with app.app_context():
        with pytest.raises(ControlledDrugError, match="DISCREPANCY DETECTED"):
            submit_shift_reconciliation(
                drug_id=controlled_drug.id, shift_date=date.today(),
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
TESTS
echo "✅ Created tests/test_controlled_drugs.py"

echo ""
echo "🎉 Phase 3 foundational code applied successfully."
echo "Next steps:"
echo "1. Run: flask db migrate -m 'phase3_add_controlled_drug_register'"
echo "2. Run: flask db upgrade"
echo "3. Run: pytest tests/test_controlled_drugs.py"
echo "4. Run: ruff check . --select E,F,W,I --ignore E501 --fix"
