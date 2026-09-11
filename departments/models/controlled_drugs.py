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
