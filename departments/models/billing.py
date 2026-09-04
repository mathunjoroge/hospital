from extensions import db  # Use absolute import for db
from departments.models.pharmacy import Drug
from datetime import datetime

class ChargeCategory(db.Model):
    """Categories of charges (Consultation, Lab, Surgery, etc.)."""
    __tablename__ = 'charge_categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

class Charge(db.Model):
    """Stores hospital charges."""
    __tablename__ = 'charges'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('charge_categories.id'), nullable=False)
    cost = db.Column(db.Numeric(10, 2), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)

    category = db.relationship('ChargeCategory', backref=db.backref('charges', lazy=True))

class Billing(db.Model):
    """Links patients to charges, tracks payment status."""
    __tablename__ = 'billing'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    charge_id = db.Column(db.Integer, db.ForeignKey('charges.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    total_cost = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(db.Integer, nullable=False, default=0)  # 0 for Pending, 1 for Paid
    billed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    receipt_number = db.Column(db.String(50), nullable=True)  # Receipt number for paid bills

    patient = db.relationship('Patient', backref=db.backref('bills', lazy=True))
    charge = db.relationship('Charge', backref=db.backref('billings', lazy=True))

    def calculate_total(self):
        """Calculate total cost based on charge."""
        self.total_cost = self.quantity * self.charge.cost

class DrugsBill(db.Model):
    """Links patients to drugs and tracks payment status."""
    __tablename__ = 'drugs_bill'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=True)  # Nullable for summary drug bills
    quantity = db.Column(db.Integer, nullable=False, default=1)
    total_cost = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(db.Integer, nullable=False, default=0)  # 0 for Pending, 1 for Paid
    billed_at = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)
    receipt_number = db.Column(db.String(50), nullable=True)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    # Relationships
    patient = db.relationship('Patient', backref=db.backref('drug_bills', lazy=True))
    drug = db.relationship('Drug', backref=db.backref('bills', lazy=True))

    def calculate_total(self):
        """Calculate total cost based on drug selling price."""
        if self.drug:
            self.total_cost = self.quantity * self.drug.selling_price
        else:
            self.total_cost = 0


class PaidBill(db.Model):
    """Stores details of paid bills."""
    __tablename__ = 'paid_bills'

    id = db.Column(db.Integer, primary_key=True)
    receipt_number = db.Column(db.String(50), unique=True, nullable=False)  # Unique receipt number
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)  # Link to Patient
    grand_total = db.Column(db.Numeric(10, 2), nullable=False)  # Total cost of all unpaid bills
    amount_paid = db.Column(db.Numeric(10, 2), nullable=False)  # Amount paid in this transaction
    balance = db.Column(db.Numeric(10, 2), nullable=False)  # Remaining balance after payment
    paid_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # Timestamp of payment
    payment_method = db.Column(db.String(50), nullable=True)  # Payment method (e.g., Cash, M-Pesa)

    # Relationships
    patient = db.relationship('Patient', backref=db.backref('paid_bills', lazy=True))

    @staticmethod
    def generate_receipt_number():
        """Generate a unique receipt number."""
        now = datetime.utcnow()
        prefix = f"REC-{now.strftime('%Y%m%d')}"
        last_paid_bill = PaidBill.query.filter(PaidBill.receipt_number.like(f"{prefix}%")).order_by(PaidBill.id.desc()).first()
        if last_paid_bill:
            try:
                last_number = int(last_paid_bill.receipt_number.split('-')[-1])
            except ValueError:
                last_number = 0
        else:
            last_number = 0
        new_number = last_number + 1
        return f"{prefix}-{new_number:04d}"

class WardBill(db.Model):
    __tablename__ = 'ward_bills'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship('Patient', backref=db.backref('ward_bills', lazy=True))

    def __repr__(self):
        return f"<WardBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"

# Lab Bills (for requested_labs)
class LabBill(db.Model):
    __tablename__ = 'lab_bills'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship('Patient', backref=db.backref('lab_bills', lazy=True))

    def __repr__(self):
        return f"<LabBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"

# Clinic Bills (for clinic_bookings)
class ClinicBill(db.Model):
    __tablename__ = 'clinic_bills'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship('Patient', backref=db.backref('clinic_bills', lazy=True))

    def __repr__(self):
        return f"<ClinicBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"

# Theatre Bills (for theatre_list)
class TheatreBill(db.Model):
    __tablename__ = 'theatre_bills'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship('Patient', backref=db.backref('theatre_bills', lazy=True))

    def __repr__(self):
        return f"<TheatreBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"

# Imaging Bills (for requested_images)
class ImagingBill(db.Model):
    __tablename__ = 'imaging_bills'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship('Patient', backref=db.backref('imaging_bills', lazy=True))

    def __repr__(self):
        return f"<ImagingBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"


 
        
