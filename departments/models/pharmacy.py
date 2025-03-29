from extensions import db 
from sqlalchemy.orm import column_property
from datetime import date,datetime
from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey
from sqlalchemy.orm import relationship
import uuid
class DrugCategory(db.Model):
    __tablename__ = 'drugs_category'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

class Drug(db.Model):
    __tablename__ = 'drugs'
    id = db.Column(db.Integer, primary_key=True)
    generic_name = db.Column(db.String(255), nullable=False)
    brand_name = db.Column(db.String(255), nullable=True)
    category_id = db.Column(db.Integer, db.ForeignKey('drugs_category.id'), nullable=False)
    dosage_form = db.Column(db.String(100), nullable=False)
    strength = db.Column(db.String(100), nullable=False)
    manufacturer = db.Column(db.String(255), nullable=True)
    buying_price = db.Column(db.Float, nullable=False)
    selling_price = db.Column(db.Float, nullable=False)
    quantity_in_stock = db.Column(db.Integer, default=0, nullable=False)  # Sum of batch quantities
    reorder_level = db.Column(db.Integer)
    
    # Relationship
    category = db.relationship('DrugCategory', backref=db.backref('drugs', lazy=True))

class Batch(db.Model):
    __tablename__ = 'batches'
    
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False)
    batch_number = db.Column(db.String(100), nullable=True)
    expiry_date = db.Column(db.Date, nullable=True)
    quantity_in_stock = db.Column(db.Integer, nullable=False, default=0)

    # Relationship
    drug = db.relationship('Drug', backref=db.backref('batches', lazy=True))
    
    def __repr__(self):
        return f"<Batch {self.batch_number} - Drug {self.drug.generic_name}>"


#purchases model
class Purchase(db.Model):
    __tablename__ = 'purchases'
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False)  # Links to Drug
    batch_id = db.Column(db.Integer, db.ForeignKey('batches.id'), nullable=False)  # Links to Batch
    purchase_date = db.Column(db.Date, default=datetime.today().date(), nullable=False)
    quantity_purchased = db.Column(db.Integer, nullable=False)
    unit_cost = db.Column(db.Float, nullable=False)
    total_cost = db.Column(db.Float, nullable=False)
    # Relationships
    drug = db.relationship('Drug', backref=db.backref('purchases', lazy=True))
    batch = db.relationship('Batch', backref=db.backref('purchase', uselist=False))
    def __repr__(self):
        return f"<Purchase {self.id} - Drug {self.drug.generic_name}, Batch {self.batch.batch_number}>"
class DispensedDrug(db.Model):
    __tablename__ = 'dispensed_drugs'

    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False)
    batch_id = db.Column(db.Integer, db.ForeignKey('batches.id'), nullable=False)  # Store batch_id instead of batch_no
    patient_id = db.Column(db.String(50), db.ForeignKey('patients.patient_id'), nullable=False)
    prescription_id = db.Column(db.String(36), nullable=False)
    quantity_dispensed = db.Column(db.Integer, nullable=False)
    date_dispensed = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    status = db.Column(db.String(20), default="0")
    receipt_number = db.Column(db.String(255), nullable=True)  # Added for billing integration

    # Relationships
    drug = relationship("Drug", backref="dispensed_drugs", lazy="subquery")
    batch = relationship("Batch", backref="dispensed_batches", lazy="subquery")
    patient = relationship("Patient", backref="dispensed_drugs", lazy="subquery")

    def __repr__(self):
        return f"<DispensedDrug {self.drug_id} - {self.quantity_dispensed} dispensed for Patient {self.patient_id}>"
        
class Expiry(db.Model):
    __tablename__ = 'expiries'
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False)
    batch_number = db.Column(db.String(100))
    quantity_removed = db.Column(db.Integer, nullable=False)
    expiry_date = db.Column(db.Date)
    removal_date = db.Column(db.Date, nullable=False)

class DrugRequest(db.Model):
    __tablename__ = 'drug_requests'
    id = db.Column(db.Integer, primary_key=True)
    request_uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))  # New UUID column
    request_date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='Pending')  # e.g., Pending, Approved, Denied, Fulfilled
    requested_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    items = db.relationship('RequestItem', backref='request', lazy=True)


class RequestItem(db.Model):
    __tablename__ = 'request_items'
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey('drug_requests.id'), nullable=False)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False)
    quantity_requested = db.Column(db.Integer, nullable=False)
    quantity_issued = db.Column(db.Integer, nullable=False, default=0)
    comments = db.Column(db.Text)
    drug = db.relationship('Drug', backref='request_items')     

             