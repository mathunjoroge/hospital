from extensions import db 
from sqlalchemy.orm import column_property
from datetime import date,datetime
class NonPharmCategory(db.Model):
    __tablename__ = 'non_pharm_categories'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)  # e.g., "Non-Pharmaceutical", "Lab Reagent", "Grocery", "Laundry"

    def __repr__(self):
        return f"<NonPharmCategory {self.name}>"
class NonPharmItem(db.Model):
    __tablename__ = 'non_pharm_items'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # e.g., "Syringe", "Rice", "Detergent"
    category_id = db.Column(db.Integer, db.ForeignKey('non_pharm_categories.id'), nullable=False)
    unit = db.Column(db.String(20), nullable=False)   # e.g., "pieces", "kg", "liters"
    unit_cost = db.Column(db.Float, nullable=False, default=0.0)  # Cost per unit
    stock_level = db.Column(db.Integer, nullable=False, default=0)
    in_dispensing = db.Column(db.Integer, nullable=False, default=0) 
    # Relationship
    category = db.relationship('NonPharmCategory', backref='items')
    # Index for faster lookups
    __table_args__ = (db.Index('idx_name_category', 'name', 'category_id'),)

    def __repr__(self):
        return f"<NonPharmItem {self.name} ({self.category.name})>"
    
class OtherOrder(db.Model):
    __tablename__ = 'other_orders'
    id = db.Column(db.Integer, primary_key=True)
    request_date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='Pending')  # e.g., Pending, Fulfilled
    requested_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('non_pharm_items.id'), nullable=False)
    quantity_requested = db.Column(db.Integer, nullable=False)
    quantity_issued = db.Column(db.Integer, nullable=False, default=0)
    notes = db.Column(db.Text)
    # Relationship
    item = db.relationship('NonPharmItem', backref='orders')

    def __repr__(self):
        return f"<OtherOrder {self.id} for {self.item.name}>"    