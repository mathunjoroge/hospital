"""
departments/models/supplier.py
───────────────────────────────
Phase F — Automated Pharmacy Inventory & Supplier Purchase Orders

Models:
  - Supplier: Vendor directory with contact details and lead time
  - PurchaseOrder: Header for purchase order lifecycle (DRAFT -> ORDERED -> RECEIVED)
  - PurchaseOrderItem: Line items referencing pharmacy drugs and ordered quantities
"""

from datetime import datetime, timezone

from extensions import db


class Supplier(db.Model):
    """Pharmacy pharmaceutical supplier / vendor entity."""
    __tablename__ = 'suppliers'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False, unique=True, index=True)
    contact_email = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(30), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    lead_time_days = db.Column(db.Integer, default=3, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    purchase_orders = db.relationship('PurchaseOrder', backref='supplier', lazy='dynamic')

    def __repr__(self):
        return f"<Supplier id={self.id} name='{self.name}'>"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'contact_email': self.contact_email,
            'phone': self.phone,
            'address': self.address,
            'lead_time_days': self.lead_time_days,
            'is_active': self.is_active,
        }


class PurchaseOrder(db.Model):
    """Header record for a Supplier Purchase Order."""
    __tablename__ = 'purchase_orders'

    id = db.Column(db.Integer, primary_key=True)
    po_number = db.Column(db.String(50), nullable=False, unique=True, index=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'), nullable=False, index=True)

    status = db.Column(
        db.String(20),
        default='DRAFT',
        nullable=False,
        index=True,
    )  # DRAFT, ORDERED, RECEIVED, CANCELLED

    total_cost = db.Column(db.Numeric(10, 2), default=0.0, nullable=False)
    notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    ordered_at = db.Column(db.DateTime(timezone=True), nullable=True)
    received_at = db.Column(db.DateTime(timezone=True), nullable=True)

    items = db.relationship('PurchaseOrderItem', backref='purchase_order', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<PurchaseOrder po_number='{self.po_number}' status='{self.status}' total={self.total_cost}>"

    def recalculate_total(self):
        """Recompute total cost from line items."""
        total = sum(float(item.quantity_ordered) * float(item.unit_cost) for item in self.items)
        self.total_cost = round(total, 2)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'po_number': self.po_number,
            'supplier_id': self.supplier_id,
            'supplier_name': self.supplier.name if self.supplier else None,
            'status': self.status,
            'total_cost': float(self.total_cost),
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'ordered_at': self.ordered_at.isoformat() if self.ordered_at else None,
            'received_at': self.received_at.isoformat() if self.received_at else None,
            'items': [item.to_dict() for item in self.items],
        }


class PurchaseOrderItem(db.Model):
    """Line item in a Purchase Order referencing a specific drug."""
    __tablename__ = 'purchase_order_items'

    id = db.Column(db.Integer, primary_key=True)
    po_id = db.Column(db.Integer, db.ForeignKey('purchase_orders.id'), nullable=False, index=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=False, index=True)

    quantity_ordered = db.Column(db.Integer, nullable=False)
    unit_cost = db.Column(db.Numeric(10, 2), nullable=False)
    quantity_received = db.Column(db.Integer, default=0, nullable=False)

    drug = db.relationship('Drug')

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'po_id': self.po_id,
            'drug_id': self.drug_id,
            'drug_name': self.drug.generic_name if self.drug else None,
            'quantity_ordered': self.quantity_ordered,
            'unit_cost': float(self.unit_cost),
            'quantity_received': self.quantity_received,
        }
