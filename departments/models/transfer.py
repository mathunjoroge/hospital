"""
departments/models/transfer.py
───────────────────────────────
Phase E — Inter-Facility Stock Transfer Engine

Models:
  - TransferOrder: Header record for inter-facility stock transfer lifecycle
  - TransferOrderItem: Line items referencing transferred drugs or non-pharm commodities
"""

from datetime import datetime, timezone

from extensions import db


class TransferOrder(db.Model):
    """Header record for an inter-facility stock transfer."""
    __tablename__ = 'transfer_orders'

    id = db.Column(db.Integer, primary_key=True)
    transfer_number = db.Column(db.String(50), nullable=False, unique=True, index=True)

    source_facility_id = db.Column(db.Integer, db.ForeignKey('facilities.id'), nullable=False, index=True)
    target_facility_id = db.Column(db.Integer, db.ForeignKey('facilities.id'), nullable=False, index=True)

    status = db.Column(
        db.String(20),
        default='DRAFT',
        nullable=False,
        index=True,
    )  # DRAFT, DISPATCHED, RECEIVED, CANCELLED

    created_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    dispatched_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    received_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    dispatched_at = db.Column(db.DateTime(timezone=True), nullable=True)
    received_at = db.Column(db.DateTime(timezone=True), nullable=True)

    # Relationships
    source_facility = db.relationship('Facility', foreign_keys=[source_facility_id])
    target_facility = db.relationship('Facility', foreign_keys=[target_facility_id])

    creator = db.relationship('User', foreign_keys=[created_by_id])
    dispatcher = db.relationship('User', foreign_keys=[dispatched_by_id])
    receiver = db.relationship('User', foreign_keys=[received_by_id])

    items = db.relationship('TransferOrderItem', backref='transfer_order', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<TransferOrder number='{self.transfer_number}' status='{self.status}'>"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'transfer_number': self.transfer_number,
            'source_facility_id': self.source_facility_id,
            'source_facility_name': self.source_facility.name if self.source_facility else None,
            'target_facility_id': self.target_facility_id,
            'target_facility_name': self.target_facility.name if self.target_facility else None,
            'status': self.status,
            'created_by_id': self.created_by_id,
            'dispatched_by_id': self.dispatched_by_id,
            'received_by_id': self.received_by_id,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'dispatched_at': self.dispatched_at.isoformat() if self.dispatched_at else None,
            'received_at': self.received_at.isoformat() if self.received_at else None,
            'items': [item.to_dict() for item in self.items],
        }


class TransferOrderItem(db.Model):
    """Line item in a Transfer Order referencing a drug or non-pharm item."""
    __tablename__ = 'transfer_order_items'

    id = db.Column(db.Integer, primary_key=True)
    transfer_id = db.Column(db.Integer, db.ForeignKey('transfer_orders.id'), nullable=False, index=True)

    item_type = db.Column(db.String(20), default='DRUG', nullable=False)  # DRUG | NON_PHARM
    drug_id = db.Column(db.Integer, db.ForeignKey('drugs.id'), nullable=True, index=True)
    non_pharm_item_id = db.Column(db.Integer, db.ForeignKey('non_pharm_items.id'), nullable=True, index=True)

    quantity_requested = db.Column(db.Integer, nullable=False)
    quantity_dispatched = db.Column(db.Integer, default=0, nullable=False)
    quantity_received = db.Column(db.Integer, default=0, nullable=False)

    batch_number = db.Column(db.String(50), nullable=True)
    expiry_date = db.Column(db.Date, nullable=True)

    drug = db.relationship('Drug')
    non_pharm_item = db.relationship('NonPharmItem')

    @property
    def item_name(self) -> str:
        if self.item_type == 'NON_PHARM' and self.non_pharm_item:
            return self.non_pharm_item.name
        elif self.drug:
            return self.drug.generic_name
        return f"Item #{self.id}"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'transfer_id': self.transfer_id,
            'item_type': self.item_type,
            'drug_id': self.drug_id,
            'non_pharm_item_id': self.non_pharm_item_id,
            'item_name': self.item_name,
            'quantity_requested': self.quantity_requested,
            'quantity_dispatched': self.quantity_dispatched,
            'quantity_received': self.quantity_received,
            'batch_number': self.batch_number,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
        }
