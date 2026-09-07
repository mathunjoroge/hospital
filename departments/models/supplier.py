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

    __tablename__ = "suppliers"

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

    purchase_orders = db.relationship(
        "PurchaseOrder", backref="supplier", lazy="dynamic"
    )

    def __repr__(self):
        return f"<Supplier id={self.id} name='{self.name}'>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "contact_email": self.contact_email,
            "phone": self.phone,
            "address": self.address,
            "lead_time_days": self.lead_time_days,
            "is_active": self.is_active,
        }


class PurchaseOrder(db.Model):
    """Header record for a Supplier Purchase Order."""

    __tablename__ = "purchase_orders"

    id = db.Column(db.Integer, primary_key=True)
    po_number = db.Column(db.String(50), nullable=False, unique=True, index=True)
    supplier_id = db.Column(
        db.Integer, db.ForeignKey("suppliers.id"), nullable=False, index=True
    )

    status = db.Column(
        db.String(20),
        default="DRAFT",
        nullable=False,
        index=True,
    )  # DRAFT, ORDERED, RECEIVED, CANCELLED

    created_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    approved_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    received_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    sod_warning = db.Column(db.Boolean, default=False, nullable=False)

    total_cost = db.Column(db.Numeric(10, 2), default=0.0, nullable=False)
    notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    ordered_at = db.Column(db.DateTime(timezone=True), nullable=True)
    received_at = db.Column(db.DateTime(timezone=True), nullable=True)

    items = db.relationship(
        "PurchaseOrderItem", backref="purchase_order", cascade="all, delete-orphan"
    )
    creator = db.relationship(
        "User",
        foreign_keys=[created_by_id],
        backref=db.backref("created_pos", lazy="dynamic"),
    )
    approver = db.relationship(
        "User",
        foreign_keys=[approved_by_id],
        backref=db.backref("approved_pos", lazy="dynamic"),
    )
    receiver = db.relationship(
        "User",
        foreign_keys=[received_by_id],
        backref=db.backref("received_pos", lazy="dynamic"),
    )

    def __repr__(self):
        return f"<PurchaseOrder po_number='{self.po_number}' status='{self.status}' total={self.total_cost}>"

    def recalculate_total(self):
        """Recompute total cost from line items."""
        total = sum(
            float(item.quantity_ordered) * float(item.unit_cost) for item in self.items
        )
        self.total_cost = round(total, 2)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "po_number": self.po_number,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier.name if self.supplier else None,
            "status": self.status,
            "created_by_id": self.created_by_id,
            "approved_by_id": self.approved_by_id,
            "received_by_id": self.received_by_id,
            "sod_warning": self.sod_warning,
            "total_cost": float(self.total_cost),
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ordered_at": self.ordered_at.isoformat() if self.ordered_at else None,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "items": [item.to_dict() for item in self.items],
        }


class PurchaseOrderItem(db.Model):
    """Line item in a Purchase Order referencing a drug or non-pharm item."""

    __tablename__ = "purchase_order_items"

    id = db.Column(db.Integer, primary_key=True)
    po_id = db.Column(
        db.Integer, db.ForeignKey("purchase_orders.id"), nullable=False, index=True
    )
    item_type = db.Column(
        db.String(20), default="DRUG", nullable=False
    )  # DRUG | NON_PHARM
    drug_id = db.Column(
        db.Integer, db.ForeignKey("drugs.id"), nullable=True, index=True
    )
    non_pharm_item_id = db.Column(
        db.Integer, db.ForeignKey("non_pharm_items.id"), nullable=True, index=True
    )
    vote_head_id = db.Column(
        db.Integer, db.ForeignKey("vote_heads.id"), nullable=True, index=True
    )

    quantity_ordered = db.Column(db.Integer, nullable=False)
    unit_cost = db.Column(db.Numeric(10, 2), nullable=False)
    quantity_received = db.Column(db.Integer, default=0, nullable=False)

    drug = db.relationship("Drug")
    non_pharm_item = db.relationship("NonPharmItem")
    vote_head = db.relationship("VoteHead")

    @property
    def item_name(self) -> str:
        if self.item_type == "NON_PHARM" and self.non_pharm_item:
            return self.non_pharm_item.name
        elif self.drug:
            return self.drug.generic_name
        return f"Item #{self.id}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "po_id": self.po_id,
            "item_type": self.item_type,
            "drug_id": self.drug_id,
            "non_pharm_item_id": self.non_pharm_item_id,
            "item_name": self.item_name,
            "quantity_ordered": self.quantity_ordered,
            "unit_cost": float(self.unit_cost),
            "quantity_received": self.quantity_received,
        }


class SupplierReturn(db.Model):
    """Header record for Return to Vendor (RTV)."""

    __tablename__ = "supplier_returns"

    id = db.Column(db.Integer, primary_key=True)
    rtv_number = db.Column(db.String(50), nullable=False, unique=True, index=True)
    supplier_id = db.Column(
        db.Integer, db.ForeignKey("suppliers.id"), nullable=False, index=True
    )
    po_id = db.Column(
        db.Integer, db.ForeignKey("purchase_orders.id"), nullable=True, index=True
    )

    status = db.Column(
        db.String(20), default="DRAFT", nullable=False
    )  # DRAFT, DISPATCHED, CREDIT_ISSUED
    reason = db.Column(
        db.String(255), nullable=True
    )  # Damaged, Expired, Wrong Specification
    total_credit_amount = db.Column(db.Numeric(10, 2), default=0.0, nullable=False)

    created_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    dispatched_at = db.Column(db.DateTime(timezone=True), nullable=True)

    supplier = db.relationship(
        "Supplier", backref=db.backref("returns", lazy="dynamic")
    )
    purchase_order = db.relationship(
        "PurchaseOrder", backref=db.backref("returns", lazy="dynamic")
    )
    creator = db.relationship("User", foreign_keys=[created_by_id])
    items = db.relationship(
        "SupplierReturnItem", backref="supplier_return", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rtv_number": self.rtv_number,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier.name if self.supplier else None,
            "po_id": self.po_id,
            "po_number": self.purchase_order.po_number if self.purchase_order else None,
            "status": self.status,
            "reason": self.reason,
            "total_credit_amount": float(self.total_credit_amount),
            "created_by_id": self.created_by_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "dispatched_at": self.dispatched_at.isoformat()
            if self.dispatched_at
            else None,
            "items": [item.to_dict() for item in self.items],
        }


class SupplierReturnItem(db.Model):
    """Line item in a Return to Vendor record."""

    __tablename__ = "supplier_return_items"

    id = db.Column(db.Integer, primary_key=True)
    supplier_return_id = db.Column(
        db.Integer, db.ForeignKey("supplier_returns.id"), nullable=False, index=True
    )
    item_type = db.Column(
        db.String(20), default="DRUG", nullable=False
    )  # DRUG | NON_PHARM
    drug_id = db.Column(
        db.Integer, db.ForeignKey("drugs.id"), nullable=True, index=True
    )
    non_pharm_item_id = db.Column(
        db.Integer, db.ForeignKey("non_pharm_items.id"), nullable=True, index=True
    )

    batch_number = db.Column(db.String(50), nullable=True)
    quantity_returned = db.Column(db.Integer, nullable=False)
    unit_cost = db.Column(db.Numeric(10, 2), nullable=False)
    reason = db.Column(db.String(255), nullable=True)

    drug = db.relationship("Drug")
    non_pharm_item = db.relationship("NonPharmItem")

    @property
    def item_name(self) -> str:
        if self.item_type == "NON_PHARM" and self.non_pharm_item:
            return self.non_pharm_item.name
        elif self.drug:
            return self.drug.generic_name
        return f"Item #{self.id}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "supplier_return_id": self.supplier_return_id,
            "item_type": self.item_type,
            "drug_id": self.drug_id,
            "non_pharm_item_id": self.non_pharm_item_id,
            "item_name": self.item_name,
            "batch_number": self.batch_number,
            "quantity_returned": self.quantity_returned,
            "unit_cost": float(self.unit_cost),
            "reason": self.reason,
        }


class StockDisposal(db.Model):
    """Header record for Quarantine & Stock Disposal Board (expired/damaged destruction)."""

    __tablename__ = "stock_disposals"

    id = db.Column(db.Integer, primary_key=True)
    disposal_number = db.Column(db.String(50), nullable=False, unique=True, index=True)
    status = db.Column(
        db.String(20), default="DRAFT", nullable=False
    )  # DRAFT, APPROVED, DISPOSED
    reason = db.Column(db.String(255), nullable=False)  # Expired, Damaged, Contaminated
    total_loss_value = db.Column(db.Numeric(10, 2), default=0.0, nullable=False)

    created_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    approved_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    disposed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    creator = db.relationship("User", foreign_keys=[created_by_id])
    approver = db.relationship("User", foreign_keys=[approved_by_id])
    items = db.relationship(
        "StockDisposalItem", backref="disposal", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "disposal_number": self.disposal_number,
            "status": self.status,
            "reason": self.reason,
            "total_loss_value": float(self.total_loss_value),
            "created_by_id": self.created_by_id,
            "approved_by_id": self.approved_by_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "disposed_at": self.disposed_at.isoformat() if self.disposed_at else None,
            "items": [item.to_dict() for item in self.items],
        }


class StockDisposalItem(db.Model):
    """Line item in a Stock Disposal record."""

    __tablename__ = "stock_disposal_items"

    id = db.Column(db.Integer, primary_key=True)
    disposal_id = db.Column(
        db.Integer, db.ForeignKey("stock_disposals.id"), nullable=False, index=True
    )
    item_type = db.Column(
        db.String(20), default="DRUG", nullable=False
    )  # DRUG | NON_PHARM
    drug_id = db.Column(
        db.Integer, db.ForeignKey("drugs.id"), nullable=True, index=True
    )
    non_pharm_item_id = db.Column(
        db.Integer, db.ForeignKey("non_pharm_items.id"), nullable=True, index=True
    )
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"), nullable=True)

    quantity_disposed = db.Column(db.Integer, nullable=False)
    unit_cost = db.Column(db.Numeric(10, 2), nullable=False)
    reason = db.Column(db.String(255), nullable=True)

    drug = db.relationship("Drug")
    non_pharm_item = db.relationship("NonPharmItem")
    batch = db.relationship("Batch")

    @property
    def item_name(self) -> str:
        if self.item_type == "NON_PHARM" and self.non_pharm_item:
            return self.non_pharm_item.name
        elif self.drug:
            return self.drug.generic_name
        return f"Item #{self.id}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "disposal_id": self.disposal_id,
            "item_type": self.item_type,
            "drug_id": self.drug_id,
            "non_pharm_item_id": self.non_pharm_item_id,
            "item_name": self.item_name,
            "batch_id": self.batch_id,
            "batch_number": self.batch.batch_number if self.batch else None,
            "quantity_disposed": self.quantity_disposed,
            "unit_cost": float(self.unit_cost),
            "reason": self.reason,
        }
