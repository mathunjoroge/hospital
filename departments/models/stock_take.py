from datetime import datetime, timezone

from extensions import db


class StockTake(db.Model):
    """
    Represents a Physical Inventory Audit / Cycle Count event.
    Compares physical stock on shelf vs book stock in database.
    Adjustments post STOCK_TAKE_ADJUSTMENT entries to StockMovement.
    """

    __tablename__ = "stock_takes"

    id = db.Column(db.Integer, primary_key=True)
    take_number = db.Column(db.String(50), unique=True, nullable=False, index=True)
    status = db.Column(
        db.String(20), nullable=False, default="COMPLETED"
    )  # COMPLETED / DRAFT
    notes = db.Column(db.String(255), nullable=True)
    total_items_counted = db.Column(db.Integer, nullable=False, default=0)
    total_variance_count = db.Column(db.Integer, nullable=False, default=0)
    created_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    created_by = db.relationship("User", foreign_keys=[created_by_id])
    items = db.relationship(
        "StockTakeItem", backref="stock_take", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "take_number": self.take_number,
            "status": self.status,
            "notes": self.notes,
            "total_items_counted": self.total_items_counted,
            "total_variance_count": self.total_variance_count,
            "created_by": self.created_by.username if self.created_by else "System",
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "items": [item.to_dict() for item in self.items],
        }


class StockTakeItem(db.Model):
    """
    Line item for a physical stock count.
    """

    __tablename__ = "stock_take_items"

    id = db.Column(db.Integer, primary_key=True)
    stock_take_id = db.Column(
        db.Integer, db.ForeignKey("stock_takes.id"), nullable=False, index=True
    )
    item_type = db.Column(
        db.String(20), nullable=False, default="DRUG"
    )  # DRUG / NON_PHARM
    drug_id = db.Column(
        db.Integer, db.ForeignKey("drugs.id"), nullable=True, index=True
    )
    non_pharm_item_id = db.Column(
        db.Integer, db.ForeignKey("non_pharm_items.id"), nullable=True, index=True
    )
    book_quantity = db.Column(db.Integer, nullable=False, default=0)
    physical_quantity = db.Column(db.Integer, nullable=False, default=0)
    variance = db.Column(db.Integer, nullable=False, default=0)  # physical - book
    reason = db.Column(
        db.String(255), nullable=True
    )  # Shrinkage, Damaged, Miscount, Spoilage

    drug = db.relationship("Drug")
    non_pharm_item = db.relationship("NonPharmItem")

    @property
    def item_name(self):
        if self.drug:
            return f"{self.drug.generic_name} ({self.drug.strength or ''})"
        if self.non_pharm_item:
            return self.non_pharm_item.name
        return "Unknown Item"

    def to_dict(self):
        return {
            "id": self.id,
            "stock_take_id": self.stock_take_id,
            "item_type": self.item_type,
            "item_name": self.item_name,
            "drug_id": self.drug_id,
            "non_pharm_item_id": self.non_pharm_item_id,
            "book_quantity": self.book_quantity,
            "physical_quantity": self.physical_quantity,
            "variance": self.variance,
            "reason": self.reason,
        }
