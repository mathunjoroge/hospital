"""
departments/models/stock_movement.py
──────────────────────────────────────
Phase C.4 — Append-only Stock Movement Ledger (Bin Card)

This model is the digital equivalent of a physical bin card or store ledger.
Every stock-changing event (receive, issue, transfer, adjust, write-off)
appends one row here. The mutable quantity_in_stock on Drug/NonPharmItem
remains as a fast-read cache but must always be derivable by summing movement
rows.
"""

from datetime import datetime, timezone

from extensions import db


class StockMovement(db.Model):
    """Append-only ledger row for every stock-changing event."""

    __tablename__ = "stock_movements"

    id = db.Column(db.Integer, primary_key=True)

    # Which item?
    item_type = db.Column(
        db.String(20), nullable=False, index=True
    )  # 'DRUG' | 'NON_PHARM'
    item_id = db.Column(db.Integer, nullable=False, index=True)

    # Optional reference to a specific Batch (for FEFO-relevant movements)
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"), nullable=True)

    # Movement kind
    movement_type = db.Column(
        db.String(30), nullable=False, index=True
    )  # RECEIVED | ISSUED | TRANSFER_OUT | TRANSFER_IN | ADJUSTMENT | WRITEOFF

    # The actual change: positive for inbound, negative for outbound
    quantity_delta = db.Column(db.Integer, nullable=False)

    # Running balance AFTER this movement was applied (denormalised for O(1) reads)
    balance_after = db.Column(db.Integer, nullable=False)

    # What caused this movement?
    reference_type = db.Column(
        db.String(50), nullable=True
    )  # PURCHASE_ORDER | DRUG_REQUEST | TRANSFER | MANUAL
    reference_id = db.Column(
        db.String(50), nullable=True
    )  # PO number, request id, transfer id

    # Who performed it, and from which facility?
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    facility_id = db.Column(db.Integer, db.ForeignKey("facilities.id"), nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    notes = db.Column(db.Text, nullable=True)

    # Relationships
    batch = db.relationship("Batch", backref=db.backref("movements", lazy="dynamic"))
    user = db.relationship(
        "User", backref=db.backref("stock_movements", lazy="dynamic")
    )

    __table_args__ = (
        db.Index("idx_sm_item", "item_type", "item_id"),
        db.Index("idx_sm_ref", "reference_type", "reference_id"),
    )

    def __repr__(self):
        sign = "+" if self.quantity_delta >= 0 else ""
        return (
            f"<StockMovement id={self.id} {self.item_type}:{self.item_id} "
            f"{self.movement_type} {sign}{self.quantity_delta} bal={self.balance_after}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "item_type": self.item_type,
            "item_id": self.item_id,
            "batch_id": self.batch_id,
            "movement_type": self.movement_type,
            "quantity_delta": self.quantity_delta,
            "balance_after": self.balance_after,
            "reference_type": self.reference_type,
            "reference_id": self.reference_id,
            "user_id": self.user_id,
            "facility_id": self.facility_id,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def record_movement(
    item_type: str,
    item_id: int,
    movement_type: str,
    quantity_delta: int,
    balance_after: int,
    reference_type: str | None = None,
    reference_id: str | None = None,
    user_id: int | None = None,
    facility_id: int | None = None,
    batch_id: int | None = None,
    notes: str | None = None,
) -> StockMovement:
    """
    Helper: create and persist a StockMovement row.
    Must be called inside an active DB session (before commit).
    """
    movement = StockMovement(
        item_type=item_type,
        item_id=item_id,
        batch_id=batch_id,
        movement_type=movement_type,
        quantity_delta=quantity_delta,
        balance_after=balance_after,
        reference_type=reference_type,
        reference_id=reference_id,
        user_id=user_id,
        facility_id=facility_id,
        notes=notes,
    )
    db.session.add(movement)
    return movement


def reconcile_stock_balance(item_type: str, item_id: int) -> dict:
    """
    Sum all StockMovement rows for this item and compare against the
    cached quantity_in_stock / stock_level field on the item itself.

    Returns:
        {
            'item_type': ...,
            'item_id': ...,
            'ledger_balance': int,   # sum of all movement deltas
            'cached_balance': int,   # fast-read field on Drug/NonPharmItem
            'match': bool,           # True if they agree
            'variance': int,         # ledger_balance - cached_balance
        }
    """
    from sqlalchemy import func

    ledger_balance = (
        db.session.query(func.coalesce(func.sum(StockMovement.quantity_delta), 0))
        .filter_by(item_type=item_type, item_id=item_id)
        .scalar()
    )

    cached_balance = 0
    if item_type == "DRUG":
        from departments.models.pharmacy import Drug

        drug = db.session.get(Drug, item_id)
        if drug:
            cached_balance = drug.quantity_in_stock
    elif item_type == "NON_PHARM":
        from departments.models.stores import NonPharmItem

        item = db.session.get(NonPharmItem, item_id)
        if item:
            cached_balance = item.stock_level

    variance = int(ledger_balance) - cached_balance
    return {
        "item_type": item_type,
        "item_id": item_id,
        "ledger_balance": int(ledger_balance),
        "cached_balance": cached_balance,
        "match": variance == 0,
        "variance": variance,
    }
