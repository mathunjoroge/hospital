"""
departments/models/budget.py
─────────────────────────────
Phase D — Institutional Budget & Vote-Head Procurement Control

Models:
  - VoteHead: Departmental / category fiscal budget allocation record
"""

from datetime import datetime, timezone

from extensions import db


class VoteHead(db.Model):
    """Institutional vote-head budget allocation entity."""
    __tablename__ = 'vote_heads'

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), nullable=False, unique=True, index=True)  # e.g., 'VOTE-PHARM-2026'
    name = db.Column(db.String(128), nullable=False)                         # e.g., 'Pharmacy Pharmaceuticals'
    department = db.Column(db.String(50), nullable=False, index=True)        # e.g., 'pharmacy', 'laboratory', 'stores'
    financial_year = db.Column(db.String(20), nullable=False, default='2026') # e.g., 'FY2025/2026'

    allocated_amount = db.Column(db.Numeric(12, 2), nullable=False, default=0.0)
    encumbered_amount = db.Column(db.Numeric(12, 2), nullable=False, default=0.0)  # Funds committed in ORDERED POs
    spent_amount = db.Column(db.Numeric(12, 2), nullable=False, default=0.0)       # Funds disbursed upon PO receiving

    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        return f"<VoteHead code='{self.code}' alloc={self.allocated_amount} avail={self.available_amount}>"

    @property
    def available_amount(self) -> float:
        """Net unencumbered available funds."""
        return float(self.allocated_amount) - float(self.encumbered_amount) - float(self.spent_amount)

    def can_encumber(self, amount: float) -> bool:
        """Check if amount fits within available balance."""
        return self.available_amount >= round(float(amount), 2)

    def encumber(self, amount: float):
        """Encumber / reserve funds upon PO ordering."""
        amt = round(float(amount), 2)
        if not self.can_encumber(amt):
            raise ValueError(f"Insufficient funds in vote-head {self.code}. Available: {self.available_amount}, requested: {amt}")
        self.encumbered_amount = float(self.encumbered_amount) + amt

    def unencumber(self, amount: float):
        """Release encumbered funds (e.g. if PO cancelled)."""
        amt = round(float(amount), 2)
        self.encumbered_amount = max(0.0, float(self.encumbered_amount) - amt)

    def record_expenditure(self, amount: float):
        """Convert encumbered amount to spent amount upon PO receipt."""
        amt = round(float(amount), 2)
        self.encumbered_amount = max(0.0, float(self.encumbered_amount) - amt)
        self.spent_amount = float(self.spent_amount) + amt

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'code': self.code,
            'name': self.name,
            'department': self.department,
            'financial_year': self.financial_year,
            'allocated_amount': float(self.allocated_amount),
            'encumbered_amount': float(self.encumbered_amount),
            'spent_amount': float(self.spent_amount),
            'available_amount': self.available_amount,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
