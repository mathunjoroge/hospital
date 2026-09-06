"""
departments/models/facility.py
───────────────────────────────
Phase B — Foundational Facility Model

Models:
  - Facility: Entity representing a healthcare facility / hospital / dispensary
"""

from datetime import datetime, timezone

from extensions import db


class Facility(db.Model):
    """Healthcare facility entity representing this installation or an external institution."""
    __tablename__ = 'facilities'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False, index=True)
    facility_code = db.Column(db.String(50), nullable=True, unique=True, index=True)  # KMHFL facility code
    facility_type = db.Column(db.String(50), nullable=False, default='Hospital')
    address = db.Column(db.String(255), nullable=True)
    contact_phone = db.Column(db.String(30), nullable=True)
    contact_email = db.Column(db.String(120), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_self = db.Column(db.Boolean, default=False, nullable=False, index=True)  # Marks the installation's home facility

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        return f"<Facility id={self.id} name='{self.name}' code='{self.facility_code}' is_self={self.is_self}>"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'facility_code': self.facility_code,
            'facility_type': self.facility_type,
            'address': self.address,
            'contact_phone': self.contact_phone,
            'contact_email': self.contact_email,
            'is_active': self.is_active,
            'is_self': self.is_self,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


def get_home_facility(create_if_missing: bool = True) -> Facility:
    """Retrieve or seed the installation's home facility record (is_self=True)."""
    facility = Facility.query.filter_by(is_self=True).first()
    if not facility and create_if_missing:
        facility = Facility(
            name="Main County Referral Hospital",
            facility_code="KMHFL-MAIN-001",
            facility_type="Level 5 County Referral Hospital",
            address="Hospital Road, Central Ward",
            is_active=True,
            is_self=True,
        )
        db.session.add(facility)
        db.session.commit()
    return facility
