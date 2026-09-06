from datetime import date, timedelta

import pytest

from app import app
from departments.models.hr import StaffCredential
from extensions import db


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client

def test_staff_credential_expiry_sorting_and_alerts(client):
    """Test staff credentials ordering soonest-first and filtering expiring credentials."""
    with app.app_context():
        today = date.today()

        c1 = StaffCredential(
            staff_name="Dr. Alice Smith",
            credential_type="KMPDC",
            credential_number="A-1001",
            expiry_date=today + timedelta(days=90)
        )
        c2 = StaffCredential(
            staff_name="Nurse Bob Jones",
            credential_type="NCK",
            credential_number="N-2002",
            expiry_date=today + timedelta(days=10)
        )
        c3 = StaffCredential(
            staff_name="Pharm. Carol Danvers",
            credential_type="PPB",
            credential_number="P-3003",
            expiry_date=today + timedelta(days=25)
        )
        db.session.add_all([c1, c2, c3])
        db.session.commit()

        # Query soonest-first
        all_creds = StaffCredential.query.order_by(StaffCredential.expiry_date.asc()).all()
        assert len(all_creds) >= 3
        # Nurse Bob Jones (10 days) should come before Carol (25 days) and Alice (90 days)
        sorted_names = [c.staff_name for c in all_creds if c.staff_name in ["Dr. Alice Smith", "Nurse Bob Jones", "Pharm. Carol Danvers"]]
        assert sorted_names == ["Nurse Bob Jones", "Pharm. Carol Danvers", "Dr. Alice Smith"]

        # Alert filter (credentials expiring within 30 days)
        cutoff_date = today + timedelta(days=30)
        expiring_creds = StaffCredential.query.filter(StaffCredential.expiry_date <= cutoff_date).all()
        expiring_names = [c.staff_name for c in expiring_creds]

        assert "Nurse Bob Jones" in expiring_names
        assert "Pharm. Carol Danvers" in expiring_names
        assert "Dr. Alice Smith" not in expiring_names
