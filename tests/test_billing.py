import pytest
from extensions import db
from departments.models.billing import ChargeCategory, Charge, Billing

def test_create_and_verify_billing_record(app):
    with app.app_context():
        category = ChargeCategory(name="Consultation Fee")
        db.session.add(category)
        db.session.commit()

        charge = Charge(
            name="General Consultation",
            category_id=category.id,
            cost=1500.00,
            description="Standard doctor consultation"
        )
        db.session.add(charge)
        db.session.commit()

        bill = Billing(
            patient_id="PTEST100",
            charge_id=charge.id,
            quantity=2,
            total_cost=3000.00,
            status=0
        )
        db.session.add(bill)
        db.session.commit()

        saved_bill = Billing.query.filter_by(patient_id="PTEST100").first()
        assert saved_bill is not None
        assert float(saved_bill.total_cost) == 3000.00
        assert saved_bill.status == 0
