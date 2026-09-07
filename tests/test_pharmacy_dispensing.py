from datetime import date, datetime

from departments.models.pharmacy import Batch, DispensedDrug, Drug, DrugCategory
from extensions import db


def test_pharmacy_dispensing_decrements_stock(app):
    with app.app_context():
        category = DrugCategory(name="Antibiotics")
        db.session.add(category)
        db.session.commit()

        drug = Drug(
            generic_name="Amoxicillin",
            brand_name="Amoxil",
            category_id=category.id,
            dosage_form="Capsule",
            strength="500mg",
            buying_price=10.0,
            selling_price=15.0,
            quantity_in_stock=100,
            reorder_level=20,
        )
        db.session.add(drug)
        db.session.commit()

        batch = Batch(
            drug_id=drug.id,
            batch_number="B12345",
            expiry_date=date(2028, 12, 31),
            quantity_in_stock=100,
        )
        db.session.add(batch)
        db.session.commit()

        # Dispense 15 capsules
        dispensed = DispensedDrug(
            drug_id=drug.id,
            batch_id=batch.id,
            patient_id="PTEST100",
            prescription_id="RX100",
            quantity_dispensed=15,
            date_dispensed=datetime.utcnow(),
        )
        batch.quantity_in_stock -= 15
        drug.quantity_in_stock -= 15

        db.session.add(dispensed)
        db.session.commit()

        updated_batch = Batch.query.get(batch.id)
        updated_drug = Drug.query.get(drug.id)

        assert updated_batch.quantity_in_stock == 85
        assert updated_drug.quantity_in_stock == 85
