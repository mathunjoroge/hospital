"""
tests/test_pharmacy_fefo.py
────────────────────────────
Unit tests for Task 3.3: Pharmacy FEFO & Inventory Management
"""

from datetime import date, timedelta

import pytest

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.pharmacy import Batch, DispensedDrug, Drug, DrugCategory
from departments.models.records import Patient
from departments.pharmacy.fefo import (
    allocate_drug_fefo,
    check_pharmacy_inventory_alerts,
    dispense_medication_fefo,
)


@pytest.fixture
def sample_inventory(app):
    category = DrugCategory(name="Antibiotics")
    db.session.add(category)
    db.session.commit()

    drug = Drug(
        generic_name="Amoxicillin FEFO",
        brand_name="Amoxil",
        category_id=category.id,
        dosage_form="Capsule",
        strength="500mg",
        buying_price=10.0,
        selling_price=20.0,
        quantity_in_stock=100,
        reorder_level=40
    )
    db.session.add(drug)
    db.session.commit()

    # Batch 1: Expiring in 20 days (Earliest) - Qty 30
    b1 = Batch(
        drug_id=drug.id,
        batch_number="BATCH-EARLY-01",
        expiry_date=date.today() + timedelta(days=20),
        quantity_in_stock=30
    )
    # Batch 2: Expiring in 180 days (Later) - Qty 70
    b2 = Batch(
        drug_id=drug.id,
        batch_number="BATCH-LATE-02",
        expiry_date=date.today() + timedelta(days=180),
        quantity_in_stock=70
    )
    db.session.add_all([b1, b2])
    db.session.commit()

    patient = Patient(
        patient_id="PTPHARM01",
        name="Pharmacy Test Patient",
        place_of_residence="Nairobi",
        sex="Male",
        date_of_birth=date(1992, 1, 1),
        marital_status="Single",
        contact="0700998877",
        next_of_kin="Kin",
        relationship_with_next_of_kin="Parent",
        next_of_kin_contact="0700998866",
        emergency_contact="0700998866"
    )
    db.session.add(patient)
    db.session.commit()

    return {"drug": drug, "b1": b1, "b2": b2, "patient": patient}


class TestFEFOAllocation:
    def test_allocates_earliest_expiring_batch_first(self, app, sample_inventory):
        drug = sample_inventory["drug"]
        b1 = sample_inventory["b1"]

        allocations = allocate_drug_fefo(drug.id, 20)
        assert len(allocations) == 1
        assert allocations[0]["batch_id"] == b1.id
        assert allocations[0]["allocated_quantity"] == 20

    def test_spans_multiple_batches_if_needed(self, app, sample_inventory):
        drug = sample_inventory["drug"]
        b1 = sample_inventory["b1"]
        b2 = sample_inventory["b2"]

        allocations = allocate_drug_fefo(drug.id, 50)
        assert len(allocations) == 2
        assert allocations[0]["batch_id"] == b1.id
        assert allocations[0]["allocated_quantity"] == 30  # Exhausts Batch 1
        assert allocations[1]["batch_id"] == b2.id
        assert allocations[1]["allocated_quantity"] == 20  # Remaining 20 from Batch 2

    def test_insufficient_stock_raises_value_error(self, app, sample_inventory):
        drug = sample_inventory["drug"]
        with pytest.raises(ValueError, match="Insufficient stock"):
            allocate_drug_fefo(drug.id, 500)


class TestDispensingAndDeduction:
    def test_dispense_deducts_batch_and_drug_stock(self, app, sample_inventory):
        drug = sample_inventory["drug"]
        b1 = sample_inventory["b1"]
        patient = sample_inventory["patient"]

        dispense_medication_fefo(patient.patient_id, drug.id, 15)

        # Refresh
        b1_fresh = Batch.query.get(b1.id)
        drug_fresh = Drug.query.get(drug.id)
        assert b1_fresh.quantity_in_stock == 15  # 30 - 15
        assert drug_fresh.quantity_in_stock == 85  # 100 - 15

        dispensed = DispensedDrug.query.filter_by(patient_id=patient.patient_id).all()
        assert len(dispensed) == 1
        assert dispensed[0].quantity_dispensed == 15


class TestInventoryAlerts:
    def test_expiry_and_low_stock_alerts(self, app, sample_inventory):
        drug = sample_inventory["drug"]
        # Reduce stock to below reorder level (40)
        drug.quantity_in_stock = 30
        db.session.commit()

        alerts = check_pharmacy_inventory_alerts(near_expiry_days=60)
        assert alerts["expiry_count"] >= 1
        assert any(b["batch_number"] == "BATCH-EARLY-01" for b in alerts["expiry_alerts"])

        assert alerts["reorder_count"] >= 1
        assert any(d["drug_id"] == drug.id for d in alerts["reorder_alerts"])


class TestFEFOEndpoints:
    def test_preview_allocation_api(self, client, sample_inventory):
        drug = sample_inventory["drug"]
        resp = client.get(f'/pharmacy/fefo/allocate?drug_id={drug.id}&quantity=25')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["allocations"]) == 1

    def test_dispense_api(self, client, sample_inventory):
        drug = sample_inventory["drug"]
        patient = sample_inventory["patient"]

        resp = client.post('/pharmacy/fefo/dispense', json={
            "patient_id": patient.patient_id,
            "drug_id": drug.id,
            "quantity": 10,
            "prescription_id": "RX-TEST-001"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_alerts_api(self, client, sample_inventory):
        resp = client.get('/pharmacy/fefo/alerts?days=60')
        assert resp.status_code == 200
        data = resp.get_json()
        assert "expiry_alerts" in data
        assert "reorder_alerts" in data
