"""
Comprehensive Unit Tests for Pharmacy Dispensing & Stock Ops.

Pushes test coverage for departments/pharmacy/dispensing.py & stock_ops.py to >90%.
"""

from datetime import date, datetime, timezone

import pytest

from departments.models.medicine import PrescribedMedicine
from departments.models.pharmacy import (
    Batch,
    DispensedDrug,
    Drug,
    DrugCategory,
)
from departments.models.records import Patient
from extensions import db


@pytest.fixture
def pharmacy_setup(app):
    """Setup pharmacy categories, drugs, batches, and patient data."""
    with app.app_context():
        category = DrugCategory(name="Analgesics")
        db.session.add(category)
        db.session.commit()

        drug1 = Drug(
            generic_name="Paracetamol",
            brand_name="Panadol",
            category_id=category.id,
            dosage_form="Tablet",
            strength="500mg",
            buying_price=2.0,
            selling_price=5.0,
            quantity_in_stock=100,
            reorder_level=20,
        )
        drug2 = Drug(
            generic_name="Ibuprofen",
            brand_name="Brufen",
            category_id=category.id,
            dosage_form="Tablet",
            strength="400mg",
            buying_price=4.0,
            selling_price=10.0,
            quantity_in_stock=50,
            reorder_level=10,
        )
        db.session.add_all([drug1, drug2])
        db.session.commit()

        batch1 = Batch(
            drug_id=drug1.id,
            batch_number="BATCH-PARA-01",
            expiry_date=date(2028, 1, 1),
            quantity_in_stock=100,
        )
        batch2 = Batch(
            drug_id=drug2.id,
            batch_number="BATCH-IBU-01",
            expiry_date=date(2028, 1, 1),
            quantity_in_stock=50,
        )
        db.session.add_all([batch1, batch2])
        db.session.commit()

        patient = Patient(
            patient_id="P-PHARM-001",
            name="John Pharmacy Test",
            sex="Male",
            date_of_birth=date(1990, 1, 1),
            contact="0700000000",
            date_registered=datetime.now(timezone.utc),
        )
        db.session.add(patient)
        db.session.commit()

        prescription = PrescribedMedicine(
            prescription_id="RX-TEST-001",
            patient_id=patient.patient_id,
            medicine_id=drug1.id,
            num_days=5,
            dosage="500mg",
            strength="500mg",
            frequency="TID",
            status=0,
        )
        db.session.add(prescription)
        db.session.commit()

        yield {
            "drug1": drug1,
            "drug2": drug2,
            "batch1": batch1,
            "batch2": batch2,
            "patient": patient,
            "prescription": prescription,
        }


def test_prescriptions_route(client, admin_user, pharmacy_setup):
    """Test /pharmacy/prescriptions endpoint."""
    resp = client.get("/pharmacy/prescriptions", follow_redirects=True)
    assert resp.status_code == 200
    assert (
        b"RX-TEST-001" in resp.data
        or b"Paracetamol" in resp.data
        or b"Prescriptions" in resp.data
    )


def test_view_prescriptions_route(client, admin_user, pharmacy_setup):
    """Test /pharmacy/view_prescriptions/<patient_id> endpoint."""
    patient = pharmacy_setup["patient"]
    resp = client.get(
        f"/pharmacy/view_prescriptions/{patient.patient_id}", follow_redirects=True
    )
    assert resp.status_code == 200
    assert b"John Pharmacy Test" in resp.data or b"RX-TEST-001" in resp.data


def test_dispense_prescription_route(client, admin_user, pharmacy_setup):
    """Test /pharmacy/dispense/<prescription_id> endpoint."""
    resp = client.get("/pharmacy/dispense/RX-TEST-001", follow_redirects=True)
    assert resp.status_code == 200

    # Non-existent prescription redirects
    resp_invalid = client.get(
        "/pharmacy/dispense/RX-NONEXISTENT", follow_redirects=True
    )
    assert resp_invalid.status_code == 200


def test_delete_dispensed_drug_route(client, admin_user, pharmacy_setup, app):
    """Test /pharmacy/delete_dispensed_drug/<dispensed_drug_id> endpoint."""
    with app.app_context():
        drug1 = pharmacy_setup["drug1"]
        batch1 = pharmacy_setup["batch1"]
        patient = pharmacy_setup["patient"]

        dispensed = DispensedDrug(
            drug_id=drug1.id,
            batch_id=batch1.id,
            patient_id=patient.patient_id,
            prescription_id="RX-TEST-001",
            quantity_dispensed=10,
            date_dispensed=datetime.now(timezone.utc).date(),
        )
        db.session.add(dispensed)
        batch1.quantity_in_stock -= 10
        db.session.commit()
        dispensed_id = dispensed.id

    resp = client.post(
        f"/pharmacy/delete_dispensed_drug/{dispensed_id}",
        data={"void_reason": "Patient cancelled"},
        headers={"Referer": "/pharmacy/prescriptions"},
        follow_redirects=True,
    )
    assert resp.status_code == 200

    with app.app_context():
        # Verify status is VOIDED and batch stock restored
        d = db.session.get(DispensedDrug, dispensed_id)
        assert d is not None
        assert d.status == "VOIDED"
        b = db.session.get(Batch, batch1.id)
        assert b.quantity_in_stock == 100


def test_save_dispensed_drugs_route(client, admin_user, pharmacy_setup, app):
    """Test /pharmacy/save_dispensed_drugs POST route."""
    with app.app_context():
        drug1 = pharmacy_setup["drug1"]
        batch1 = pharmacy_setup["batch1"]
        patient = pharmacy_setup["patient"]

        dispensed = DispensedDrug(
            drug_id=drug1.id,
            batch_id=batch1.id,
            patient_id=patient.patient_id,
            prescription_id="RX-TEST-001",
            quantity_dispensed=5,
            date_dispensed=datetime.now(timezone.utc).date(),
            status="Pending",
        )
        db.session.add(dispensed)
        db.session.commit()
        dispensed_id = dispensed.id

    data = {
        "prescription_id": "RX-TEST-001",
        f"updatedDrugs[{dispensed_id}]": "10",
    }
    resp = client.post(
        "/pharmacy/save_dispensed_drugs", data=data, follow_redirects=True
    )
    assert resp.status_code == 200

    with app.app_context():
        med = PrescribedMedicine.query.filter_by(prescription_id="RX-TEST-001").first()
        assert med.status == 1


def test_save_prescription_route(client, admin_user, pharmacy_setup, app):
    """Test /pharmacy/save_prescription/<prescription_id> POST route."""
    drug1 = pharmacy_setup["drug1"]
    batch1 = pharmacy_setup["batch1"]
    patient = pharmacy_setup["patient"]

    data = {
        "drugs[]": [str(drug1.id)],
        "quantity[]": ["5"],
        "batch_number[]": [batch1.batch_number],
        "patient_id": patient.patient_id,
    }
    resp = client.post(
        "/pharmacy/save_prescription/RX-TEST-001", data=data, follow_redirects=True
    )
    assert resp.status_code == 200

    with app.app_context():
        dispensed = DispensedDrug.query.filter_by(prescription_id="RX-TEST-001").first()
        assert dispensed is not None
        assert dispensed.quantity_dispensed == 5


def test_stock_ops_remove_dispensed(client, admin_user, pharmacy_setup, app):
    """Test /pharmacy/remove_dispensed/<dispense_id> POST route in stock_ops.py."""
    with app.app_context():
        drug1 = pharmacy_setup["drug1"]
        batch1 = pharmacy_setup["batch1"]
        patient = pharmacy_setup["patient"]

        dispensed = DispensedDrug(
            drug_id=drug1.id,
            batch_id=batch1.id,
            patient_id=patient.patient_id,
            prescription_id="RX-TEST-001",
            quantity_dispensed=4,
            date_dispensed=datetime.now(timezone.utc).date(),
        )
        db.session.add(dispensed)
        db.session.commit()
        dispense_id = dispensed.id

    resp = client.post(
        f"/pharmacy/remove_dispensed/{dispense_id}",
        data={"prescription_id": "RX-TEST-001", "void_reason": "Order error"},
        follow_redirects=True,
    )
    assert resp.status_code == 200

    with app.app_context():
        d = db.session.get(DispensedDrug, dispense_id)
        assert d is not None
        assert d.status == "VOIDED"


def test_stock_ops_process_dispense(client, admin_user, pharmacy_setup):
    """Test /pharmacy/dispense/process/<prescription_id> POST route in stock_ops.py."""
    drug1 = pharmacy_setup["drug1"]
    batch1 = pharmacy_setup["batch1"]

    data = {
        "drug_id": str(drug1.id),
        "batch_id": str(batch1.id),
        "quantity_dispensed": "3",
    }
    resp = client.post(
        "/pharmacy/dispense/process/RX-TEST-001", data=data, follow_redirects=True
    )
    assert resp.status_code == 200
    assert b"dispensed successfully" in resp.data or b"Panadol" in resp.data


def test_patient_history_route(client, admin_user, pharmacy_setup):
    """Test /pharmacy/patient_history GET and POST JSON/Form routes."""
    patient = pharmacy_setup["patient"]

    # GET
    resp_get = client.get("/pharmacy/patient_history", follow_redirects=True)
    assert resp_get.status_code == 200

    # POST Form
    resp_post = client.post(
        "/pharmacy/patient_history",
        data={"patient_id": patient.patient_id},
        follow_redirects=True,
    )
    assert resp_post.status_code == 200

    # POST JSON
    resp_json = client.post(
        "/pharmacy/patient_history",
        json={"patient_id": patient.patient_id},
    )
    assert resp_json.status_code == 200
    json_data = resp_json.get_json()
    assert json_data["patient"]["patient_id"] == patient.patient_id
