"""
tests/test_inpatient_mar.py
────────────────────────────
Unit tests for Task 3.6: Inpatient ADT & Medication Administration Record (MAR)
"""

from datetime import datetime

import pytest

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.billing import Invoice, InvoiceLineItem
from departments.models.medicine import AdmittedPatient, Ward
from departments.models.nursing import MedicationAdmin
from departments.models.records import Patient


@pytest.fixture
def mar_app(app):
    return app


@pytest.fixture
def mar_client(mar_app):
    return mar_app.test_client()


@pytest.fixture
def mar_data(mar_app):
    patient = Patient(
        patient_id="PT-MAR-01",
        name="MAR Test Patient",
        place_of_residence="Nairobi",
        sex="Male",
        date_of_birth=datetime(1985, 5, 5).date(),  # noqa: DTZ001
        marital_status="Married",
        contact="0700112233",
        next_of_kin="Kin",
        relationship_with_next_of_kin="Spouse",
        next_of_kin_contact="0700112233",
        emergency_contact="0700112233",
    )
    db.session.add(patient)

    ward = Ward(
        name="General Male Ward",
        sex="Male",
        number_of_beds=20,
        occupied_beds=5,
        daily_charge=1500.0,
    )
    db.session.add(ward)
    db.session.commit()

    admission = AdmittedPatient(
        patient_id=patient.patient_id,
        ward_id=ward.id,
        admission_criteria="Severe Malaria",
        admitted_by=1,
    )
    db.session.add(admission)
    db.session.commit()

    return {"patient": patient, "ward": ward, "admission": admission}


class TestInpatientMAR:
    def test_ward_occupancy(self, mar_client, mar_data):
        resp = mar_client.get("/nursing/mar/occupancy")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "occupancy" in data

        ward = next(w for w in data["occupancy"] if w["ward_id"] == mar_data["ward"].id)
        assert ward["name"] == "General Male Ward"
        assert ward["available_beds"] == 15

    def test_chart_medication(self, mar_client, mar_data):
        resp = mar_client.post(
            "/nursing/mar/chart",
            json={
                "patient_id": mar_data["patient"].patient_id,
                "medication": "Paracetamol IV",
                "dosage": "1000mg",
                "nurse_id": 2,
            },
        )

        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True

        # Verify DB
        record = MedicationAdmin.query.get(data["record_id"])
        assert record is not None
        assert record.medication == "Paracetamol IV"
        assert record.dosage == "1000mg"

    def test_auto_billing(self, mar_client, mar_data):
        resp = mar_client.post("/nursing/mar/auto_bill")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["patients_billed"] >= 1

        # Verify invoice created
        invoice = Invoice.query.filter_by(
            patient_id=mar_data["patient"].patient_id
        ).first()
        assert invoice is not None
        assert invoice.grand_total >= 1500.0

        line_item = InvoiceLineItem.query.filter_by(invoice_id=invoice.id).first()
        assert line_item is not None
        assert "Daily Ward Charge" in line_item.description
