"""
tests/test_eprescribing.py
───────────────────────────
Unit tests for Task 3.2: Clinical Consultation & E-Prescribing System
"""

import pytest
from datetime import date
try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.records import Patient
from departments.models.nursing import NursingNote
from departments.models.medicine import PrescribedMedicine, SOAPNote
from departments.models.billing import Invoice
from departments.medicine.prescribe import (
    search_icd10,
    check_drug_safety
)


@pytest.fixture
def sample_patient(app):
    patient = Patient(
        patient_id="PTPRESCRIBE01",
        name="Prescribe Test Patient",
        place_of_residence="Nairobi",
        sex="Female",
        date_of_birth=date(1988, 3, 15),
        marital_status="Single",
        contact="0711223344",
        next_of_kin="Kin Name",
        relationship_with_next_of_kin="Sister",
        next_of_kin_contact="0711223355",
        emergency_contact="0711223355"
    )
    db.session.add(patient)
    db.session.commit()

    # Document Penicillin allergy
    note = NursingNote(
        patient_id=patient.patient_id,
        nurse_id=1,
        note="Initial Triage Assessment",
        allergies="Penicillin, Sulfa"
    )
    db.session.add(note)
    db.session.commit()
    return patient


class TestICD10Search:
    def test_search_by_keyword(self):
        res = search_icd10("hypertension")
        assert len(res) >= 1
        assert res[0]["code"] == "I10"

    def test_search_by_code(self):
        res = search_icd10("J06.9")
        assert len(res) >= 1
        assert "upper respiratory" in res[0]["description"].lower()


class TestSafetyChecks:
    def test_allergy_warning_detection(self, app, sample_patient):
        report = check_drug_safety(sample_patient.patient_id, ["Amoxicillin 500mg"])
        assert report["has_warnings"] is True
        assert report["critical_block"] is True
        assert any(a["type"] == "ALLERGY_WARNING" for a in report["alerts"])

    def test_ddi_warning_detection(self, app, sample_patient):
        report = check_drug_safety(sample_patient.patient_id, ["Warfarin 5mg", "Aspirin 81mg"])
        assert report["has_warnings"] is True
        assert any(a["type"] == "DRUG_INTERACTION" for a in report["alerts"])

    def test_safe_drug_no_warnings(self, app, sample_patient):
        report = check_drug_safety(sample_patient.patient_id, ["Paracetamol 500mg"])
        assert report["has_warnings"] is False
        assert report["critical_block"] is False


class TestEPrescribingEndpoints:
    def test_soap_consultation_endpoint(self, client, sample_patient):
        resp = client.post('/medicine/prescribe/soap', json={
            "patient_id": sample_patient.patient_id,
            "subjective": "Patient reports 3-day history of dry cough and fever.",
            "objective": "Temp 38.2C, Chest clear.",
            "assessment": "Acute Upper Respiratory Infection",
            "icd10_code": "J06.9",
            "plan": "Rest, oral fluids, paracetamol PRN."
        })
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert data["icd10_code"] == "J06.9"

    def test_signoff_blocked_on_critical_allergy(self, client, sample_patient):
        resp = client.post('/medicine/prescribe/signoff', json={
            "patient_id": sample_patient.patient_id,
            "prescriptions": [{"name": "Amoxicillin 500mg", "dosage": "TDS", "cost": 300.0}]
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["requires_override"] is True

    def test_signoff_success_creates_invoice_line_item(self, client, sample_patient):
        resp = client.post('/medicine/prescribe/signoff', json={
            "patient_id": sample_patient.patient_id,
            "prescriptions": [
                {"name": "Paracetamol 500mg", "dosage": "1 tab TDS", "duration": "5 days", "cost": 150.0},
                {"name": "Multivitamins", "dosage": "1 tab OD", "duration": "10 days", "cost": 200.0}
            ]
        })
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert data["prescribed_count"] == 2
        assert data["total_charge"] == 350.0

        # Verify draft invoice created
        inv = Invoice.query.filter_by(patient_id=sample_patient.patient_id).first()
        assert inv is not None
        assert float(inv.grand_total) == 350.0
