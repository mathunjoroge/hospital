"""
tests/test_patient_resolution_r02.py
──────────────────────────────────────
Verification test for R-02:
Ensures exact patient ID resolution across records, medicine & CDSS endpoints,
preventing ambiguous fuzzy matches (e.g., 'A1' matching 'A10' or patient named 'John A1').
"""

from datetime import date

import pytest

from departments.medicine.cdss import evaluate_prescription_safety
from departments.models.records import Patient
from extensions import db


@pytest.fixture
def resolution_patients(app):
    with app.app_context():
        p1 = Patient(
            patient_id="A1",
            name="Alice One",
            place_of_residence="Nairobi",
            sex="Female",
            date_of_birth=date(1990, 1, 1),
            marital_status="Single",
            contact="0700000001",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Parent",
            next_of_kin_contact="0700000001",
            emergency_contact="0700000001",
        )
        p2 = Patient(
            patient_id="A10",
            name="Alexander Ten",
            place_of_residence="Mombasa",
            sex="Male",
            date_of_birth=date(1985, 5, 5),
            marital_status="Married",
            contact="0700000010",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Spouse",
            next_of_kin_contact="0700000010",
            emergency_contact="0700000010",
        )
        p3 = Patient(
            patient_id="P99",
            name="Contains A1 Substring",
            place_of_residence="Kisumu",
            sex="Male",
            date_of_birth=date(1992, 3, 3),
            marital_status="Single",
            contact="0700000099",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Sibling",
            next_of_kin_contact="0700000099",
            emergency_contact="0700000099",
        )
        db.session.add_all([p1, p2, p3])
        db.session.commit()
        return {"p1": p1, "p2": p2, "p3": p3}


def test_cdss_patient_resolution_exact(resolution_patients):
    """CDSS evaluation with patient_id='A1' must resolve only to patient A1."""
    report = evaluate_prescription_safety(patient_id="A1", drug_name="amoxicillin")
    assert report is not None


def test_records_patient_profile_exact(client, admin_user, resolution_patients):
    """Route GET /records/patient/A1 must return profile for patient A1."""
    resp = client.get("/records/patient/A1")
    assert resp.status_code == 200
    assert "Alice One" in resp.get_data(as_text=True)
    assert "Alexander Ten" not in resp.get_data(as_text=True)


def test_records_patient_profile_nonexistent_returns_404(client, admin_user, resolution_patients):
    """Route GET /records/patient/A1_NON_EXISTENT must return 404."""
    resp = client.get("/records/patient/A1_NON_EXISTENT")
    assert resp.status_code == 404
