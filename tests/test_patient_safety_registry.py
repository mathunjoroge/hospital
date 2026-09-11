from datetime import date, datetime, timezone

import pytest

from app import app
from departments.medicine.prescribe import check_drug_safety
from departments.models.records import Patient, PatientAllergy, PatientProblem
from extensions import db


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client, app.app_context():
        db.create_all()
        yield client


@pytest.fixture
def sample_patient(client):
    with app.app_context():
        p = Patient.query.filter_by(patient_id="P-TEST-ALLERGY").first()
        if not p:
            p = Patient(
                patient_id="P-TEST-ALLERGY",
                name="Allergy Test Patient",
                place_of_residence="Nairobi",
                sex="Female",
                date_of_birth=date(1990, 1, 1),
                marital_status="Single",
                contact="0712345678",
                next_of_kin="Kin",
                relationship_with_next_of_kin="Sister",
                next_of_kin_contact="0787654321",
                emergency_contact="0711111111",
            )
            db.session.add(p)
            db.session.commit()
        return p.patient_id


def test_structured_allergy_registry_safety_check(client, sample_patient):
    """Test that structured PatientAllergy record triggers safety warning even with empty NursingNote text."""
    with app.app_context():
        # 1. Ensure no free-text nursing notes exist for sample patient
        # 2. Add structured allergy to Amoxicillin
        allergy = PatientAllergy(
            patient_id=sample_patient,
            allergen="Amoxicillin",
            category="DRUG",
            reaction="Severe Skin Rash & Bronchospasm",
            severity="CRITICAL",
        )
        db.session.add(allergy)
        db.session.commit()

        # 3. Check prescribing safety for Amoxicillin
        report = check_drug_safety(sample_patient, ["Amoxicillin"])

        assert report["has_warnings"] is True
        assert report["critical_block"] is True
        assert len(report["alerts"]) > 0
        assert "Amoxicillin" in report["alerts"][0]["message"]
        assert report["alerts"][0]["type"] == "ALLERGY_WARNING"


def test_patient_problem_list_lifecycle(client, sample_patient):
    """Test active problem creation, status updates, and resolution tracking."""
    with app.app_context():
        # Add active problem
        p1 = PatientProblem(
            patient_id=sample_patient,
            icd10_code="E11.9",
            description="Type 2 Diabetes Mellitus",
            status="ACTIVE",
            onset_date=date(2023, 6, 15),
        )
        p2 = PatientProblem(
            patient_id=sample_patient,
            icd10_code="J06.9",
            description="Acute upper respiratory infection",
            status="ACTIVE",
        )
        db.session.add_all([p1, p2])
        db.session.commit()

        # Verify initial active problems count
        active_problems = PatientProblem.query.filter_by(
            patient_id=sample_patient, status="ACTIVE"
        ).all()
        assert len(active_problems) == 2

        # Mark acute infection as RESOLVED
        p2.status = "RESOLVED"
        p2.resolved_date = datetime.now(timezone.utc).date()
        db.session.commit()

        # Re-query problem list
        active_now = PatientProblem.query.filter_by(
            patient_id=sample_patient, status="ACTIVE"
        ).all()
        resolved_now = PatientProblem.query.filter_by(
            patient_id=sample_patient, status="RESOLVED"
        ).all()

        assert len(active_now) == 1
        assert active_now[0].icd10_code == "E11.9"
        assert len(resolved_now) == 1
        assert resolved_now[0].description == "Acute upper respiratory infection"
