"""
tests/test_triage_esi.py
────────────────────────
Unit tests for Task 3.1: Emergency Severity Index (ESI 1-5) & Triage Workflow
"""

from datetime import date

import pytest

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.records import Patient
from departments.nursing.triage import calculate_esi_level, validate_vitals


@pytest.fixture
def sample_patient(app):
    patient = Patient(
        patient_id="PTTRIAGE01",
        name="Triage Test Patient",
        place_of_residence="Nairobi",
        sex="Male",
        date_of_birth=date(1990, 5, 10),
        marital_status="Single",
        contact="0700112233",
        next_of_kin="Kin Name",
        relationship_with_next_of_kin="Brother",
        next_of_kin_contact="0700112244",
        emergency_contact="0700112244",
    )
    db.session.add(patient)
    db.session.commit()
    return patient


class TestVitalsValidation:
    def test_adult_normal_vitals(self):
        res = validate_vitals(age_years=30, hr=75, rr=16, sbp=120, temp=36.8, spo2=98)
        assert res["is_abnormal"] is False
        assert res["risk_level"] == "NORMAL"

    def test_pediatric_abnormal_hr(self):
        # Infant (<1 year) with HR 190 (normal 100-160)
        res = validate_vitals(age_years=0.5, hr=190, rr=40, sbp=85, temp=37.0, spo2=97)
        assert res["is_abnormal"] is True
        assert any("Heart Rate" in w for w in res["warnings"])

    def test_critical_hypoxia_warning(self):
        res = validate_vitals(age_years=45, hr=90, rr=20, sbp=120, temp=37.0, spo2=84)
        assert res["is_abnormal"] is True
        assert res["risk_level"] == "CRITICAL"
        assert any("CRITICAL HYPOXIA" in w for w in res["warnings"])


class TestESICalculation:
    def test_esi_1_resuscitation(self):
        vitals = {"pulse": 150, "respiratory_rate": 35, "oxygen_saturation": 82}
        esi, desc = calculate_esi_level(
            vitals, chief_complaint="Unresponsive", resources_needed=3, age_years=40
        )
        assert esi == 1
        assert "ESI Level 1" in desc

    def test_esi_2_emergent_chest_pain(self):
        vitals = {"pulse": 110, "respiratory_rate": 22, "oxygen_saturation": 95}
        esi, desc = calculate_esi_level(
            vitals,
            chief_complaint="Severe chest pain",
            resources_needed=2,
            age_years=50,
        )
        assert esi == 2

    def test_esi_3_urgent_multi_resource(self):
        vitals = {"pulse": 72, "respiratory_rate": 16, "oxygen_saturation": 98}
        esi, desc = calculate_esi_level(
            vitals, chief_complaint="Abdominal pain", resources_needed=2, age_years=30
        )
        assert esi == 3

    def test_esi_4_less_urgent_single_resource(self):
        vitals = {"pulse": 70, "respiratory_rate": 14, "oxygen_saturation": 99}
        esi, desc = calculate_esi_level(
            vitals, chief_complaint="Ankle sprain", resources_needed=1, age_years=25
        )
        assert esi == 4

    def test_esi_5_non_urgent_no_resource(self):
        vitals = {"pulse": 68, "respiratory_rate": 15, "oxygen_saturation": 98}
        esi, desc = calculate_esi_level(
            vitals,
            chief_complaint="Medication refill",
            resources_needed=0,
            age_years=20,
        )
        assert esi == 5


class TestTriageEndpoints:
    def test_assess_patient_endpoint(self, client, sample_patient):
        resp = client.post(
            "/nursing/triage/assess",
            json={
                "patient_id": sample_patient.patient_id,
                "chief_complaint": "Severe shortness of breath",
                "resources_needed": 2,
                "vitals": {
                    "temperature": 38.5,
                    "pulse": 125,
                    "blood_pressure_systolic": 140,
                    "blood_pressure_diastolic": 90,
                    "respiratory_rate": 28,
                    "oxygen_saturation": 91,
                },
            },
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert data["esi_level"] in (1, 2)
        assert data["priority_status"] == "ESCALATED"

    def test_triage_queue_ordering(self, client, sample_patient):
        # Create non-urgent assessment (ESI 4)
        client.post(
            "/nursing/triage/assess",
            json={
                "patient_id": sample_patient.patient_id,
                "chief_complaint": "Minor cut",
                "resources_needed": 1,
                "vitals": {
                    "pulse": 70,
                    "respiratory_rate": 16,
                    "oxygen_saturation": 98,
                },
            },
        )

        resp = client.get("/nursing/triage/queue")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "queue" in data
        assert data["total_waiting"] >= 1
