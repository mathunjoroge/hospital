"""
tests/test_lis_panic_alerts.py
───────────────────────────────
Unit tests for Task 3.4: LIS Panic Alerts & 2-Tier Result Verification
"""

import pytest
from datetime import datetime
try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.records import Patient
from departments.models.laboratory import LabResult
from departments.models.medicine import LabTest
from departments.models.nursing import Notifications
from departments.laboratory.panic_alerts import evaluate_panic_level

@pytest.fixture
def sample_lab_setup(app):
    patient = Patient(
        patient_id="PT-LIS-01",
        name="LIS Test Patient",
        place_of_residence="Nairobi",
        sex="Female",
        date_of_birth=datetime(1990, 1, 1).date(),
        marital_status="Single",
        contact="0700998877",
        next_of_kin="Kin",
        relationship_with_next_of_kin="Sibling",
        next_of_kin_contact="0700998866",
        emergency_contact="0700998866"
    )
    db.session.add(patient)
    
    lab_test = LabTest(
        test_name="Complete Blood Count",
        cost=1000.0,
        description="CBC"
    )
    db.session.add(lab_test)
    db.session.commit()
    
    return {"patient": patient, "lab_test": lab_test}


class TestPanicThresholds:
    def test_evaluate_panic_level_normal(self):
        status, msg = evaluate_panic_level("hemoglobin", 14.0)
        assert status == "NORMAL"
        assert "Normal" in msg
        
    def test_evaluate_panic_level_abnormal_low(self):
        status, msg = evaluate_panic_level("potassium", 3.2)
        assert status == "ABNORMAL"
        assert "Abnormal Low" in msg
        
    def test_evaluate_panic_level_panic_low(self):
        status, msg = evaluate_panic_level("potassium", 2.5)
        assert status == "PANIC_CRITICAL"
        assert "CRITICAL PANIC LOW" in msg

    def test_evaluate_panic_level_panic_high(self):
        status, msg = evaluate_panic_level("blood_glucose", 450.0)
        assert status == "PANIC_CRITICAL"
        assert "CRITICAL PANIC HIGH" in msg


class TestLISAPI:
    def test_enter_lab_result_normal(self, client, sample_lab_setup):
        patient = sample_lab_setup["patient"]
        lab_test = sample_lab_setup["lab_test"]
        
        resp = client.post('/laboratory/lis/enter', json={
            "patient_id": patient.patient_id,
            "lab_test_id": lab_test.id,
            "parameter_name": "potassium",
            "result_value": 4.5,
            "tech_id": 2
        })
        
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert data["panic_status"] == "NORMAL"
        assert data["status"] == "PENDING_VERIFICATION"
        
        result_id = data["result_id"]
        db_res = LabResult.query.filter_by(result_id=result_id).first()
        assert db_res is not None
        assert db_res.panic_status == "NORMAL"

    def test_enter_lab_result_panic_critical(self, client, sample_lab_setup):
        patient = sample_lab_setup["patient"]
        lab_test = sample_lab_setup["lab_test"]
        
        resp = client.post('/laboratory/lis/enter', json={
            "patient_id": patient.patient_id,
            "lab_test_id": lab_test.id,
            "parameter_name": "potassium",
            "result_value": 7.0,  # Panic high
            "tech_id": 2
        })
        
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["panic_status"] == "PANIC_CRITICAL"

    def test_verify_result_dispatches_alert(self, client, sample_lab_setup):
        # Create a PANIC_CRITICAL result first
        patient = sample_lab_setup["patient"]
        lab_test = sample_lab_setup["lab_test"]
        
        res_uuid = "RES-TEST-PANIC"
        lab_res = LabResult(
            patient_id=patient.patient_id,
            lab_test_id=lab_test.id,
            result_id=res_uuid,
            result="potassium: 7.0",
            status="PENDING_VERIFICATION",
            panic_status="PANIC_CRITICAL",
            panic_message="CRITICAL PANIC HIGH: 7.0 mmol/L",
            updated_by=2
        )
        db.session.add(lab_res)
        db.session.commit()
        
        # Verify it
        resp = client.post('/laboratory/lis/verify', json={
            "result_id": res_uuid,
            "verifier_id": 5,
            "action": "VERIFY"
        })
        
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "VERIFIED"
        assert data["panic_alert_sent"] is True
        
        # Check if notification was created
        notification = Notifications.query.filter_by(receiver_id=5).first()
        assert notification is not None
        assert "CRITICAL LAB PANIC ALERT" in notification.message

    def test_get_panic_alerts(self, client, sample_lab_setup):
        resp = client.get('/laboratory/lis/panic_alerts')
        assert resp.status_code == 200
        data = resp.get_json()
        assert "panic_alerts" in data
        assert isinstance(data["panic_alerts"], list)
