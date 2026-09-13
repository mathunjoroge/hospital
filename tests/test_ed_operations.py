"""
tests/test_ed_operations.py
─────────────────────────────
Unit and integration tests for ED Operations Management, door-to-triage,
door-to-doctor, ED LOS tracking, ESI 2 re-evaluations, and boarding alerts (Gap #9).
"""

from datetime import date, datetime, timedelta, timezone

import pytest
from flask_jwt_extended import create_access_token
from werkzeug.security import generate_password_hash

from app import app
from departments.models.nursing import TriageAssessment
from departments.models.records import Patient
from departments.models.user import User
from departments.nursing.ed_engine import EDOperationsEngine
from extensions import db


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client


@pytest.fixture
def auth_headers(client):
    with app.app_context():
        user = User.query.filter_by(username="nursing_ed_test").first()
        if not user:
            user = User(
                username="nursing_ed_test",
                password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
                role="nursing",
            )
            db.session.add(user)
            db.session.commit()
        token = create_access_token(identity=str(user.id))
    return {"Authorization": f"Bearer {token}"}


def _make_patient(patient_id: str, name: str = "Test ED Patient", sex: str = "Female") -> Patient:
    """Insert a minimal patient into DB context."""
    p = Patient.query.filter_by(patient_id=patient_id).first()
    if not p:
        p = Patient(
            patient_id=patient_id,
            name=name,
            sex=sex,
            date_of_birth=date(1990, 1, 1),
            emergency_contact="0700000000",
        )
        db.session.add(p)
        db.session.commit()
    return p


def test_ed_patient_lifecycle_flow(client):
    """Test full ED operational lifecycle: arrival -> triage -> bed -> doctor -> re-eval -> disposition."""
    with app.app_context():
        # 1. Setup Patient
        _make_patient("ED-PAT-100", name="Emergency Patient 1", sex="Male")

        # 2. Record Arrival
        assessment = EDOperationsEngine.record_arrival(patient_id="ED-PAT-100", chief_complaint="Chest Pain")
        assert assessment.id is not None
        assert assessment.arrival_at is not None
        assert assessment.priority_status == "WAITING"

        # 3. Complete Triage (ESI 2)
        triage_assessment = EDOperationsEngine.complete_triage(
            assessment_id=assessment.id, esi_level=2, vitals_warning="High BP"
        )
        assert triage_assessment.triage_completed_at is not None
        assert triage_assessment.esi_level == 2
        assert triage_assessment.re_evaluation_due_at is not None
        assert triage_assessment.priority_status == "ESCALATED"

        # 4. Assign Bed
        bed_assessment = EDOperationsEngine.assign_bed(assessment_id=assessment.id, bed_label="RESUS-2")
        assert bed_assessment.bed_label == "RESUS-2"
        assert bed_assessment.bed_assigned_at is not None

        # 5. Seen by Doctor
        doc_assessment = EDOperationsEngine.mark_seen_by_doctor(assessment_id=assessment.id)
        assert doc_assessment.seen_by_doctor_at is not None
        assert doc_assessment.priority_status == "SEEN"

        # 6. ESI Re-evaluation
        reeval_assessment = EDOperationsEngine.record_re_evaluation(
            assessment_id=assessment.id, notes="Patient stable following sublingual nitroglycerin."
        )
        assert reeval_assessment.last_re_evaluation_at is not None
        assert "sublingual nitroglycerin" in reeval_assessment.re_evaluation_notes

        # 7. Discharge / Disposition
        disp_assessment = EDOperationsEngine.discharge_patient(assessment_id=assessment.id, disposition="ADMITTED")
        assert disp_assessment.disposition == "ADMITTED"
        assert disp_assessment.disposition_at is not None
        assert disp_assessment.priority_status == "DISPOSITIONED"
        assert disp_assessment.re_evaluation_due_at is None


def test_ed_dashboard_metrics_and_alerts(client):
    """Test ED KPIs, ESI 2 overdue detection, and boarding alert (>4 hours)."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        _make_patient("ED-PAT-201", name="Overdue ESI2", sex="Female")
        _make_patient("ED-PAT-202", name="Boarding Patient", sex="Male")

        # ESI 2 patient whose re-evaluation is overdue (due 20 minutes ago)
        a1 = TriageAssessment(
            patient_id="ED-PAT-201",
            nurse_id=1,
            esi_level=2,
            chief_complaint="Severe Dyspnea",
            arrival_at=now - timedelta(minutes=45),
            triage_completed_at=now - timedelta(minutes=35),
            re_evaluation_due_at=now - timedelta(minutes=20),
            priority_status="ESCALATED",
        )

        # Patient with LOS > 4.5 hours (270 minutes) -> Boarding Alert
        a2 = TriageAssessment(
            patient_id="ED-PAT-202",
            nurse_id=1,
            esi_level=3,
            chief_complaint="Abdominal Pain",
            arrival_at=now - timedelta(minutes=270),
            triage_completed_at=now - timedelta(minutes=250),
            seen_by_doctor_at=now - timedelta(minutes=200),
            bed_label="BAY-4",
            priority_status="SEEN",
        )

        db.session.add_all([a1, a2])
        db.session.commit()

        metrics = EDOperationsEngine.get_ed_dashboard_metrics()
        assert metrics["total_active"] >= 2
        assert metrics["esi_counts"][2] >= 1
        assert metrics["esi_counts"][3] >= 1

        # Check ESI 2 overdue list
        assert metrics["re_eval_overdue_count"] >= 1
        overdue_ids = [item["patient_id"] for item in metrics["re_eval_overdue"]]
        assert "ED-PAT-201" in overdue_ids

        # Check Boarding alerts list (>4h stay)
        assert metrics["boarding_alerts_count"] >= 1
        boarding_ids = [item["patient_id"] for item in metrics["boarding_alerts"]]
        assert "ED-PAT-202" in boarding_ids


def test_ed_routes_api(client, auth_headers):
    """Test ED HTTP API endpoints for arrival, bed assignment, doctor mark, metrics, and boarding alerts."""
    with app.app_context():
        _make_patient("ED-API-99", name="API Test Patient", sex="Female")

    # 1. Post Arrival
    res1 = client.post("/nursing/ed/arrive", json={
        "patient_id": "ED-API-99",
        "chief_complaint": "Acute Headache"
    }, headers=auth_headers)
    assert res1.status_code == 201
    ass_id = res1.get_json()["assessment_id"]

    # 2. Post Triage Complete
    res2 = client.post(f"/nursing/ed/triage-complete/{ass_id}", json={
        "esi_level": 2,
        "vitals_warning": "Elevated BP"
    }, headers=auth_headers)
    assert res2.status_code == 200

    # 3. Post Assign Bed
    res3 = client.post(f"/nursing/ed/assign-bed/{ass_id}", json={
        "bed_label": "BAY-9"
    }, headers=auth_headers)
    assert res3.status_code == 200
    assert res3.get_json()["bed_label"] == "BAY-9"

    # 4. Post Mark Seen by Doctor
    res4 = client.post(f"/nursing/ed/seen-by-doctor/{ass_id}", headers=auth_headers)
    assert res4.status_code == 200

    # 5. Post Re-evaluation
    res5 = client.post(f"/nursing/ed/re-evaluate/{ass_id}", json={
        "notes": "Patient headache improving after analgesia."
    }, headers=auth_headers)
    assert res5.status_code == 200

    # 6. GET ED Metrics API
    res6 = client.get("/nursing/api/ed/metrics", headers=auth_headers)
    assert res6.status_code == 200
    assert res6.get_json()["total_active"] >= 1

    # 7. GET Boarding Alerts API
    res7 = client.get("/nursing/api/ed/boarding-alerts", headers=auth_headers)
    assert res7.status_code == 200

    # 8. GET Dashboard UI HTML
    res8 = client.get("/nursing/ed/dashboard", headers=auth_headers)
    assert res8.status_code == 200
    assert b"ED Operations Console" in res8.data
