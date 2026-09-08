"""
tests/test_worldclass_wiring.py
────────────────────────────────
Verifies the previously-orphaned "world-class" modules are now correctly
wired end-to-end:

  - appointments blueprint (booking, check-in, live queue)
  - referrals blueprint (initiate, status update, discharge summary)
  - consent /api/check endpoint (bridges Patient.id -> Patient.patient_id)

These blueprints existed in the codebase with full models/engine/routes
but were never registered in app.py, so none of this was previously
reachable. This test would 404 on every assertion before that fix.
"""
from datetime import date, datetime, timedelta, timezone

import pytest

from app import app as flask_app
from departments.models.compliance import grant_patient_consent
from departments.models.records import Patient
from departments.models.user import User
from extensions import db


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    flask_app.config["RATELIMIT_ENABLED"] = False
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"

    with flask_app.app_context():
        db.engine.dispose()
        db.create_all()

        user = User(id=1, username="wiring_test_user", password="x", role="admin")
        patient = Patient(
            patient_id="P-WIRE-001",
            name="Wiring Test Patient",
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
        db.session.add_all([user, patient])
        db.session.commit()

        with flask_app.test_client() as test_client:
            with test_client.session_transaction() as sess:
                sess["_user_id"] = "1"
                sess["_fresh"] = True
            yield test_client

        db.session.remove()
        db.drop_all()
        db.engine.dispose()


def _patient_pk():
    with flask_app.app_context():
        return Patient.query.filter_by(patient_id="P-WIRE-001").first().id


# ── Appointments ──────────────────────────────────────────────────────


def test_book_checkin_and_live_queue(client):
    patient_pk = _patient_pk()
    start = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    resp = client.post(
        "/appointments/api/book",
        json={
            "patient_id": patient_pk,
            "provider_id": 1,
            "start_time": start,
            "duration_minutes": 20,
            "reason": "Follow-up",
        },
    )
    assert resp.status_code == 201, resp.get_json()
    appointment_id = resp.get_json()["appointment_id"]

    # Check the patient in -> moves them onto the live queue
    resp = client.post(f"/appointments/api/check-in/{appointment_id}")
    assert resp.status_code == 200, resp.get_json()

    # The HTMX rows endpoint backing the Live Queue dashboard
    resp = client.get("/appointments/api/queue/1/rows")
    assert resp.status_code == 200
    assert f"Patient {patient_pk}" in resp.get_data(as_text=True)
    assert "Queue is empty" not in resp.get_data(as_text=True)

    # The JSON queue endpoint
    resp = client.get("/appointments/api/queue/1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["waiting_count"] == 1
    assert str(body["queue"][0]["patient_id"]) == str(patient_pk)


def test_double_booking_rejected(client):
    patient_pk = _patient_pk()
    start = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    payload = {
        "patient_id": patient_pk,
        "provider_id": 2,
        "start_time": start,
        "duration_minutes": 30,
    }
    first = client.post("/appointments/api/book", json=payload)
    assert first.status_code == 201

    second = client.post("/appointments/api/book", json=payload)
    assert second.status_code == 409


# ── Referrals & discharge ─────────────────────────────────────────────


def test_referral_lifecycle(client):
    patient_pk = _patient_pk()

    resp = client.post(
        "/referrals/api/initiate",
        json={
            "patient_id": patient_pk,
            "referring_facility": "Nakuru County Hospital",
            "receiving_facility": "Kenyatta National Hospital",
            "reason": "Specialist cardiology review",
            "clinical_summary": "Stable, referred for echo.",
        },
    )
    assert resp.status_code == 201, resp.get_json()
    referral_id = resp.get_json()["referral_id"]
    assert resp.get_json()["current_status"] == "PENDING"

    resp = client.patch(
        f"/referrals/api/status/{referral_id}", json={"status": "ACCEPTED"}
    )
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["current_status"] == "ACCEPTED"


def test_discharge_summary_generation(client):
    patient_pk = _patient_pk()
    admission = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()

    resp = client.post(
        "/referrals/api/discharge",
        json={
            "patient_id": patient_pk,
            "primary_diagnosis": "Malaria, uncomplicated",
            "admission_date": admission,
            "follow_up_instructions": "Return if fever persists beyond 48h.",
        },
    )
    assert resp.status_code == 201, resp.get_json()
    assert "summary_id" in resp.get_json()


# ── Consent check (the previously-missing /api/check route) ───────────


def test_consent_check_not_granted_by_default(client):
    patient_pk = _patient_pk()
    resp = client.get(
        f"/consent/api/check?patient_id={patient_pk}&consent_type=TREATMENT"
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "NOT_GRANTED"


def test_consent_check_active_after_grant(client):
    patient_pk = _patient_pk()
    with flask_app.app_context():
        grant_patient_consent(patient_id="P-WIRE-001", consent_type="TREATMENT")

    resp = client.get(
        f"/consent/api/check?patient_id={patient_pk}&consent_type=TREATMENT"
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ACTIVE"


def test_consent_check_missing_patient_404(client):
    resp = client.get("/consent/api/check?patient_id=999999&consent_type=TREATMENT")
    assert resp.status_code == 404


def test_consent_check_missing_params_400(client):
    resp = client.get("/consent/api/check")
    assert resp.status_code == 400
