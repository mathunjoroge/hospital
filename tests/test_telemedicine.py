"""
tests/test_telemedicine.py
───────────────────────────
Phase E — Telemedicine & Virtual Consultation Engine Test Suite
"""

import pytest
from werkzeug.security import generate_password_hash

from departments.models.user import User
from extensions import db


@pytest.fixture
def doctor_user(app):
    with app.app_context():
        u = User(
            username="teledoc_001",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="medicine",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def unauthorized_user(app):
    with app.app_context():
        u = User(
            username="records_clerk_001",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="records",
        )
        db.session.add(u)
        db.session.commit()
        yield u


def test_create_telemedicine_session_success(client, doctor_user):
    """POST /telemedicine/session/create creates a session with SCHEDULED status."""
    client.post("/login", data={"username": doctor_user.username, "password": "Password123!"})
    resp = client.post(
        "/telemedicine/session/create",
        json={"patient_id": "P-TELE-101", "appointment_id": 42},
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert "session" in data
    assert data["session"]["status"] == "SCHEDULED"
    assert data["session"]["patient_id"] == "P-TELE-101"
    assert data["session"]["room_token"].startswith("room_")


def test_create_telemedicine_session_missing_patient(client, doctor_user):
    """POST /telemedicine/session/create returns 400 if patient_id is absent."""
    client.post("/login", data={"username": doctor_user.username, "password": "Password123!"})
    resp = client.post("/telemedicine/session/create", json={})
    assert resp.status_code == 400


def test_session_lifecycle_and_notes(client, app, doctor_user):
    """Test start, notes update, and complete lifecycle endpoints."""
    client.post("/login", data={"username": doctor_user.username, "password": "Password123!"})

    # 1. Create
    c_resp = client.post(
        "/telemedicine/session/create",
        json={"patient_id": "P-TELE-102"},
    )
    session_uuid = c_resp.get_json()["session"]["session_uuid"]

    # 2. Start
    s_resp = client.post(f"/telemedicine/session/{session_uuid}/start")
    assert s_resp.status_code == 200
    assert s_resp.get_json()["session"]["status"] == "ACTIVE"

    # 3. Notes
    n_resp = client.post(
        f"/telemedicine/session/{session_uuid}/notes",
        json={"notes": "Patient presents with mild respiratory symptoms. Prescribed bed rest."},
    )
    assert n_resp.status_code == 200
    assert "respiratory" in n_resp.get_json()["notes"]

    # 4. Complete
    comp_resp = client.post(
        f"/telemedicine/session/{session_uuid}/complete",
        json={"notes": "Finalized consultation."},
    )
    assert comp_resp.status_code == 200
    assert comp_resp.get_json()["session"]["status"] == "COMPLETED"


def test_room_ui_access_control(client, app, doctor_user, unauthorized_user):
    """GET /telemedicine/room/<uuid> returns 200 for participant, 403 for unauthorized."""
    client.post("/login", data={"username": doctor_user.username, "password": "Password123!"})
    c_resp = client.post(
        "/telemedicine/session/create",
        json={"patient_id": "P-TELE-103"},
    )
    session_uuid = c_resp.get_json()["session"]["session_uuid"]

    # Doctor access -> 200
    doc_resp = client.get(f"/telemedicine/room/{session_uuid}")
    assert doc_resp.status_code == 200
    assert b"Virtual Telemedicine Consultation" in doc_resp.data

    # Unauthorized access -> 403
    client.post("/login", data={"username": unauthorized_user.username, "password": "Password123!"})
    unauth_resp = client.get(f"/telemedicine/room/{session_uuid}")
    assert unauth_resp.status_code == 403


def test_list_sessions_endpoint(client, doctor_user):
    """GET /telemedicine/sessions lists the doctor's sessions."""
    client.post("/login", data={"username": doctor_user.username, "password": "Password123!"})
    client.post("/telemedicine/session/create", json={"patient_id": "P-TELE-104"})

    resp = client.get("/telemedicine/sessions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sessions" in data
    assert len(data["sessions"]) >= 1
