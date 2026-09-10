"""
tests/test_telemedicine.py
───────────────────────────
Phase E — Telemedicine & Virtual Consultation Engine Test Suite

Tests include feature-flag quarantine verification (ENABLE_TELEMEDICINE).
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


# ── Feature-flag quarantine tests ──────────────────────────────────


def test_telemedicine_blocked_when_flag_off(client, app, doctor_user):
    """All telemedicine routes return 403 when ENABLE_TELEMEDICINE is False (default)."""
    app.config["ENABLE_TELEMEDICINE"] = False
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )

    resp = client.post(
        "/telemedicine/session/create", json={"patient_id": "P-BLOCK-01"}
    )
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["code"] == "FEATURE_DISABLED"

    resp2 = client.get("/telemedicine/sessions")
    assert resp2.status_code == 403


# ── Feature-enabled tests ─────────────────────────────────────────


def test_create_telemedicine_session_success(client, app, doctor_user):
    """POST /telemedicine/session/create creates a session with SCHEDULED status."""
    app.config["ENABLE_TELEMEDICINE"] = True
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )
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


def test_create_telemedicine_session_missing_patient(client, app, doctor_user):
    """POST /telemedicine/session/create returns 400 if patient_id is absent."""
    app.config["ENABLE_TELEMEDICINE"] = True
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )
    resp = client.post("/telemedicine/session/create", json={})
    assert resp.status_code == 400


def test_session_lifecycle_and_notes(client, app, doctor_user):
    """Test start, notes update, and complete lifecycle endpoints."""
    app.config["ENABLE_TELEMEDICINE"] = True
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )

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
        json={
            "notes": "Patient presents with mild respiratory symptoms. Prescribed bed rest."
        },
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
    app.config["ENABLE_TELEMEDICINE"] = True
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )
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
    client.post(
        "/login",
        data={"username": unauthorized_user.username, "password": "Password123!"},
    )
    unauth_resp = client.get(f"/telemedicine/room/{session_uuid}")
    assert unauth_resp.status_code == 403


def test_list_sessions_endpoint(client, app, doctor_user):
    """GET /telemedicine/sessions lists the doctor's sessions."""
    app.config["ENABLE_TELEMEDICINE"] = True
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )
    client.post("/telemedicine/session/create", json={"patient_id": "P-TELE-104"})

    resp = client.get("/telemedicine/sessions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sessions" in data
    assert len(data["sessions"]) >= 1


# ── T3.1 Encounter lifecycle tests ───────────────────────────────

def test_start_session_creates_telehealth_encounter(client, app, doctor_user):
    """Starting a telemedicine session must create a TELEHEALTH encounter in IN_CONSULTATION."""
    app.config["ENABLE_TELEMEDICINE"] = True
    client.post("/login", data={"username": doctor_user.username, "password": "Password123!"})

    # Create + start
    c_resp = client.post("/telemedicine/session/create", json={"patient_id": "P-ENC-01"})
    session_uuid = c_resp.get_json()["session"]["session_uuid"]
    client.post(f"/telemedicine/session/{session_uuid}/start")

    with app.app_context():
        from departments.models.encounter import Encounter
        from departments.models.telemedicine import TelemedicineSession
        sess = TelemedicineSession.query.filter_by(session_uuid=session_uuid).first()
        assert sess.encounter_id is not None, "encounter_id must be set after start"
        enc = Encounter.query.get(sess.encounter_id)
        assert enc is not None
        assert enc.encounter_type == "TELEHEALTH"
        assert enc.stage == "IN_CONSULTATION"
        assert enc.status == "ACTIVE"
        assert enc.patient_id == "P-ENC-01"


def test_complete_session_discharges_encounter(client, app, doctor_user):
    """Completing a telemedicine session must close the linked encounter."""
    app.config["ENABLE_TELEMEDICINE"] = True
    client.post("/login", data={"username": doctor_user.username, "password": "Password123!"})

    c_resp = client.post("/telemedicine/session/create", json={"patient_id": "P-ENC-02"})
    session_uuid = c_resp.get_json()["session"]["session_uuid"]
    client.post(f"/telemedicine/session/{session_uuid}/start")
    client.post(f"/telemedicine/session/{session_uuid}/complete", json={"notes": "All good."})

    with app.app_context():
        from departments.models.encounter import Encounter
        from departments.models.telemedicine import TelemedicineSession
        sess = TelemedicineSession.query.filter_by(session_uuid=session_uuid).first()
        enc = Encounter.query.get(sess.encounter_id)
        assert enc.stage == "DISCHARGED"
        assert enc.status == "DISCHARGED"
        assert enc.ended_at is not None


def test_create_session_does_not_open_encounter(client, app, doctor_user):
    """Creating a session (SCHEDULED) must NOT create an encounter — only starting does."""
    app.config["ENABLE_TELEMEDICINE"] = True
    client.post("/login", data={"username": doctor_user.username, "password": "Password123!"})

    c_resp = client.post("/telemedicine/session/create", json={"patient_id": "P-ENC-03"})
    session_uuid = c_resp.get_json()["session"]["session_uuid"]

    with app.app_context():
        from departments.models.telemedicine import TelemedicineSession
        sess = TelemedicineSession.query.filter_by(session_uuid=session_uuid).first()
        assert sess.encounter_id is None, "Encounter must not exist until session starts"
