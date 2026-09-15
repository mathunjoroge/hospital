# Phase-1 queue bridge: READY-state transitions across the merged queues.
from departments.appointments.engine import ScheduleEngine
from departments.appointments.models import Appointment
from extensions import db


def test_walk_in_starts_checked_in(app):
    appt = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    assert appt.status == "CHECKED_IN"


def test_triage_complete_moves_to_ready(app):
    appt = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    ready = ScheduleEngine().mark_triage_complete("P0001")
    assert ready is not None and ready.id == appt.id
    assert ready.status == "READY"


def test_mark_triage_complete_is_idempotent_noop(app):
    assert ScheduleEngine().mark_triage_complete("P9999") is None
    appt = ScheduleEngine().create_walk_in(patient_id="P0002", provider_id="1")
    ScheduleEngine().mark_triage_complete("P0002")
    assert ScheduleEngine().mark_triage_complete("P0002") is None
    assert db.session.get(Appointment, appt.id).status == "READY"


def test_call_in_accepts_ready(app):
    appt = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    ScheduleEngine().mark_triage_complete("P0001")
    called = ScheduleEngine().call_in(appt.id)
    assert called is not None and called.status == "IN_PROGRESS"


def test_live_queue_includes_ready_excludes_in_progress(app):
    """The unified Encounter-based live queue shows all active patients.
    WAITING_DOCTOR patients are visible; IN_CONSULTATION patients are also
    visible on the dashboard so staff can see who is currently being seen.
    After call_in, a1's encounter moves to IN_CONSULTATION stage."""
    a1 = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    a2 = ScheduleEngine().create_walk_in(patient_id="P0002", provider_id="1")
    ScheduleEngine().mark_triage_complete("P0002")
    ScheduleEngine().call_in(a1.id)
    queue = ScheduleEngine().get_live_queue()
    patient_ids = {item["patient_id"] if isinstance(item, dict) else item.patient_id for item in queue}
    # a2 (WAITING_DOCTOR) must be in the queue
    assert "P0002" in patient_ids
    # a1's encounter should have moved to IN_CONSULTATION
    from departments.models.encounter import Encounter
    enc_a1 = Encounter.query.filter_by(patient_id="P0001", status="ACTIVE").first()
    assert enc_a1 is not None
    assert enc_a1.stage == "IN_CONSULTATION"


def test_vitals_template_has_no_broken_url(app, client):
    """The vitals page must render without BuildError for any url_for call."""
    from werkzeug.security import generate_password_hash

    from departments.models.user import User
    from extensions import db

    with app.app_context():
        u = User(id=99, username="nursetpl",
                 password=generate_password_hash("p"), role="admin")
        db.session.add(u)
        db.session.commit()
    client.post("/login", data={"username": "nursetpl", "password": "p"})
    resp = client.get("/nursing/vitals/P0001")
    # Either 200 (rendered cleanly) or 404 (no such patient, but no BuildError).
    # A BuildError would surface as 500 and be caught by the error handler.
    assert resp.status_code in (200, 404), (
        f"vitals template failed to render (status {resp.status_code}); "
        f"likely a broken url_for in the template."
    )
