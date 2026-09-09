# Phase-1 queue bridge: READY-state transitions across the merged queues.
from departments.appointments.engine import ScheduleEngine
from departments.appointments.models import Appointment


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
    assert Appointment.query.get(appt.id).status == "READY"


def test_call_in_accepts_ready(app):
    appt = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    ScheduleEngine().mark_triage_complete("P0001")
    called = ScheduleEngine().call_in(appt.id)
    assert called is not None and called.status == "IN_PROGRESS"


def test_live_queue_includes_ready_excludes_in_progress(app):
    a1 = ScheduleEngine().create_walk_in(patient_id="P0001", provider_id="1")
    a2 = ScheduleEngine().create_walk_in(patient_id="P0002", provider_id="1")
    ScheduleEngine().mark_triage_complete("P0002")
    ScheduleEngine().call_in(a1.id)
    ids = {a.id for a in ScheduleEngine().get_live_queue()}
    assert a2.id in ids and a1.id not in ids

def test_vitals_template_has_no_broken_url(app, client):
    """The vitals page must render without BuildError for any url_for call."""
    from departments.models.user import User
    from werkzeug.security import generate_password_hash
    with app.app_context():
        u = User(id=99, username="nursetpl",
                 password=generate_password_hash("p"), role="admin")
        from extensions import db; db.session.add(u); db.session.commit()
        client.post("/login", data={"username":"nursetpl","password":"p"})
        resp = client.get("/nursing/vitals/P0001")
        # Either 200 (rendered cleanly) or 404 (no such patient, but no BuildError).
        # A BuildError would surface as 500 and be caught by the error handler.
        assert resp.status_code in (200, 404), (
            f"vitals template failed to render (status {resp.status_code}); "
            f"likely a broken url_for in the template."
        )    
