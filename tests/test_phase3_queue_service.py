# Phase 3: QueueService reads department queues from Encounter.stage.
from datetime import date

from departments.appointments.engine import ScheduleEngine
from departments.models.encounter import Encounter
from departments.models.records import Patient, PatientWaitingList
from departments.shared import queue_service
from departments.shared.queue_constants import QueueStatus
from extensions import db


def _patient(pid):
    p = Patient(
        patient_id=pid, name=f"Test {pid}", sex="F",
        date_of_birth=date(1990, 1, 1),
    )
    db.session.add(p)
    db.session.add(PatientWaitingList(patient_id=pid, seen=QueueStatus.WAITING_TRIAGE))
    db.session.commit()
    return p


def test_nursing_queue_shows_registered_only(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    out = queue_service.queue_for("nursing")
    assert len(out) == 1
    assert out[0].stage == "REGISTERED"
    assert out[0].patient.name == "Test P0001"


def test_nursing_queue_excludes_patients_past_triage(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    ScheduleEngine().mark_triage_complete("P0001")
    assert queue_service.queue_for("nursing") == []


def test_medicine_queue_includes_waiting_and_in_consult(app):
    _patient("P0001")
    _patient("P0002")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    ScheduleEngine().create_walk_in(patient_id="P0002")
    # Triage both patients so they move from REGISTERED to WAITING_DOCTOR
    ScheduleEngine().mark_triage_complete("P0001")
    ScheduleEngine().mark_triage_complete("P0002")
    enc1 = Encounter.query.filter_by(patient_id="P0001").first()
    enc1.set_stage("IN_CONSULTATION")
    db.session.commit()
    out = queue_service.queue_for("medicine")
    ids = {e.patient_id for e in out}
    # P0001 is IN_CONSULTATION, P0002 is WAITING_DOCTOR - both in medicine queue
    assert "P0001" in ids and "P0002" in ids


def test_medicine_queue_excludes_discharged(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    enc = Encounter.query.filter_by(patient_id="P0001").first()
    enc.close()
    db.session.commit()
    assert queue_service.queue_for("medicine") == []


def test_count_for_returns_correct_totals(app):
    _patient("P0001")
    _patient("P0002")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    ScheduleEngine().create_walk_in(patient_id="P0002")
    assert queue_service.count_for("nursing") == 2
    ScheduleEngine().mark_triage_complete("P0001")
    assert queue_service.count_for("nursing") == 1
    assert queue_service.count_for("medicine") == 1


def test_unknown_department_returns_empty(app):
    assert queue_service.queue_for("radiology") == []
    assert queue_service.count_for("radiology") == 0
