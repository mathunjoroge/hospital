# Phase 4: PatientWaitingList writes are retired. Encounter is the source of truth.
from datetime import date

from departments.appointments.engine import ScheduleEngine
from departments.models.encounter import Encounter
from departments.models.records import Patient, PatientWaitingList
from departments.shared.queue_constants import QueueStatus
from extensions import db


def _patient(pid):
    p = Patient(
        patient_id=pid, name=f"Test {pid}", sex="F",
        date_of_birth=date(1990, 1, 1),
    )
    db.session.add(p)
    db.session.commit()
    return p


def test_registration_does_not_create_waiting_list_row(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    # Phase 4: no PatientWaitingList row should be created
    assert PatientWaitingList.query.filter_by(patient_id="P0001").first() is None
    # But Encounter should exist
    enc = Encounter.query.filter_by(patient_id="P0001").first()
    assert enc is not None
    assert enc.stage == "REGISTERED"


def test_encounter_has_seen_property_for_backward_compat(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    enc = Encounter.query.filter_by(patient_id="P0001").first()
    # The .seen property should map stage to legacy QueueStatus integer
    assert enc.seen == QueueStatus.WAITING_TRIAGE
    ScheduleEngine().mark_triage_complete("P0001")
    assert enc.seen == QueueStatus.VITALS_DONE


def test_encounter_has_last_updated_property(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    enc = Encounter.query.filter_by(patient_id="P0001").first()
    # The .last_updated property should return started_at
    assert enc.last_updated == enc.started_at
