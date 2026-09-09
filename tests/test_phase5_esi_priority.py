# Phase 5: ESI acuity drives queue priority.
from datetime import date

from departments.appointments.engine import ScheduleEngine
from departments.models.encounter import Encounter
from departments.models.records import Patient
from departments.shared import queue_service
from extensions import db


def _patient(pid):
    p = Patient(
        patient_id=pid, name=f'Test {pid}', sex='F',
        date_of_birth=date(1990, 1, 1),
    )
    db.session.add(p)
    db.session.commit()
    return p


def _triage(pid, esi):
    enc = Encounter.query.filter_by(patient_id=pid).first()
    enc.set_stage('WAITING_DOCTOR')
    enc.esi_level = esi


def test_esi_1_jumps_queue_over_earlier_arrivals(app):
    _patient('P0001')
    ScheduleEngine().create_walk_in(patient_id='P0001')
    _patient('P0002')
    ScheduleEngine().create_walk_in(patient_id='P0002')
    _triage('P0001', 4)
    _triage('P0002', 1)
    db.session.commit()

    ids = [e.patient_id for e in queue_service.queue_for('medicine')]
    assert ids[0] == 'P0002', f'ESI 1 should be first, got {ids}'
    assert ids[1] == 'P0001'


def test_esi_2_jumps_queue_over_esi_3(app):
    _patient('P0001')
    ScheduleEngine().create_walk_in(patient_id='P0001')
    _patient('P0002')
    ScheduleEngine().create_walk_in(patient_id='P0002')
    _triage('P0001', 3)  # Arrived first, non-emergent
    _triage('P0002', 2)  # Arrived second, emergent
    db.session.commit()

    ids = [e.patient_id for e in queue_service.queue_for('medicine')]
    assert ids == ['P0002', 'P0001'], f'ESI 2 should jump ESI 3, got {ids}'


def test_same_acuity_is_fifo(app):
    _patient('P0001')
    ScheduleEngine().create_walk_in(patient_id='P0001')
    _patient('P0002')
    ScheduleEngine().create_walk_in(patient_id='P0002')
    _triage('P0001', 3)
    _triage('P0002', 3)
    db.session.commit()

    ids = [e.patient_id for e in queue_service.queue_for('medicine')]
    assert ids == ['P0001', 'P0002'], f'Same acuity should be FIFO, got {ids}'
