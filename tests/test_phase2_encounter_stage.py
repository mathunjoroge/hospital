# Phase 2: Encounter stage machine + single visit-closure authority.
from departments.appointments.engine import ScheduleEngine
from departments.models.billing import (
    Invoice, InvoiceLineItem, InvoiceStatus, Payment, PaymentMethod,
)
from departments.models.encounter import Encounter
from departments.models.medicine import LabTest, RequestedLab
from departments.models.records import Patient, PatientWaitingList
from departments.models.user import User
from departments.shared import visit_closure
from departments.shared.queue_constants import QueueStatus
from extensions import db
from datetime import date
from werkzeug.security import generate_password_hash


def _patient(pid):
    p = Patient(patient_id=pid, name=f"Test {pid}", sex="F", date_of_birth=date(1990, 1, 1))
    db.session.add(p)
    db.session.add(PatientWaitingList(patient_id=pid, seen=QueueStatus.WAITING_TRIAGE))
    db.session.commit()
    return p


def test_stage_machine_legal_chain(app):
    enc = Encounter(patient_id="P1", encounter_type="OPD", status="ACTIVE")
    db.session.add(enc); db.session.commit()
    for stage in ["REGISTERED", "WAITING_DOCTOR", "IN_CONSULTATION",
                  "AWAITING_RESULTS", "IN_CONSULTATION", "AWAITING_PHARMACY",
                  "AWAITING_BILLING"]:
        assert enc.set_stage(stage) is True, stage
    assert enc.stage == "AWAITING_BILLING"


def test_stage_machine_refuses_illegal_jumps(app):
    enc = Encounter(patient_id="P1", encounter_type="OPD", status="ACTIVE",
                    stage="REGISTERED")
    db.session.add(enc); db.session.commit()
    assert enc.set_stage("DISCHARGED") is False
    assert enc.set_stage("AWAITING_PHARMACY") is False
    assert enc.stage == "REGISTERED"
    enc.close()
    assert enc.stage == "DISCHARGED"
    assert enc.set_stage("IN_CONSULTATION") is False


def test_walk_in_and_triage_set_stages(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    enc = visit_closure.active_encounter("P0001")
    assert enc.stage == "REGISTERED"
    ScheduleEngine().mark_triage_complete("P0001")
    assert enc.stage == "WAITING_DOCTOR"


def test_visit_not_closed_while_labs_pending(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    lt = LabTest(test_name="CBC", cost=500); db.session.add(lt); db.session.commit()
    db.session.add(RequestedLab(patient_id="P0001", lab_test_id=lt.id, status=0))
    db.session.commit()
    assert visit_closure.maybe_close_encounter("P0001") is False
    assert visit_closure.active_encounter("P0001").status == "ACTIVE"


def test_visit_closes_when_no_work_no_debt(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    assert visit_closure.maybe_close_encounter("P0001") is True
    enc = Encounter.query.filter_by(patient_id="P0001").first()
    assert enc.status == "DISCHARGED" and enc.ended_at is not None
    entry = PatientWaitingList.query.filter_by(patient_id="P0001").first()
    assert entry.seen == QueueStatus.DISCHARGED


def test_full_payment_closes_scoped_encounter(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    enc = visit_closure.active_encounter("P0001")
    inv = Invoice(patient_id="P0001", encounter_id=enc.id, status=InvoiceStatus.ISSUED)
    db.session.add(inv); db.session.commit()
    db.session.add(InvoiceLineItem(invoice_id=inv.id, description="Consult",
                                   category="consult", quantity=1,
                                   unit_price=1000, total=1000))
    db.session.commit()
    inv.recalculate()
    db.session.add(Payment(invoice_id=inv.id, patient_id="P0001", amount=1000,
                           method=PaymentMethod.CASH))
    db.session.commit()
    inv.recalculate()
    db.session.refresh(enc)
    assert inv.status == InvoiceStatus.PAID
    assert enc.status == "DISCHARGED"


def test_post_consult_charges_still_scope_to_open_encounter(app):
    from departments.billing.sync import get_or_create_open_invoice
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    enc = visit_closure.active_encounter("P0001")
    inv = get_or_create_open_invoice("P0001")
    assert inv.encounter_id == enc.id


def test_discharge_route_force_closes(app, client):
    with app.app_context():
        db.session.add(User(id=1, username="doc1",
                            password=generate_password_hash("password123"),
                            role="admin"))
        db.session.commit()
    client.post("/login", data={"username": "doc1", "password": "password123"},
                follow_redirects=True)
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    lt = LabTest(test_name="CBC", cost=500); db.session.add(lt); db.session.commit()
    db.session.add(RequestedLab(patient_id="P0001", lab_test_id=lt.id, status=0))
    db.session.commit()
    resp = client.post("/medicine/visit-discharge/P0001", follow_redirects=True)
    assert resp.status_code == 200
    enc = Encounter.query.filter_by(patient_id="P0001").first()
    assert enc.status == "DISCHARGED"