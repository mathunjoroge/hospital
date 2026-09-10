"""
tests/test_theatre_booking_stages.py
──────────────────────────────────────
T3.2 — Theatre Booking Stages:
  - Booking a patient to theatre creates Encounter(type=SURGICAL, stage=PRE_OP)
  - Recording post-op notes advances PRE_OP -> INTRA_OP -> POST_OP -> DISCHARGED
  - Stage machine enforces PRE_OP -> INTRA_OP (illegal skips are refused)
  - TheatreList.encounter_id is set at booking time
"""
from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from departments.models.encounter import Encounter
from departments.models.medicine import TheatreList, TheatreProcedure
from departments.models.records import Patient
from departments.models.user import User
from extensions import db


@pytest.fixture
def doctor(app):
    with app.app_context():
        u = User(
            username="theatre_doc",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="doctor",
        )
        db.session.add(u)
        db.session.commit()
        yield u


def _patient(pid: str) -> Patient:
    p = Patient(patient_id=pid, name=f"Test {pid}", sex="M",
                date_of_birth=date(1980, 5, 15))
    db.session.add(p)
    db.session.commit()
    return p


def _procedure(name: str = "Appendectomy") -> TheatreProcedure:
    proc = TheatreProcedure(name=name, type="General", cost=5000.0)
    db.session.add(proc)
    db.session.commit()
    return proc


class TestSurgicalStageMachine:
    def test_pre_op_to_intra_op_allowed(self, app):
        """PRE_OP -> INTRA_OP is a legal transition."""
        with app.app_context():
            _patient("P-TH-SM-01")
            enc = Encounter(patient_id="P-TH-SM-01", encounter_type="SURGICAL",
                            stage="PRE_OP", status="ACTIVE")
            db.session.add(enc)
            db.session.commit()
            assert enc.set_stage("INTRA_OP") is True
            assert enc.stage == "INTRA_OP"

    def test_intra_op_to_post_op_allowed(self, app):
        """INTRA_OP -> POST_OP is a legal transition."""
        with app.app_context():
            _patient("P-TH-SM-02")
            enc = Encounter(patient_id="P-TH-SM-02", encounter_type="SURGICAL",
                            stage="INTRA_OP", status="ACTIVE")
            db.session.add(enc)
            db.session.commit()
            assert enc.set_stage("POST_OP") is True
            assert enc.stage == "POST_OP"

    def test_post_op_to_discharged_allowed(self, app):
        """POST_OP -> DISCHARGED is legal (via close())."""
        with app.app_context():
            _patient("P-TH-SM-03")
            enc = Encounter(patient_id="P-TH-SM-03", encounter_type="SURGICAL",
                            stage="POST_OP", status="ACTIVE")
            db.session.add(enc)
            db.session.commit()
            enc.close()
            assert enc.stage == "DISCHARGED"
            assert enc.status == "DISCHARGED"
            assert enc.ended_at is not None

    def test_pre_op_to_post_op_illegal(self, app):
        """Skipping INTRA_OP (PRE_OP -> POST_OP directly) is refused."""
        with app.app_context():
            _patient("P-TH-SM-04")
            enc = Encounter(patient_id="P-TH-SM-04", encounter_type="SURGICAL",
                            stage="PRE_OP", status="ACTIVE")
            db.session.add(enc)
            db.session.commit()
            assert enc.set_stage("POST_OP") is False
            assert enc.stage == "PRE_OP"

    def test_full_surgical_lifecycle(self, app):
        """Walk the full PRE_OP -> INTRA_OP -> POST_OP -> DISCHARGED chain."""
        with app.app_context():
            _patient("P-TH-SM-07")
            enc = Encounter(patient_id="P-TH-SM-07", encounter_type="SURGICAL",
                            stage="PRE_OP", status="ACTIVE")
            db.session.add(enc)
            db.session.commit()
            assert enc.set_stage("INTRA_OP") is True
            assert enc.set_stage("POST_OP") is True
            enc.close()
            assert enc.stage == "DISCHARGED"
            assert enc.status == "DISCHARGED"


class TestTheatreBookingRoute:
    def test_booking_creates_surgical_encounter_at_pre_op(self, client, app, doctor):
        """POST /medicine/add-to-theatre creates a SURGICAL encounter in PRE_OP."""
        client.post("/login", data={"username": doctor.username, "password": "Password123!"})
        with app.app_context():
            _patient("P-TH-R-01")
            proc = _procedure("Appendectomy")
        resp = client.post("/medicine/add-to-theatre", data={
            "patient_id": "P-TH-R-01",
            "procedure_id": str(proc.id),
            "created_by": str(doctor.id),
            "notes_on_book": "Routine appendectomy",
        })
        assert resp.status_code in (200, 302)
        with app.app_context():
            entry = TheatreList.query.filter_by(patient_id="P-TH-R-01").first()
            assert entry is not None, "TheatreList entry not created"
            assert entry.encounter_id is not None, "encounter_id must be set"
            enc = Encounter.query.get(entry.encounter_id)
            assert enc is not None
            assert enc.encounter_type == "SURGICAL"
            assert enc.stage == "PRE_OP"
            assert enc.status == "ACTIVE"

    def test_post_op_notes_advances_to_discharged(self, client, app, doctor):
        """POST /medicine/update-post-op/<id> advances PRE_OP -> INTRA_OP -> POST_OP -> DISCHARGED."""
        client.post("/login", data={"username": doctor.username, "password": "Password123!"})
        with app.app_context():
            _patient("P-TH-R-02")
            proc = _procedure("Cholecystectomy")
        client.post("/medicine/add-to-theatre", data={
            "patient_id": "P-TH-R-02",
            "procedure_id": str(proc.id),
            "created_by": str(doctor.id),
        })
        with app.app_context():
            entry = TheatreList.query.filter_by(patient_id="P-TH-R-02").first()
            entry_id = entry.id
            enc_id = entry.encounter_id
        resp = client.post(f"/medicine/update-post-op/{entry_id}", data={
            "notes_on_post_op": "Procedure completed without complications.",
        })
        assert resp.status_code in (200, 302)
        with app.app_context():
            enc = Encounter.query.get(enc_id)
            assert enc.stage == "DISCHARGED", f"Expected DISCHARGED, got {enc.stage}"
            assert enc.status == "DISCHARGED"
            assert enc.ended_at is not None

    def test_transition_route_moves_pre_op_to_intra_op(self, client, app, doctor):
        """POST /medicine/theatre-transition/<id>/INTRA_OP advances PRE_OP -> INTRA_OP."""
        client.post("/login", data={"username": doctor.username, "password": "Password123!"})
        with app.app_context():
            _patient("P-TH-R-05")
            proc = _procedure("Hernia Repair")
        client.post("/medicine/add-to-theatre", data={
            "patient_id": "P-TH-R-05",
            "procedure_id": str(proc.id),
            "created_by": str(doctor.id),
        })
        with app.app_context():
            entry = TheatreList.query.filter_by(patient_id="P-TH-R-05").first()
            entry_id = entry.id
        resp = client.post(f"/medicine/theatre-transition/{entry_id}/INTRA_OP")
        assert resp.status_code in (200, 302)
        with app.app_context():
            enc = TheatreList.query.get(entry_id).encounter
            assert enc.stage == "INTRA_OP"
