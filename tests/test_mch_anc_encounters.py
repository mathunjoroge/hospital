"""
tests/test_mch_anc_encounters.py
──────────────────────────────────
T3.4 — MCH ANC Encounter Lifecycle:
  - ANC visit creates an ANC encounter in IN_CONSULTATION
  - Immunizations recorded during the visit attach to the encounter
  - Closing the visit discharges the encounter
"""
from datetime import datetime

import pytest
from werkzeug.security import generate_password_hash

from departments.mch.engine import MchEngine
from departments.mch.models import AncVisit, ImmunizationRecord
from departments.models.encounter import Encounter
from departments.models.records import Patient, PatientWaitingList
from departments.models.user import User
from departments.shared.queue_constants import QueueStatus
from extensions import db


# ── fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def nursing_user(app):
    with app.app_context():
        u = User(
            username="mch_nurse_001",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="nursing",
        )
        db.session.add(u)
        db.session.commit()
        yield u


def _patient(patient_id: str):
    p = Patient(
        patient_id=patient_id,
        name=f"Test {patient_id}",
        sex="F",
        date_of_birth=datetime(1995, 4, 20),
    )
    db.session.add(p)
    db.session.add(PatientWaitingList(patient_id=patient_id, seen=QueueStatus.WAITING_TRIAGE))
    db.session.commit()
    return p


engine = MchEngine()


# ── T3.4 unit tests (engine layer) ────────────────────────────────


def test_anc_visit_creates_encounter(app):
    """log_anc_visit must open an ANC encounter in IN_CONSULTATION."""
    with app.app_context():
        _patient("P-ANC-01")
        visit = engine.log_anc_visit(
            patient_id="P-ANC-01",
            visit_number=1,
            gestation_weeks=12,
        )
        assert visit.encounter_id is not None, "encounter_id must be set on AncVisit"

        enc = Encounter.query.get(visit.encounter_id)
        assert enc is not None
        assert enc.encounter_type == "ANC"
        assert enc.stage == "IN_CONSULTATION"
        assert enc.status == "ACTIVE"
        assert enc.patient_id == "P-ANC-01"


def test_anc_encounter_chief_complaint_contains_visit_number(app):
    """The ANC encounter chief_complaint should capture visit number and gestation."""
    with app.app_context():
        _patient("P-ANC-02")
        visit = engine.log_anc_visit(
            patient_id="P-ANC-02",
            visit_number=3,
            gestation_weeks=28,
        )
        enc = Encounter.query.get(visit.encounter_id)
        assert "3" in enc.chief_complaint
        assert "28" in enc.chief_complaint


def test_immunization_attaches_to_active_anc_encounter(app):
    """Immunizations recorded while an ANC encounter is open must get the encounter_id."""
    with app.app_context():
        _patient("P-ANC-03")
        # Open ANC encounter via visit
        visit = engine.log_anc_visit(
            patient_id="P-ANC-03",
            visit_number=1,
            gestation_weeks=10,
        )
        # Record an immunization
        record = engine.record_immunization(
            child_patient_id="P-ANC-03",
            vaccine_name="BCG",
            dose_number=1,
        )
        assert record.encounter_id == visit.encounter_id, (
            "Immunization must be scoped to the open ANC encounter"
        )


def test_close_anc_visit_discharges_encounter(app):
    """close_anc_visit must discharge the linked encounter."""
    with app.app_context():
        _patient("P-ANC-04")
        visit = engine.log_anc_visit(
            patient_id="P-ANC-04",
            visit_number=2,
            gestation_weeks=24,
        )
        enc_id = visit.encounter_id

        engine.close_anc_visit(visit.id)

        enc = Encounter.query.get(enc_id)
        assert enc.stage == "DISCHARGED"
        assert enc.status == "DISCHARGED"
        assert enc.ended_at is not None


def test_immunization_without_open_encounter_has_null_encounter_id(app):
    """If no active encounter exists, immunization.encounter_id is None (no crash)."""
    with app.app_context():
        _patient("P-ANC-05")
        # No ANC visit opened — no active encounter
        record = engine.record_immunization(
            child_patient_id="P-ANC-05",
            vaccine_name="BCG",
            dose_number=1,
        )
        assert record.encounter_id is None


def test_multiple_anc_visits_get_separate_encounters(app):
    """Each ANC visit should create a distinct encounter."""
    with app.app_context():
        _patient("P-ANC-06")

        # Close the first visit before logging the second so it doesn't count as active
        v1 = engine.log_anc_visit("P-ANC-06", visit_number=1, gestation_weeks=14)
        engine.close_anc_visit(v1.id)

        v2 = engine.log_anc_visit("P-ANC-06", visit_number=2, gestation_weeks=22)

        assert v1.encounter_id != v2.encounter_id, (
            "Each ANC visit must have a distinct encounter"
        )
        enc1 = Encounter.query.get(v1.encounter_id)
        enc2 = Encounter.query.get(v2.encounter_id)
        assert enc1.stage == "DISCHARGED"
        assert enc2.stage == "IN_CONSULTATION"


# ── T3.4 API route tests ───────────────────────────────────────────


def test_anc_visit_api_creates_encounter(client, app, nursing_user):
    """POST /mch/api/anc-visit creates an ANC visit with a linked encounter."""
    client.post("/login", data={"username": nursing_user.username, "password": "Password123!"})

    with app.app_context():
        _patient("P-ANC-API-01")

    resp = client.post(
        "/mch/api/anc-visit",
        json={"patient_id": "P-ANC-API-01", "visit_number": 1, "gestation_weeks": 16},
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["status"] == "success"

    with app.app_context():
        visit = AncVisit.query.filter_by(patient_id="P-ANC-API-01").first()
        assert visit is not None
        assert visit.encounter_id is not None
        enc = Encounter.query.get(visit.encounter_id)
        assert enc.encounter_type == "ANC"


def test_close_anc_visit_api(client, app, nursing_user):
    """POST /mch/api/anc-visit/<id>/close discharges the encounter."""
    client.post("/login", data={"username": nursing_user.username, "password": "Password123!"})

    with app.app_context():
        _patient("P-ANC-API-02")
        visit = engine.log_anc_visit("P-ANC-API-02", visit_number=1, gestation_weeks=20)
        visit_id = visit.id
        enc_id = visit.encounter_id

    resp = client.post(f"/mch/api/anc-visit/{visit_id}/close")
    assert resp.status_code == 200

    with app.app_context():
        enc = Encounter.query.get(enc_id)
        assert enc.stage == "DISCHARGED"
