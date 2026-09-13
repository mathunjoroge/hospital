"""
tests/test_theatre_operations_engine.py
────────────────────────────────────────
Unit and integration tests for Johns Hopkins-Grade Operating Theatre Operations,
WHO Surgical Safety Checklist gates, intraoperative vitals & fluid balance, PACU Aldrete readiness, and OR analytics (Gap #12).
"""

from datetime import date
import pytest
from werkzeug.security import generate_password_hash

from app import app
from extensions import db
from departments.models.user import User
from departments.models.records import Patient
from departments.models.medicine import TheatreList, TheatreProcedure
from departments.models.theatre import (
    AnaestheticRecord,
    PostOpNote,
    SurgicalInstrumentCount,
    WhoSurgicalChecklist,
)
from departments.theatre.theatre_engine import TheatreOperationsEngine


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client


def _make_theatre_setup(app) -> int:
    """Helper to create patient, procedure, and theatre entry."""
    with app.app_context():
        p = Patient.query.filter_by(patient_id="OR-PAT-001").first()
        if not p:
            p = Patient(
                patient_id="OR-PAT-001",
                name="Surgical Patient 1",
                sex="Female",
                date_of_birth=date(1992, 5, 15),
                emergency_contact="0711111111",
            )
            db.session.add(p)

        proc = TheatreProcedure.query.filter_by(name="Laparoscopic Appendectomy").first()
        if not proc:
            proc = TheatreProcedure(name="Laparoscopic Appendectomy", cost=45000.0)
            db.session.add(proc)
        db.session.commit()

        entry = TheatreList(
            patient_id="OR-PAT-001",
            procedure_id=proc.id,
            status=0,
        )
        db.session.add(entry)
        db.session.commit()
        return entry.id


def test_intraop_vitals_streaming(app):
    """Test streaming intraoperative vital sign snapshots into AnaestheticRecord."""
    entry_id = _make_theatre_setup(app)
    with app.app_context():
        record = TheatreOperationsEngine.record_intraop_vitals(
            entry_id=entry_id, hr=75, bp_systolic=120, bp_diastolic=80, spo2=99, etco2=35, agent_concentration=1.8
        )
        assert record.id is not None
        assert len(record.vitals_series) == 1
        snapshot = record.vitals_series[0]
        assert snapshot["hr"] == 75
        assert snapshot["bp_sys"] == 120
        assert snapshot["spo2"] == 99


def test_fluid_balance_calculation(app):
    """Test intraoperative fluid balance arithmetic (Intake - Loss)."""
    entry_id = _make_theatre_setup(app)
    with app.app_context():
        record = AnaestheticRecord(
            theatre_entry_id=entry_id,
            patient_id="OR-PAT-001",
            crystalloids_ml=1500,
            colloids_ml=500,
            blood_products_ml=250,
            estimated_blood_loss_ml=300,
            urine_output_ml=450,
        )
        db.session.add(record)
        db.session.commit()

        balance = TheatreOperationsEngine.calculate_fluid_balance(entry_id)
        assert balance["total_intake_ml"] == 2250
        assert balance["total_loss_ml"] == 750
        assert balance["net_balance_ml"] == 1500


def test_who_checklist_gate_enforcement(app):
    """Test WHO checklist gates for INTRA_OP and POST_OP stage transitions."""
    entry_id = _make_theatre_setup(app)
    with app.app_context():
        # Before Sign-In -> Gate fails
        passed, msg = TheatreOperationsEngine.evaluate_who_checklist_gate(entry_id, "INTRA_OP")
        assert passed is False
        assert "Sign In" in msg

        # Complete Sign-In -> Gate passes
        chk = WhoSurgicalChecklist(theatre_entry_id=entry_id, patient_id="OR-PAT-001", sign_in_completed=True)
        db.session.add(chk)
        db.session.commit()

        passed2, msg2 = TheatreOperationsEngine.evaluate_who_checklist_gate(entry_id, "INTRA_OP")
        assert passed2 is True

        # Before Sign-Out / Instrument count -> POST_OP gate fails
        passed3, msg3 = TheatreOperationsEngine.evaluate_who_checklist_gate(entry_id, "POST_OP")
        assert passed3 is False

        # Complete Sign-Out and Instrument count reconciliation
        chk.sign_out_completed = True
        cnt = SurgicalInstrumentCount(
            theatre_entry_id=entry_id,
            sponges_initial=10, sponges_closing_skin=10,
            needles_initial=5, needles_closing_skin=5,
            instruments_initial=20, instruments_closing_skin=20,
        )
        cnt.calculate_reconciliation()
        db.session.add(cnt)
        db.session.commit()

        passed4, msg4 = TheatreOperationsEngine.evaluate_who_checklist_gate(entry_id, "POST_OP")
        assert passed4 is True


def test_pacu_aldrete_discharge_readiness(app):
    """Test PACU Aldrete Recovery Score evaluation threshold (>= 9)."""
    entry_id = _make_theatre_setup(app)
    with app.app_context():
        note = PostOpNote(
            theatre_entry_id=entry_id,
            patient_id="OR-PAT-001",
            preop_diagnosis="Appendicitis",
            postop_diagnosis="Acute Appendicitis",
            procedure_performed="Laparoscopic Appendectomy",
            surgical_findings="Inflamed appendix removed",
            aldrete_activity=2,
            aldrete_respiration=2,
            aldrete_circulation=2,
            aldrete_consciousness=2,
            aldrete_spo2=2,
        )
        db.session.add(note)
        db.session.commit()

        readiness = TheatreOperationsEngine.evaluate_pacu_discharge_readiness(entry_id)
        assert readiness["score"] == 10
        assert readiness["is_fit_for_discharge"] is True

        # Lower score below threshold
        note.aldrete_consciousness = 0
        db.session.commit()

        readiness2 = TheatreOperationsEngine.evaluate_pacu_discharge_readiness(entry_id)
        assert readiness2["score"] == 8
        assert readiness2["is_fit_for_discharge"] is False


def test_or_dashboard_metrics(app):
    """Test OR dashboard metrics calculation."""
    _make_theatre_setup(app)
    with app.app_context():
        metrics = TheatreOperationsEngine.get_or_dashboard_metrics()
        assert metrics["total_cases"] >= 1
        assert "who_compliance_pct" in metrics
        assert "or_cases" in metrics


def test_theatre_routes_api_integration(client):
    """Test Theatre API endpoints for vitals, fluid balance, PACU readiness, OR dashboard, and metrics."""
    with app.app_context():
        entry_id = _make_theatre_setup(app)
        user = db.session.query(User).filter_by(username="theatre_admin_test").first()
        if not user:
            user = User(
                username="theatre_admin_test",
                password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
                role="admin",
            )
            db.session.add(user)
            db.session.commit()
        user_id = user.id

    with client.session_transaction() as sess:
        sess["_user_id"] = str(user_id)
        sess["_fresh"] = True

    # 1. Post Vitals API
    res1 = client.post(f"/theatre/api/vitals/{entry_id}", json={
        "hr": 82, "bp_sys": 118, "bp_dia": 76, "spo2": 98, "etco2": 36, "agent_conc": 1.5
    })
    assert res1.status_code == 200
    assert res1.get_json()["status"] == "success"

    # 2. Get Fluid Balance API
    res2 = client.get(f"/theatre/api/fluid-balance/{entry_id}")
    assert res2.status_code == 200
    assert "net_balance_ml" in res2.get_json()

    # 3. Get PACU Readiness API
    res3 = client.get(f"/theatre/api/pacu-readiness/{entry_id}")
    assert res3.status_code == 200
    assert "is_fit_for_discharge" in res3.get_json()

    # 4. Get OR Dashboard HTML
    res4 = client.get("/theatre/dashboard")
    assert res4.status_code == 200
    assert b"Operating Theatre &amp; OR Flow Console" in res4.data

    # 5. Get OR Metrics JSON API
    res5 = client.get("/theatre/api/metrics")
    assert res5.status_code == 200
    assert res5.get_json()["total_cases"] >= 1

