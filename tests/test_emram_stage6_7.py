"""
tests/test_emram_stage6_7.py
──────────────────────────────
Unit and Integration tests for Gap #4 — HIMSS EMRAM Stage 6–7 Enterprise Analytics & Closed-Loop Engine.
"""

from datetime import datetime, timezone

from departments.analytics.emram_engine import (
    ClosedLoopAuditEngine,
    EMRAMEngine,
    EMRAMStage6Metrics,
    EMRAMStage7Metrics,
)
from departments.models.encounter import Encounter
from departments.models.medicine import PrescribedMedicine
from departments.models.nursing import MedicationAdmin
from departments.models.records import Patient
from extensions import db


def test_emram_stage6_metrics_structure(app):
    """Test EMRAMStage6Metrics.evaluate() structure and scoring."""
    with app.app_context():
        res = EMRAMStage6Metrics.evaluate()
        assert "overall_score_pct" in res
        assert "status" in res
        assert res["status"] in ["STAGE_6_CERTIFIED", "INPROGRESS"]
        assert "pillars" in res
        assert "bcma_medication_closed_loop" in res["pillars"]
        assert "lims_specimen_closed_loop" in res["pillars"]
        assert "cpoe_cdss_safety_coverage" in res["pillars"]
        assert "nursing_vitals_closed_loop" in res["pillars"]
        assert "ccda_interoperability_summary" in res["pillars"]


def test_emram_stage7_metrics_structure(app):
    """Test EMRAMStage7Metrics.evaluate() structure and scoring."""
    with app.app_context():
        res = EMRAMStage7Metrics.evaluate()
        assert "overall_score_pct" in res
        assert "status" in res
        assert "pillars" in res
        assert "data_warehouse_etl_coverage" in res["pillars"]
        assert "population_health_analytics" in res["pillars"]
        assert "hie_ccda_fhir_interoperability" in res["pillars"]
        assert "cryptographic_audit_integrity" in res["pillars"]


def test_emram_master_scorecard(app):
    """Test EMRAMEngine.get_full_emram_scorecard()."""
    with app.app_context():
        scorecard = EMRAMEngine.get_full_emram_scorecard()
        assert "emram_level" in scorecard
        assert scorecard["emram_level"] in [5, 6, 7]
        assert "stage6" in scorecard
        assert "stage7" in scorecard
        assert "evaluated_at" in scorecard


def test_closed_loop_audit_timeline(app):
    """Test ClosedLoopAuditEngine.get_patient_closed_loop_timeline()."""
    with app.app_context():
        patient = Patient.query.filter_by(patient_id="PAT-EMRAM-999").first()
        if not patient:
            patient = Patient(
                patient_id="PAT-EMRAM-999",
                name="EMRAM Test Patient",
                sex="M",
                date_of_birth=datetime(1990, 1, 1).date(),
                place_of_residence="Nairobi",
                marital_status="Single",
                contact="0700000000",
                next_of_kin="Kin",
                relationship_with_next_of_kin="Sibling",
                next_of_kin_contact="0700000001",
                emergency_contact="0700000002",
            )
            db.session.add(patient)
            db.session.commit()

        # Add encounter
        enc = Encounter(
            patient_id=patient.patient_id,
            encounter_type="IPD",
            started_at=datetime.now(timezone.utc),
            status="ACTIVE",
        )
        db.session.add(enc)

        # Add prescription
        rx = PrescribedMedicine(
            prescription_id="RX-EMRAM-001",
            patient_id=patient.patient_id,
            medicine_id=1,
            dosage="500mg",
            strength="500mg",
            frequency="BD",
            num_days=3,
        )
        db.session.add(rx)

        # Add BCMA admin
        adm = MedicationAdmin(
            patient_id=patient.patient_id,
            medication="Amoxicillin 500mg",
            dosage="500mg",
            prescribed_medicine_id=rx.id,
            recorded_by=1,
            time_administered=datetime.now(timezone.utc),
            scan_verified=True,
        )
        db.session.add(adm)
        db.session.commit()

        timeline_res = ClosedLoopAuditEngine.get_patient_closed_loop_timeline(
            "PAT-EMRAM-999"
        )
        assert timeline_res["found"] is True
        assert timeline_res["patient_id"] == "PAT-EMRAM-999"
        assert len(timeline_res["timeline"]) >= 3


def test_emram_dashboard_ui_route(client, admin_user):
    """Test GET /analytics/emram-dashboard UI route."""
    resp = client.get("/analytics/emram-dashboard")
    assert resp.status_code == 200
    assert b"HIMSS EMRAM Stage" in resp.data
    assert b"Closed-Loop Clinical Audit Explorer" in resp.data


def test_emram_api_status_route(client, admin_user):
    """Test GET /analytics/api/emram-status API route."""
    resp = client.get("/analytics/api/emram-status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert "scorecard" in data
    assert "stage6" in data["scorecard"]


def test_closed_loop_trail_api_route(client, admin_user):
    """Test GET /analytics/api/closed-loop-trail/<patient_id> API route."""
    resp = client.get("/analytics/api/closed-loop-trail/PAT-EMRAM-999")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert "audit_trail" in data
