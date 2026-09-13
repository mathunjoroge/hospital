"""
tests/test_nicu_pediatrics_workstation.py
────────────────────────────────────────────
Unit & integration tests for Gap #10: NICU & Pediatrics Growth Charts / APGAR Workstation.
"""

from datetime import date

from departments.mch.nicu_pediatrics_engine import NicuPediatricsEngine
from departments.models.records import Patient
from extensions import db


def test_apgar_score_calculation(app):
    """Test APGAR score computation & risk tier categorization."""
    with app.app_context():
        # 1. Normal APGAR 9/10
        rec_normal = NicuPediatricsEngine.calculate_apgar_score(
            patient_id="PAT_NICU_01",
            time_interval="1_MIN",
            appearance=2,
            pulse=2,
            grimace=2,
            activity=1,
            respiration=2,
        )
        assert rec_normal.total_score == 9
        assert rec_normal.risk_category == "NORMAL"

        # 2. Moderate depression APGAR 5/10
        rec_mod = NicuPediatricsEngine.calculate_apgar_score(
            patient_id="PAT_NICU_01",
            time_interval="5_MIN",
            appearance=1,
            pulse=1,
            grimace=1,
            activity=1,
            respiration=1,
        )
        assert rec_mod.total_score == 5
        assert rec_mod.risk_category == "MODERATE_DEPRESSION"

        # 3. Severe depression APGAR 2/10
        rec_sev = NicuPediatricsEngine.calculate_apgar_score(
            patient_id="PAT_NICU_01",
            time_interval="10_MIN",
            appearance=0,
            pulse=1,
            grimace=0,
            activity=0,
            respiration=1,
        )
        assert rec_sev.total_score == 2
        assert rec_sev.risk_category == "SEVERE_DEPRESSION"


def test_phototherapy_risk_nomogram(app):
    """Test Bhutani Total Serum Bilirubin (TSB) phototherapy risk nomogram evaluation."""
    with app.app_context():
        # High Risk TSB (16.0 mg/dL at 36 hours)
        eval_high = NicuPediatricsEngine.evaluate_phototherapy_risk(
            patient_id="PAT_NICU_02",
            age_hours=36,
            serum_bilirubin_mg_dl=16.0,
            gestational_weeks=38,
        )
        assert eval_high.risk_zone == "HIGH_RISK"
        assert eval_high.phototherapy_indicated is True

        # Low Risk TSB (4.0 mg/dL at 48 hours)
        eval_low = NicuPediatricsEngine.evaluate_phototherapy_risk(
            patient_id="PAT_NICU_02",
            age_hours=48,
            serum_bilirubin_mg_dl=4.0,
            gestational_weeks=40,
        )
        assert eval_low.risk_zone == "LOW_RISK"
        assert eval_low.phototherapy_indicated is False


def test_pediatric_growth_zscore_calculation(app):
    """Test WHO Growth Z-score percentile logic & nutritional status categorization."""
    with app.app_context():
        # Normal 6-month-old infant (weight 7.8kg)
        growth_normal = NicuPediatricsEngine.calculate_growth_percentiles(
            patient_id="PAT_PED_01",
            age_months=6.0,
            weight_kg=7.8,
            height_cm=65.0,
            head_circumference_cm=42.0,
        )
        assert growth_normal.nutritional_status == "NORMAL"
        assert growth_normal.weight_for_age_zscore is not None

        # Severe Acute Malnutrition (weight 3.5kg at 6 months)
        growth_sam = NicuPediatricsEngine.calculate_growth_percentiles(
            patient_id="PAT_PED_01",
            age_months=6.0,
            weight_kg=3.5,
        )
        assert growth_sam.nutritional_status == "SEVERE_ACUTE_MALNUTRITION"
        assert growth_sam.weight_for_age_zscore < -3.0


def test_nicu_pediatrics_api_endpoints(client, app, admin_user):
    """Test HTTP API endpoints for APGAR, Phototherapy, and Growth Charts."""
    with app.app_context():
        pat = Patient(patient_id="PAT_NICU_API", name="Baby API Doe", sex="Male", date_of_birth=date(2026, 9, 1))
        db.session.add(pat)
        db.session.commit()

    # 1. Test POST /mch/api/apgar
    resp_apgar = client.post(
        "/mch/api/apgar",
        json={
            "patient_id": "PAT_NICU_API",
            "time_interval": "1_MIN",
            "appearance": 2,
            "pulse": 2,
            "grimace": 2,
            "activity": 2,
            "respiration": 1,
        },
    )
    assert resp_apgar.status_code == 201
    data_apgar = resp_apgar.get_json()
    assert data_apgar["total_score"] == 9
    assert data_apgar["risk_category"] == "NORMAL"

    # 2. Test GET /mch/api/apgar/<patient_id>
    resp_apgar_get = client.get("/mch/api/apgar/PAT_NICU_API")
    assert resp_apgar_get.status_code == 200
    assert resp_apgar_get.get_json()["count"] >= 1

    # 3. Test POST /mch/api/phototherapy
    resp_photo = client.post(
        "/mch/api/phototherapy",
        json={
            "patient_id": "PAT_NICU_API",
            "age_hours": 48,
            "serum_bilirubin_mg_dl": 14.5,
            "gestational_weeks": 37,
        },
    )
    assert resp_photo.status_code == 201
    data_photo = resp_photo.get_json()
    assert data_photo["phototherapy_indicated"] is True

    # 4. Test POST /mch/api/growth-chart
    resp_growth = client.post(
        "/mch/api/growth-chart",
        json={
            "patient_id": "PAT_NICU_API",
            "age_months": 1.0,
            "weight_kg": 4.1,
            "height_cm": 54.0,
        },
    )
    assert resp_growth.status_code == 201
    assert resp_growth.get_json()["status"] == "success"

    # 5. Test GET /mch/nicu-workstation/<patient_id>
    resp_ui = client.get("/mch/nicu-workstation/PAT_NICU_API")
    assert resp_ui.status_code == 200
    assert b"NICU &amp; Pediatrics Workstation" in resp_ui.data
