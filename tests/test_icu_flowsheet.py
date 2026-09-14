"""
tests/test_icu_flowsheet.py
────────────────────────────
Unit and integration tests for ICU / HDU Flowsheet Workstation engine and endpoints.
"""

from departments.icu.icu_engine import (
    calculate_fluid_balance,
    calculate_gcs,
    calculate_map,
    generate_flowsheet_matrix,
)
from departments.models.icu import ICUFlowsheetEntry, ICUFluidBalance
from extensions import db


def test_calculate_map():
    """Test Mean Arterial Pressure calculation."""
    assert calculate_map(120, 80) == 93.3
    assert calculate_map(150, 90) == 110.0
    assert calculate_map(None, 80) is None
    assert calculate_map(100, 110) is None  # Invalid systolic < diastolic


def test_calculate_gcs():
    """Test Glasgow Coma Scale scoring and severity classification."""
    res_severe = calculate_gcs(2, 1, 3)
    assert res_severe["total"] == 6
    assert "Severe" in res_severe["severity"]

    res_mod = calculate_gcs(3, 3, 4)
    assert res_mod["total"] == 10
    assert "Moderate" in res_mod["severity"]

    res_mild = calculate_gcs(4, 5, 6)
    assert res_mild["total"] == 15
    assert "Mild" in res_mild["severity"]

    res_inc = calculate_gcs(None, 5, 6)
    assert res_inc["valid"] is False


def test_calculate_fluid_balance():
    """Test I/O fluid balance math and oliguria alert."""
    fb = calculate_fluid_balance(
        iv_fluids=1000,
        blood_products=250,
        enteral=500,
        medications=50,
        urine=300,
        drains=100,
        ng_emesis=50,
        stool=0,
        weight_kg=70.0,
        period_hours=1.0,
    )
    assert fb["total_input_ml"] == 1800.0
    assert fb["total_output_ml"] == 450.0
    assert fb["net_balance_ml"] == 1350.0
    assert fb["urine_rate_ml_kg_hr"] == 4.29
    assert fb["is_oliguria"] is False

    # Low urine output test (Oliguria trigger: < 0.5 mL/kg/hr)
    fb_low = calculate_fluid_balance(
        urine=20.0,
        weight_kg=70.0,
        period_hours=1.0,
    )
    assert fb_low["urine_rate_ml_kg_hr"] == 0.29
    assert fb_low["is_oliguria"] is True
    assert "Oliguria" in fb_low["oliguria_warning"]


def test_generate_flowsheet_matrix(app):
    """Test 24h flowsheet matrix generation."""
    with app.app_context():
        entry = ICUFlowsheetEntry(
            patient_id="PAT_ICU_01",
            nurse_id=1,
            heart_rate=95,
            bp_systolic=125,
            bp_diastolic=85,
            spo2=97,
            temperature=37.2,
            ventilator_mode="AC/VC",
            fio2=40.0,
            peep=5.0,
            gcs_eye=4,
            gcs_verbal=5,
            gcs_motor=6,
            gcs_total=15,
        )
        fluid = ICUFluidBalance(
            patient_id="PAT_ICU_01",
            nurse_id=1,
            iv_fluids_ml=500,
            total_input_ml=500,
            urine_output_ml=200,
            total_output_ml=200,
            net_balance_ml=300,
        )
        db.session.add_all([entry, fluid])
        db.session.commit()

        matrix = generate_flowsheet_matrix("PAT_ICU_01", hours=24)
        assert matrix["patient_id"] == "PAT_ICU_01"
        assert len(matrix["vitals_entries"]) >= 1
        assert len(matrix["fluid_entries"]) >= 1
        assert matrix["summary"]["net_24h_balance_ml"] == 300.0
        assert matrix["summary"]["latest_gcs"]["score"] == 15


def test_icu_routes(client, app):
    """Test ICU flowsheet HTTP routes and API endpoints."""
    # Test GET flowsheet UI page
    resp_ui = client.get("/icu/flowsheet/PAT_ICU_TEST")
    assert resp_ui.status_code == 200
    assert b"ICU / HDU Flowsheet Workstation" in resp_ui.data

    # Test POST log vitals
    resp_vitals = client.post(
        "/icu/flowsheet/PAT_ICU_TEST/vitals",
        json={
            "heart_rate": 88,
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "spo2": 98,
            "ventilator_mode": "SIMV",
            "gcs_eye": 4,
            "gcs_verbal": 4,
            "gcs_motor": 6,
        },
    )
    assert resp_vitals.status_code == 201
    v_data = resp_vitals.get_json()
    assert v_data["success"] is True
    assert v_data["map"] == 93.3
    assert v_data["gcs"] == 14

    # Test POST log fluid balance
    resp_fluid = client.post(
        "/icu/flowsheet/PAT_ICU_TEST/fluid",
        json={
            "iv_fluids": 1000,
            "urine": 400,
            "patient_weight_kg": 70.0,
        },
    )
    assert resp_fluid.status_code == 201
    f_data = resp_fluid.get_json()
    assert f_data["success"] is True
    assert f_data["net_balance_ml"] == 600.0

    # Test GET JSON API matrix
    resp_api = client.get("/icu/api/flowsheet/PAT_ICU_TEST")
    assert resp_api.status_code == 200
    matrix = resp_api.get_json()
    assert matrix["patient_id"] == "PAT_ICU_TEST"
    assert len(matrix["vitals_entries"]) >= 1
