"""
tests/test_oncology_chemotherapy.py
───────────────────────────────────
Unit and integration tests for Oncology BSA calculation, chemotherapy protocol dosing,
cumulative toxicity cap enforcement, and builder endpoints.
"""

from departments.medicine.chemotherapy_engine import (
    calculate_bsa,
    calculate_regimen_doses,
)
from departments.models.oncology_models import ChemotherapyRegimenOrder
from extensions import db


def test_calculate_bsa():
    """Test Body Surface Area (BSA) Mosteller and DuBois formulas."""
    bsa_mosteller = calculate_bsa(170, 70, formula="mosteller")
    assert bsa_mosteller == 1.82

    bsa_dubois = calculate_bsa(170, 70, formula="dubois")
    assert bsa_dubois == 1.81

    # Invalid biometrics fallback
    assert calculate_bsa(0, 0) == 1.73


def test_calculate_regimen_doses_folfox():
    """Test FOLFOX6 protocol dose calculations."""
    res = calculate_regimen_doses("PAT_ONCO_01", "FOLFOX6", 170, 70)
    assert res["protocol_name"] == "FOLFOX6"
    assert res["bsa_m2"] == 1.82
    assert len(res["drugs"]) == 4

    oxali = [d for d in res["drugs"] if d["drug_name"] == "Oxaliplatin"][0]
    assert oxali["calculated_dose"] == round(1.82 * 85.0, 1)  # 154.7 mg
    assert oxali["cap_exceeded"] is False


def test_vincristine_single_dose_cap():
    """Test Vincristine 2.0 mg single dose cap enforcement in CHOP protocol."""
    # Large BSA (e.g. 200cm, 120kg -> BSA ~ 2.58 m2)
    # Vincristine 1.4 mg/m2 * 2.58 = 3.61 mg -> should be capped at 2.0 mg
    res = calculate_regimen_doses("PAT_ONCO_02", "CHOP", 200, 120)
    vinc = [d for d in res["drugs"] if d["drug_name"] == "Vincristine"][0]
    assert vinc["calculated_dose"] == 2.0
    assert "Capped" in vinc["dose_display"]


def test_doxorubicin_lifetime_toxicity_cap(app):
    """Test Doxorubicin 450 mg/m² cumulative lifetime toxicity cap alert."""
    with app.app_context():
        # Insert historical orders exceeding 450 mg/m2
        order1 = ChemotherapyRegimenOrder(
            patient_id="PAT_CARDIO_01",
            physician_id=1,
            protocol_name="AC-T",
            weight_kg=70.0,
            height_cm=170.0,
            bsa_m2=1.82,
            calculated_doses_json='[{"drug_name": "Doxorubicin", "calculated_dose": 800.0}]',
        )
        db.session.add(order1)
        db.session.commit()

        # Calculate new AC-T order
        res = calculate_regimen_doses("PAT_CARDIO_01", "AC-T", 170, 70)
        assert res["has_toxicity_warning"] is True
        assert any("Doxorubicin" in w for w in res["toxicity_warnings"])
        dox = [d for d in res["drugs"] if d["drug_name"] == "Doxorubicin"][0]
        assert dox["cap_exceeded"] is True


def test_chemo_builder_endpoints(client, app):
    """Test Chemotherapy Builder UI and API endpoints."""
    # Test GET Chemo Builder UI Page
    resp_ui = client.get("/medicine/oncology/chemo-builder/PAT_ONCO_TEST")
    assert resp_ui.status_code == 200
    assert b"Chemotherapy Protocol Builder" in resp_ui.data

    # Test GET calculate chemo API
    resp_calc = client.get("/medicine/oncology/api/calculate-chemo?patient_id=PAT_ONCO_TEST&protocol=AC-T&height=170&weight=70")
    assert resp_calc.status_code == 200
    calc_data = resp_calc.get_json()
    assert calc_data["protocol_name"] == "AC-T"
    assert calc_data["bsa_m2"] == 1.82

    # Test POST save chemo order API
    resp_save = client.post(
        "/medicine/oncology/api/save-chemo-order",
        json={
            "patient_id": "PAT_ONCO_TEST",
            "protocol_name": "ABVD",
            "height_cm": 175,
            "weight_kg": 75,
            "bsa_formula": "mosteller",
            "cycle_number": 1,
            "total_cycles": 6,
        },
    )
    assert resp_save.status_code == 201
    save_data = resp_save.get_json()
    assert save_data["success"] is True
    assert save_data["protocol_name"] == "ABVD"
