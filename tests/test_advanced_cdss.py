"""
tests/test_advanced_cdss.py
────────────────────────────
Unit and integration tests for Johns Hopkins-Grade Advanced CDSS Engine.
"""

import pytest

from departments.clinical_safety.cdss_advanced import (
    AlertFatigueManager,
    HepaticDosingEngine,
    PediatricDosingEngine,
    PregnancySafetyEngine,
    RenalDosingEngine,
    calculate_crcl,
    calculate_egfr,
)
from departments.clinical_safety.engine import ClinicalSafetyEngine


def test_egfr_and_crcl_calculation():
    """Test eGFR (CKD-EPI 2021) and CrCl (Cockcroft-Gault) formulas."""
    egfr_normal = calculate_egfr(creatinine_mg_dl=0.9, age_years=40, is_female=False)
    assert egfr_normal > 80.0

    egfr_impairment = calculate_egfr(creatinine_mg_dl=2.5, age_years=65, is_female=True)
    assert egfr_impairment < 30.0

    crcl = calculate_crcl(
        creatinine_mg_dl=1.5, age_years=60, weight_kg=70.0, is_female=False
    )
    assert 30.0 < crcl < 80.0


def test_renal_dosing_metformin_block():
    """Test Metformin is blocked when eGFR < 30 mL/min."""
    alerts = RenalDosingEngine.evaluate("Metformin 500mg", egfr=25.0)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "CRITICAL"
    assert alerts[0]["action"] == "BLOCK"
    assert "eGFR < 30" in alerts[0]["message"]


def test_renal_dosing_ciprofloxacin_warning():
    """Test Ciprofloxacin dosage reduction warning when eGFR < 30 mL/min."""
    alerts = RenalDosingEngine.evaluate("Ciprofloxacin 500mg", egfr=20.0)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "HIGH"
    assert "Ciprofloxacin" in alerts[0]["message"]


def test_hepatic_dosing_warnings():
    """Test hepatic impairment dosing warnings for Paracetamol & Methotrexate."""
    para_alerts = HepaticDosingEngine.evaluate(
        "Paracetamol 1000mg", has_hepatic_impairment=True
    )
    assert len(para_alerts) == 1
    assert "2.0 g/day" in para_alerts[0]["message"]

    mtx_alerts = HepaticDosingEngine.evaluate("Methotrexate 15mg", is_cirrhotic=True)
    assert len(mtx_alerts) == 1
    assert mtx_alerts[0]["severity"] == "CRITICAL"


def test_pediatric_age_contraindications():
    """Test Doxycycline in children < 8 yrs and Aspirin in children < 16 yrs."""
    doxy_alerts = PediatricDosingEngine.evaluate(
        "Doxycycline 100mg", dose_mg=100, weight_kg=20, age_years=5
    )
    assert len(doxy_alerts) == 1
    assert doxy_alerts[0]["type"] == "PEDIATRIC_AGE_ALERT"
    assert "tooth discoloration" in doxy_alerts[0]["message"]

    aspirin_alerts = PediatricDosingEngine.evaluate(
        "Aspirin 300mg", dose_mg=300, weight_kg=30, age_years=10
    )
    assert len(aspirin_alerts) == 1
    assert aspirin_alerts[0]["severity"] == "CRITICAL"
    assert "Reye's Syndrome" in aspirin_alerts[0]["message"]


def test_pediatric_weight_dose_high_warning():
    """Test warning when prescribed pediatric dose exceeds recommended weight-based dose."""
    # Paracetamol 15mg/kg for 10kg child = 150mg rec. Prescribed 300mg (>25% high).
    alerts = PediatricDosingEngine.evaluate(
        "Paracetamol Syrup", dose_mg=300, weight_kg=10, age_years=3
    )
    assert any(a["type"] == "PEDIATRIC_DOSE_HIGH" for a in alerts)


def test_pregnancy_category_x_contraindications():
    """Test Category X drug contraindications in pregnancy."""
    war_alerts = PregnancySafetyEngine.evaluate(
        "Warfarin 5mg", is_pregnant=True, trimester=1
    )
    assert len(war_alerts) == 1
    assert war_alerts[0]["severity"] == "CRITICAL"
    assert "Category X" in war_alerts[0]["message"]

    mtx_alerts = PregnancySafetyEngine.evaluate("Methotrexate 2.5mg", is_pregnant=True)
    assert len(mtx_alerts) == 1
    assert mtx_alerts[0]["severity"] == "CRITICAL"


def test_pregnancy_3rd_trimester_nsaid_warning():
    """Test NSAID warning in 3rd trimester pregnancy."""
    nsaid_t1 = PregnancySafetyEngine.evaluate(
        "Ibuprofen 400mg", is_pregnant=True, trimester=1
    )
    assert len(nsaid_t1) == 0

    nsaid_t3 = PregnancySafetyEngine.evaluate(
        "Ibuprofen 400mg", is_pregnant=True, trimester=3
    )
    assert len(nsaid_t3) == 1
    assert "ductus arteriosus" in nsaid_t3[0]["message"]


def test_alert_fatigue_manager():
    """Test CRITICAL alerts pass through while duplicate moderate alerts are suppressed (shadow_mode=False)."""
    mgr = AlertFatigueManager(suppression_window_hours=24, shadow_mode=False)
    raw_alerts = [
        {
            "type": "ALLERGY_WARNING",
            "severity": "CRITICAL",
            "drug": "penicillin",
            "message": "Allergy block",
        },
        {
            "type": "HEPATIC_DOSING_ALERT",
            "severity": "MODERATE",
            "drug": "paracetamol",
            "message": "Hepatic warning",
        },
    ]

    # First evaluation: both alerts pass
    filtered = mgr.process_and_filter(raw_alerts, patient_id="P123")
    assert len(filtered) == 2

    # Simulate clinician overrode moderate alert
    overrides = [
        {
            "alert_type": "HEPATIC_DOSING_ALERT",
            "drug": "paracetamol",
            "created_at": pytest.importorskip("datetime").datetime.now(
                pytest.importorskip("datetime").timezone.utc
            ),
        }
    ]
    filtered_suppressed = mgr.process_and_filter(
        raw_alerts, patient_id="P123", recent_overrides=overrides
    )

    # Moderate alert suppressed, CRITICAL alert retained
    assert len(filtered_suppressed) == 1
    assert filtered_suppressed[0]["severity"] == "CRITICAL"


def test_alert_fatigue_manager_shadow_mode():
    """Test shadow mode (P1-13) retains all alerts while logging suppression decision."""
    mgr = AlertFatigueManager(suppression_window_hours=24, shadow_mode=True)
    raw_alerts = [
        {
            "type": "HEPATIC_DOSING_ALERT",
            "severity": "MODERATE",
            "drug": "paracetamol",
            "message": "Hepatic warning",
        },
    ]
    overrides = [
        {
            "alert_type": "HEPATIC_DOSING_ALERT",
            "drug": "paracetamol",
            "created_at": pytest.importorskip("datetime").datetime.now(
                pytest.importorskip("datetime").timezone.utc
            ),
        }
    ]
    filtered_shadow = mgr.process_and_filter(
        raw_alerts, patient_id="P123", recent_overrides=overrides
    )
    # In shadow mode, alert is retained despite recent override
    assert len(filtered_shadow) == 1



def test_clinical_safety_engine_check_by_names_integration(app):
    """Integration test for ClinicalSafetyEngine using app context."""
    with app.app_context():
        engine = ClinicalSafetyEngine()

        # Test renal block
        result_renal = engine.check_by_names("P9999", ["Metformin 500mg"], egfr=20.0)
        assert result_renal["critical_block"] is True
        assert any(
            "METFORMIN CONTRAINDICATED" in a["message"] for a in result_renal["alerts"]
        )

        # Test pregnancy Cat X block
        result_preg = engine.check_by_names(
            "P9999", ["Methotrexate 10mg"], is_pregnant=True
        )
        assert result_preg["critical_block"] is True
        assert any(
            "PREGNANCY CONTRAINDICATION" in a["message"] for a in result_preg["alerts"]
        )
