"""
tests/test_cdss.py
───────────────────
Phase G — Clinical Decision Support System (CDSS) Test Suite
"""

from departments.medicine.cdss import (
    calculate_dosing_adjustment,
    check_drug_interactions,
    check_patient_allergies,
)
from departments.models.records import Patient


def test_check_drug_interactions_high_risk():
    """Detects Warfarin + Aspirin high risk interaction."""
    warnings = check_drug_interactions(["Warfarin 5mg", "Aspirin 75mg", "Paracetamol"])
    assert len(warnings) >= 1
    w = warnings[0]
    assert w["severity"] == "HIGH"
    assert "Warfarin" in w["title"] or "Bleeding" in w["title"]
    assert set(w["interacting_drugs"]) == {"aspirin", "warfarin"}


def test_check_drug_interactions_no_interaction():
    """Returns empty list when no interaction rules match."""
    warnings = check_drug_interactions(["Amoxicillin 500mg", "Paracetamol 500mg"])
    assert warnings == []


def test_patient_allergy_screening(app):
    """Detects penicillin allergy match from patient record."""
    with app.app_context():
        p = Patient(
            patient_id="P-CDSS-001",
            name="Allergy Test Patient",
            relationship_with_next_of_kin="Penicillin allergy documented",
        )
        alert = check_patient_allergies(p, "Amoxicillin")
        assert alert is not None
        assert alert["severity"] == "HIGH"
        assert alert["allergen_class"] == "penicillin"


def test_renal_dosing_guidance():
    """Calculates renal dose guidance for Metformin when eGFR < 45."""
    guidance = calculate_dosing_adjustment("Metformin 500mg", egfr=30.0)
    assert guidance is not None
    assert guidance["severity"] == "MODERATE"
    assert "Lactic Acidosis" in guidance["guidance"]


def test_cdss_evaluate_endpoint(client):
    """POST /medicine/prescribe/cdss/evaluate returns structured CDSS safety report."""
    resp = client.post(
        "/medicine/prescribe/cdss/evaluate",
        json={
            "drug_name": "Aspirin",
            "existing_meds": ["Warfarin"],
            "egfr": 40.0,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["has_warnings"] is True
    assert data["high_risk"] is True
    assert len(data["warnings"]) >= 1
