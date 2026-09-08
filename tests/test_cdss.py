"""
tests/test_cdss.py
───────────────────
Phase G — Clinical Decision Support System (CDSS) Test Suite

Includes P0 regression tests for the DrugCentral unreachable-host safety fix:
  - Simulated OperationalError (timeout) → local matrix fires, result < 1s
  - Simulated circuit-breaker-open → local matrix fires immediately
  - Warfarin + Aspirin detected by local matrix when DrugCentral is down
"""

import time
from unittest.mock import patch

import pytest

from departments.medicine.cdss import (
    calculate_dosing_adjustment,
    check_drug_interactions,
    check_patient_allergies,
    query_drugcentral_ddi,
)
from departments.models.records import Patient
from departments.shared.drugcentral import DrugCentralUnavailable


# ---------------------------------------------------------------------------
# Existing CDSS unit tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# P0 regression: DrugCentral unreachable host safety
# ---------------------------------------------------------------------------


def test_drugcentral_timeout_returns_empty_within_bound():
    """
    P0 regression: when get_drugcentral_connection raises OperationalError
    (simulating a TCP timeout / unreachable host), query_drugcentral_ddi must
    return [] within a short wall-clock window — NOT hang.
    """
    import psycopg2

    with patch(
        "departments.shared.drugcentral.get_drugcentral_connection",
        side_effect=DrugCentralUnavailable(
            psycopg2.OperationalError("could not connect to server: Connection timed out")
        ),
    ):
        t0 = time.monotonic()
        result = query_drugcentral_ddi("warfarin", "aspirin")
        elapsed = time.monotonic() - t0

    assert result == [], "Expected empty list when DrugCentral is unreachable"
    assert elapsed < 1.0, (
        f"query_drugcentral_ddi took {elapsed:.3f}s with unreachable host — "
        "expected < 1.0s"
    )


def test_drugcentral_circuit_breaker_open_returns_empty_immediately():
    """
    P0 regression: when the circuit breaker is open (DrugCentralUnavailable
    raised before a connection is attempted), query_drugcentral_ddi must
    return [] immediately.
    """
    with patch(
        "departments.shared.drugcentral.get_drugcentral_connection",
        side_effect=DrugCentralUnavailable("Circuit breaker open"),
    ):
        t0 = time.monotonic()
        result = query_drugcentral_ddi("sildenafil", "nitroglycerin")
        elapsed = time.monotonic() - t0

    assert result == []
    assert elapsed < 0.5, (
        f"Circuit-breaker path took {elapsed:.3f}s — expected near-instant"
    )


def test_local_matrix_fires_when_drugcentral_down():
    """
    P0 regression: check_drug_interactions() must detect Warfarin + Aspirin
    via the local KNOWN_INTERACTIONS matrix even when DrugCentral is completely
    unreachable, and must do so within 1 second.
    """
    with patch(
        "departments.shared.drugcentral.get_drugcentral_connection",
        side_effect=DrugCentralUnavailable("host unreachable"),
    ):
        t0 = time.monotonic()
        warnings = check_drug_interactions(["Warfarin 5mg", "Aspirin 75mg"])
        elapsed = time.monotonic() - t0

    assert len(warnings) >= 1, (
        "Local KNOWN_INTERACTIONS matrix did not fire when DrugCentral was down"
    )
    assert any(w["severity"] == "HIGH" for w in warnings), (
        "Expected HIGH severity warning for Warfarin + Aspirin from local matrix"
    )
    assert any(w["source"] == "Local Fallback Matrix" for w in warnings), (
        "Warning source should be 'Local Fallback Matrix' when DrugCentral is unavailable"
    )
    assert elapsed < 1.0, (
        f"check_drug_interactions took {elapsed:.3f}s with unreachable DrugCentral "
        "— expected < 1.0s (local fallback should be near-instant)"
    )


def test_connect_timeout_is_set():
    """
    P0 regression: DRUGCENTRAL_DB_PARAMS must include connect_timeout so that
    psycopg2 does not use the kernel default (which can hang for ~2 minutes).
    """
    from departments.shared.drugcentral import DRUGCENTRAL_DB_PARAMS

    assert "connect_timeout" in DRUGCENTRAL_DB_PARAMS, (
        "DRUGCENTRAL_DB_PARAMS is missing 'connect_timeout' — "
        "without it a TCP connect to an unreachable host can block for minutes"
    )
    assert isinstance(DRUGCENTRAL_DB_PARAMS["connect_timeout"], int)
    assert DRUGCENTRAL_DB_PARAMS["connect_timeout"] > 0
    assert DRUGCENTRAL_DB_PARAMS["connect_timeout"] <= 10, (
        "connect_timeout should be short (≤10s) for a clinical safety path"
    )


def test_query_drugcentral_ddi_unexpected_exception_returns_empty():
    """
    Broad exception guard: any unexpected error (e.g. DB schema change, cursor
    error) must also return [] rather than propagating and crashing the
    prescribing endpoint.
    """
    with patch(
        "departments.shared.drugcentral.get_drugcentral_connection",
        side_effect=RuntimeError("unexpected internal error"),
    ):
        result = query_drugcentral_ddi("metformin", "gentamicin")
    assert result == []
