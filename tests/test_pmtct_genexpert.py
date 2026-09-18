"""
Unit tests for PMTCT, Viral Load Suppression (TX_PVLS), and GeneXpert DR-TB decision tree.
"""

from datetime import datetime, timezone

from departments.api.dhis2_exporter import aggregate_monthly_khis_data
from departments.hiv_art.moh_731_729b import aggregate_pmtct_tx_pvls_monthly
from departments.tb_dots.ntldp_tb_tpt import classify_dr_tb_regimen_from_genexpert


def test_genexpert_dr_tb_decision_tree():
    """Verify GeneXpert automated regimen decision logic across clinical scenarios."""
    # Scenario 1: MTB Not Detected
    neg = classify_dr_tb_regimen_from_genexpert("MTB_NOT_DETECTED", "NOT_TESTED")
    assert neg["recommended_code"] == "NO_TB_DETECTED"

    # Scenario 2: DS-TB (RIF Susceptible)
    ds_adult = classify_dr_tb_regimen_from_genexpert(
        "MTB_DETECTED", "RIF_SUSCEPTIBLE", age_years=30
    )
    assert ds_adult["recommended_code"] == "DS-TB-ADULT"

    ds_ped = classify_dr_tb_regimen_from_genexpert(
        "MTB_DETECTED", "RIF_SUSCEPTIBLE", age_years=8
    )
    assert ds_ped["recommended_code"] == "DS-TB-PED"

    # Scenario 3: MDR/RR-TB FQ Susceptible -> BPaLM
    bpalm = classify_dr_tb_regimen_from_genexpert(
        "MTB_DETECTED", "RIF_RESISTANT", "FQ_SUSCEPTIBLE"
    )
    assert bpalm["recommended_code"] == "DR-BPaLM"
    assert bpalm["regimen_acronym"] == "BPaLM"

    # Scenario 4: Pre-XDR TB FQ Resistant -> BPaL
    bpal = classify_dr_tb_regimen_from_genexpert(
        "MTB_DETECTED", "RIF_RESISTANT", "FQ_RESISTANT"
    )
    assert bpal["recommended_code"] == "DR-BPaL"
    assert bpal["regimen_acronym"] == "BPaL"

    # Scenario 5: Prior BDQ exposure -> Individualized Longer DR-TB
    indiv = classify_dr_tb_regimen_from_genexpert(
        "MTB_DETECTED", "RIF_RESISTANT", prior_bdq_exposure=True
    )
    assert indiv["recommended_code"] == "DR-INDIVIDUALIZED"


def test_aggregate_pmtct_tx_pvls_monthly(app):
    """Test PMTCT and TX_PVLS metrics calculation."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        result = aggregate_pmtct_tx_pvls_monthly(now.year, now.month)
        assert "pmtct_art_count" in result
        assert "hei_prophylaxis_count" in result
        assert "eid_6wk_pcr_count" in result
        assert "tx_pvls_eligible" in result
        assert "tx_pvls_suppressed" in result
        assert "suppression_rate_pct" in result


def test_dhis2_exporter_pmtct_genexpert(app):
    """Test DHIS2 export includes PMTCT, TX_PVLS, and NTLD-P TB data elements."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        export = aggregate_monthly_khis_data(now.year, now.month)
        assert "pmtct_pvls" in export
        assert "tb_tpt_regimens" in export

        data_elements = {
            elem["dataElement"]: elem["value"] for elem in export["data_elements"]
        }
        assert "MOH731_PMTCT_ART_COUNT" in data_elements
        assert "MOH731_TX_PVLS_SUPPRESSED" in data_elements
