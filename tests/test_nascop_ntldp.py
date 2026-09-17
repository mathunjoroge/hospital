"""
Unit tests for NASCOP ARV and NTLD-P TB/TPT Master Clinical Regimens.
"""
from datetime import datetime, timezone

from departments.api.dhis2_exporter import aggregate_monthly_khis_data
from departments.hiv_art.moh_731_729b import (
    NASCOP_REGIMEN_CATALOG,
    classify_nascop_regimen,
)
from departments.models.clinical_regimens import MasterClinicalRegimen
from departments.tb_dots.ntldp_tb_tpt import (
    NTLDP_TB_TPT_CATALOG,
    aggregate_ntldp_tb_tpt_monthly,
    classify_ntldp_regimen,
    seed_master_clinical_regimens_catalog,
)


def test_nascop_regimen_catalog_completeness():
    """Verify NASCOP ARV catalog contains all 16 specified codes and correct definitions."""
    codes = [item["regimen_code"] for item in NASCOP_REGIMEN_CATALOG]
    assert len(codes) == 16
    assert "AF1A" in codes  # TLD
    assert "AF1B" in codes  # TAFLD
    assert "AF1C" in codes  # TLE
    assert "AF2A" in codes  # ALD
    assert "AF2B" in codes  # ZLD
    assert "AS1A" in codes
    assert "AS1B" in codes
    assert "AS2A" in codes
    assert "AS5A" in codes
    assert "AS6A" in codes
    assert "TL3A" in codes
    assert "TL3B" in codes  # TAFLD + DRV/r
    assert "TL3C" in codes  # ETR + DRV/r + DTG
    assert "PF1A" in codes
    assert "PF1B" in codes
    assert "PF2A" in codes

    tafld = classify_nascop_regimen("AF1B")
    assert tafld is not None
    assert "TAFLD" in tafld["regimen_name"]

    tle = classify_nascop_regimen("AF1C")
    assert tle is not None
    assert "TLE" in tle["regimen_name"]


def test_ntldp_tb_tpt_catalog_completeness():
    """Verify NTLD-P catalog contains all DS-TB, DR-TB (BPaLM, BPaL), and TPT regimens."""
    codes = [item["nascop_ntldp_code"] for item in NTLDP_TB_TPT_CATALOG]
    assert len(codes) == 10
    assert "DS-TB-ADULT" in codes
    assert "DS-TB-PED" in codes
    assert "DR-BPaLM" in codes
    assert "DR-BPaL" in codes
    assert "DR-INDIVIDUALIZED" in codes
    assert "TPT-3HP" in codes
    assert "TPT-1HP" in codes
    assert "TPT-3RH" in codes
    assert "TPT-6H" in codes
    assert "TPT-6LFX" in codes

    bpalm = classify_ntldp_regimen("DR-BPaLM")
    assert bpalm["regimen_acronym"] == "BPaLM"

    tpt_3hp = classify_ntldp_regimen(drugs_text="Rifapentine + Isoniazid weekly")
    assert tpt_3hp["nascop_ntldp_code"] == "TPT-3HP"


def test_seed_master_clinical_regimens_catalog(app):
    """Verify seeding populates master_clinical_regimens DB table."""
    with app.app_context():
        seed_master_clinical_regimens_catalog()
        count = MasterClinicalRegimen.query.count()
        assert count >= 26  # 16 NASCOP + 10 NTLD-P

        rec = MasterClinicalRegimen.query.filter_by(nascop_ntldp_code="AF1B").first()
        assert rec is not None
        assert rec.program_domain == "HIV"
        assert "TAFLD" in rec.drug_components

        tb_rec = MasterClinicalRegimen.query.filter_by(nascop_ntldp_code="DR-BPaLM").first()
        assert tb_rec is not None
        assert tb_rec.program_domain == "TB"
        assert tb_rec.regimen_acronym == "BPaLM"


def test_aggregate_ntldp_tb_tpt_monthly(app):
    """Test aggregation of NTLD-P TB/TPT metrics."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        result = aggregate_ntldp_tb_tpt_monthly(now.year, now.month)
        assert "total_tb_active_patients" in result
        assert "total_tpt_active_patients" in result
        assert len(result["regimen_details"]) == 10


def test_dhis2_exporter_integration(app):
    """Test DHIS2 monthly aggregation including HIV ARV and NTLD-P TB/TPT regimens."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        export = aggregate_monthly_khis_data(now.year, now.month)
        assert "tb_tpt_regimens" in export
        assert "hiv_arv_regimens" in export
        assert export["tb_tpt_regimens"]["total_tb_active_patients"] >= 0
