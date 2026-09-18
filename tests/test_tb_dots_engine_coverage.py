"""
Comprehensive Unit Tests for TB/DOTS Engine.

Pushes test coverage for departments/tb_dots/engine.py to >90%.
"""

from datetime import datetime, timedelta, timezone

import pytest

from departments.tb_dots.engine import (
    create_tb_enrollment,
    get_current_regimen,
    get_dose_summary,
    get_latest_chest_xray,
    get_latest_hiv_status,
    get_latest_sputum_result,
    get_tb_formulary,
    is_tb_regimen_valid,
    log_tb_formulary_change,
    record_chest_xray,
    record_dose_taken,
    record_hiv_status,
    record_sputum_result,
    update_tb_regimen,
)
from departments.tb_dots.models import (
    TBRegimen,
)
from extensions import db


@pytest.fixture
def sample_tb_regimens(app):
    """Fixture providing line 1 and line 2 TB regimens."""
    with app.app_context():
        now = datetime.now(timezone.utc).date()
        regimen1 = TBRegimen(
            id="REG-1ST-001",
            regimen_code="2RHZE/4RH",
            regimen_name="Standard 1st Line TB Regimen",
            line_of_therapy=1,
            drugs="Rifampicin,Isoniazid,Pyrazinamide,Ethambutol",
            duration_months=6,
            effective_from=now - timedelta(days=30),
            effective_to=now + timedelta(days=365),
        )
        regimen2 = TBRegimen(
            id="REG-2ND-001",
            regimen_code="6Lfx-Eto-Cs-Z",
            regimen_name="Standard 2nd Line MDR-TB Regimen",
            line_of_therapy=2,
            drugs="Levofloxacin,Ethionamide,Cycloserine,Pyrazinamide",
            duration_months=24,
            effective_from=now - timedelta(days=30),
            effective_to=now + timedelta(days=365),
        )
        db.session.add_all([regimen1, regimen2])
        db.session.commit()
        yield regimen1, regimen2


def test_create_tb_enrollment_success(app, sample_tb_regimens):
    with app.app_context():
        reg1, _ = sample_tb_regimens
        ok, msg, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_001",
            tb_number="TB/2026/0001",
            hiv_status="negative",
            tb_classification="pulmonary",
            bacteriological_status="confirmed",
            current_regimen_id=reg1.id,
        )
        assert ok is True
        assert "created successfully" in msg
        assert enrollment is not None
        assert enrollment.patient_id == "TB_PAT_001"
        assert enrollment.tb_number == "TB/2026/0001"


def test_create_tb_enrollment_duplicate_patient(app, sample_tb_regimens):
    with app.app_context():
        create_tb_enrollment(
            patient_id="TB_PAT_DUP",
            tb_number="TB/2026/0002",
        )
        ok, msg, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_DUP",
            tb_number="TB/2026/0003",
        )
        assert ok is False
        assert "already has an active TB enrollment" in msg
        assert enrollment is None


def test_create_tb_enrollment_duplicate_tb_number(app, sample_tb_regimens):
    with app.app_context():
        create_tb_enrollment(
            patient_id="TB_PAT_NUM1",
            tb_number="TB/2026/SAME",
        )
        ok, msg, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_NUM2",
            tb_number="TB/2026/SAME",
        )
        assert ok is False
        assert "already assigned to another patient" in msg


def test_create_tb_enrollment_invalid_regimen(app):
    with app.app_context():
        ok, msg, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_INV_REG",
            tb_number="TB/2026/0004",
            current_regimen_id="NON_EXISTENT_REG",
        )
        assert ok is False
        assert "not found" in msg


def test_create_tb_enrollment_invalid_hiv_status(app):
    with app.app_context():
        ok, msg, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_INV_HIV",
            tb_number="TB/2026/0005",
            hiv_status="invalid_status",
        )
        assert ok is False
        assert "Invalid HIV status" in msg


def test_update_tb_regimen_initial(app, sample_tb_regimens):
    with app.app_context():
        reg1, _ = sample_tb_regimens
        ok, msg, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_REG_INIT",
            tb_number="TB/2026/0006",
        )
        assert ok is True

        ok, msg, updated = update_tb_regimen(
            enrollment_id=enrollment.id,
            new_regimen_id=reg1.id,
            change_reason="Initial assignment",
            approved_by="DOC_001",
        )
        assert ok is True
        assert updated.current_regimen_id == reg1.id


def test_update_tb_regimen_line_advancement(app, sample_tb_regimens):
    with app.app_context():
        reg1, reg2 = sample_tb_regimens
        ok, msg, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_ADVANCE",
            tb_number="TB/2026/0007",
            current_regimen_id=reg1.id,
        )

        # Fail due to short reason
        ok, msg, _ = update_tb_regimen(
            enrollment_id=enrollment.id,
            new_regimen_id=reg2.id,
            change_reason="Short",
            approved_by="DOC_001",
        )
        assert ok is False
        assert "Clinical reason required" in msg

        # Success with detailed justification
        ok, msg, updated = update_tb_regimen(
            enrollment_id=enrollment.id,
            new_regimen_id=reg2.id,
            change_reason="Treatment failure on 1st line regimen after 3 months",
            approved_by="DOC_001",
        )
        assert ok is True
        assert updated.current_regimen_id == reg2.id


def test_update_tb_regimen_line_reduction(app, sample_tb_regimens):
    with app.app_context():
        reg1, reg2 = sample_tb_regimens
        ok, msg, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_REDUCE",
            tb_number="TB/2026/0008",
            current_regimen_id=reg2.id,
        )

        # Fail due to short reason (<20 chars)
        ok, msg, _ = update_tb_regimen(
            enrollment_id=enrollment.id,
            new_regimen_id=reg1.id,
            change_reason="Patient recovered",
            approved_by="DOC_001",
        )
        assert ok is False
        assert "Strong clinical justification required" in msg

        # Success with long justification
        ok, msg, updated = update_tb_regimen(
            enrollment_id=enrollment.id,
            new_regimen_id=reg1.id,
            change_reason="Culture confirmed sensitivity restored to 1st line drugs following de-escalation protocol",
            approved_by="DOC_001",
        )
        assert ok is True
        assert updated.current_regimen_id == reg1.id


def test_update_tb_regimen_not_found(app, sample_tb_regimens):
    with app.app_context():
        reg1, _ = sample_tb_regimens
        ok, msg, _ = update_tb_regimen(
            enrollment_id="INVALID_ENROLLMENT",
            new_regimen_id=reg1.id,
            change_reason="Testing invalid",
            approved_by="DOC_001",
        )
        assert ok is False
        assert "not found" in msg


def test_record_dose_taken(app):
    with app.app_context():
        _, _, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_DOSE",
            tb_number="TB/2026/0009",
        )

        # Record first dose
        ok, msg, dose1 = record_dose_taken(
            enrollment_id=enrollment.id,
            taken_as_directly_observed=True,
        )
        assert ok is True
        assert dose1.dose_number == 1
        assert dose1.taken_as_directly_observed is True

        # Record second dose
        ok, msg, dose2 = record_dose_taken(
            enrollment_id=enrollment.id,
            taken_as_directly_observed=False,
        )
        assert ok is True
        assert dose2.dose_number == 2
        assert dose2.taken_as_directly_observed is False


def test_record_sputum_result(app):
    with app.app_context():
        _, _, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_SPUTUM",
            tb_number="TB/2026/0010",
        )

        ok, msg, sputum = record_sputum_result(
            enrollment_id=enrollment.id,
            specimen_type="sputum",
            specimen_number=1,
            smear_result="3+",
            culture_result="positive",
            culture_species="M. tuberculosis",
            drug_susceptibility="Rifampicin resistant",
        )
        assert ok is True
        assert sputum.smear_result == "3+"
        assert sputum.culture_result == "positive"


def test_record_chest_xray(app):
    with app.app_context():
        _, _, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_XRAY",
            tb_number="TB/2026/0011",
        )

        ok, msg, xray = record_chest_xray(
            enrollment_id=enrollment.id,
            finding="Cavitary lesion right upper lobe",
            severity="advanced",
            progression="worsened",
        )
        assert ok is True
        assert xray.finding == "Cavitary lesion right upper lobe"
        assert xray.severity == "advanced"


def test_record_hiv_status(app):
    with app.app_context():
        _, _, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_HIV",
            tb_number="TB/2026/0012",
        )

        # Invalid result
        ok, msg, _ = record_hiv_status(
            enrollment_id=enrollment.id,
            test_type="rapid",
            result="invalid_result",
        )
        assert ok is False

        # Valid result
        ok, msg, hiv = record_hiv_status(
            enrollment_id=enrollment.id,
            test_type="rapid",
            result="positive",
            cd4_count=350,
        )
        assert ok is True
        assert hiv.result == "positive"
        assert hiv.cd4_count == 350


def test_getters_and_formulary(app, sample_tb_regimens):
    with app.app_context():
        reg1, _ = sample_tb_regimens
        _, _, enrollment = create_tb_enrollment(
            patient_id="TB_PAT_GETTERS",
            tb_number="TB/2026/0013",
            current_regimen_id=reg1.id,
        )

        record_dose_taken(enrollment.id, taken_as_directly_observed=True)
        record_sputum_result(
            enrollment.id, specimen_type="sputum", specimen_number=1, smear_result="2+"
        )
        record_chest_xray(enrollment.id, finding="Infiltrates")
        record_hiv_status(enrollment.id, test_type="rapid", result="negative")

        assert get_current_regimen(enrollment.id).id == reg1.id
        assert get_latest_sputum_result(enrollment.id).smear_result == "2+"
        assert get_latest_chest_xray(enrollment.id).finding == "Infiltrates"
        assert get_latest_hiv_status(enrollment.id).result == "negative"
        assert len(get_dose_summary(enrollment.id)) == 1

        formulary = get_tb_formulary()
        assert len(formulary) >= 2
        assert is_tb_regimen_valid(reg1.id) is True
        assert is_tb_regimen_valid("NON_EXISTENT") is False
        assert (
            log_tb_formulary_change(reg1.id, "update", "ADMIN_USER", "Updated notes")
            is True
        )
