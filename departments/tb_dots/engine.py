"""
TB/DOTS Workflow Engine.

Aligns with WHO TB treatment guidelines and national protocols.
Enforces TB enrollment, regimen management, dose monitoring,
and clinical workflows for TB treatment and care.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from extensions import db

from .models import (
    ChestXRay,
    DoseTaken,
    HIVStatus,
    SputumResult,
    TBEnrollment,
    TBRegimen,
)

logger = logging.getLogger(__name__)


def create_tb_enrollment(
    patient_id: str,
    tb_number: str,
    hiv_status: Optional[str] = None,
    art_enrollment_id: Optional[str] = None,
    tb_classification: Optional[str] = None,
    site_of_disease: Optional[str] = None,
    bacteriological_status: Optional[str] = None,
    treatment_start_date: Optional[datetime] = None,
    facility_enrolled_at: Optional[str] = None,
    encounter_id: Optional[int] = None,
    current_regimen_id: Optional[str] = None
) -> Tuple[bool, str, Optional[TBEnrollment]]:
    """
    Create a new TB enrollment for a patient.

    Args:
        patient_id: Patient identifier
        tb_number: Unique TB number for the patient
        hiv_status: HIV status (positive, negative, unknown, refused_test)
        art_enrollment_id: Link to ART enrollment if co-infected
        tb_classification: pulmonary or extrapulmonary
        site_of_disease: Site of disease for extrapulmonary TB
        bacteriological_status: confirmed, clinical, etc.
        treatment_start_date: Date TB treatment was started (defaults to now)
        facility_enrolled_at: Facility where enrollment occurred
        encounter_id: Associated encounter ID
        current_regimen_id: Initial TB regimen ID

    Returns:
        Tuple of (success, message, enrollment_object)
    """
    try:
        # Check for duplicate enrollment (active enrollment without end date)
        existing_enrollment = TBEnrollment.query.filter_by(
            patient_id=patient_id
        ).first()

        if existing_enrollment:
            return False, f"Patient {patient_id} already has an active TB enrollment", None

        # Validate TB number uniqueness
        if TBEnrollment.query.filter_by(tb_number=tb_number).first():
            return False, f"TB number {tb_number} is already assigned to another patient", None

        # Validate regimen if provided
        if current_regimen_id:
            regimen = TBRegimen.query.get(current_regimen_id)
            if not regimen:
                return False, f"TB regimen with ID {current_regimen_id} not found", None

        # Validate HIV status if provided
        if hiv_status and hiv_status not in ['positive', 'negative', 'unknown', 'refused_test']:
            return False, "Invalid HIV status. Must be one of: positive, negative, unknown, refused_test", None

        # Set defaults
        if treatment_start_date is None:
            treatment_start_date = datetime.now(timezone.utc)

        # Create enrollment
        enrollment = TBEnrollment(
            patient_id=patient_id,
            tb_number=tb_number,
            hiv_status=hiv_status,
            art_enrollment_id=art_enrollment_id,
            tb_classification=tb_classification,
            site_of_disease=site_of_disease,
            bacteriological_status=bacteriological_status,
            treatment_start_date=treatment_start_date,
            facility_enrolled_at=facility_enrolled_at,
            encounter_id=encounter_id,
            current_regimen_id=current_regimen_id
        )

        db.session.add(enrollment)
        db.session.commit()

        logger.info(f"Created TB enrollment for patient {patient_id} with TB number {tb_number}")
        return True, "TB enrollment created successfully", enrollment

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error creating TB enrollment: {e}")
        return False, f"Failed to create TB enrollment: {str(e)}", None


def update_tb_regimen(
    enrollment_id: str,
    new_regimen_id: str,
    change_reason: str,
    approved_by: str,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[TBEnrollment]]:
    """
    Update a patient's TB regimen with validation for line changes.

    Args:
        enrollment_id: TB enrollment ID
        new_regimen_id: New regimen ID to switch to
        change_reason: Clinical reason for regimen change
        approved_by: User ID of clinician approving the change
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, enrollment_object)
    """
    try:
        enrollment = TBEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"TB enrollment {enrollment_id} not found", None

        new_regimen = TBRegimen.query.get(new_regimen_id)
        if not new_regimen:
            return False, f"TB regimen {new_regimen_id} not found", None

        # Get current regimen for comparison
        current_regimen = enrollment.current_regimen
        if not current_regimen:
            # First regimen assignment - no line change validation needed
            enrollment.current_regimen_id = new_regimen_id
            db.session.commit()
            logger.info(f"Set initial regimen {new_regimen.regimen_code} for enrollment {enrollment_id}")
            return True, "Initial regimen assigned successfully", enrollment

        # Check if this is a line change
        current_line = current_regimen.line_of_therapy
        new_line = new_regimen.line_of_therapy

        if new_line > current_line:
            # Moving to higher line (e.g., 1st to 2nd line) - requires clinical justification
            if not change_reason or len(change_reason.strip()) < 10:
                return False, "Clinical reason required for line advancement", None

            logger.warning(f"Advancing patient {enrollment.patient_id} from line {current_line} to {new_line}")
            # In a real system, this might trigger additional review or notification

        elif new_line < current_line:
            # Moving to lower line - requires strong justification
            if not change_reason or len(change_reason.strip()) < 20:
                return False, "Strong clinical justification required for line reduction", None

            logger.warning(f"Reducing patient {enrollment.patient_id} from line {current_line} to {new_line}")

        # Same line or downward move - update regimen
        enrollment.current_regimen_id = new_regimen_id
        db.session.commit()

        logger.info(f"Updated regimen for enrollment {enrollment_id} to {new_regimen.regimen_code}")
        return True, "TB regimen updated successfully", enrollment

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error updating TB regimen: {e}")
        return False, f"Failed to update TB regimen: {str(e)}", None


def record_dose_taken(
    enrollment_id: str,
    taken_as_directly_observed: bool = False,
    date_taken: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[DoseTaken]]:
    """
    Record a TB dose taken (either self-administered or directly observed).

    Args:
        enrollment_id: TB enrollment ID
        taken_as_directly_observed: Whether the dose was taken under direct observation (DOT)
        date_taken: Date and time the dose was taken (defaults to now)
        encounter_id: Associated encounter ID (if DOT)

    Returns:
        Tuple of (success, message, dose_object)
    """
    try:
        enrollment = TBEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"TB enrollment {enrollment_id} not found", None

        if date_taken is None:
            date_taken = datetime.now(timezone.utc)

        # Determine the next dose number for this enrollment
        last_dose = DoseTaken.query.filter_by(tb_enrollment_id=enrollment_id)\
            .order_by(DoseTaken.date_taken.desc())\
            .first()
        dose_number = (last_dose.dose_number + 1) if last_dose and last_dose.dose_number else 1

        # Create dose taken record
        dose = DoseTaken(
            tb_enrollment_id=enrollment_id,
            dose_number=dose_number,
            taken_as_directly_observed=taken_as_directly_observed,
            date_taken=date_taken,
            encounter_id=encounter_id
        )

        db.session.add(dose)
        db.session.commit()

        logger.info(f"Recorded dose {dose_number} for enrollment {enrollment_id} (DOT: {taken_as_directly_observed})")
        return True, "Dose taken recorded successfully", dose

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording dose taken: {e}")
        return False, f"Failed to record dose taken: {str(e)}", None


def record_sputum_result(
    enrollment_id: str,
    specimen_type: str,
    specimen_number: int,
    smear_result: Optional[str] = None,
    culture_result: Optional[str] = None,
    culture_species: Optional[str] = None,
    drug_susceptibility: Optional[str] = None,
    test_date: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[SputumResult]]:
    """
    Record a TB sputum test result.

    Args:
        enrollment_id: TB enrollment ID
        specimen_type: Type of specimen (sputum, gastric, etc.)
        specimen_number: Specimen number in series (1, 2, 3)
        smear_result: Smear microscopy result (negative, scanty, 1+, 2+, 3+)
        culture_result: Culture result (negative, positive, contaminated)
        culture_species: Species identified if culture positive
        drug_susceptibility: Drug susceptibility test results
        test_date: Date of test (defaults to now)
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, sputum_result_object)
    """
    try:
        enrollment = TBEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"TB enrollment {enrollment_id} not found", None

        if test_date is None:
            test_date = datetime.now(timezone.utc)

        sputum_result = SputumResult(
            tb_enrollment_id=enrollment_id,
            specimen_type=specimen_type,
            specimen_number=specimen_number,
            smear_result=smear_result,
            culture_result=culture_result,
            culture_species=culture_species,
            drug_susceptibility=drug_susceptibility,
            test_date=test_date,
            encounter_id=encounter_id
        )

        db.session.add(sputum_result)
        db.session.commit()

        logger.info(f"Recorded sputum result for enrollment {enrollment_id}: smear {smear_result}, culture {culture_result}")
        return True, "Sputum result recorded successfully", sputum_result

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording sputum result: {e}")
        return False, f"Failed to record sputum result: {str(e)}", None


def record_chest_xray(
    enrollment_id: str,
    finding: Optional[str] = None,
    severity: Optional[str] = None,
    progression: Optional[str] = None,
    test_date: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[ChestXRay]]:
    """
    Record a TB chest X-ray result.

    Args:
        enrollment_id: TB enrollment ID
        finding: X-ray finding (normal, abnormal, cavitary, etc.)
        severity: Severity of findings (minimal, moderate, advanced)
        progression: Change from previous X-ray (improved, worsened, unchanged)
        test_date: Date of test (defaults to now)
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, chest_xray_object)
    """
    try:
        enrollment = TBEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"TB enrollment {enrollment_id} not found", None

        if test_date is None:
            test_date = datetime.now(timezone.utc)

        chest_xray = ChestXRay(
            tb_enrollment_id=enrollment_id,
            finding=finding,
            severity=severity,
            progression=progression,
            test_date=test_date,
            encounter_id=encounter_id
        )

        db.session.add(chest_xray)
        db.session.commit()

        logger.info(f"Recorded chest X-ray for enrollment {enrollment_id}: {finding}")
        return True, "Chest X-ray recorded successfully", chest_xray

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording chest X-ray: {e}")
        return False, f"Failed to record chest X-ray: {str(e)}", None


def record_hiv_status(
    enrollment_id: str,
    test_type: str,
    result: str,
    cd4_count: Optional[int] = None,
    test_date: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[HIVStatus]]:
    """
    Record an HIV status test result for a TB patient.

    Args:
        enrollment_id: TB enrollment ID
        test_type: Type of HIV test (rapid, ELISA, Western blot)
        result: HIV test result (positive, negative, indeterminate)
        cd4_count: CD4 count if available (cells/µL)
        test_date: Date of test (defaults to now)
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, hiv_status_object)
    """
    try:
        if result not in ['positive', 'negative', 'indeterminate']:
            return False, "HIV test result must be one of: positive, negative, indeterminate", None

        enrollment = TBEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"TB enrollment {enrollment_id} not found", None

        if test_date is None:
            test_date = datetime.now(timezone.utc)

        hiv_status = HIVStatus(
            tb_enrollment_id=enrollment_id,
            test_type=test_type,
            result=result,
            cd4_count=cd4_count,
            test_date=test_date,
            encounter_id=encounter_id
        )

        db.session.add(hiv_status)
        db.session.commit()

        logger.info(f"Recorded HIV status for enrollment {enrollment_id}: {result}")
        return True, "HIV status recorded successfully", hiv_status

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording HIV status: {e}")
        return False, f"Failed to record HIV status: {str(e)}", None


def get_current_regimen(enrollment_id: str) -> Optional[TBRegimen]:
    """
    Get the current TB regimen for an enrollment.

    Args:
        enrollment_id: TB enrollment ID

    Returns:
        Current TB regimen or None if not found
    """
    try:
        enrollment = TBEnrollment.query.get(enrollment_id)
        if not enrollment:
            return None
        return enrollment.current_regimen
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting current regimen: {e}")
        return None


def get_latest_sputum_result(enrollment_id: str) -> Optional[SputumResult]:
    """
    Get the most recent sputum test for an enrollment.

    Args:
        enrollment_id: TB enrollment ID

    Returns:
        Most recent sputum result or None if not found
    """
    try:
        return SputumResult.query.filter_by(tb_enrollment_id=enrollment_id)\
            .order_by(SputumResult.test_date.desc())\
            .first()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting latest sputum result: {e}")
        return None


def get_latest_chest_xray(enrollment_id: str) -> Optional[ChestXRay]:
    """
    Get the most recent chest X-ray for an enrollment.

    Args:
        enrollment_id: TB enrollment ID

    Returns:
        Most recent chest X-ray or None if not found
    """
    try:
        return ChestXRay.query.filter_by(tb_enrollment_id=enrollment_id)\
            .order_by(ChestXRay.test_date.desc())\
            .first()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting latest chest X-ray: {e}")
        return None


def get_latest_hiv_status(enrollment_id: str) -> Optional[HIVStatus]:
    """
    Get the most recent HIV status test for an enrollment.

    Args:
        enrollment_id: TB enrollment ID

    Returns:
        Most recent HIV status or None if not found
    """
    try:
        return HIVStatus.query.filter_by(tb_enrollment_id=enrollment_id)\
            .order_by(HIVStatus.test_date.desc())\
            .first()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting latest HIV status: {e}")
        return None


def get_dose_summary(enrollment_id: str, limit: int = 30) -> List[DoseTaken]:
    """
    Get recent doses taken for an enrollment.

    Args:
        enrollment_id: TB enrollment ID
        limit: Maximum number of doses to return

    Returns:
        List of doses taken ordered by date (most recent first)
    """
    try:
        return DoseTaken.query.filter_by(tb_enrollment_id=enrollment_id)\
            .order_by(DoseTaken.date_taken.desc())\
            .limit(limit)\
            .all()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting dose summary: {e}")
        return []


# Formulary management functions
def get_tb_formulary() -> List[TBRegimen]:
    """
    Get all active TB regimens in the formulary.

    Returns:
        List of active TB regimens
    """
    try:
        now = datetime.now(timezone.utc).date()
        return TBRegimen.query.filter(
            db.or_(TBRegimen.effective_to.is_(None), TBRegimen.effective_to >= now),
            TBRegimen.effective_from <= now
        ).order_by(TBRegimen.line_of_therapy, TBRegimen.regimen_code).all()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting TB formulary: {e}")
        return []


def is_tb_regimen_valid(regimen_id: str) -> bool:
    """
    Check if a TB regimen is currently valid/effective.

    Args:
        regimen_id: TB regimen ID to check

    Returns:
        True if regimen is valid, False otherwise
    """
    try:
        regimen = TBRegimen.query.get(regimen_id)
        if not regimen:
            return False

        now = datetime.now(timezone.utc).date()
        if regimen.effective_to and regimen.effective_to < now:
            return False
        if regimen.effective_from > now:
            return False
        return True
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error checking TB regimen validity: {e}")
        return False


def log_tb_formulary_change(
    regimen_id: str,
    change_type: str,  # 'create', 'update', 'retire'
    changed_by: str,
    change_notes: Optional[str] = None
) -> bool:
    """
    Log a change to the TB formulary for audit purposes.

    Args:
        regimen_id: TB regimen ID that was changed
        change_type: Type of change made
        changed_by: User ID of person making the change
        change_notes: Optional notes about the change

    Returns:
        True if logged successfully, False otherwise
    """
    try:
        # In a full implementation, this would write to a formulary_change_log table
        # For now, we'll just log it
        logger.info(
            f"TB formulary change: {change_type} regimen {regimen_id} "
            f"by {changed_by}. Notes: {change_notes or 'None'}"
        )
        return True
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error logging TB formulary change: {e}")
        return False
