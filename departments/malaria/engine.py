"""
Malaria Workflow Engine.

Aligns with WHO malaria treatment guidelines and national protocols.
Enforces malaria case management, treatment administration,
and clinical workflows for malaria diagnosis and care.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from extensions import db

from .models import MalariaCase, MalariaLabResult, MalariaRegimen, MalariaTreatment

logger = logging.getLogger(__name__)


def create_malaria_case(
    patient_id: str,
    case_number: str,
    malaria_species: Optional[str] = None,
    parasite_density: Optional[int] = None,
    diagnosis_method: Optional[str] = None,
    severity: Optional[str] = None,
    pregnancy_status: Optional[str] = None,
    treatment_start_date: Optional[datetime] = None,
    facility_diagnosed_at: Optional[str] = None,
    encounter_id: Optional[int] = None,
    current_regimen_id: Optional[str] = None
) -> Tuple[bool, str, Optional[MalariaCase]]:
    """
    Create a new malaria case for a patient.

    Args:
        patient_id: Patient identifier
        case_number: Unique malaria case number for the patient
        malaria_species: Plasmodium species (falciparum, vivax, ovale, malariae, knowlesi, mixed)
        parasite_density: Parasite density (parasites/µL for microscopy, % for RDT)
        diagnosis_method: Method used for diagnosis (microscopy, RDT, PCR)
        severity: Case severity (uncomplicated, severe)
        pregnancy_status: Pregnancy status (not_pregnant, pregnant_first_trimester, pregnant_second_trimester, pregnant_third_trimester, postpartum)
        treatment_start_date: Date malaria treatment was started (defaults to now)
        facility_diagnosed_at: Facility where diagnosis occurred
        encounter_id: Associated encounter ID
        current_regimen_id: Initial malaria regimen ID

    Returns:
        Tuple of (success, message, case_object)
    """
    try:
        # Check for duplicate case (active case without end date)
        existing_case = MalariaCase.query.filter_by(
            patient_id=patient_id
        ).first()

        if existing_case:
            return False, f"Patient {patient_id} already has an active malaria case", None

        # Validate case number uniqueness
        if MalariaCase.query.filter_by(case_number=case_number).first():
            return False, f"Malaria case number {case_number} is already assigned to another patient", None

        # Validate regimen if provided
        if current_regimen_id:
            regimen = MalariaRegimen.query.get(current_regimen_id)
            if not regimen:
                return False, f"Malaria regimen with ID {current_regimen_id} not found", None

        # Validate malaria species if provided
        valid_species = ['falciparum', 'vivax', 'ovale', 'malariae', 'knowlesi', 'mixed']
        if malaria_species and malaria_species not in valid_species:
            return False, f"Invalid malaria species. Must be one of: {', '.join(valid_species)}", None

        # Validate diagnosis method if provided
        valid_methods = ['microscopy', 'RDT', 'PCR']
        if diagnosis_method and diagnosis_method not in valid_methods:
            return False, f"Invalid diagnosis method. Must be one of: {', '.join(valid_methods)}", None

        # Validate severity if provided
        valid_severity = ['uncomplicated', 'severe']
        if severity and severity not in valid_severity:
            return False, f"Invalid severity. Must be one of: {', '.join(valid_severity)}", None

        # Validate pregnancy status if provided
        valid_pregnancy = ['not_pregnant', 'pregnant_first_trimester', 'pregnant_second_trimester', 'pregnant_third_trimester', 'postpartum']
        if pregnancy_status and pregnancy_status not in valid_pregnancy:
            return False, f"Invalid pregnancy status. Must be one of: {', '.join(valid_pregnancy)}", None

        # Set defaults
        if treatment_start_date is None:
            treatment_start_date = datetime.now(timezone.utc)

        # Create case
        case = MalariaCase(
            patient_id=patient_id,
            case_number=case_number,
            malaria_species=malaria_species,
            parasite_density=parasite_density,
            diagnosis_method=diagnosis_method,
            severity=severity,
            pregnancy_status=pregnancy_status,
            treatment_start_date=treatment_start_date,
            facility_diagnosed_at=facility_diagnosed_at,
            encounter_id=encounter_id,
            current_regimen_id=current_regimen_id
        )

        db.session.add(case)
        db.session.commit()

        logger.info(f"Created malaria case for patient {patient_id} with case number {case_number}")
        return True, "Malaria case created successfully", case

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error creating malaria case: {e}")
        return False, f"Failed to create malaria case: {str(e)}", None


def update_malaria_regimen(
    case_id: str,
    new_regimen_id: str,
    change_reason: str,
    approved_by: str,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[MalariaCase]]:
    """
    Update a patient's malaria regimen with validation for line changes.

    Args:
        case_id: Malaria case ID
        new_regimen_id: New regimen ID to switch to
        change_reason: Clinical reason for regimen change
        approved_by: User ID of clinician approving the change
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, case_object)
    """
    try:
        case = MalariaCase.query.get(case_id)
        if not case:
            return False, f"Malaria case {case_id} not found", None

        new_regimen = MalariaRegimen.query.get(new_regimen_id)
        if not new_regimen:
            return False, f"Malaria regimen {new_regimen_id} not found", None

        # Get current regimen for comparison
        current_regimen = case.current_regimen
        if not current_regimen:
            # First regimen assignment - no line change validation needed
            case.current_regimen_id = new_regimen_id
            db.session.commit()
            logger.info(f"Set initial regimen {new_regimen.regimen_code} for case {case_id}")
            return True, "Initial regimen assigned successfully", case

        # Check if this is a line change
        current_line = current_regimen.line_of_therapy
        new_line = new_regimen.line_of_therapy

        if new_line > current_line:
            # Moving to higher line (e.g., 1st to 2nd line) - requires clinical justification
            if not change_reason or len(change_reason.strip()) < 10:
                return False, "Clinical reason required for line advancement", None

            logger.warning(f"Advancing patient {case.patient_id} from line {current_line} to {new_line}")
            # In a real system, this might trigger additional review or notification

        elif new_line < current_line:
            # Moving to lower line - requires strong justification
            if not change_reason or len(change_reason.strip()) < 20:
                return False, "Strong clinical justification required for line reduction", None

            logger.warning(f"Reducing patient {case.patient_id} from line {current_line} to {new_line}")

        # Same line or downward move - update regimen
        case.current_regimen_id = new_regimen_id
        db.session.commit()

        logger.info(f"Updated regimen for case {case_id} to {new_regimen.regimen_code}")
        return True, "Malaria regimen updated successfully", case

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error updating malaria regimen: {e}")
        return False, f"Failed to update malaria regimen: {str(e)}", None


def record_treatment_administered(
    case_id: str,
    administered_as_directly_observed: bool = False,
    date_administered: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[MalariaTreatment]]:
    """
    Record a malaria treatment dose administered (either self-observed or directly observed).

    Args:
        case_id: Malaria case ID
        administered_as_directly_observed: Whether the treatment was administered under direct observation (DOT)
        date_administered: Date and time the treatment was administered (defaults to now)
        encounter_id: Associated encounter ID (if DOT)

    Returns:
        Tuple of (success, message, treatment_object)
    """
    try:
        case = MalariaCase.query.get(case_id)
        if not case:
            return False, f"Malaria case {case_id} not found", None

        if date_administered is None:
            date_administered = datetime.now(timezone.utc)

        # Determine the next treatment number for this case
        last_treatment = MalariaTreatment.query.filter_by(malaria_case_id=case_id)\
            .order_by(MalariaTreatment.date_administered.desc())\
            .first()
        dose_number = (last_treatment.dose_number + 1) if last_treatment and last_treatment.dose_number else 1

        # Create treatment administered record
        treatment = MalariaTreatment(
            malaria_case_id=case_id,
            dose_number=dose_number,
            administered_as_directly_observed=administered_as_directly_observed,
            date_administered=date_administered,
            encounter_id=encounter_id
        )

        db.session.add(treatment)
        db.session.commit()

        logger.info(f"Recorded treatment dose {dose_number} for case {case_id} (DOT: {administered_as_directly_observed})")
        return True, "Treatment administered recorded successfully", treatment

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording treatment administered: {e}")
        return False, f"Failed to record treatment administered: {str(e)}", None


def record_lab_result(
    case_id: str,
    test_type: str,
    result_value: Optional[str] = None,
    result_interpretation: Optional[str] = None,
    test_date: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[MalariaLabResult]]:
    """
    Record a malaria laboratory test result.

    Args:
        case_id: Malaria case ID
        test_type: Type of test (microscopy, RDT, PCR, hemoglobin, etc.)
        result_value: Result value (e.g., parasite density, hemoglobin level)
        result_interpretation: Result interpretation (positive, negative, etc.)
        test_date: Date of test (defaults to now)
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, lab_result_object)
    """
    try:
        case = MalariaCase.query.get(case_id)
        if not case:
            return False, f"Malaria case {case_id} not found", None

        if test_date is None:
            test_date = datetime.now(timezone.utc)

        lab_result = MalariaLabResult(
            malaria_case_id=case_id,
            test_type=test_type,
            result_value=result_value,
            result_interpretation=result_interpretation,
            test_date=test_date,
            encounter_id=encounter_id
        )

        db.session.add(lab_result)
        db.session.commit()

        logger.info(f"Recorded lab result for case {case_id}: {test_type} {result_value} ({result_interpretation})")
        return True, "Lab result recorded successfully", lab_result

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording lab result: {e}")
        return False, f"Failed to record lab result: {str(e)}", None


def get_current_regimen(case_id: str) -> Optional[MalariaRegimen]:
    """
    Get the current malaria regimen for a case.

    Args:
        case_id: Malaria case ID

    Returns:
        Current malaria regimen or None if not found
    """
    try:
        case = MalariaCase.query.get(case_id)
        if not case:
            return None
        return case.current_regimen
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting current regimen: {e}")
        return None


def get_latest_lab_result(case_id: str) -> Optional[MalariaLabResult]:
    """
    Get the most recent lab test for a case.

    Args:
        case_id: Malaria case ID

    Returns:
        Most recent lab result or None if not found
    """
    try:
        return MalariaLabResult.query.filter_by(malaria_case_id=case_id)\
            .order_by(MalariaLabResult.test_date.desc())\
            .first()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting latest lab result: {e}")
        return None


def get_treatment_summary(case_id: str, limit: int = 10) -> List[MalariaTreatment]:
    """
    Get recent treatments administered for a case.

    Args:
        case_id: Malaria case ID
        limit: Maximum number of treatments to return

    Returns:
        List of treatments ordered by date (most recent first)
    """
    try:
        return MalariaTreatment.query.filter_by(malaria_case_id=case_id)\
            .order_by(MalariaTreatment.date_administered.desc())\
            .limit(limit)\
            .all()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting treatment summary: {e}")
        return []


# Formulary management functions
def get_malaria_formulary() -> List[MalariaRegimen]:
    """
    Get all active malaria regimens in the formulary.

    Returns:
        List of active malaria regimens
    """
    try:
        now = datetime.now(timezone.utc).date()
        return MalariaRegimen.query.filter(
            db.or_(MalariaRegimen.effective_to.is_(None), MalariaRegimen.effective_to >= now),
            MalariaRegimen.effective_from <= now
        ).order_by(MalariaRegimen.line_of_therapy, MalariaRegimen.regimen_code).all()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting malaria formulary: {e}")
        return []


def is_malaria_regimen_valid(regimen_id: str) -> bool:
    """
    Check if a malaria regimen is currently valid/effective.

    Args:
        regimen_id: Malaria regimen ID to check

    Returns:
        True if regimen is valid, False otherwise
    """
    try:
        regimen = MalariaRegimen.query.get(regimen_id)
        if not regimen:
            return False

        now = datetime.now(timezone.utc).date()
        if regimen.effective_to and regimen.effective_to < now:
            return False
        if regimen.effective_from > now:
            return False
        return True
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error checking malaria regimen validity: {e}")
        return False


def log_malaria_formulary_change(
    regimen_id: str,
    change_type: str,  # 'create', 'update', 'retire'
    changed_by: str,
    change_notes: Optional[str] = None
) -> bool:
    """
    Log a change to the malaria formulary for audit purposes.

    Args:
        regimen_id: Malaria regimen ID that was changed
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
            f"Malaria formulary change: {change_type} regimen {regimen_id} "
            f"by {changed_by}. Notes: {change_notes or 'None'}"
        )
        return True
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error logging malaria formulary change: {e}")
        return False
