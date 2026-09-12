"""
HIV/ART Workflow Engine.

Aligns with Kenya's MoH HIV/AIDS guidelines and ART protocols.
Enforces ART enrollment, regimen management, adherence monitoring,
and clinical workflows for HIV treatment and care.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from extensions import db

from .models import (
    AdherenceVisit,
    ARTEnrollment,
    ARTRegimen,
    CD4Count,
    ViralLoad,
    WHOStage,
)

logger = logging.getLogger(__name__)


def create_art_enrollment(
    patient_id: str,
    art_number: str,
    baseline_cd4: Optional[int] = None,
    baseline_who_stage: Optional[int] = None,
    art_start_date: Optional[datetime] = None,
    facility_enrolled_at: Optional[str] = None,
    encounter_id: Optional[int] = None,
    current_regimen_id: Optional[str] = None
) -> Tuple[bool, str, Optional[ARTEnrollment]]:
    """
    Create a new ART enrollment for a patient.

    Args:
        patient_id: Patient identifier
        art_number: Unique ART number for the patient
        baseline_cd4: Baseline CD4 count at enrollment (cells/µL)
        baseline_who_stage: Baseline WHO clinical stage (1-4)
        art_start_date: Date ART was started (defaults to now)
        facility_enrolled_at: Facility where enrollment occurred
        encounter_id: Associated encounter ID
        current_regimen_id: Initial ART regimen ID

    Returns:
        Tuple of (success, message, enrollment_object)
    """
    try:
        # Check for duplicate enrollment (active enrollment without end date)
        existing_enrollment = ARTEnrollment.query.filter_by(
            patient_id=patient_id
        ).first()

        if existing_enrollment:
            return False, f"Patient {patient_id} already has an active ART enrollment", None

        # Validate ART number uniqueness
        if ARTEnrollment.query.filter_by(art_number=art_number).first():
            return False, f"ART number {art_number} is already assigned to another patient", None

        # Validate regimen if provided
        if current_regimen_id:
            regimen = ARTRegimen.query.get(current_regimen_id)
            if not regimen:
                return False, f"ART regimen with ID {current_regimen_id} not found", None

        # Set defaults
        if art_start_date is None:
            art_start_date = datetime.now(timezone.utc)

        # Create enrollment
        enrollment = ARTEnrollment(
            patient_id=patient_id,
            art_number=art_number,
            baseline_cd4=baseline_cd4,
            baseline_who_stage=baseline_who_stage,
            art_start_date=art_start_date,
            facility_enrolled_at=facility_enrolled_at,
            encounter_id=encounter_id,
            current_regimen_id=current_regimen_id
        )

        db.session.add(enrollment)
        db.session.commit()

        logger.info(f"Created ART enrollment for patient {patient_id} with ART number {art_number}")
        return True, "ART enrollment created successfully", enrollment

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error creating ART enrollment: {e}")
        return False, f"Failed to create ART enrollment: {str(e)}", None


def update_art_regimen(
    enrollment_id: str,
    new_regimen_id: str,
    change_reason: str,
    approved_by: str,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[ARTEnrollment]]:
    """
    Update a patient's ART regimen with validation for line changes.

    Args:
        enrollment_id: ART enrollment ID
        new_regimen_id: New regimen ID to switch to
        change_reason: Clinical reason for regimen change
        approved_by: User ID of clinician approving the change
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, enrollment_object)
    """
    try:
        enrollment = ARTEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"ART enrollment {enrollment_id} not found", None

        new_regimen = ARTRegimen.query.get(new_regimen_id)
        if not new_regimen:
            return False, f"ART regimen {new_regimen_id} not found", None

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
        return True, "ART regimen updated successfully", enrollment

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error updating ART regimen: {e}")
        return False, f"Failed to update ART regimen: {str(e)}", None


def record_adherence_visit(
    enrollment_id: str,
    pills_dispensed: int,
    pills_returned: int,
    visit_date: Optional[datetime] = None,
    viral_load_ordered: bool = False,
    cd4_ordered: bool = False,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[AdherenceVisit]]:
    """
    Record an adherence visit and calculate adherence percentage.

    Args:
        enrollment_id: ART enrollment ID
        pills_dispensed: Number of pills dispensed at last visit
        pills_returned: Number of pills returned at this visit
        visit_date: Date of visit (defaults to now)
        viral_load_ordered: Whether viral load test was ordered
        cd4_ordered: Whether CD4 test was ordered
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, visit_object)
    """
    try:
        enrollment = ARTEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"ART enrollment {enrollment_id} not found", None

        if visit_date is None:
            visit_date = datetime.now(timezone.utc)

        # Calculate adherence
        if pills_dispensed <= 0:
            adherence_percentage = 0.0
        else:
            pills_taken = max(0, pills_dispensed - pills_returned)
            adherence_percentage = min(100.0, (pills_taken / pills_dispensed) * 100)

        # Determine adherence category
        if adherence_percentage >= 95:
            adherence_category = "good"
        elif adherence_percentage >= 80:
            adherence_category = "fair"
        else:
            adherence_category = "poor"

        # Create adherence visit record
        visit = AdherenceVisit(
            art_enrollment_id=enrollment_id,
            pills_dispensed=pills_dispensed,
            pills_returned=pills_returned,
            adherence_percentage=adherence_percentage,
            adherence_category=adherence_category,
            viral_load_ordered=viral_load_ordered,
            cd4_ordered=cd4_ordered,
            visit_date=visit_date,
            encounter_id=encounter_id
        )

        db.session.add(visit)
        db.session.commit()

        logger.info(f"Recorded adherence visit for enrollment {enrollment_id}: {adherence_percentage:.1f}% ({adherence_category})")
        return True, "Adherence visit recorded successfully", visit

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording adherence visit: {e}")
        return False, f"Failed to record adherence visit: {str(e)}", None


def check_missed_visits() -> List[AdherenceVisit]:
    """
    Check for missed adherence visits (no visit in last 7 days).
    This function would typically be called by a Celery periodic task.

    Returns:
        List of adherence visits that are considered missed
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)

        # Find enrollments without visits in the last 7 days
        # Get the most recent visit for each enrollment
        subquery = db.session.query(
            AdherenceVisit.art_enrollment_id,
            db.func.max(AdherenceVisit.visit_date).label('last_visit')
        ).group_by(AdherenceVisit.art_enrollment_id).subquery()

        missed_visits = db.session.query(AdherenceVisit).join(
            subquery,
            db.and_(
                AdherenceVisit.art_enrollment_id == subquery.c.art_enrollment_id,
                AdherenceVisit.visit_date == subquery.c.last_visit
            )
        ).filter(
            subquery.c.last_visit < cutoff_date
        ).all()

        logger.info(f"Found {len(missed_visits)} patients with missed adherence visits")
        return missed_visits

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error checking missed visits: {e}")
        return []


def record_viral_load(
    enrollment_id: str,
    viral_load_copies: Optional[int],
    test_type: str = "routine",
    test_date: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[ViralLoad]]:
    """
    Record a viral load test result.

    Args:
        enrollment_id: ART enrollment ID
        viral_load_copies: Viral load in copies/mL (None if undetectable)
        test_type: Type of test (routine, diagnostic, confirmation)
        test_date: Date of test (defaults to now)
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, viral_load_object)
    """
    try:
        enrollment = ARTEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"ART enrollment {enrollment_id} not found", None

        if test_date is None:
            test_date = datetime.now(timezone.utc)

        viral_load = ViralLoad(
            art_enrollment_id=enrollment_id,
            viral_load_copies=viral_load_copies,
            test_type=test_type,
            test_date=test_date,
            encounter_id=encounter_id
        )

        db.session.add(viral_load)
        db.session.commit()

        result_str = f"{viral_load_copies} copies/mL" if viral_load_copies is not None else "undetectable"
        logger.info(f"Recorded viral load for enrollment {enrollment_id}: {result_str}")
        return True, "Viral load recorded successfully", viral_load

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording viral load: {e}")
        return False, f"Failed to record viral load: {str(e)}", None


def record_cd4_count(
    enrollment_id: str,
    cd4_count: Optional[int],
    cd4_percent: Optional[float] = None,
    test_date: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[CD4Count]]:
    """
    Record a CD4 count test result.

    Args:
        enrollment_id: ART enrollment ID
        cd4_count: CD4 count in cells/µL
        cd4_percent: CD4 percentage (optional)
        test_date: Date of test (defaults to now)
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, cd4_object)
    """
    try:
        enrollment = ARTEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"ART enrollment {enrollment_id} not found", None

        if test_date is None:
            test_date = datetime.now(timezone.utc)

        cd4 = CD4Count(
            art_enrollment_id=enrollment_id,
            cd4_count=cd4_count,
            cd4_percent=cd4_percent,
            test_date=test_date,
            encounter_id=encounter_id
        )

        db.session.add(cd4)
        db.session.commit()

        logger.info(f"Recorded CD4 count for enrollment {enrollment_id}: {cd4_count} cells/µL")
        return True, "CD4 count recorded successfully", cd4

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording CD4 count: {e}")
        return False, f"Failed to record CD4 count: {str(e)}", None


def record_who_stage(
    enrollment_id: str,
    who_stage: int,
    defining_conditions: Optional[str] = None,
    assessment_date: Optional[datetime] = None,
    encounter_id: Optional[int] = None
) -> Tuple[bool, str, Optional[WHOStage]]:
    """
    Record a WHO clinical staging assessment.

    Args:
        enrollment_id: ART enrollment ID
        who_stage: WHO clinical stage (1-4)
        defining_conditions: Conditions that define the stage
        assessment_date: Date of assessment (defaults to now)
        encounter_id: Associated encounter ID

    Returns:
        Tuple of (success, message, who_stage_object)
    """
    try:
        if who_stage < 1 or who_stage > 4:
            return False, "WHO stage must be between 1 and 4", None

        enrollment = ARTEnrollment.query.get(enrollment_id)
        if not enrollment:
            return False, f"ART enrollment {enrollment_id} not found", None

        if assessment_date is None:
            assessment_date = datetime.now(timezone.utc)

        who_stage_record = WHOStage(
            art_enrollment_id=enrollment_id,
            who_stage=who_stage,
            defining_conditions=defining_conditions,
            assessment_date=assessment_date,
            encounter_id=encounter_id
        )

        db.session.add(who_stage_record)
        db.session.commit()

        logger.info(f"Recorded WHO stage {who_stage} for enrollment {enrollment_id}")
        return True, "WHO stage recorded successfully", who_stage_record

    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        db.session.rollback()
        logger.error(f"Error recording WHO stage: {e}")
        return False, f"Failed to record WHO stage: {str(e)}", None


def get_current_regimen(enrollment_id: str) -> Optional[ARTRegimen]:
    """
    Get the current ART regimen for an enrollment.

    Args:
        enrollment_id: ART enrollment ID

    Returns:
        Current ART regimen or None if not found
    """
    try:
        enrollment = ARTEnrollment.query.get(enrollment_id)
        if not enrollment:
            return None
        return enrollment.current_regimen
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting current regimen: {e}")
        return None


def get_latest_viral_load(enrollment_id: str) -> Optional[ViralLoad]:
    """
    Get the most recent viral load test for an enrollment.

    Args:
        enrollment_id: ART enrollment ID

    Returns:
        Most recent viral load or None if not found
    """
    try:
        return ViralLoad.query.filter_by(art_enrollment_id=enrollment_id)\
            .order_by(ViralLoad.test_date.desc())\
            .first()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting latest viral load: {e}")
        return None


def get_latest_cd4_count(enrollment_id: str) -> Optional[CD4Count]:
    """
    Get the most recent CD4 count for an enrollment.

    Args:
        enrollment_id: ART enrollment ID

    Returns:
        Most recent CD4 count or None if not found
    """
    try:
        return CD4Count.query.filter_by(art_enrollment_id=enrollment_id)\
            .order_by(CD4Count.test_date.desc())\
            .first()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting latest CD4 count: {e}")
        return None


def get_latest_who_stage(enrollment_id: str) -> Optional[WHOStage]:
    """
    Get the most recent WHO stage assessment for an enrollment.

    Args:
        enrollment_id: ART enrollment ID

    Returns:
        Most recent WHO stage or None if not found
    """
    try:
        return WHOStage.query.filter_by(art_enrollment_id=enrollment_id)\
            .order_by(WHOStage.assessment_date.desc())\
            .first()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting latest WHO stage: {e}")
        return None


def get_adherence_summary(enrollment_id: str, limit: int = 12) -> List[AdherenceVisit]:
    """
    Get recent adherence visits for an enrollment.

    Args:
        enrollment_id: ART enrollment ID
        limit: Maximum number of visits to return

    Returns:
        List of adherence visits ordered by date (most recent first)
    """
    try:
        return AdherenceVisit.query.filter_by(art_enrollment_id=enrollment_id)\
            .order_by(AdherenceVisit.visit_date.desc())\
            .limit(limit)\
            .all()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting adherence summary: {e}")
        return []


# Formulary management functions
def get_art_formulary() -> List[ARTRegimen]:
    """
    Get all active ART regimens in the formulary.

    Returns:
        List of active ART regimens
    """
    try:
        now = datetime.now(timezone.utc).date()
        return ARTRegimen.query.filter(
            db.or_(ARTRegimen.effective_to.is_(None), ARTRegimen.effective_to >= now),
            ARTRegimen.effective_from <= now
        ).order_by(ARTRegimen.line_of_therapy, ARTRegimen.regimen_code).all()
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error getting ART formulary: {e}")
        return []


def is_regimen_valid(regimen_id: str) -> bool:
    """
    Check if an ART regimen is currently valid/effective.

    Args:
        regimen_id: ART regimen ID to check

    Returns:
        True if regimen is valid, False otherwise
    """
    try:
        regimen = ARTRegimen.query.get(regimen_id)
        if not regimen:
            return False

        now = datetime.now(timezone.utc).date()
        if regimen.effective_to and regimen.effective_to < now:
            return False
        if regimen.effective_from > now:
            return False
        return True
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error checking regimen validity: {e}")
        return False


def log_formulary_change(
    regimen_id: str,
    change_type: str,  # 'create', 'update', 'retire'
    changed_by: str,
    change_notes: Optional[str] = None
) -> bool:
    """
    Log a change to the ART formulary for audit purposes.

    Args:
        regimen_id: ART regimen ID that was changed
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
            f"ART formulary change: {change_type} regimen {regimen_id} "
            f"by {changed_by}. Notes: {change_notes or 'None'}"
        )
        return True
    except Exception as e:  # noqa: BLE001  # Broad catch intentional: return error message to caller
        logger.error(f"Error logging formulary change: {e}")
        return False
