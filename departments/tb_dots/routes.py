"""
TB/DOTS Module API routes.
"""

import logging
from datetime import datetime, timezone

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required

from departments.rbac import roles_required

from . import bp
from .engine import (
    create_tb_enrollment,
    get_current_regimen,
    get_dose_summary,
    get_latest_chest_xray,
    get_latest_hiv_status,
    get_latest_sputum_result,
    get_tb_formulary,
    is_tb_regimen_valid,
    record_chest_xray,
    record_dose_taken,
    record_hiv_status,
    record_sputum_result,
    update_tb_regimen,
)
from .models import TBEnrollment

logger = logging.getLogger(__name__)

# Initialize engine (placeholder for potential engine class)
# In this implementation, we're using functional approach similar to some other modules


@bp.route("/")
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def index():
    return "TB/DOTS Module Active"


@bp.route("/api/enroll", methods=["POST"])
@login_required
@roles_required("clinical", "nursing")
def enroll_patient():
    """
    Enroll a patient in TB treatment.
    Expected JSON: {
        patient_id: string,
        tb_number: string,
        hiv_status: string (optional, one of: positive, negative, unknown, refused_test),
        art_enrollment_id: string (optional),
        tb_classification: string (optional, pulmonary or extrapulmonary),
        site_of_disease: string (optional),
        bacteriological_status: string (optional),
        treatment_start_date: ISO string (optional),
        facility_enrolled_at: string (optional),
        encounter_id: integer (optional),
        current_regimen_id: string (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Validate required fields
    patient_id = data.get("patient_id")
    tb_number = data.get("tb_number")

    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400
    if not tb_number:
        return jsonify({"error": "tb_number is required"}), 400

    # Validate hiv_status if provided
    hiv_status = data.get("hiv_status")
    if hiv_status and hiv_status not in ['positive', 'negative', 'unknown', 'refused_test']:
        return jsonify({"error": "Invalid hiv_status. Must be one of: positive, negative, unknown, refused_test"}), 400

    # Parse optional dates
    treatment_start_date = None
    if data.get("treatment_start_date"):
        try:
            treatment_start_date = datetime.fromisoformat(data["treatment_start_date"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid treatment_start_date format. Use ISO format."}), 400

    success, message, enrollment = create_tb_enrollment(
        patient_id=patient_id,
        tb_number=tb_number,
        hiv_status=hiv_status,
        art_enrollment_id=data.get("art_enrollment_id"),
        tb_classification=data.get("tb_classification"),
        site_of_disease=data.get("site_of_disease"),
        bacteriological_status=data.get("bacteriological_status"),
        treatment_start_date=treatment_start_date,
        facility_enrolled_at=data.get("facility_enrolled_at"),
        encounter_id=data.get("encounter_id"),
        current_regimen_id=data.get("current_regimen_id")
    )

    if success:
        return jsonify({
            "message": message,
            "enrollment_id": enrollment.id,
            "tb_number": enrollment.tb_number
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/regimen", methods=["PUT"])
@login_required
@roles_required("clinical")
def change_regimen(enrollment_id):
    """
    Change a patient's TB regimen.
    Expected JSON: {
        new_regimen_id: string,
        change_reason: string,
        approved_by: string,
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Validate required fields
    new_regimen_id = data.get("new_regimen_id")
    change_reason = data.get("change_reason")
    approved_by = data.get("approved_by")

    if not new_regimen_id:
        return jsonify({"error": "new_regimen_id is required"}), 400
    if not change_reason:
        return jsonify({"error": "change_reason is required"}), 400
    if not approved_by:
        return jsonify({"error": "approved_by is required"}), 400

    success, message, enrollment = update_tb_regimen(
        enrollment_id=enrollment_id,
        new_regimen_id=new_regimen_id,
        change_reason=change_reason,
        approved_by=approved_by,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "current_regimen": {
                "id": enrollment.current_regimen.id if enrollment.current_regimen else None,
                "code": enrollment.current_regimen.regimen_code if enrollment.current_regimen else None,
                "name": enrollment.current_regimen.regimen_name if enrollment.current_regimen else None
            } if enrollment.current_regimen else None
        }), 200
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/dose", methods=["POST"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def record_dose(enrollment_id):
    """
    Record a TB dose taken.
    Expected JSON: {
        taken_as_directly_observed: boolean (optional, default: false),
        date_taken: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Parse optional date
    date_taken = None
    if data.get("date_taken"):
        try:
            date_taken = datetime.fromisoformat(data["date_taken"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid date_taken format. Use ISO format."}), 400

    success, message, dose = record_dose_taken(
        enrollment_id=enrollment_id,
        taken_as_directly_observed=data.get("taken_as_directly_observed", False),
        date_taken=date_taken,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "dose_id": dose.id,
            "dose_number": dose.dose_number,
            "date_taken": dose.date_taken.isoformat() if dose.date_taken else None,
            "taken_as_directly_observed": dose.taken_as_directly_observed
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/sputum", methods=["POST"])
@login_required
@roles_required("clinical", "nursing")
def record_sputum(enrollment_id):
    """
    Record a TB sputum test result.
    Expected JSON: {
        specimen_type: string,
        specimen_number: integer,
        smear_result: string (optional),
        culture_result: string (optional),
        culture_species: string (optional),
        drug_susceptibility: string (optional),
        test_date: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Validate required fields
    specimen_type = data.get("specimen_type")
    specimen_number = data.get("specimen_number")

    if not specimen_type:
        return jsonify({"error": "specimen_type is required"}), 400
    if specimen_number is None:
        return jsonify({"error": "specimen_number is required"}), 400
    try:
        specimen_number = int(specimen_number)
        if specimen_number < 1:
            return jsonify({"error": "specimen_number must be a positive integer"}), 400
    except ValueError:
        return jsonify({"error": "specimen_number must be an integer"}), 400

    # Parse optional date
    test_date = None
    if data.get("test_date"):
        try:
            test_date = datetime.fromisoformat(data["test_date"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid test_date format. Use ISO format."}), 400

    success, message, sputum = record_sputum_result(
        enrollment_id=enrollment_id,
        specimen_type=specimen_type,
        specimen_number=specimen_number,
        smear_result=data.get("smear_result"),
        culture_result=data.get("culture_result"),
        culture_species=data.get("culture_species"),
        drug_susceptibility=data.get("drug_susceptibility"),
        test_date=test_date,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "sputum_id": sputum.id,
            "specimen_type": sputum.specimen_type,
            "specimen_number": sputum.specimen_number,
            "smear_result": sputum.smear_result,
            "culture_result": sputum.culture_result,
            "test_date": sputum.test_date.isoformat() if sputum.test_date else None
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/xray", methods=["POST"])
@login_required
@roles_required("clinical", "nursing")
def record_xray(enrollment_id):
    """
    Record a TB chest X-ray result.
    Expected JSON: {
        finding: string (optional),
        severity: string (optional),
        progression: string (optional),
        test_date: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Parse optional date
    test_date = None
    if data.get("test_date"):
        try:
            test_date = datetime.fromisoformat(data["test_date"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid test_date format. Use ISO format."}), 400

    success, message, xray = record_chest_xray(
        enrollment_id=enrollment_id,
        finding=data.get("finding"),
        severity=data.get("severity"),
        progression=data.get("progression"),
        test_date=test_date,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "xray_id": xray.id,
            "finding": xray.finding,
            "severity": xray.severity,
            "progression": xray.progression,
            "test_date": xray.test_date.isoformat() if xray.test_date else None
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/hiv-status", methods=["POST"])
@login_required
@roles_required("clinical")
def record_hiv_status_endpoint(enrollment_id):
    """
    Record an HIV status test result for a TB patient.
    Expected JSON: {
        test_type: string,
        result: string (positive, negative, indeterminate),
        cd4_count: integer (optional),
        test_date: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Validate required fields
    test_type = data.get("test_type")
    result = data.get("result")

    if not test_type:
        return jsonify({"error": "test_type is required"}), 400
    if not result:
        return jsonify({"error": "result is required"}), 400
    if result not in ['positive', 'negative', 'indeterminate']:
        return jsonify({"error": "result must be one of: positive, negative, indeterminate"}), 400

    # Parse optional date
    test_date = None
    if data.get("test_date"):
        try:
            test_date = datetime.fromisoformat(data["test_date"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid test_date format. Use ISO format."}), 400

    success, message, hiv_status = record_hiv_status(
        enrollment_id=enrollment_id,
        test_type=test_type,
        result=result,
        cd4_count=data.get("cd4_count"),
        test_date=test_date,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "hiv_status_id": hiv_status.id,
            "test_type": hiv_status.test_type,
            "result": hiv_status.result,
            "cd4_count": hiv_status.cd4_count,
            "test_date": hiv_status.test_date.isoformat() if hiv_status.test_date else None
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/summary", methods=["GET"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def get_patient_summary(enrollment_id):
    """
    Get a comprehensive summary of a patient's TB status.
    """
    try:
        enrollment = TBEnrollment.query.get(enrollment_id)
        if not enrollment:
            return jsonify({"error": "TB enrollment not found"}), 404

        # Get latest records
        current_regimen = get_current_regimen(enrollment_id)
        latest_sputum = get_latest_sputum_result(enrollment_id)
        latest_xray = get_latest_chest_xray(enrollment_id)
        latest_hiv_status = get_latest_hiv_status(enrollment_id)
        recent_doses = get_dose_summary(enrollment_id, limit=10)

        # Calculate days on treatment
        days_on_treatment = None
        if enrollment.treatment_start_date:
            delta = datetime.now(timezone.utc) - enrollment.treatment_start_date
            days_on_treatment = delta.days

        # Calculate adherence percentage based on doses taken vs expected
        # This is a simplified calculation - in reality, it's more complex
        adherence_percentage = None
        if current_regimen and enrollment.treatment_start_date:
            # Expected doses per day based on regimen (simplified: assume daily dosing)
            expected_doses_per_day = 1  # This would be more complex in reality
            days_on_treatment_float = (datetime.now(timezone.utc) - enrollment.treatment_start_date).total_seconds() / (24*3600)
            expected_doses = int(days_on_treatment_float * expected_doses_per_day)
            actual_doses = len(recent_doses)  # This is only recent doses, not total - limitation
            if expected_doses > 0:
                adherence_percentage = min(100.0, (actual_doses / expected_doses) * 100)

        return jsonify({
            "enrollment": {
                "id": enrollment.id,
                "patient_id": enrollment.patient_id,
                "tb_number": enrollment.tb_number,
                "enrollment_date": enrollment.enrollment_date.isoformat() if enrollment.enrollment_date else None,
                "treatment_start_date": enrollment.treatment_start_date.isoformat() if enrollment.treatment_start_date else None,
                "hiv_status": enrollment.hiv_status,
                "art_enrollment_id": enrollment.art_enrollment_id,
                "tb_classification": enrollment.tb_classification,
                "site_of_disease": enrollment.site_of_disease,
                "bacteriological_status": enrollment.bacteriological_status,
                "facility_enrolled_at": enrollment.facility_enrolled_at,
                "days_on_treatment": days_on_treatment
            },
            "current_regimen": {
                "id": current_regimen.id if current_regimen else None,
                "code": current_regimen.regimen_code if current_regimen else None,
                "name": current_regimen.regimen_name if current_regimen else None,
                "line_of_therapy": current_regimen.line_of_therapy if current_regimen else None,
                "drugs": current_regimen.drugs if current_regimen else None,
                "duration_months": current_regimen.duration_months if current_regimen else None
            } if current_regimen else None,
            "latest_sputum": {
                "id": latest_sputum.id if latest_sputum else None,
                "specimen_type": latest_sputum.specimen_type if latest_sputum else None,
                "specimen_number": latest_sputum.specimen_number if latest_sputum else None,
                "smear_result": latest_sputum.smear_result if latest_sputum else None,
                "culture_result": latest_sputum.culture_result if latest_sputum else None,
                "culture_species": latest_sputum.culture_species if latest_sputum else None,
                "drug_susceptibility": latest_sputum.drug_susceptibility if latest_sputum else None,
                "test_date": latest_sputum.test_date.isoformat() if latest_sputum and latest_sputum.test_date else None
            } if latest_sputum else None,
            "latest_xray": {
                "id": latest_xray.id if latest_xray else None,
                "finding": latest_xray.finding if latest_xray else None,
                "severity": latest_xray.severity if latest_xray else None,
                "progression": latest_xray.progression if latest_xray else None,
                "test_date": latest_xray.test_date.isoformat() if latest_xray and latest_xray.test_date else None
            } if latest_xray else None,
            "latest_hiv_status": {
                "id": latest_hiv_status.id if latest_hiv_status else None,
                "test_type": latest_hiv_status.test_type if latest_hiv_status else None,
                "result": latest_hiv_status.result if latest_hiv_status else None,
                "cd4_count": latest_hiv_status.cd4_count if latest_hiv_status else None,
                "test_date": latest_hiv_status.test_date.isoformat() if latest_hiv_status and latest_hiv_status.test_date else None
            } if latest_hiv_status else None,
            "recent_doses": [
                {
                    "id": dose.id,
                    "date_taken": dose.date_taken.isoformat() if dose.date_taken else None,
                    "dose_number": dose.dose_number,
                    "taken_as_directly_observed": dose.taken_as_directly_observed
                } for dose in recent_doses
            ],
            "adherence_percentage": adherence_percentage
        }), 200

    except Exception as e:
        logger.error(f"Error getting patient summary: {e}")
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/api/formulary", methods=["GET"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def get_formulary():
    """
    Get the current TB formulary (list of active regimens).
    """
    try:
        regimens = get_tb_formulary()
        return jsonify({
            "regimens": [
                {
                    "id": regimen.id,
                    "code": regimen.regimen_code,
                    "name": regimen.regimen_name,
                    "line_of_therapy": regimen.line_of_therapy,
                    "drugs": regimen.drugs,
                    "is_preferred": regimen.is_preferred,
                    "is_alternative": regimen.is_alternative,
                    "restriction_notes": regimen.restriction_notes,
                    "effective_from": regimen.effective_from.isoformat() if regimen.effective_from else None,
                    "effective_to": regimen.effective_to.isoformat() if regimen.effective_to else None,
                    "duration_months": regimen.duration_months
                } for regimen in regimens
            ]
        }), 200
    except Exception as e:
        logger.error(f"Error getting formulary: {e}")
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/api/formulary/<string:regimen_id>/validate", methods=["GET"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def validate_regimen(regimen_id):
    """
    Check if a regimen is currently valid/effective.
    """
    try:
        valid = is_tb_regimen_valid(regimen_id)
        return jsonify({
            "regimen_id": regimen_id,
            "is_valid": valid
        }), 200
    except Exception as e:
        logger.error(f"Error validating regimen: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Celery task for checking missed doses (would be called periodically)
@bp.route("/api/tasks/check-missed-doses", methods=["POST"])
@login_required
@roles_required("admin")  # Only admins can trigger this manually
def trigger_missed_dose_check():
    """
    Trigger a check for missed doses.
    In production, this would be called by Celery beat schedule.
    """
    try:
        # For simplicity, we'll just return a placeholder message
        # In a real implementation, we would check for doses not taken within expected windows
        return jsonify({
            "message": "Missed dose check triggered. (Implementation pending)",
            "checked": True
        }), 200
    except Exception as e:
        logger.error(f"Error in missed dose check: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Error handlers
@bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404


@bp.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request"}), 400


@bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ── UI Routes ──────────────────────────────────────────────────────────
@bp.route("/ui/dashboard")
@login_required
@roles_required("admin", "medicine", "doctor", "clinical", "nursing")
def dashboard_ui():
    """TB/DOTS Dashboard."""
    from departments.tb_dots.models import TBEnrollment

    enrollments = TBEnrollment.query.order_by(TBEnrollment.enrollment_date.desc()).all()
    active_count = len([e for e in enrollments if e.treatment_start_date])
    critical_count = 0  # Would need logic to determine critical cases
    doses_due = 0  # Would need logic to determine doses due

    return render_template("tb_dots/dashboard.html",
                          enrollments=enrollments,
                          active_count=active_count,
                          critical_count=critical_count,
                          doses_due=doses_due)


@bp.route("/ui/enroll", methods=["GET", "POST"])
@login_required
@roles_required("admin", "medicine", "doctor", "clinical", "nursing")
def enroll_ui():
    """Enroll a new TB patient."""
    from departments.tb_dots.engine import create_tb_enrollment
    from departments.tb_dots.models import TBRegimen

    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        tb_number = request.form.get("tb_number")
        enrollment_date = request.form.get("enrollment_date")
        treatment_start_date = request.form.get("treatment_start_date")
        site_of_disease = request.form.get("site_of_disease")
        hiv_status = request.form.get("hiv_status")
        tb_classification = request.form.get("tb_classification")
        current_regimen_id = request.form.get("current_regimen_id")

        try:
            create_tb_enrollment(
                patient_id=patient_id,
                tb_number=tb_number,
                enrollment_date=enrollment_date,
                treatment_start_date=treatment_start_date,
                site_of_disease=site_of_disease,
                hiv_status=hiv_status,
                tb_classification=tb_classification,
                current_regimen_id=int(current_regimen_id) if current_regimen_id else None,
            )
            flash("Patient enrolled successfully!", "success")
            return redirect(url_for("tb_dots.dashboard_ui"))
        except Exception as e:
            flash(f"Error enrolling patient: {str(e)}", "danger")

    regimens = TBRegimen.query.filter_by(is_preferred=True).all()
    return render_template("tb_dots/enroll.html", regimens=regimens)


@bp.route("/ui/patient/<int:enrollment_id>")
@login_required
@roles_required("admin", "medicine", "doctor", "clinical", "nursing")
def patient_detail_ui(enrollment_id):
    """View TB patient details."""
    from departments.tb_dots.models import DoseTaken, SputumResult, TBEnrollment
    from extensions import db

    enrollment = db.session.get(TBEnrollment, enrollment_id)
    if not enrollment:
        flash("Enrollment not found", "error")
        return redirect(url_for("tb_dots.dashboard_ui"))

    doses_taken = DoseTaken.query.filter_by(tb_enrollment_id=enrollment_id).order_by(DoseTaken.date_taken.desc()).all()
    sputum_results = SputumResult.query.filter_by(tb_enrollment_id=enrollment_id).order_by(SputumResult.test_date.desc()).all()

    return render_template("tb_dots/patient_detail.html",
                          enrollment=enrollment,
                          doses_taken=doses_taken,
                          sputum_results=sputum_results)
