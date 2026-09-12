"""
Malaria Module API routes.
"""

import logging
from datetime import datetime, timezone

from flask import jsonify, request
from flask_login import login_required

from departments.rbac import roles_required

from . import bp
from .engine import (
    create_malaria_case,
    get_current_regimen,
    get_latest_lab_result,
    get_malaria_formulary,
    get_treatment_summary,
    is_malaria_regimen_valid,
    record_lab_result,
    record_treatment_administered,
    update_malaria_regimen,
)
from .models import MalariaCase

logger = logging.getLogger(__name__)

# Initialize engine (placeholder for potential engine class)
# In this implementation, we're using functional approach similar to some other modules


@bp.route("/")
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def index():
    return "Malaria Module Active"


@bp.route("/api/enroll", methods=["POST"])
@login_required
@roles_required("clinical", "nursing")
def register_case():
    """
    Register a new malaria case.
    Expected JSON: {
        patient_id: string,
        case_number: string,
        malaria_species: string (optional),
        parasite_density: integer (optional),
        diagnosis_method: string (optional),
        severity: string (optional),
        pregnancy_status: string (optional),
        treatment_start_date: ISO string (optional),
        facility_diagnosed_at: string (optional),
        encounter_id: integer (optional),
        current_regimen_id: string (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Validate required fields
    patient_id = data.get("patient_id")
    case_number = data.get("case_number")

    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400
    if not case_number:
        return jsonify({"error": "case_number is required"}), 400

    # Validate malaria_species if provided
    malaria_species = data.get("malaria_species")
    if malaria_species:
        valid_species = ['falciparum', 'vivax', 'ovale', 'malariae', 'knowlesi', 'mixed']
        if malaria_species not in valid_species:
            return jsonify({"error": f"Invalid malaria_species. Must be one of: {', '.join(valid_species)}"}), 400

    # Validate diagnosis_method if provided
    diagnosis_method = data.get("diagnosis_method")
    if diagnosis_method:
        valid_methods = ['microscopy', 'RDT', 'PCR']
        if diagnosis_method not in valid_methods:
            return jsonify({"error": f"Invalid diagnosis_method. Must be one of: {', '.join(valid_methods)}"}), 400

    # Validate severity if provided
    severity = data.get("severity")
    if severity:
        valid_severity = ['uncomplicated', 'severe']
        if severity not in valid_severity:
            return jsonify({"error": f"Invalid severity. Must be one of: {', '.join(valid_severity)}"}), 400

    # Validate pregnancy_status if provided
    pregnancy_status = data.get("pregnancy_status")
    if pregnancy_status:
        valid_pregnancy = ['not_pregnant', 'pregnant_first_trimester', 'pregnant_second_trimester', 'pregnant_third_trimester', 'postpartum']
        if pregnancy_status not in valid_pregnancy:
            return jsonify({"error": f"Invalid pregnancy_status. Must be one of: {', '.join(valid_pregnancy)}"}), 400

    # Parse optional dates
    treatment_start_date = None
    if data.get("treatment_start_date"):
        try:
            treatment_start_date = datetime.fromisoformat(data["treatment_start_date"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid treatment_start_date format. Use ISO format."}), 400

    success, message, case = create_malaria_case(
        patient_id=patient_id,
        case_number=case_number,
        malaria_species=malaria_species,
        parasite_density=data.get("parasite_density"),
        diagnosis_method=diagnosis_method,
        severity=severity,
        pregnancy_status=pregnancy_status,
        treatment_start_date=treatment_start_date,
        facility_diagnosed_at=data.get("facility_diagnosed_at"),
        encounter_id=data.get("encounter_id"),
        current_regimen_id=data.get("current_regimen_id")
    )

    if success:
        return jsonify({
            "message": message,
            "case_id": case.id,
            "case_number": case.case_number
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:case_id>/regimen", methods=["PUT"])
@login_required
@roles_required("clinical")
def change_regimen(case_id):
    """
    Change a patient's malaria regimen.
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

    success, message, case = update_malaria_regimen(
        case_id=case_id,
        new_regimen_id=new_regimen_id,
        change_reason=change_reason,
        approved_by=approved_by,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "current_regimen": {
                "id": case.current_regimen.id if case.current_regimen else None,
                "code": case.current_regimen.regimen_code if case.current_regimen else None,
                "name": case.current_regimen.regimen_name if case.current_regimen else None
            } if case.current_regimen else None
        }), 200
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:case_id>/treatment", methods=["POST"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def record_treatment(case_id):
    """
    Record a malaria treatment dose administered.
    Expected JSON: {
        administered_as_directly_observed: boolean (optional, default: false),
        date_administered: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Parse optional date
    date_administered = None
    if data.get("date_administered"):
        try:
            date_administered = datetime.fromisoformat(data["date_administered"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid date_administered format. Use ISO format."}), 400

    success, message, treatment = record_treatment_administered(
        case_id=case_id,
        administered_as_directly_observed=data.get("administered_as_directly_observed", False),
        date_administered=date_administered,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "treatment_id": treatment.id,
            "dose_number": treatment.dose_number,
            "date_administered": treatment.date_administered.isoformat() if treatment.date_administered else None,
            "administered_as_directly_observed": treatment.administered_as_directly_observed
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:case_id>/lab", methods=["POST"])
@login_required
@roles_required("clinical", "nursing")
def record_lab(case_id):
    """
    Record a malaria laboratory test result.
    Expected JSON: {
        test_type: string,
        result_value: string (optional),
        result_interpretation: string (optional),
        test_date: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Validate required fields
    test_type = data.get("test_type")
    if not test_type:
        return jsonify({"error": "test_type is required"}), 400

    # Parse optional date
    test_date = None
    if data.get("test_date"):
        try:
            test_date = datetime.fromisoformat(data["test_date"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid test_date format. Use ISO format."}), 400

    success, message, lab_result = record_lab_result(
        case_id=case_id,
        test_type=test_type,
        result_value=data.get("result_value"),
        result_interpretation=data.get("result_interpretation"),
        test_date=test_date,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "lab_result_id": lab_result.id,
            "test_type": lab_result.test_type,
            "result_value": lab_result.result_value,
            "result_interpretation": lab_result.result_interpretation,
            "test_date": lab_result.test_date.isoformat() if lab_result.test_date else None
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:case_id>/summary", methods=["GET"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def get_case_summary(case_id):
    """
    Get a comprehensive summary of a patient's malaria case.
    """
    try:
        case = MalariaCase.query.get(case_id)
        if not case:
            return jsonify({"error": "Malaria case not found"}), 404

        # Get latest records
        current_regimen = get_current_regimen(case_id)
        latest_lab = get_latest_lab_result(case_id)
        recent_treatments = get_treatment_summary(case_id, limit=5)

        # Calculate days since diagnosis
        days_since_diagnosis = None
        if case.diagnosis_date:
            delta = datetime.now(timezone.utc) - case.diagnosis_date
            days_since_diagnosis = delta.days

        # Calculate treatment completion percentage (simplified)
        treatment_completion = None
        if current_regimen and case.treatment_start_date:
            # Expected doses based on regimen (simplified: assume daily dosing)
            expected_doses_per_day = 1  # This would be more complex in reality
            days_on_treatment_float = (datetime.now(timezone.utc) - case.treatment_start_date).total_seconds() / (24*3600)
            expected_doses = int(days_on_treatment_float * expected_doses_per_day)
            actual_doses = len(recent_treatments)  # This is only recent treatments, not total - limitation
            if expected_doses > 0:
                treatment_completion = min(100.0, (actual_doses / expected_doses) * 100)

        return jsonify({
            "case": {
                "id": case.id,
                "patient_id": case.patient_id,
                "case_number": case.case_number,
                "diagnosis_date": case.diagnosis_date.isoformat() if case.diagnosis_date else None,
                "malaria_species": case.malaria_species,
                "parasite_density": case.parasite_density,
                "diagnosis_method": case.diagnosis_method,
                "severity": case.severity,
                "pregnancy_status": case.pregnancy_status,
                "facility_diagnosed_at": case.facility_diagnosed_at,
                "days_since_diagnosis": days_since_diagnosis
            },
            "current_regimen": {
                "id": current_regimen.id if current_regimen else None,
                "code": current_regimen.regimen_code if current_regimen else None,
                "name": current_regimen.regimen_name if current_regimen else None,
                "line_of_therapy": current_regimen.line_of_therapy if current_regimen else None,
                "drugs": current_regimen.drugs if current_regimen else None,
                "duration_days": current_regimen.duration_days if current_regimen else None
            } if current_regimen else None,
            "latest_lab": {
                "id": latest_lab.id if latest_lab else None,
                "test_type": latest_lab.test_type if latest_lab else None,
                "result_value": latest_lab.result_value if latest_lab else None,
                "result_interpretation": latest_lab.result_interpretation if latest_lab else None,
                "test_date": latest_lab.test_date.isoformat() if latest_lab and latest_lab.test_date else None
            } if latest_lab else None,
            "recent_treatments": [
                {
                    "id": treatment.id,
                    "date_administered": treatment.date_administered.isoformat() if treatment.date_administered else None,
                    "dose_number": treatment.dose_number,
                    "administered_as_directly_observed": treatment.administered_as_directly_observed
                } for treatment in recent_treatments
            ],
            "treatment_completion": treatment_completion
        }), 200

    except Exception as e:
        logger.error(f"Error getting case summary: {e}")
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/api/formulary", methods=["GET"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def get_formulary():
    """
    Get the current malaria formulary (list of active regimens).
    """
    try:
        regimens = get_malaria_formulary()
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
                    "duration_days": regimen.duration_days
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
        valid = is_malaria_regimen_valid(regimen_id)
        return jsonify({
            "regimen_id": regimen_id,
            "is_valid": valid
        }), 200
    except Exception as e:
        logger.error(f"Error validating regimen: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Celery task for checking missed treatments (would be called periodically)
@bp.route("/api/tasks/check-missed-treatments", methods=["POST"])
@login_required
@roles_required("admin")  # Only admins can trigger this manually
def trigger_missed_treatment_check():
    """
    Trigger a check for missed treatments.
    In production, this would be called by Celery beat schedule.
    """
    try:
        # For simplicity, we'll just return a placeholder message
        # In a real implementation, we would check for treatments not taken within expected windows
        return jsonify({
            "message": "Missed treatment check triggered. (Implementation pending)",
            "checked": True
        }), 200
    except Exception as e:
        logger.error(f"Error in missed treatment check: {e}")
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
def dashboard_ui():
    """Malaria Dashboard."""
    from departments.malaria.models import MalariaCase
    from extensions import db
    
    cases = MalariaCase.query.order_by(MalariaCase.diagnosis_date.desc()).all()
    active_count = len([c for c in cases if c.treatment_start_date])
    severe_count = len([c for c in cases if c.severity == 'severe'])
    treated_count = 0  # Would need date logic
    
    return render_template("malaria/dashboard.html",
                          cases=cases,
                          active_count=active_count,
                          severe_count=severe_count,
                          treated_count=treated_count)


@bp.route("/ui/new-case", methods=["GET", "POST"])
@login_required
def new_case_ui():
    """Register a new malaria case."""
    from departments.malaria.models import MalariaRegimen
    from departments.malaria.engine import create_malaria_case
    from extensions import db
    
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        case_number = request.form.get("case_number")
        diagnosis_date = request.form.get("diagnosis_date")
        malaria_species = request.form.get("malaria_species")
        severity = request.form.get("severity")
        diagnosis_method = request.form.get("diagnosis_method")
        treatment_start_date = request.form.get("treatment_start_date")
        current_regimen_id = request.form.get("current_regimen_id")
        
        try:
            result = create_malaria_case(
                patient_id=patient_id,
                case_number=case_number,
                diagnosis_date=diagnosis_date,
                malaria_species=malaria_species,
                severity=severity,
                diagnosis_method=diagnosis_method,
                treatment_start_date=treatment_start_date,
                current_regimen_id=int(current_regimen_id) if current_regimen_id else None,
            )
            flash("Malaria case registered successfully!", "success")
            return redirect(url_for("malaria.dashboard_ui"))
        except Exception as e:
            flash(f"Error registering case: {str(e)}", "danger")
    
    regimens = MalariaRegimen.query.filter_by(is_preferred=True).all()
    return render_template("malaria/new_case.html", regimens=regimens)


@bp.route("/ui/case/<int:case_id>")
@login_required
def case_detail_ui(case_id):
    """View malaria case details."""
    from departments.malaria.models import MalariaCase, MalariaTreatment, MalariaLabResult
    from extensions import db
    
    case = db.session.get(MalariaCase, case_id)
    if not case:
        flash("Case not found", "error")
        return redirect(url_for("malaria.dashboard_ui"))
    
    treatments = MalariaTreatment.query.filter_by(malaria_case_id=case_id).all()
    lab_results = MalariaLabResult.query.filter_by(malaria_case_id=case_id).order_by(MalariaLabResult.test_date.desc()).all()
    
    return render_template("malaria/case_detail.html",
                          case=case,
                          treatments=treatments,
                          lab_results=lab_results)
