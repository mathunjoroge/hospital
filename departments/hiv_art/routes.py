"""
HIV/ART Module API routes.
"""

import logging
from datetime import datetime, timezone

from flask import jsonify, request
from flask_login import login_required

from departments.rbac import roles_required

from . import bp
from .engine import (
    check_missed_visits,
    create_art_enrollment,
    get_adherence_summary,
    get_art_formulary,
    get_current_regimen,
    get_latest_cd4_count,
    get_latest_viral_load,
    get_latest_who_stage,
    is_regimen_valid,
    record_adherence_visit,
    record_cd4_count,
    record_viral_load,
    record_who_stage,
    update_art_regimen,
)
from .models import ARTEnrollment

logger = logging.getLogger(__name__)

# Initialize engine (placeholder for potential engine class)
# In this implementation, we're using functional approach similar to some other modules


@bp.route("/")
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def index():
    return "HIV/ART Module Active"


@bp.route("/api/enroll", methods=["POST"])
@login_required
@roles_required("clinical", "nursing")
def enroll_patient():
    """
    Enroll a patient in ART treatment.
    Expected JSON: {
        patient_id: string,
        art_number: string,
        baseline_cd4: integer (optional),
        baseline_who_stage: integer (1-4, optional),
        art_start_date: ISO string (optional),
        facility_enrolled_at: string (optional),
        encounter_id: integer (optional),
        current_regimen_id: string (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Validate required fields
    patient_id = data.get("patient_id")
    art_number = data.get("art_number")

    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400
    if not art_number:
        return jsonify({"error": "art_number is required"}), 400

    # Parse optional dates
    art_start_date = None
    if data.get("art_start_date"):
        try:
            art_start_date = datetime.fromisoformat(data["art_start_date"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid art_start_date format. Use ISO format."}), 400

    success, message, enrollment = create_art_enrollment(
        patient_id=patient_id,
        art_number=art_number,
        baseline_cd4=data.get("baseline_cd4"),
        baseline_who_stage=data.get("baseline_who_stage"),
        art_start_date=art_start_date,
        facility_enrolled_at=data.get("facility_enrolled_at"),
        encounter_id=data.get("encounter_id"),
        current_regimen_id=data.get("current_regimen_id")
    )

    if success:
        return jsonify({
            "message": message,
            "enrollment_id": enrollment.id,
            "art_number": enrollment.art_number
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/regimen", methods=["PUT"])
@login_required
@roles_required("clinical")
def change_regimen(enrollment_id):
    """
    Change a patient's ART regimen.
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

    success, message, enrollment = update_art_regimen(
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


@bp.route("/api/<string:enrollment_id>/adherence", methods=["POST"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def record_adherence(enrollment_id):
    """
    Record an adherence visit.
    Expected JSON: {
        pills_dispensed: integer,
        pills_returned: integer,
        visit_date: ISO string (optional),
        viral_load_ordered: boolean (optional),
        cd4_ordered: boolean (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    # Validate required fields
    pills_dispensed = data.get("pills_dispensed")
    pills_returned = data.get("pills_returned")

    if pills_dispensed is None:
        return jsonify({"error": "pills_dispensed is required"}), 400
    if pills_returned is None:
        return jsonify({"error": "pills_returned is required"}), 400
    if pills_returned > pills_dispensed:
        return jsonify({"error": "pills_returned cannot exceed pills_dispensed"}), 400

    # Parse optional date
    visit_date = None
    if data.get("visit_date"):
        try:
            visit_date = datetime.fromisoformat(data["visit_date"].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid visit_date format. Use ISO format."}), 400

    success, message, visit = record_adherence_visit(
        enrollment_id=enrollment_id,
        pills_dispensed=pills_dispensed,
        pills_returned=pills_returned,
        visit_date=visit_date,
        viral_load_ordered=data.get("viral_load_ordered", False),
        cd4_ordered=data.get("cd4_ordered", False),
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "visit_id": visit.id,
            "adherence_percentage": visit.adherence_percentage,
            "adherence_category": visit.adherence_category
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/viral-load", methods=["POST"])
@login_required
@roles_required("clinical", "nursing")
def record_viral_load_endpoint(enrollment_id):
    """
    Record a viral load test result.
    Expected JSON: {
        viral_load_copies: integer (null for undetectable),
        test_type: string (optional, default: "routine"),
        test_date: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    viral_load_copies = data.get("viral_load_copies")
    # Allow None/null for undetectable

    success, message, viral_load = record_viral_load(
        enrollment_id=enrollment_id,
        viral_load_copies=viral_load_copies,
        test_type=data.get("test_type", "routine"),
        test_date=datetime.fromisoformat(data["test_date"].replace('Z', '+00:00')) if data.get("test_date") else None,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "viral_load_id": viral_load.id,
            "viral_load_copies": viral_load.viral_load_copies,
            "test_date": viral_load.test_date.isoformat() if viral_load.test_date else None
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/cd4", methods=["POST"])
@login_required
@roles_required("clinical", "nursing")
def record_cd4_endpoint(enrollment_id):
    """
    Record a CD4 count test result.
    Expected JSON: {
        cd4_count: integer,
        cd4_percent: float (optional),
        test_date: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    cd4_count = data.get("cd4_count")
    if cd4_count is None:
        return jsonify({"error": "cd4_count is required"}), 400

    success, message, cd4 = record_cd4_count(
        enrollment_id=enrollment_id,
        cd4_count=cd4_count,
        cd4_percent=data.get("cd4_percent"),
        test_date=datetime.fromisoformat(data["test_date"].replace('Z', '+00:00')) if data.get("test_date") else None,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "cd4_id": cd4.id,
            "cd4_count": cd4.cd4_count,
            "cd4_percent": cd4.cd4_percent,
            "test_date": cd4.test_date.isoformat() if cd4.test_date else None
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/who-stage", methods=["POST"])
@login_required
@roles_required("clinical")
def record_who_stage_endpoint(enrollment_id):
    """
    Record a WHO clinical staging assessment.
    Expected JSON: {
        who_stage: integer (1-4),
        defining_conditions: string (optional),
        assessment_date: ISO string (optional),
        encounter_id: integer (optional)
    }
    """
    data = request.get_json(silent=True) or {}

    who_stage = data.get("who_stage")
    if who_stage is None:
        return jsonify({"error": "who_stage is required"}), 400
    try:
        who_stage = int(who_stage)
        if who_stage < 1 or who_stage > 4:
            return jsonify({"error": "who_stage must be between 1 and 4"}), 400
    except ValueError:
        return jsonify({"error": "who_stage must be an integer"}), 400

    success, message, who_stage_record = record_who_stage(
        enrollment_id=enrollment_id,
        who_stage=who_stage,
        defining_conditions=data.get("defining_conditions"),
        assessment_date=datetime.fromisoformat(data["assessment_date"].replace('Z', '+00:00')) if data.get("assessment_date") else None,
        encounter_id=data.get("encounter_id")
    )

    if success:
        return jsonify({
            "message": message,
            "who_stage_id": who_stage_record.id,
            "who_stage": who_stage_record.who_stage,
            "assessment_date": who_stage_record.assessment_date.isoformat() if who_stage_record.assessment_date else None
        }), 201
    else:
        return jsonify({"error": message}), 400


@bp.route("/api/<string:enrollment_id>/summary", methods=["GET"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def get_patient_summary(enrollment_id):
    """
    Get a comprehensive summary of a patient's ART status.
    """
    try:
        enrollment = ARTEnrollment.query.get(enrollment_id)
        if not enrollment:
            return jsonify({"error": "ART enrollment not found"}), 404

        # Get latest records
        current_regimen = get_current_regimen(enrollment_id)
        latest_viral_load = get_latest_viral_load(enrollment_id)
        latest_cd4 = get_latest_cd4_count(enrollment_id)
        latest_who_stage = get_latest_who_stage(enrollment_id)
        recent_adherence = get_adherence_summary(enrollment_id, limit=5)

        # Calculate days on ART
        days_on_art = None
        if enrollment.art_start_date:
            delta = datetime.now(timezone.utc) - enrollment.art_start_date
            days_on_art = delta.days

        return jsonify({
            "enrollment": {
                "id": enrollment.id,
                "patient_id": enrollment.patient_id,
                "art_number": enrollment.art_number,
                "enrollment_date": enrollment.enrollment_date.isoformat() if enrollment.enrollment_date else None,
                "art_start_date": enrollment.art_start_date.isoformat() if enrollment.art_start_date else None,
                "baseline_cd4": enrollment.baseline_cd4,
                "baseline_who_stage": enrollment.baseline_who_stage,
                "facility_enrolled_at": enrollment.facility_enrolled_at,
                "days_on_art": days_on_art
            },
            "current_regimen": {
                "id": current_regimen.id if current_regimen else None,
                "code": current_regimen.regimen_code if current_regimen else None,
                "name": current_regimen.regimen_name if current_regimen else None,
                "line_of_therapy": current_regimen.line_of_therapy if current_regimen else None,
                "arv_drugs": current_regimen.arv_drugs if current_regimen else None
            } if current_regimen else None,
            "latest_viral_load": {
                "id": latest_viral_load.id if latest_viral_load else None,
                "copies": latest_viral_load.viral_load_copies,
                "test_type": latest_viral_load.test_type,
                "test_date": latest_viral_load.test_date.isoformat() if latest_viral_load.test_date else None
            } if latest_viral_load else None,
            "latest_cd4": {
                "id": latest_cd4.id if latest_cd4 else None,
                "count": latest_cd4.cd4_count,
                "percent": latest_cd4.cd4_percent,
                "test_date": latest_cd4.test_date.isoformat() if latest_cd4.test_date else None
            } if latest_cd4 else None,
            "latest_who_stage": {
                "id": latest_who_stage.id if latest_who_stage else None,
                "stage": latest_who_stage.who_stage,
                "defining_conditions": latest_who_stage.defining_conditions,
                "assessment_date": latest_who_stage.assessment_date.isoformat() if latest_who_stage.assessment_date else None
            } if latest_who_stage else None,
            "recent_adherence": [
                {
                    "id": visit.id,
                    "visit_date": visit.visit_date.isoformat() if visit.visit_date else None,
                    "adherence_percentage": visit.adherence_percentage,
                    "adherence_category": visit.adherence_category,
                    "pills_dispensed": visit.pills_dispensed,
                    "pills_returned": visit.pills_returned
                } for visit in recent_adherence
            ]
        }), 200

    except Exception as e:
        logger.error(f"Error getting patient summary: {e}")
        return jsonify({"error": "Internal server error"}), 500


@bp.route("/api/formulary", methods=["GET"])
@login_required
@roles_required("clinical", "nursing", "pharmacy")
def get_formulary():
    """
    Get the current ART formulary (list of active regimens).
    """
    try:
        regimens = get_art_formulary()
        return jsonify({
            "regimens": [
                {
                    "id": regimen.id,
                    "code": regimen.regimen_code,
                    "name": regimen.regimen_name,
                    "line_of_therapy": regimen.line_of_therapy,
                    "arv_drugs": regimen.arv_drugs,
                    "is_preferred": regimen.is_preferred,
                    "is_alternative": regimen.is_alternative,
                    "restriction_notes": regimen.restriction_notes,
                    "effective_from": regimen.effective_from.isoformat() if regimen.effective_from else None,
                    "effective_to": regimen.effective_to.isoformat() if regimen.effective_to else None
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
        valid = is_regimen_valid(regimen_id)
        return jsonify({
            "regimen_id": regimen_id,
            "is_valid": valid
        }), 200
    except Exception as e:
        logger.error(f"Error validating regimen: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Celery task for checking missed adherence visits (would be called periodically)
@bp.route("/api/tasks/check-missed-visits", methods=["POST"])
@login_required
@roles_required("admin")  # Only admins can trigger this manually
def trigger_missed_visit_check():
    """
    Trigger a check for missed adherence visits.
    In production, this would be called by Celery beat schedule.
    """
    try:
        missed = check_missed_visits()
        return jsonify({
            "message": f"Checked for missed visits. Found {len(missed)} patients with missed visits.",
            "missed_visit_count": len(missed),
            "patients": [
                {
                    "enrollment_id": visit.enrollment_id,
                    "patient_id": visit.enrollment.patient_id if visit.enrollment else None,
                    "art_number": visit.enrollment.art_number if visit.enrollment else None,
                    "last_visit_date": visit.visit_date.isoformat() if visit.visit_date else None
                } for visit in missed[:10]  # Limit to first 10 for response size
            ]
        }), 200
    except Exception as e:
        logger.error(f"Error in missed visit check: {e}")
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
    """HIV/ART Dashboard."""
    from departments.hiv_art.models import ARTEnrollment
    from extensions import db
    
    enrollments = ARTEnrollment.query.order_by(ARTEnrollment.enrollment_date.desc()).all()
    active_count = len([e for e in enrollments if e.art_start_date])
    pending_vl = 0  # Would need logic to determine pending VL
    adherence_due = 0  # Would need logic to determine due visits
    
    return render_template("hiv_art/dashboard.html",
                          enrollments=enrollments,
                          active_count=active_count,
                          pending_vl=pending_vl,
                          adherence_due=adherence_due)


@bp.route("/ui/enroll", methods=["GET", "POST"])
@login_required
def enroll_ui():
    """Enroll a new ART patient."""
    from departments.hiv_art.models import ARTRegimen
    from departments.hiv_art.engine import create_art_enrollment
    from extensions import db
    
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        art_number = request.form.get("art_number")
        enrollment_date = request.form.get("enrollment_date")
        art_start_date = request.form.get("art_start_date")
        baseline_who_stage = request.form.get("baseline_who_stage")
        baseline_cd4 = request.form.get("baseline_cd4")
        current_regimen_id = request.form.get("current_regimen_id")
        
        try:
            result = create_art_enrollment(
                patient_id=patient_id,
                art_number=art_number,
                enrollment_date=enrollment_date,
                art_start_date=art_start_date,
                baseline_who_stage=int(baseline_who_stage) if baseline_who_stage else None,
                baseline_cd4=float(baseline_cd4) if baseline_cd4 else None,
                current_regimen_id=int(current_regimen_id) if current_regimen_id else None,
            )
            flash("Patient enrolled successfully!", "success")
            return redirect(url_for("hiv_art.dashboard_ui"))
        except Exception as e:
            flash(f"Error enrolling patient: {str(e)}", "danger")
    
    regimens = ARTRegimen.query.filter_by(is_preferred=True).all()
    return render_template("hiv_art/enroll.html", regimens=regimens)


@bp.route("/ui/patient/<int:enrollment_id>")
@login_required
def patient_detail_ui(enrollment_id):
    """View ART patient details."""
    from departments.hiv_art.models import ARTEnrollment, ViralLoad, AdherenceVisit
    from extensions import db
    
    enrollment = db.session.get(ARTEnrollment, enrollment_id)
    if not enrollment:
        flash("Enrollment not found", "error")
        return redirect(url_for("hiv_art.dashboard_ui"))
    
    viral_loads = ViralLoad.query.filter_by(art_enrollment_id=enrollment_id).order_by(ViralLoad.test_date.desc()).all()
    adherence_visits = AdherenceVisit.query.filter_by(art_enrollment_id=enrollment_id).order_by(AdherenceVisit.visit_date.desc()).all()
    
    return render_template("hiv_art/patient_detail.html",
                          enrollment=enrollment,
                          viral_loads=viral_loads,
                          adherence_visits=adherence_visits)
