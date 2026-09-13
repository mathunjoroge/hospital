"""
departments/nursing/ed_routes.py
──────────────────────────────
Flask routes for ED Operations Management & Real-Time Console.
"""

import logging
from flask import Blueprint, jsonify, render_template, request

from departments.api.auth import jwt_or_session_required
from departments.nursing.ed_engine import EDOperationsEngine
from departments.rbac import roles_required

logger = logging.getLogger(__name__)

ed_bp = Blueprint("ed", __name__, template_folder="templates")


@ed_bp.route("/nursing/ed/arrive", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine", "records")
def record_ed_arrival():
    """API endpoint to record patient arrival in ED."""
    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    chief_complaint = data.get("chief_complaint", "ED Arrival")
    nurse_id = data.get("nurse_id", 1)

    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400

    try:
        assessment = EDOperationsEngine.record_arrival(
            patient_id=patient_id, nurse_id=nurse_id, chief_complaint=chief_complaint
        )
        return jsonify({
            "success": True,
            "assessment_id": assessment.id,
            "patient_id": assessment.patient_id,
            "arrival_at": assessment.arrival_at.isoformat() if assessment.arrival_at else None,
            "priority_status": assessment.priority_status,
        }), 201
    except Exception as e:
        logger.exception("Error recording ED arrival")
        return jsonify({"error": str(e)}), 400


@ed_bp.route("/nursing/ed/triage-complete/<int:assessment_id>", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine")
def complete_ed_triage(assessment_id):
    """API endpoint to mark triage completed and compute door-to-triage time."""
    data = request.get_json() or {}
    esi_level = int(data.get("esi_level", 3))
    vitals_warning = data.get("vitals_warning")

    try:
        assessment = EDOperationsEngine.complete_triage(
            assessment_id=assessment_id, esi_level=esi_level, vitals_warning=vitals_warning
        )
        return jsonify({
            "success": True,
            "assessment_id": assessment.id,
            "esi_level": assessment.esi_level,
            "triage_completed_at": assessment.triage_completed_at.isoformat() if assessment.triage_completed_at else None,
            "re_evaluation_due_at": assessment.re_evaluation_due_at.isoformat() if assessment.re_evaluation_due_at else None,
        }), 200
    except Exception as e:
        logger.exception("Error completing ED triage")
        return jsonify({"error": str(e)}), 400


@ed_bp.route("/nursing/ed/assign-bed/<int:assessment_id>", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine")
def assign_ed_bed(assessment_id):
    """API endpoint to assign an ED bed/bay."""
    data = request.get_json() or {}
    bed_label = data.get("bed_label", "BAY-1")

    try:
        assessment = EDOperationsEngine.assign_bed(assessment_id=assessment_id, bed_label=bed_label)
        return jsonify({
            "success": True,
            "assessment_id": assessment.id,
            "bed_label": assessment.bed_label,
            "bed_assigned_at": assessment.bed_assigned_at.isoformat() if assessment.bed_assigned_at else None,
        }), 200
    except Exception as e:
        logger.exception("Error assigning ED bed")
        return jsonify({"error": str(e)}), 400


@ed_bp.route("/nursing/ed/seen-by-doctor/<int:assessment_id>", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "medicine", "nursing")
def mark_ed_seen_by_doctor(assessment_id):
    """API endpoint to mark patient seen by clinician and calculate door-to-doctor time."""
    try:
        assessment = EDOperationsEngine.mark_seen_by_doctor(assessment_id=assessment_id)
        return jsonify({
            "success": True,
            "assessment_id": assessment.id,
            "seen_by_doctor_at": assessment.seen_by_doctor_at.isoformat() if assessment.seen_by_doctor_at else None,
            "priority_status": assessment.priority_status,
        }), 200
    except Exception as e:
        logger.exception("Error marking seen by doctor")
        return jsonify({"error": str(e)}), 400


@ed_bp.route("/nursing/ed/re-evaluate/<int:assessment_id>", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine")
def record_ed_re_evaluation(assessment_id):
    """API endpoint to submit ESI 2 re-evaluation note."""
    data = request.get_json() or {}
    notes = data.get("notes", "ESI Re-evaluation completed.")
    new_esi_level = data.get("new_esi_level")
    if new_esi_level is not None:
        new_esi_level = int(new_esi_level)

    try:
        assessment = EDOperationsEngine.record_re_evaluation(
            assessment_id=assessment_id, notes=notes, new_esi_level=new_esi_level
        )
        return jsonify({
            "success": True,
            "assessment_id": assessment.id,
            "esi_level": assessment.esi_level,
            "last_re_evaluation_at": assessment.last_re_evaluation_at.isoformat() if assessment.last_re_evaluation_at else None,
            "re_evaluation_due_at": assessment.re_evaluation_due_at.isoformat() if assessment.re_evaluation_due_at else None,
        }), 200
    except Exception as e:
        logger.exception("Error recording ESI re-evaluation")
        return jsonify({"error": str(e)}), 400


@ed_bp.route("/nursing/ed/discharge/<int:assessment_id>", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine")
def discharge_ed_patient(assessment_id):
    """API endpoint to record ED patient disposition."""
    data = request.get_json() or {}
    disposition = data.get("disposition", "DISCHARGED")

    try:
        assessment = EDOperationsEngine.discharge_patient(assessment_id=assessment_id, disposition=disposition)
        return jsonify({
            "success": True,
            "assessment_id": assessment.id,
            "disposition": assessment.disposition,
            "disposition_at": assessment.disposition_at.isoformat() if assessment.disposition_at else None,
        }), 200
    except Exception as e:
        logger.exception("Error recording ED disposition")
        return jsonify({"error": str(e)}), 400


@ed_bp.route("/nursing/ed/dashboard", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine", "records")
def get_ed_dashboard_ui():
    """Render real-time ED Operations Console dashboard."""
    metrics = EDOperationsEngine.get_ed_dashboard_metrics()
    return render_template("nursing/ed_dashboard.html", metrics=metrics)


@ed_bp.route("/nursing/api/ed/metrics", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine", "records", "api")
def get_ed_metrics_api():
    """JSON API endpoint returning real-time ED KPIs and patient queue."""
    metrics = EDOperationsEngine.get_ed_dashboard_metrics()
    return jsonify(metrics), 200


@ed_bp.route("/nursing/api/ed/boarding-alerts", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine", "records", "api")
def get_ed_boarding_alerts_api():
    """JSON API endpoint returning patients exceeding 4-hour LOS threshold."""
    metrics = EDOperationsEngine.get_ed_dashboard_metrics()
    return jsonify({
        "boarding_alerts_count": metrics["boarding_alerts_count"],
        "boarding_alerts": metrics["boarding_alerts"],
    }), 200
