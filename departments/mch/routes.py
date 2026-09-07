"""
MCH, ANC, and Immunization API routes.
"""

from flask import jsonify, request
from flask_login import login_required

from . import bp
from .engine import MchEngine

_engine = MchEngine()


@bp.route("/")
@login_required
def index():
    return "MCH & Immunization Module Active"


@bp.route("/api/anc-visit", methods=["POST"])
@login_required
def log_anc_visit():
    """
    Logs an ANC visit and automatically calculates the next appointment date.
    """
    data = request.get_json(silent=True) or {}

    patient_id = data.get("patient_id")
    visit_number = data.get("visit_number")
    gestation_weeks = data.get("gestation_weeks")

    if not all([patient_id, visit_number, gestation_weeks]):
        return jsonify(
            {"error": "patient_id, visit_number, and gestation_weeks are required"}
        ), 400

    try:
        visit = _engine.log_anc_visit(
            patient_id=patient_id,
            visit_number=visit_number,
            gestation_weeks=gestation_weeks,
            high_risk_factors=data.get("high_risk_factors"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            "status": "success",
            "visit_id": visit.id,
            "next_appointment_date": visit.next_appointment_date.isoformat()
            if visit.next_appointment_date
            else None,
        }
    ), 201


@bp.route("/api/immunize", methods=["POST"])
@login_required
def record_immunization():
    """
    Records a vaccine administration, enforcing dose sequencing.
    """
    data = request.get_json(silent=True) or {}

    child_patient_id = data.get("child_patient_id")
    vaccine_name = data.get("vaccine_name")
    dose_number = data.get("dose_number")

    if not all([child_patient_id, vaccine_name, dose_number]):
        return jsonify(
            {"error": "child_patient_id, vaccine_name, and dose_number are required"}
        ), 400

    try:
        record = _engine.record_immunization(
            child_patient_id=child_patient_id,
            vaccine_name=vaccine_name,
            dose_number=dose_number,
            batch_number=data.get("batch_number"),
        )
    except ValueError as e:
        # Use 400 for validation errors (e.g. out of sequence, already given)
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            "status": "success",
            "record_id": record.id,
            "message": "%s Dose %s recorded successfully."
            % (vaccine_name, dose_number),
        }
    ), 201


@bp.route("/api/child/<int:child_patient_id>/schedule", methods=["GET"])
@login_required
def get_child_schedule(child_patient_id: int):
    """
    Calculates due/overdue vaccines based on child's age.
    Query param: ?age_weeks=12
    """
    age_weeks = request.args.get("age_weeks", type=int)

    if age_weeks is None or age_weeks < 0:
        return jsonify(
            {"error": "age_weeks query parameter is required and must be >= 0"}
        ), 400

    due_vaccines = _engine.get_due_vaccines(age_weeks, child_patient_id)

    return jsonify(
        {
            "child_patient_id": child_patient_id,
            "age_weeks": age_weeks,
            "due_count": len(due_vaccines),
            "vaccines": due_vaccines,
        }
    ), 200
