"""
Referrals and Discharge API routes.
"""

from datetime import datetime, timezone

from flask import jsonify, request
from flask_login import login_required

from . import bp
from .engine import DischargeEngine, ReferralEngine

_referral_engine = ReferralEngine()
_discharge_engine = DischargeEngine()


@bp.route("/")
@login_required
def index():
    return "Referrals & Continuity of Care Module Active"


@bp.route("/api/initiate", methods=["POST"])
@login_required
def initiate_referral():
    """
    Initiates a new inter-facility referral.
    """
    data = request.get_json(silent=True) or {}

    patient_id = data.get("patient_id")
    referring = data.get("referring_facility")
    receiving = data.get("receiving_facility")
    reason = data.get("reason")
    summary = data.get("clinical_summary")

    if not all([patient_id, referring, receiving, reason, summary]):
        return jsonify(
            {"error": "Missing required fields for referral initiation"}
        ), 400

    referral = _referral_engine.initiate(
        patient_id=patient_id,
        referring_facility=referring,
        receiving_facility=receiving,
        reason=reason,
        clinical_summary=summary,
    )

    return jsonify(
        {
            "status": "success",
            "referral_id": referral.id,
            "current_status": referral.status,
        }
    ), 201


@bp.route("/api/status/<string:referral_id>", methods=["PATCH"])
@login_required
def update_referral_status(referral_id: str):
    """
    Updates the status of an existing referral (e.g., PENDING -> ACCEPTED).
    """
    data = request.get_json(silent=True) or {}
    new_status = data.get("status")

    if not new_status:
        return jsonify({"error": "status is required"}), 400

    try:
        referral = _referral_engine.update_status(referral_id, new_status)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not referral:
        return jsonify({"error": "Referral not found"}), 404

    return jsonify(
        {
            "status": "success",
            "referral_id": referral.id,
            "current_status": referral.status,
        }
    ), 200


@bp.route("/api/discharge", methods=["POST"])
@login_required
def generate_discharge_summary():
    """
    Generates a structured discharge summary for a patient leaving the facility.
    """
    data = request.get_json(silent=True) or {}

    patient_id = data.get("patient_id")
    primary_diagnosis = data.get("primary_diagnosis")
    admission_date_str = data.get("admission_date")

    if not all([patient_id, primary_diagnosis, admission_date_str]):
        return jsonify(
            {"error": "patient_id, primary_diagnosis, and admission_date are required"}
        ), 400

    try:
        admission_date = datetime.fromisoformat(admission_date_str)
        if admission_date.tzinfo is None:
            admission_date = admission_date.replace(tzinfo=timezone.utc)
    except ValueError:
        return jsonify({"error": "Invalid admission_date format. Use ISO 8601."}), 400

    try:
        summary = _discharge_engine.generate_summary(
            patient_id=patient_id,
            appointment_id=data.get("appointment_id"),
            admission_date=admission_date,
            primary_diagnosis=primary_diagnosis,
            discharge_medications=data.get("discharge_medications"),
            follow_up_instructions=data.get("follow_up_instructions"),
            referred_to=data.get("referred_to"),
            secondary_diagnoses=data.get("secondary_diagnoses"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            "status": "success",
            "summary_id": summary.id,
            "discharge_date": summary.discharge_date.isoformat(),
        }
    ), 201
