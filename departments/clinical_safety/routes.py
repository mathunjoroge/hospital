"""
Clinical Safety API routes.

These endpoints power the Clinical Decision Support (CDS) engine,
providing real-time safety checks during prescription workflows.
"""

from flask import jsonify, request
from flask_login import current_user, login_required

from . import bp
from .engine import ClinicalSafetyEngine

# Singleton engine instance
_engine = ClinicalSafetyEngine()


@bp.route("/")
@login_required
def index():
    return "Clinical Safety & CDS Module Active - Phase 4 MVP"


@bp.route("/api/check", methods=["POST"])
@login_required
def check_safety():
    """
    Run safety checks for a prescription.

    Request body:
        {
            "patient_id": 123,
            "drug_ids": [45, 67, 89]
        }

    Response:
        {
            "has_alerts": true,
            "has_critical": false,
            "can_proceed": true,
            "total_alerts": 1,
            "alerts": [...],
            "checked_at": "2025-01-01T12:00:00+00:00"
        }
    """
    data = request.get_json() or {}

    patient_id = data.get("patient_id")
    drug_ids = data.get("drug_ids", [])

    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400

    if not drug_ids:
        return jsonify({"error": "drug_ids list is required"}), 400

    # Run the safety engine
    result = _engine.check_prescription(
        patient_id=patient_id,
        drug_ids=drug_ids,
        clinician_id=current_user.id,
    )

    return jsonify(result.to_dict()), 200


@bp.route("/api/override", methods=["POST"])
@login_required
def log_override():
    """
    Log a safety alert override.

    Request body:
        {
            "patient_id": 123,
            "alert_type": "ALLERGY",
            "alert_message": "...",
            "justification": "Patient confirmed allergy was resolved in 2020"
        }
    """
    data = request.get_json() or {}

    patient_id = data.get("patient_id")
    alert_type = data.get("alert_type")
    alert_message = data.get("alert_message")
    justification = data.get("justification")

    if not all([patient_id, alert_type, alert_message, justification]):
        return jsonify(
            {
                "error": (
                    "patient_id, alert_type, alert_message, "
                    "and justification are all required"
                )
            }
        ), 400

    # Log the override
    override = _engine.log_override(
        patient_id=patient_id,
        clinician_id=current_user.id,
        alert_type=alert_type,
        alert_message=alert_message,
        justification=justification,
    )

    return jsonify(
        {
            "status": "success",
            "message": "Override logged successfully",
            "override_id": override.id,
        }
    ), 201


@bp.route("/api/overrides/<int:patient_id>", methods=["GET"])
@login_required
def get_patient_overrides(patient_id: int):
    """
    Retrieve all safety overrides for a patient (audit trail).
    """
    from .models import SafetyAlertOverride

    overrides = (
        SafetyAlertOverride.query.filter_by(patient_id=patient_id)
        .order_by(SafetyAlertOverride.overridden_at.desc())
        .all()
    )

    return jsonify(
        {
            "patient_id": patient_id,
            "total_overrides": len(overrides),
            "overrides": [
                {
                    "id": o.id,
                    "alert_type": o.alert_type,
                    "alert_message": o.alert_message,
                    "justification": o.justification,
                    "clinician_id": o.clinician_id,
                    "overridden_at": o.overridden_at.isoformat(),
                }
                for o in overrides
            ],
        }
    ), 200
