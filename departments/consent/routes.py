from flask import jsonify, request
from flask_login import login_required

from departments.models.compliance import PatientConsent
from departments.models.records import Patient

from . import bp


@bp.route("/")
@login_required
def index():
    return "Consent Module Active - Phase 1 MVP"


@bp.route("/api/patient/<int:patient_id>", methods=["GET"])
@login_required
def get_consents(patient_id: int):
    return jsonify(
        {
            "patient_id": patient_id,
            "message": "Consent retrieval endpoint active. Full query logic in Phase 1.1.",
        }
    )


@bp.route("/api/check", methods=["GET"])
@login_required
def check_consent():
    """
    Consent pre-flight check used by the clinical workbenches.

    Query params:
        patient_id (int, required): Patient.id, the internal numeric key
            used throughout the appointments/clinical-safety/referrals
            modules (as opposed to Patient.patient_id, the "P0001"-style
            business identifier used by billing/insurance/consent records).
        consent_type (str, required): e.g. "TREATMENT", "ai_diagnosis".

    Response: { "status": "ACTIVE" | "NOT_GRANTED" | "REVOKED" }
    """
    patient_id_raw = request.args.get("patient_id")
    consent_type = request.args.get("consent_type")

    if not patient_id_raw or not consent_type:
        return jsonify({"error": "patient_id and consent_type are required"}), 400

    try:
        patient_id = int(patient_id_raw)
    except ValueError:
        return jsonify({"error": "patient_id must be an integer"}), 400

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    consent = PatientConsent.query.filter_by(
        patient_id=patient.patient_id,  # bridge to the string business ID
        consent_type=consent_type,
    ).first()

    if consent is None:
        status = "NOT_GRANTED"
    elif consent.is_granted and consent.revoked_at is None:
        status = "ACTIVE"
    else:
        status = "REVOKED"

    return jsonify({"patient_id": patient_id, "consent_type": consent_type, "status": status})


@bp.route("/api/grant", methods=["POST"])
@login_required
def grant_consent():
    _data = request.get_json() or {}
    return jsonify({"status": "success", "message": "Consent grant stub active."}), 201


@bp.route("/api/revoke", methods=["POST"])
@login_required
def revoke_consent():
    _data = request.get_json() or {}
    return jsonify(
        {"status": "success", "message": "Consent revocation stub active."}
    ), 200
