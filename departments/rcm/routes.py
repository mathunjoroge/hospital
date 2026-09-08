"""
Revenue Cycle Management API routes.
"""

from datetime import datetime, timezone

from flask import jsonify, request
from flask_login import login_required
from departments.rbac import roles_required

from . import bp
from .engine import RevenueCycleEngine

_engine = RevenueCycleEngine()


@bp.route("/")
@login_required
@roles_required("billing", "admin")
def index():
    return "Revenue Cycle Management Module Active"


@bp.route("/api/preauth", methods=["POST"])
@login_required
@roles_required("billing", "admin")
def submit_preauth():
    """
    Submits a pre-authorization request for SHA or private insurance.
    """
    data = request.get_json(silent=True) or {}

    patient_id = data.get("patient_id")
    insurance_scheme_id = data.get("insurance_scheme_id")
    procedure_code = data.get("procedure_code")
    estimated_amount = data.get("estimated_amount")

    if not all([patient_id, insurance_scheme_id, procedure_code, estimated_amount]):
        return jsonify({"error": "Missing required fields for pre-authorization"}), 400

    preauth = _engine.submit_preauth(
        patient_id=patient_id,
        insurance_scheme_id=insurance_scheme_id,
        procedure_code=procedure_code,
        estimated_amount=estimated_amount,
        clinical_justification=data.get("clinical_justification"),
    )

    return jsonify(
        {
            "status": "success",
            "preauth_id": preauth.id,
            "current_status": preauth.status,
        }
    ), 201


@bp.route("/api/claim/scrub", methods=["POST"])
@login_required
@roles_required("billing", "admin")
def scrub_claim():
    """
    Validates a claim before submission. Returns any errors found.
    """
    data = request.get_json(silent=True) or {}

    try:
        errors = _engine.scrub_claim(
            patient_id=data.get("patient_id"),
            billed_amount=data.get("billed_amount", 0),
            service_start_date=datetime.fromisoformat(data["service_start_date"]),
            service_end_date=datetime.fromisoformat(data["service_end_date"]),
            primary_diagnosis_icd10=data.get("primary_diagnosis_icd10"),
        )
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid date format. Use ISO 8601."}), 400

    if errors:
        return jsonify({"is_clean": False, "errors": errors}), 422

    return jsonify({"is_clean": True, "errors": []}), 200


@bp.route("/api/claim/submit", methods=["POST"])
@login_required
@roles_required("billing", "admin")
def submit_claim():
    """
    Submits a claim. Automatically scrubs the claim first.
    """
    data = request.get_json(silent=True) or {}

    try:
        service_start = datetime.fromisoformat(data["service_start_date"])
        service_end = datetime.fromisoformat(data["service_end_date"])

        if service_start.tzinfo is None:
            service_start = service_start.replace(tzinfo=timezone.utc)
        if service_end.tzinfo is None:
            service_end = service_end.replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid date format. Use ISO 8601."}), 400

    claim = _engine.submit_claim(
        patient_id=data.get("patient_id"),
        billed_amount=data.get("billed_amount", 0),
        service_start_date=service_start,
        service_end_date=service_end,
        primary_diagnosis_icd10=data.get("primary_diagnosis_icd10"),
        secondary_diagnosis_icd10=data.get("secondary_diagnosis_icd10"),
        insurance_scheme_id=data.get("insurance_scheme_id"),
    )

    if not claim:
        return jsonify(
            {
                "error": "Claim failed scrubbing validation.",
                "message": "Fix validation errors before submitting.",
            }
        ), 422

    return jsonify(
        {
            "status": "success",
            "claim_id": claim.id,
            "current_status": claim.status,
        }
    ), 201


@bp.route("/api/denial/appeal", methods=["POST"])
@login_required
@roles_required("billing", "admin")
def submit_appeal():
    """
    Initiates an appeal for a denied claim.
    """
    data = request.get_json(silent=True) or {}

    claim_id = data.get("claim_id")
    denial_code = data.get("denial_code")
    denial_reason = data.get("denial_reason")
    appeal_justification = data.get("appeal_justification")

    if not all([claim_id, denial_code, denial_reason, appeal_justification]):
        return jsonify({"error": "Missing required fields for appeal"}), 400

    try:
        denial = _engine.appeal_claim(
            claim_id=claim_id,
            denial_code=denial_code,
            denial_reason=denial_reason,
            appeal_justification=appeal_justification,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not denial:
        return jsonify({"error": "Claim not found"}), 404

    return jsonify(
        {
            "status": "success",
            "denial_id": denial.id,
            "appeal_status": denial.appeal_status,
        }
    ), 201
