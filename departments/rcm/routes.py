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


# ── RCM Pre-Submission Scrubber & EDI 837/835 Endpoints ───────────────────────


@bp.route("/api/scrub/<claim_id>", methods=["POST"])
@login_required
def scrub_claim_record(claim_id: str):
    """POST /rcm/api/scrub/<claim_id> — Runs Pre-Claim Scrubber rules & calculates denial risk score."""
    from departments.rcm.claims_scrubber_engine import ClaimsScrubberEngine

    try:
        report = ClaimsScrubberEngine.scrub_claim(claim_id)
        return jsonify(report), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/api/edi837/<claim_id>", methods=["GET"])
@login_required
def generate_edi837_route(claim_id: str):
    """GET /rcm/api/edi837/<claim_id> — Generates X12 837P Professional Claim transaction text."""
    from departments.rcm.claims_scrubber_engine import ClaimsScrubberEngine

    try:
        edi_text = ClaimsScrubberEngine.generate_edi_837(claim_id)
        return jsonify({"claim_id": claim_id, "format": "X12_837P", "edi_content": edi_text}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/api/edi835/process", methods=["POST"])
@login_required
def process_edi835_route():
    """POST /rcm/api/edi835/process — Parses X12 835 Electronic Remittance Advice (ERA) & applies payments/denials."""
    from departments.rcm.claims_scrubber_engine import ClaimsScrubberEngine

    data = request.get_json(silent=True) or {}
    edi_content = data.get("edi_content")

    if not edi_content:
        return jsonify({"error": "edi_content is required."}), 400

    try:
        result = ClaimsScrubberEngine.parse_and_apply_edi_835(edi_content)
        return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/denial-risk/<claim_id>", methods=["GET"])
@login_required
def get_denial_risk_route(claim_id: str):
    """GET /rcm/api/denial-risk/<claim_id> — Retrieves pre-claim denial risk score and error details."""
    from departments.rcm.models import ClaimSubmission

    claim = ClaimSubmission.query.get_or_404(claim_id)
    return jsonify({
        "claim_id": claim.id,
        "status": claim.status,
        "scrubbing_status": claim.scrubbing_status,
        "denial_risk_score": claim.denial_risk_score,
        "errors": claim.scrubbing_errors_json,
    }), 200


@bp.route("/claims-console", methods=["GET"])
@login_required
def claims_console_ui():
    """Render RCM Pre-Submission Claims Scrubbing & EDI Console UI."""
    from flask import render_template

    from departments.rcm.models import ClaimSubmission

    claims = ClaimSubmission.query.order_by(ClaimSubmission.created_at.desc()).all()
    clean_count = sum(1 for c in claims if c.scrubbing_status == "CLEAN")
    error_count = sum(1 for c in claims if c.scrubbing_status == "HAS_ERRORS")

    return render_template(
        "rcm/claims_console.html",
        claims=claims,
        clean_count=clean_count,
        error_count=error_count,
        total_claims=len(claims),
    )

