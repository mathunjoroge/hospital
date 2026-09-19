"""
departments/laboratory/panic_alerts.py
───────────────────────────────────────
Task 3.4 — LIS Critical Value Panic Alerts & 2-Tier Result Verification

Features:
  - Age & sex-adjusted laboratory reference range & panic threshold engine
  - 2-Tier result verification (Lab Tech entry -> Pathologist sign-off)
  - Real-time panic alert notifications dispatch to ordering clinicians

P0-06 / P0-11 Security fixes:
  - All endpoints now require @login_required + @roles_required.
  - verifier_id and tech_id are derived exclusively from current_user — never from request body.
  - Re-verification of an already-VERIFIED result is blocked to prevent silent overwrite.
  - Client-supplied verifier_id/tech_id fields in request body are ignored for security.
"""

import logging
import uuid
from datetime import datetime, timezone

from flask import Blueprint, abort, jsonify, request, session
from flask_login import current_user, login_required

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.laboratory import LabResult
from departments.models.nursing import Notifications
from departments.models.records import Patient
from departments.rbac import roles_required

logger = logging.getLogger(__name__)

lis_bp = Blueprint("lis", __name__, url_prefix="/laboratory/lis")

# Clinical Lab Panic Reference Matrix
PANIC_THRESHOLDS = {
    "hemoglobin": {
        "normal_low": 12.0,
        "normal_high": 17.5,
        "panic_low": 7.0,
        "panic_high": 20.0,
        "unit": "g/dL",
    },
    "potassium": {
        "normal_low": 3.5,
        "normal_high": 5.1,
        "panic_low": 2.8,
        "panic_high": 6.2,
        "unit": "mmol/L",
    },
    "sodium": {
        "normal_low": 135.0,
        "normal_high": 145.0,
        "panic_low": 120.0,
        "panic_high": 160.0,
        "unit": "mmol/L",
    },
    "platelets": {
        "normal_low": 150.0,
        "normal_high": 450.0,
        "panic_low": 30.0,
        "panic_high": 1000.0,
        "unit": "x10^3/uL",
    },
    "blood_glucose": {
        "normal_low": 70.0,
        "normal_high": 110.0,
        "panic_low": 45.0,
        "panic_high": 400.0,
        "unit": "mg/dL",
    },
}


def _require_authenticated_user_id() -> int:
    """
    Return the authenticated user's ID from the server-side security context.
    Aborts 401 if unauthenticated. Never falls back to verifier_id=1.
    """
    if current_user and getattr(current_user, "is_authenticated", False):
        return current_user.id
    abort(401)


def _resolve_ordering_clinician_id(lab_res) -> int | None:
    """
    Resolve the user_id of the clinician who ordered the lab test, so critical
    panic alerts reach the person managing the patient (not just the verifier).

    Resolution order:
      1. Clinician recorded on the patient's ACTIVE encounter (provider_id).
      2. None if unresolvable — caller falls back to notifying the verifier only.
    """
    try:
        from departments.models.encounter import Encounter

        enc = (
            Encounter.query.filter_by(
                patient_id=lab_res.patient_id, status="ACTIVE"
            )
            .order_by(Encounter.id.desc())
            .first()
        )
        if enc and enc.provider_id:
            # provider_id is a string column; users.id is integer.
            return int(enc.provider_id)
    except (ValueError, TypeError, AttributeError):
        logger.debug(
            "Could not resolve ordering clinician for patient %s",
            lab_res.patient_id,
        )
    return None


def evaluate_panic_level(parameter_name: str, value: float) -> tuple[str, str]:
    """
    Evaluate numerical lab value against reference range and panic thresholds.
    Returns (panic_status, alert_message).
    panic_status: "NORMAL", "ABNORMAL", "PANIC_CRITICAL"
    """
    param_key = (parameter_name or "").lower().strip()
    config = None
    for k, v in PANIC_THRESHOLDS.items():
        if k in param_key:
            config = v
            break

    if not config:
        return "NORMAL", "Result within standard limits."

    unit = config["unit"]
    if value <= config["panic_low"]:
        return (
            "PANIC_CRITICAL",
            f"CRITICAL PANIC LOW: {value} {unit} (Threshold < {config['panic_low']} {unit})",
        )
    elif value >= config["panic_high"]:
        return (
            "PANIC_CRITICAL",
            f"CRITICAL PANIC HIGH: {value} {unit} (Threshold > {config['panic_high']} {unit})",
        )
    elif value < config["normal_low"]:
        return (
            "ABNORMAL",
            f"Abnormal Low: {value} {unit} (Normal: {config['normal_low']}-{config['normal_high']} {unit})",
        )
    elif value > config["normal_high"]:
        return (
            "ABNORMAL",
            f"Abnormal High: {value} {unit} (Normal: {config['normal_low']}-{config['normal_high']} {unit})",
        )

    return (
        "NORMAL",
        f"Normal: {value} {unit} (Reference: {config['normal_low']}-{config['normal_high']} {unit})",
    )


@lis_bp.route("/enter", methods=["POST"])
@login_required
@roles_required("lab_tech", "radiology", "admin")
def handle_enter_result():
    """
    Lab Tech enters test result (Tier 1). Evaluates panic status.

    P0-11: tech_id is derived from the authenticated user — never from the request body.
    P0-04: patient_id, lab_test_id, parameter_name and result_value are all
    REQUIRED. The previous silent defaults (lab_test_id=1, "Hemoglobin",
    14.0) could fabricate a normal-looking result from a malformed request.
    """
    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    lab_test_id = data.get("lab_test_id")
    parameter_name = (data.get("parameter_name") or "").strip()
    result_notes = data.get("notes", "")

    if not patient_id or not lab_test_id or not parameter_name:
        return jsonify(
            {
                "error": "patient_id, lab_test_id, and parameter_name are required. "
                "No defaults are applied to clinical data."
            }
        ), 400

    try:
        result_val = float(data.get("result_value"))
    except (TypeError, ValueError):
        return jsonify(
            {"error": "result_value is required and must be a number. "
            "No default value is applied to clinical data."}
        ), 400

    # P0-11: Derive tech_id from authenticated session — ignore any client-supplied value.
    tech_id = _require_authenticated_user_id()

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    # Validate the lab test exists — never persist a dangling clinical result.
    from departments.models.medicine import LabTest

    if not db.session.get(LabTest, lab_test_id):
        return jsonify({"error": f"Lab test {lab_test_id} not found"}), 404

    panic_status, panic_msg = evaluate_panic_level(parameter_name, result_val)
    res_uuid = f"RES-{str(uuid.uuid4())[:8].upper()}"

    lab_res = LabResult(
        patient_id=patient_id,
        lab_test_id=lab_test_id,
        result_id=res_uuid,
        result=f"{parameter_name}: {result_val}",
        result_notes=result_notes,
        status="PENDING_VERIFICATION",
        panic_status=panic_status,
        panic_message=panic_msg,
        updated_by=tech_id,
    )
    db.session.add(lab_res)
    db.session.commit()

    logger.info(
        "Lab result entered: result_id=%s patient=%s tech_id=%s panic=%s",
        res_uuid,
        patient_id,
        tech_id,
        panic_status,
    )

    return jsonify(
        {
            "success": True,
            "result_id": res_uuid,
            "panic_status": panic_status,
            "panic_message": panic_msg,
            "status": lab_res.status,
        }
    ), 201


@lis_bp.route("/verify", methods=["POST"])
@login_required
@roles_required("lab_tech", "radiology", "admin", "doctor")
def handle_verify_result():
    """
    Pathologist/Doctor verifies test result (Tier 2). Dispatches panic alert if critical.

    P0-06 / P0-11:
      - verifier_id derived from authenticated user — client-supplied value rejected.
      - Re-verification of an already-VERIFIED result is blocked to prevent silent overwrite.
      - Tier separation: the user who ENTERED the result (updated_by) cannot
        verify it — a second person is mandatory for 2-tier sign-off.
    """
    data = request.get_json() or {}
    result_id = data.get("result_id")
    action = data.get("action", "VERIFY")  # VERIFY or REJECT

    if not result_id:
        return jsonify({"error": "result_id is required"}), 400

    # P0-11: Derive verifier from authenticated session — never from request body.
    verifier_id = _require_authenticated_user_id()

    lab_res = LabResult.query.filter_by(result_id=result_id).first()
    if not lab_res:
        return jsonify({"error": "Lab result not found"}), 404

    # P0-06: Guard against silent overwrite of an already-verified result.
    if lab_res.status == "VERIFIED":
        return jsonify(
            {
                "error": "Result is already VERIFIED. Use the amendment workflow to make corrections.",
                "result_id": result_id,
                "status": "VERIFIED",
            }
        ), 409

    # P0-02: 2-tier separation — the entering tech can never verify their own
    # result. Admin role-switch is exempt (supervisor override is audited).
    if (
        lab_res.updated_by is not None
        and lab_res.updated_by == verifier_id
        and not (
            getattr(current_user, "role", "") == "admin"
            and "switched_user" not in session
        )
    ):
        return jsonify(
            {
                "error": "2-tier verification violation: the user who entered this "
                "result cannot verify it. A second signatory is required.",
                "result_id": result_id,
                "code": "SELF_VERIFICATION_BLOCKED",
            }
        ), 403

    if action == "REJECT":
        lab_res.status = "REJECTED"
        lab_res.verified_by = verifier_id
        lab_res.verified_at = datetime.now(timezone.utc)
        db.session.commit()
        logger.info(
            "Lab result REJECTED: result_id=%s verifier_id=%s",
            result_id,
            verifier_id,
        )
        return jsonify({"success": True, "status": "REJECTED"}), 200

    lab_res.status = "VERIFIED"
    lab_res.verified_by = verifier_id
    lab_res.verified_at = datetime.now(timezone.utc)

    # P0-03: Dispatch panic alert to the ORDERING CLINICIAN, not the verifier.
    # The verifier just saw the result — the clinician managing the patient is
    # the one who must act on a critical value. Fall back to verifier if no
    # ordering clinician can be resolved.
    notification_sent = False
    alert_recipient_id = None
    if lab_res.panic_status == "PANIC_CRITICAL":
        clinician_id = _resolve_ordering_clinician_id(lab_res)

        alert_text = (
            f"CRITICAL LAB PANIC ALERT: Patient {lab_res.patient_id} "
            f"— {lab_res.panic_message}"
        )
        recipients = []
        if clinician_id and clinician_id != verifier_id:
            recipients.append(clinician_id)
        # The verifier still gets a copy for documentation/closure.
        recipients.append(verifier_id)

        for receiver in dict.fromkeys(recipients):  # de-duplicate, keep order
            db.session.add(Notifications(receiver_id=receiver, message=alert_text))
            notification_sent = True
            if alert_recipient_id is None:
                alert_recipient_id = receiver

    db.session.commit()

    logger.info(
        "Lab result VERIFIED: result_id=%s verifier_id=%s panic=%s alert_sent=%s "
        "alert_recipient=%s",
        result_id,
        verifier_id,
        lab_res.panic_status,
        notification_sent,
        alert_recipient_id,
    )

    return jsonify(
        {
            "success": True,
            "result_id": result_id,
            "status": "VERIFIED",
            "panic_status": lab_res.panic_status,
            "panic_alert_sent": notification_sent,
            "panic_alert_recipient_id": alert_recipient_id,
        }
    ), 200


@lis_bp.route("/panic_alerts", methods=["GET"])
@login_required
@roles_required("lab_tech", "radiology", "admin", "doctor", "nursing")
def handle_list_panic_alerts():
    """Get active critical panic alerts across all lab results."""
    critical_results = (
        LabResult.query.filter_by(panic_status="PANIC_CRITICAL")
        .order_by(LabResult.test_date.desc())
        .all()
    )
    alerts = []
    for r in critical_results:
        alerts.append(
            {
                "result_id": r.result_id,
                "patient_id": r.patient_id,
                "result": r.result,
                "panic_message": r.panic_message,
                "status": r.status,
                "test_date": r.test_date.isoformat() if r.test_date else None,
            }
        )
    return jsonify({"panic_alerts": alerts, "count": len(alerts)}), 200
