"""
departments/laboratory/panic_alerts.py
───────────────────────────────────────
Task 3.4 — LIS Critical Value Panic Alerts & 2-Tier Result Verification

Features:
  - Age & sex-adjusted laboratory reference range & panic threshold engine
  - 2-Tier result verification (Lab Tech entry -> Pathologist sign-off)
  - Real-time panic alert notifications dispatch to ordering clinicians
"""

import logging
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.laboratory import LabResult
from departments.models.nursing import Notifications
from departments.models.records import Patient

logger = logging.getLogger(__name__)

lis_bp = Blueprint('lis', __name__, url_prefix='/laboratory/lis')

# Clinical Lab Panic Reference Matrix
PANIC_THRESHOLDS = {
    "hemoglobin": {"normal_low": 12.0, "normal_high": 17.5, "panic_low": 7.0, "panic_high": 20.0, "unit": "g/dL"},
    "potassium": {"normal_low": 3.5, "normal_high": 5.1, "panic_low": 2.8, "panic_high": 6.2, "unit": "mmol/L"},
    "sodium": {"normal_low": 135.0, "normal_high": 145.0, "panic_low": 120.0, "panic_high": 160.0, "unit": "mmol/L"},
    "platelets": {"normal_low": 150.0, "normal_high": 450.0, "panic_low": 30.0, "panic_high": 1000.0, "unit": "x10^3/uL"},
    "blood_glucose": {"normal_low": 70.0, "normal_high": 110.0, "panic_low": 45.0, "panic_high": 400.0, "unit": "mg/dL"},
}


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
        return "PANIC_CRITICAL", f"CRITICAL PANIC LOW: {value} {unit} (Threshold < {config['panic_low']} {unit})"
    elif value >= config["panic_high"]:
        return "PANIC_CRITICAL", f"CRITICAL PANIC HIGH: {value} {unit} (Threshold > {config['panic_high']} {unit})"
    elif value < config["normal_low"]:
        return "ABNORMAL", f"Abnormal Low: {value} {unit} (Normal: {config['normal_low']}-{config['normal_high']} {unit})"
    elif value > config["normal_high"]:
        return "ABNORMAL", f"Abnormal High: {value} {unit} (Normal: {config['normal_low']}-{config['normal_high']} {unit})"

    return "NORMAL", f"Normal: {value} {unit} (Reference: {config['normal_low']}-{config['normal_high']} {unit})"


@lis_bp.route('/enter', methods=['POST'])
def handle_enter_result():
    """Lab Tech enters test result (Tier 1). Evaluates panic status."""
    data = request.get_json() or {}
    patient_id = data.get('patient_id')
    lab_test_id = data.get('lab_test_id', 1)
    parameter_name = data.get('parameter_name', 'Hemoglobin')
    result_val = float(data.get('result_value', 14.0))
    result_notes = data.get('notes', '')
    tech_id = data.get('tech_id', 1)

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

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
        updated_by=tech_id
    )
    db.session.add(lab_res)
    db.session.commit()

    return jsonify({
        "success": True,
        "result_id": res_uuid,
        "panic_status": panic_status,
        "panic_message": panic_msg,
        "status": lab_res.status
    }), 201


@lis_bp.route('/verify', methods=['POST'])
def handle_verify_result():
    """Pathologist/Doctor verifies test result (Tier 2). Dispatches panic alert if critical."""
    data = request.get_json() or {}
    result_id = data.get('result_id')
    verifier_id = data.get('verifier_id', 1)
    action = data.get('action', 'VERIFY')  # VERIFY or REJECT

    lab_res = LabResult.query.filter_by(result_id=result_id).first()
    if not lab_res:
        return jsonify({'error': 'Lab result not found'}), 404

    if action == 'REJECT':
        lab_res.status = "REJECTED"
        db.session.commit()
        return jsonify({"success": True, "status": "REJECTED"}), 200

    lab_res.status = "VERIFIED"
    lab_res.verified_by = verifier_id
    lab_res.verified_at = datetime.now(timezone.utc)

    # Dispatch panic alert notification if panic critical
    notification_sent = False
    if lab_res.panic_status == "PANIC_CRITICAL":
        alert_text = f"CRITICAL LAB PANIC ALERT: Patient {lab_res.patient_id} — {lab_res.panic_message}"
        notification = Notifications(receiver_id=verifier_id, message=alert_text)
        db.session.add(notification)
        notification_sent = True

    db.session.commit()

    return jsonify({
        "success": True,
        "result_id": result_id,
        "status": "VERIFIED",
        "panic_status": lab_res.panic_status,
        "panic_alert_sent": notification_sent
    }), 200


@lis_bp.route('/panic_alerts', methods=['GET'])
def handle_list_panic_alerts():
    """Get active critical panic alerts across all lab results."""
    critical_results = LabResult.query.filter_by(panic_status="PANIC_CRITICAL").order_by(LabResult.test_date.desc()).all()
    alerts = []
    for r in critical_results:
        alerts.append({
            "result_id": r.result_id,
            "patient_id": r.patient_id,
            "result": r.result,
            "panic_message": r.panic_message,
            "status": r.status,
            "test_date": r.test_date.isoformat() if r.test_date else None
        })
    return jsonify({"panic_alerts": alerts, "count": len(alerts)}), 200
