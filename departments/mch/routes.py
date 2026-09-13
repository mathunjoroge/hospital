"""
MCH, ANC, and Immunization API routes.
"""

from flask import jsonify, request
from flask_login import login_required

from departments.rbac import roles_required

from . import bp
from .cold_chain import ColdChainEngine
from .engine import MchEngine

_engine = MchEngine()
_cc = ColdChainEngine()


@bp.route("/")
@login_required
@roles_required("nursing")
def index():
    return "MCH & Immunization Module Active"


@bp.route("/api/anc-visit", methods=["POST"])
@login_required
@roles_required("nursing")
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
@roles_required("nursing")
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
            "message": f"{vaccine_name} Dose {dose_number} recorded successfully.",
        }
    ), 201


@bp.route("/api/child/<int:child_patient_id>/schedule", methods=["GET"])
@login_required
@roles_required("nursing")
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


@bp.route("/api/anc-visit/<visit_id>/close", methods=["POST"])
@login_required
@roles_required("nursing")
def close_anc_visit(visit_id: str):
    """
    Close an ANC visit: discharges the linked encounter so the patient
    can be billed and the encounter stage reflects DISCHARGED.
    """
    visit = _engine.close_anc_visit(visit_id)
    if not visit:
        return jsonify({"error": "ANC visit not found"}), 404

    return jsonify({"status": "success", "visit_id": visit.id, "message": "ANC visit closed."}), 200


# ── Cold-Chain Routes ──────────────────────────────────────────────────────


@bp.route("/api/cold-chain/receive-batch", methods=["POST"])
@login_required
@roles_required("nursing", "pharmacy")
def receive_vaccine_batch():
    """
    POST /mch/api/cold-chain/receive-batch

    Register receipt of a vaccine batch into cold-chain stock.

    Request body (JSON):
      vaccine_name, batch_number, manufacturer, quantity (vials),
      doses_per_vial, expiry_date (YYYY-MM-DD), storage_location,
      supplied_by (optional)
    """
    data = request.get_json(silent=True) or {}

    required = ["vaccine_name", "batch_number", "manufacturer",
                "quantity", "doses_per_vial", "expiry_date", "storage_location"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        from datetime import date
        expiry_date = date.fromisoformat(data["expiry_date"])
    except ValueError:
        return jsonify({"error": "expiry_date must be in YYYY-MM-DD format."}), 400

    try:
        batch = _cc.receive_vaccine_batch(
            vaccine_name=data["vaccine_name"],
            batch_number=data["batch_number"],
            manufacturer=data["manufacturer"],
            quantity=int(data["quantity"]),
            doses_per_vial=int(data["doses_per_vial"]),
            expiry_date=expiry_date,
            storage_location=data["storage_location"],
            supplied_by=data.get("supplied_by"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "status": "success",
        "batch_id": batch.id,
        "vaccine_name": batch.vaccine_name,
        "batch_number": batch.batch_number,
        "quantity_vials": batch.quantity_vials,
        "expiry_date": batch.expiry_date.isoformat(),
    }), 201


@bp.route("/api/cold-chain/temperature-log", methods=["POST"])
@login_required
@roles_required("nursing", "pharmacy")
def log_temperature():
    """
    POST /mch/api/cold-chain/temperature-log

    Record a temperature reading for a cold-chain storage location.

    Request body (JSON):
      storage_location, temperature_celsius,
      sensor_id (optional), notes (optional)
    """
    data = request.get_json(silent=True) or {}

    storage_location = data.get("storage_location")
    temperature_celsius = data.get("temperature_celsius")

    if not storage_location or temperature_celsius is None:
        return jsonify({"error": "storage_location and temperature_celsius are required."}), 400

    try:
        log = _cc.log_temperature(
            storage_location=storage_location,
            temperature_celsius=float(temperature_celsius),
            sensor_id=data.get("sensor_id"),
            notes=data.get("notes"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "status": "success",
        "log_id": log.id,
        "is_breach": log.is_breach,
        "breach_type": log.breach_type,
        "temperature_celsius": log.temperature_celsius,
        "storage_location": log.storage_location,
        "recorded_at": log.recorded_at.isoformat(),
    }), 201


@bp.route("/api/cold-chain/stock", methods=["GET"])
@login_required
@roles_required("nursing", "pharmacy")
def cold_chain_stock():
    """
    GET /mch/api/cold-chain/stock?vaccine_name=BCG

    Returns current cold-chain vaccine inventory (FEFO-sorted).
    Optional query param: vaccine_name to filter by specific vaccine.
    """
    vaccine_name = request.args.get("vaccine_name") or None
    summary = _cc.get_stock_summary(vaccine_name=vaccine_name)
    return jsonify({"count": len(summary), "stock": summary}), 200


@bp.route("/api/cold-chain/temperature-history/<storage_location>", methods=["GET"])
@login_required
@roles_required("nursing", "pharmacy")
def temperature_history(storage_location: str):
    """
    GET /mch/api/cold-chain/temperature-history/<location>

    Returns recent temperature logs for a storage location.
    Query params:
      - limit (int, default 100)
      - breaches_only (bool, default false)
    """
    limit = request.args.get("limit", default=100, type=int)
    breaches_only = request.args.get("breaches_only", "false").lower() == "true"

    logs = _cc.get_temperature_history(
        storage_location=storage_location,
        limit=limit,
        breaches_only=breaches_only,
    )
    return jsonify({"storage_location": storage_location, "count": len(logs), "logs": logs}), 200


@bp.route("/api/cold-chain/alerts/near-expiry", methods=["GET"])
@login_required
@roles_required("nursing", "pharmacy")
def near_expiry_alerts():
    """
    GET /mch/api/cold-chain/alerts/near-expiry?days=30

    Returns vaccine batches expiring within `days` days (default 30).
    """
    days = request.args.get("days", default=30, type=int)
    alerts = _cc.get_near_expiry_alerts(days_threshold=days)
    return jsonify({"threshold_days": days, "count": len(alerts), "alerts": alerts}), 200
