"""
departments/renal/routes.py
────────────────────────────
HTTP routes for the Renal / Dialysis Unit.

Scope: DECISIONS_PENDING.md §23 decisions #1, #2, #4, #7 only.
  ✅  Log HD/CRRT sessions (manual, no slot scheduler)
  ✅  Update session status (triggers billing on COMPLETED via event_listeners)
  ✅  Log vascular access records
  ✅  JSON list / detail views per patient

NOT implemented here:
  ❌  Kt/V adequacy calculation (#3)
  ❌  Prescription / anticoagulation workflow (#5)
  ❌  CDSS pharmacy hook (#6)
"""

import logging
from datetime import date, datetime

from flask import jsonify, render_template, request

from departments.renal.engine import (
    create_session,
    get_patient_access_records,
    get_patient_sessions,
    log_access_record,
    session_summary,
    update_session_status,
)

from . import bp as renal_bp

logger = logging.getLogger(__name__)


# ── Utility ───────────────────────────────────────────────────────────────────

def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    try:
        return date.fromisoformat(val)
    except (ValueError, TypeError):
        return None


def _parse_datetime(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None


def _float(data: dict, key: str) -> float | None:
    v = data.get(key)
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ── Session routes ────────────────────────────────────────────────────────────

@renal_bp.route("/", methods=["GET"])
@renal_bp.route("/sessions", methods=["GET"])
@renal_bp.route("/sessions/<string:patient_id>", methods=["GET"])
def list_sessions(patient_id: str = "P001"):
    """
    GET /renal/sessions/<patient_id>
    List all dialysis sessions for a patient, newest first.
    Renders HTML console for browser requests, JSON for API clients.
    """
    sessions = get_patient_sessions(patient_id)
    raw_status = request.args.get("status")
    if raw_status:
        sessions = [s for s in sessions if str(s.status).upper() == raw_status.upper()]

    wants_html = "text/html" in request.headers.get("Accept", "")
    if wants_html:
        access_records = get_patient_access_records(patient_id)
        formatted_access = [
            {
                "id": r.id,
                "access_type": r.access_type,
                "insertion_date": r.insertion_date.isoformat() if r.insertion_date else None,
                "site_description": r.site_description,
                "complication_notes": r.complication_notes,
                "dialysis_session_id": r.dialysis_session_id,
            }
            for r in access_records
        ]
        return render_template(
            "renal/sessions.html",
            patient_id=patient_id,
            sessions=[session_summary(s) for s in sessions],
            access_records=formatted_access,
        )

    return jsonify({
        "patient_id": patient_id,
        "count": len(sessions),
        "sessions": [session_summary(s) for s in sessions],
    }), 200




@renal_bp.route("/sessions/<string:patient_id>", methods=["POST"])
def log_session(patient_id: str):
    """
    POST /renal/sessions/<patient_id>
    Create a new HD or CRRT session log.

    Required JSON fields: modality, session_date
    Optional: start_time, blood_flow_rate, dialysate_flow_rate,
              ultrafiltration_volume, pre_weight, post_weight, notes
    """
    data = request.get_json(silent=True) or {}

    modality = (data.get("modality") or "").strip().upper()
    if not modality:
        return jsonify({"error": "modality is required (HD or CRRT)"}), 400

    raw_date = data.get("session_date")
    session_date = _parse_date(raw_date)
    if session_date is None:
        return jsonify({"error": "session_date is required (ISO format: YYYY-MM-DD)"}), 400

    # nurse_id: prefer authenticated user; fall back to body param for API callers
    from flask_login import current_user
    nurse_id = (
        current_user.id
        if current_user and getattr(current_user, "is_authenticated", False)
        else int(data.get("nurse_id", 1))
    )

    try:
        sess = create_session(
            patient_id=patient_id,
            nurse_id=nurse_id,
            modality=modality,
            session_date=session_date,
            start_time=_parse_datetime(data.get("start_time")),
            blood_flow_rate=_float(data, "blood_flow_rate"),
            dialysate_flow_rate=_float(data, "dialysate_flow_rate"),
            ultrafiltration_volume=_float(data, "ultrafiltration_volume"),
            pre_weight=_float(data, "pre_weight"),
            post_weight=_float(data, "post_weight"),
            status=data.get("status", "SCHEDULED"),
            notes=data.get("notes"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    return jsonify({"success": True, "session": session_summary(sess)}), 201


@renal_bp.route("/sessions/<int:session_id>/status", methods=["PATCH"])
def update_status(session_id: int):
    """
    PATCH /renal/sessions/<session_id>/status
    Transition a session to a new status.

    Required JSON: { "status": "COMPLETED" | "IN_PROGRESS" | "TERMINATED_EARLY" }
    Optional:      { "end_time": "<ISO datetime>" }

    When status → COMPLETED, event_listeners.py fires a flat billing charge.
    """
    data = request.get_json(silent=True) or {}
    new_status = (data.get("status") or "").strip().upper()
    if not new_status:
        return jsonify({"error": "status is required"}), 400

    end_time = _parse_datetime(data.get("end_time"))

    try:
        sess = update_session_status(session_id, new_status, end_time=end_time)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify({"success": True, "session": session_summary(sess)}), 200


# ── Vascular access routes ────────────────────────────────────────────────────

@renal_bp.route("/access/<string:patient_id>", methods=["GET"])
def list_access_records(patient_id: str):
    """
    GET /renal/access/<patient_id>
    List vascular access records for a patient.
    """
    records = get_patient_access_records(patient_id)
    wants_html = "text/html" in request.headers.get("Accept", "")
    if wants_html:
        sessions = get_patient_sessions(patient_id)
        formatted_access = [
            {
                "id": r.id,
                "access_type": r.access_type,
                "insertion_date": r.insertion_date.isoformat() if r.insertion_date else None,
                "site_description": r.site_description,
                "complication_notes": r.complication_notes,
                "dialysis_session_id": r.dialysis_session_id,
            }
            for r in records
        ]
        return render_template(
            "renal/sessions.html",
            patient_id=patient_id,
            sessions=[session_summary(s) for s in sessions],
            access_records=formatted_access,
        )

    return jsonify({
        "patient_id": patient_id,
        "count": len(records),
        "records": [
            {
                "id": r.id,
                "access_type": r.access_type,
                "insertion_date": r.insertion_date.isoformat() if r.insertion_date else None,
                "site_description": r.site_description,
                "complication_notes": r.complication_notes,
                "dialysis_session_id": r.dialysis_session_id,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
    }), 200




@renal_bp.route("/access/<string:patient_id>", methods=["POST"])
def add_access_record(patient_id: str):
    """
    POST /renal/access/<patient_id>
    Log a vascular access record (decision #4).

    Required JSON: access_type  (AVF | AVG | Tunnelled Catheter | Temporary Catheter)
    Optional:      insertion_date, site_description, complication_notes, dialysis_session_id
    """
    data = request.get_json(silent=True) or {}
    access_type = (data.get("access_type") or "").strip()
    if not access_type:
        return jsonify({"error": "access_type is required"}), 400

    try:
        record = log_access_record(
            patient_id=patient_id,
            access_type=access_type,
            dialysis_session_id=data.get("dialysis_session_id"),
            insertion_date=_parse_date(data.get("insertion_date")),
            site_description=data.get("site_description"),
            complication_notes=data.get("complication_notes"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    return jsonify({
        "success": True,
        "record": {
            "id": record.id,
            "access_type": record.access_type,
            "insertion_date": record.insertion_date.isoformat() if record.insertion_date else None,
            "site_description": record.site_description,
            "complication_notes": record.complication_notes,
            "dialysis_session_id": record.dialysis_session_id,
        },
    }), 201
