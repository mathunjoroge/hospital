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

P0-08 / P0-11 Security fixes:
  - All routes now require @login_required + @roles_required.
  - Removed data.get("nurse_id", 1) fallback — nurse_id derived exclusively
    from the authenticated security context (current_user.id).
  - Client-supplied nurse/staff IDs are rejected.
"""

import logging
from datetime import date, datetime, timedelta

from flask import abort, jsonify, render_template, request
from flask_login import current_user, login_required

from departments.models.records import Patient
from departments.rbac import roles_required
from departments.renal.engine import (
    assign_chair_time,
    create_session,
    get_patient_access_records,
    get_patient_sessions,
    get_unit_sessions,
    log_access_record,
    session_summary,
    update_session_status,
)
from extensions import db

from . import bp as renal_bp

logger = logging.getLogger(__name__)

# Allowed clinical roles for renal unit access
_RENAL_ROLES = ("renal", "nursing", "admin", "doctor")

# Statuses shown on the unit-wide schedule board by default (pending work)
_UNIT_BOARD_STATUSES = {"SCHEDULED", "IN_PROGRESS"}

# Day-sheet group order: clinical shift order, unscheduled (no chair time) last
_SHIFT_GROUP_ORDER = ("MORNING", "AFTERNOON", "EVENING", None)


def _require_authenticated_user_id() -> int:
    """
    Return the authenticated user's ID from the server-side security context.
    Aborts 401 if unauthenticated. Never falls back to nurse_id=1 or body params.
    """
    if current_user and getattr(current_user, "is_authenticated", False):
        return current_user.id
    abort(401)


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
@login_required
@roles_required(*_RENAL_ROLES)
def list_sessions(patient_id: str | None = None):
    """
    GET /renal/sessions                     → unit-wide schedule board
    GET /renal/sessions?status=SCHEDULED    → board filtered by status
    GET /renal/sessions?start=...&end=...   → board filtered by date range
    GET /renal/sessions/<patient_id>        → per-patient console

    Status filter values: UPCOMING (default: SCHEDULED + IN_PROGRESS,
    soonest first), ALL, or a single status such as SCHEDULED / IN_PROGRESS /
    COMPLETED / TERMINATED_EARLY.

    Date filters: start / end (inclusive, ISO YYYY-MM-DD). Invalid values are
    ignored. Applies to both the unit board and the per-patient console.

    Renders HTML for browser requests, JSON for API clients.
    NOTE: previously the no-patient view silently showed demo patient P001;
    it now shows the unit-wide board. Per-patient behaviour is unchanged.
    """
    wants_html = "text/html" in request.headers.get("Accept", "")
    raw_status = (request.args.get("status") or "").strip().upper()
    start_date = _parse_date(request.args.get("start"))
    end_date = _parse_date(request.args.get("end"))
    date_context = {
        "filter_start": start_date.isoformat() if start_date else None,
        "filter_end": end_date.isoformat() if end_date else None,
    }

    if patient_id is None:
        # ── Unit-wide schedule board ─────────────────────────────────────
        sessions = _board_sessions(raw_status, start_date, end_date)
        is_day_sheet = bool(
            date_context["filter_start"]
            and date_context["filter_start"] == date_context["filter_end"]
        )
        if wants_html:
            return render_template(
                "renal/sessions.html",
                patient_id=None,
                patient=None,
                sessions=sessions,
                access_records=[],
                selected_status=raw_status or "UPCOMING",
                today=date.today().isoformat(),
                week_end=(date.today() + timedelta(days=6)).isoformat(),
                is_day_sheet=is_day_sheet,
                session_groups=_group_by_shift(sessions) if is_day_sheet else [],
                **date_context,
            )
        return jsonify(
            {
                "scope": "unit",
                "count": len(sessions),
                "date_range": date_context,
                "sessions": sessions,
            }
        ), 200

    # ── Per-patient console ───────────────────────────────────────────────
    sessions = get_patient_sessions(patient_id)
    if raw_status and raw_status != "ALL":
        sessions = [s for s in sessions if str(s.status).upper() == raw_status.upper()]
    if start_date is not None:
        sessions = [s for s in sessions if s.session_date and s.session_date >= start_date]
    if end_date is not None:
        sessions = [s for s in sessions if s.session_date and s.session_date <= end_date]

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    rows = []
    for s in sessions:
        summary = session_summary(s)
        summary["patient_name"] = patient.name if patient else None
        rows.append(summary)

    if wants_html:
        access_records = get_patient_access_records(patient_id)
        formatted_access = [
            {
                "id": r.id,
                "access_type": r.access_type,
                "insertion_date": r.insertion_date.isoformat()
                if r.insertion_date
                else None,
                "site_description": r.site_description,
                "complication_notes": r.complication_notes,
                "dialysis_session_id": r.dialysis_session_id,
            }
            for r in access_records
        ]
        is_day_sheet = bool(
            date_context["filter_start"]
            and date_context["filter_start"] == date_context["filter_end"]
        )
        return render_template(
            "renal/sessions.html",
            patient_id=patient_id,
            patient=patient,
            sessions=rows,
            access_records=formatted_access,
            selected_status=raw_status or None,
            today=date.today().isoformat(),
            week_end=(date.today() + timedelta(days=6)).isoformat(),
            is_day_sheet=is_day_sheet,
            session_groups=_group_by_shift(rows) if is_day_sheet else [],
            **date_context,
        )

    return jsonify(
        {
            "patient_id": patient_id,
            "patient_name": patient.name if patient else None,
            "date_range": date_context,
            "count": len(rows),
            "sessions": rows,
        }
    ), 200


def _group_by_shift(rows: list[dict]) -> list[tuple[str | None, list[dict]]]:
    """
    Group session rows by shift for the day-sheet view.

    Groups appear in clinical order (MORNING → AFTERNOON → EVENING →
    TIME TBD); within a group, rows sort by chair start time, then patient ID.
    """
    groups: dict[str | None, list[dict]] = {label: [] for label in _SHIFT_GROUP_ORDER}
    for row in rows:
        groups.setdefault(row.get("shift"), []).append(row)

    result = []
    for label in _SHIFT_GROUP_ORDER:
        items = groups.get(label, [])
        if not items:
            continue
        items.sort(
            key=lambda r: (
                r.get("start_time_hm") or "99:99",
                r.get("patient_id") or "",
            )
        )
        result.append((label, items))
    return result


def _board_sessions(
    status: str, start_date: date | None = None, end_date: date | None = None
) -> list[dict]:
    """Unit-wide board rows for the given filters, with patient names."""
    if status in ("", "UPCOMING"):
        statuses: set[str] | None = _UNIT_BOARD_STATUSES
    elif status == "ALL":
        statuses = None
    else:
        statuses = {status}

    sessions = get_unit_sessions(
        statuses=statuses, start_date=start_date, end_date=end_date
    )
    patient_ids = {s.patient_id for s in sessions}
    names = {}
    if patient_ids:
        names = {
            p.patient_id: p.name
            for p in Patient.query.filter(Patient.patient_id.in_(patient_ids)).all()
        }

    rows = []
    for s in sessions:
        summary = session_summary(s)
        summary["patient_name"] = names.get(s.patient_id)
        rows.append(summary)
    return rows


@renal_bp.route("/api/search-patients", methods=["GET"])
@login_required
@roles_required(*_RENAL_ROLES)
def search_patients():
    """
    GET /renal/api/search-patients?q=<term>
    Select2-style patient search for the console header (ID or name).
    Only active (non-merged) patients are returned.
    """
    term = (request.args.get("q") or "").strip()
    if len(term) < 2:
        return jsonify([])

    patients = (
        Patient.query.filter(
            Patient.is_active.is_(True),
            db.or_(
                Patient.patient_id.ilike(f"%{term}%"),
                Patient.name.ilike(f"%{term}%"),
            ),
        )
        .order_by(Patient.patient_id)
        .limit(15)
        .all()
    )
    return jsonify(
        [
            {"id": p.patient_id, "text": f"{p.name} ({p.patient_id})"}
            for p in patients
        ]
    )


@renal_bp.route("/sessions/<string:patient_id>", methods=["POST"])
@login_required
@roles_required(*_RENAL_ROLES)
def log_session(patient_id: str):
    """
    POST /renal/sessions/<patient_id>
    Create a new HD or CRRT session log.

    P0-11: nurse_id is derived exclusively from the authenticated security context.
    Client-supplied nurse_id in the request body is ignored.

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
        return jsonify(
            {"error": "session_date is required (ISO format: YYYY-MM-DD)"}
        ), 400

    # P0-11: Derive nurse_id from authenticated user only — never from request body.
    nurse_id = _require_authenticated_user_id()

    try:
        sess = create_session(
            patient_id=patient_id,
            nurse_id=nurse_id,
            modality=modality,
            session_date=session_date,
            start_time=_parse_datetime(data.get("start_time")),
            end_time=_parse_datetime(data.get("end_time")),
            blood_flow_rate=_float(data, "blood_flow_rate"),
            dialysate_flow_rate=_float(data, "dialysate_flow_rate"),
            ultrafiltration_volume=_float(data, "ultrafiltration_volume"),
            pre_weight=_float(data, "pre_weight"),
            post_weight=_float(data, "post_weight"),
            pre_bun=_float(data, "pre_bun"),
            post_bun=_float(data, "post_bun"),
            status=data.get("status", "SCHEDULED"),
            notes=data.get("notes"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    logger.info(
        "Renal session created: patient=%s session_id=%s nurse_id=%s modality=%s",
        patient_id,
        sess.id,
        nurse_id,
        modality,
    )

    return jsonify({"success": True, "session": session_summary(sess)}), 201


@renal_bp.route("/sessions/<int:session_id>/status", methods=["PATCH"])
@login_required
@roles_required(*_RENAL_ROLES)
def update_status(session_id: int):
    """
    PATCH /renal/sessions/<session_id>/status
    Transition a session to a new status.
    """
    data = request.get_json(silent=True) or {}
    new_status = (data.get("status") or "").strip().upper()
    if not new_status:
        return jsonify({"error": "status is required"}), 400

    end_time = _parse_datetime(data.get("end_time"))
    post_bun = _float(data, "post_bun")
    post_weight = _float(data, "post_weight")

    try:
        sess = update_session_status(
            session_id,
            new_status,
            end_time=end_time,
            post_bun=post_bun,
            post_weight=post_weight,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404

    logger.info(
        "Renal session status updated: session_id=%s new_status=%s actor=%s",
        session_id,
        new_status,
        current_user.id,
    )

    return jsonify({"success": True, "session": session_summary(sess)}), 200


@renal_bp.route("/sessions/<int:session_id>/chair-time", methods=["PATCH"])
@login_required
@roles_required(*_RENAL_ROLES)
def set_chair_time(session_id: int):
    """
    PATCH /renal/sessions/<session_id>/chair-time
    Assign (or reassign) the chair start time for a session — e.g. slotting a
    Records booking (date only) into the day sheet.

    JSON body: {"chair_time": "HH:MM", "end_time_hm": "HH:MM" (optional)}
    Both times are anchored to the session's own session_date.
    """
    data = request.get_json(silent=True) or {}
    chair_time = (data.get("chair_time") or "").strip()
    if not chair_time:
        return jsonify({"error": "chair_time is required (24-hour HH:MM)"}), 400
    end_time_hm = (data.get("end_time_hm") or "").strip() or None

    try:
        sess = assign_chair_time(
            session_id, chair_time=chair_time, end_time_hm=end_time_hm
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404

    logger.info(
        "Chair time assigned: session_id=%s start=%s actor=%s",
        session_id,
        sess.start_time,
        current_user.id,
    )
    return jsonify({"success": True, "session": session_summary(sess)}), 200


# ── Prescription routes (Section 23 #5) ──────────────────────────────────────


@renal_bp.route("/prescriptions/<string:patient_id>", methods=["GET"])
@login_required
@roles_required(*_RENAL_ROLES)
def list_prescriptions(patient_id: str):
    """
    GET /renal/prescriptions/<patient_id>
    List Nephrology Dialysis Prescriptions for a patient.
    """
    from departments.renal.engine import get_patient_prescriptions

    prescriptions = get_patient_prescriptions(patient_id)
    return jsonify(
        {
            "patient_id": patient_id,
            "count": len(prescriptions),
            "prescriptions": [
                {
                    "id": p.id,
                    "patient_id": p.patient_id,
                    "nephrologist_id": p.nephrologist_id,
                    "dialysate_flow_rate": p.dialysate_flow_rate,
                    "blood_flow_rate": p.blood_flow_rate,
                    "dialysate_composition": p.dialysate_composition,
                    "heparin_bolus_units": p.heparin_bolus_units,
                    "heparin_infusion_rate": p.heparin_infusion_rate,
                    "target_uf_liters": p.target_uf_liters,
                    "duration_hours": p.duration_hours,
                    "status": p.status,
                    "notes": p.notes,
                    "created_at": p.created_at.isoformat(),
                }
                for p in prescriptions
            ],
        }
    ), 200


@renal_bp.route("/prescriptions/<string:patient_id>", methods=["POST"])
@login_required
@roles_required("doctor", "admin", "renal")
def add_prescription(patient_id: str):
    """
    POST /renal/prescriptions/<patient_id>
    Create a new Nephrology Dialysis Prescription.
    """
    from departments.renal.engine import create_prescription

    data = request.get_json(silent=True) or {}
    nephrologist_id = _require_authenticated_user_id()

    try:
        p = create_prescription(
            patient_id=patient_id,
            nephrologist_id=nephrologist_id,
            dialysate_flow_rate=_float(data, "dialysate_flow_rate") or 500.0,
            blood_flow_rate=_float(data, "blood_flow_rate") or 300.0,
            dialysate_composition=data.get("dialysate_composition", "K 2.0, Ca 1.25, Na 138"),
            heparin_bolus_units=_float(data, "heparin_bolus_units"),
            heparin_infusion_rate=_float(data, "heparin_infusion_rate"),
            target_uf_liters=_float(data, "target_uf_liters"),
            duration_hours=_float(data, "duration_hours") or 4.0,
            notes=data.get("notes"),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422

    return jsonify(
        {
            "success": True,
            "prescription": {
                "id": p.id,
                "patient_id": p.patient_id,
                "nephrologist_id": p.nephrologist_id,
                "dialysate_flow_rate": p.dialysate_flow_rate,
                "blood_flow_rate": p.blood_flow_rate,
                "dialysate_composition": p.dialysate_composition,
                "heparin_bolus_units": p.heparin_bolus_units,
                "heparin_infusion_rate": p.heparin_infusion_rate,
                "target_uf_liters": p.target_uf_liters,
                "duration_hours": p.duration_hours,
                "status": p.status,
                "created_at": p.created_at.isoformat(),
            },
        }
    ), 201


# ── Vascular access routes ────────────────────────────────────────────────────



@renal_bp.route("/access/<string:patient_id>", methods=["GET"])
@login_required
@roles_required(*_RENAL_ROLES)
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
                "insertion_date": r.insertion_date.isoformat()
                if r.insertion_date
                else None,
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

    return jsonify(
        {
            "patient_id": patient_id,
            "count": len(records),
            "records": [
                {
                    "id": r.id,
                    "access_type": r.access_type,
                    "insertion_date": r.insertion_date.isoformat()
                    if r.insertion_date
                    else None,
                    "site_description": r.site_description,
                    "complication_notes": r.complication_notes,
                    "dialysis_session_id": r.dialysis_session_id,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ],
        }
    ), 200


@renal_bp.route("/access/<string:patient_id>", methods=["POST"])
@login_required
@roles_required(*_RENAL_ROLES)
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

    return jsonify(
        {
            "success": True,
            "record": {
                "id": record.id,
                "access_type": record.access_type,
                "insertion_date": record.insertion_date.isoformat()
                if record.insertion_date
                else None,
                "site_description": record.site_description,
                "complication_notes": record.complication_notes,
                "dialysis_session_id": record.dialysis_session_id,
            },
        }
    ), 201
