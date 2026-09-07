"""
departments/telemedicine/routes.py
───────────────────────────────────
Phase E — Telemedicine Routes & Web UI

Routes:
  POST /telemedicine/session/create          — create virtual room token & session
  GET  /telemedicine/room/<session_uuid>      — Web UI consultation canvas
  POST /telemedicine/session/<uuid>/start    — activate session
  POST /telemedicine/session/<uuid>/notes    — save clinical notes
  POST /telemedicine/session/<uuid>/complete — end session
  GET  /telemedicine/sessions                 — list doctor's active & past sessions
"""

import secrets

from flask import current_app, jsonify, render_template, request
from flask_login import login_required

from departments.api.audit import log_audit_event
from departments.models.telemedicine import TelemedicineSession
from departments.rbac import get_effective_user, roles_required
from extensions import db

from . import bp


@bp.before_request
def check_telemedicine_enabled():
    """Quarantine Telemedicine module behind feature flag (ENABLE_TELEMEDICINE=False by default)."""
    if not current_app.config.get("ENABLE_TELEMEDICINE", False):
        return jsonify(
            {
                "error": "Telemedicine module is currently disabled by system policy.",
                "code": "FEATURE_DISABLED",
            }
        ), 403


@bp.route("/session/create", methods=["POST"])
@login_required
@roles_required("doctor", "medicine", "admin")
def create_session():
    """
    Create a new telemedicine virtual session.
    JSON payload:
      - patient_id (str, required)
      - appointment_id (int, optional)
      - scheduled_start (iso str, optional)
    """
    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400

    user = get_effective_user()
    room_token = f"room_{secrets.token_urlsafe(16)}"

    session = TelemedicineSession(
        doctor_id=user.id,
        patient_id=patient_id,
        appointment_id=data.get("appointment_id"),
        status="SCHEDULED",
        room_token=room_token,
    )
    db.session.add(session)
    db.session.commit()

    log_audit_event(
        action="CREATE_TELEMEDICINE_SESSION",
        resource_type="TelemedicineSession",
        resource_id=session.session_uuid,
        details={"patient_id": patient_id, "doctor_id": user.id},
    )

    return jsonify(
        {
            "message": "Telemedicine session created successfully",
            "session": session.to_dict(),
        }
    ), 201


@bp.route("/room/<session_uuid>", methods=["GET"])
@login_required
def room_ui(session_uuid):
    """Render the WebRTC consultation video canvas and clinical interface."""
    user = get_effective_user()
    session = TelemedicineSession.query.filter_by(
        session_uuid=session_uuid
    ).first_or_404()

    if not session.is_participant(user):
        return jsonify({"error": "Unauthorized to join this consultation room"}), 403

    return render_template(
        "telemedicine/room.html",
        session=session,
        user=user,
    )


@bp.route("/session/<session_uuid>/start", methods=["POST"])
@login_required
def start_session(session_uuid):
    """Mark telemedicine session as ACTIVE."""
    user = get_effective_user()
    session = TelemedicineSession.query.filter_by(
        session_uuid=session_uuid
    ).first_or_404()

    if not session.is_participant(user):
        return jsonify({"error": "Unauthorized"}), 403

    session.start_session()
    db.session.commit()

    log_audit_event(
        action="START_TELEMEDICINE_SESSION",
        resource_type="TelemedicineSession",
        resource_id=session.session_uuid,
    )

    return jsonify({"message": "Session started", "session": session.to_dict()})


@bp.route("/session/<session_uuid>/notes", methods=["POST"])
@login_required
@roles_required("doctor", "medicine", "admin")
def save_notes(session_uuid):
    """Save consultation clinical notes during or after session."""
    data = request.get_json() or {}
    notes = data.get("notes", "").strip()

    session = TelemedicineSession.query.filter_by(
        session_uuid=session_uuid
    ).first_or_404()
    session.clinical_notes = notes
    db.session.commit()

    return jsonify(
        {"message": "Clinical notes updated", "notes": session.clinical_notes}
    )


@bp.route("/session/<session_uuid>/complete", methods=["POST"])
@login_required
@roles_required("doctor", "medicine", "admin")
def complete_session(session_uuid):
    """End virtual consultation and mark COMPLETED."""
    data = request.get_json() or {}
    notes = data.get("notes")

    session = TelemedicineSession.query.filter_by(
        session_uuid=session_uuid
    ).first_or_404()
    session.end_session(notes=notes)
    db.session.commit()

    log_audit_event(
        action="COMPLETE_TELEMEDICINE_SESSION",
        resource_type="TelemedicineSession",
        resource_id=session.session_uuid,
    )

    return jsonify({"message": "Session completed", "session": session.to_dict()})


@bp.route("/sessions", methods=["GET"])
@login_required
def list_sessions():
    """List telemedicine sessions for current doctor/patient."""
    user = get_effective_user()

    if user.role in ("doctor", "medicine", "admin"):
        query = TelemedicineSession.query.filter_by(doctor_id=user.id)
    else:
        # Patient portal user lookup by patient_id
        patient_id = getattr(user, "patient_id", None)
        query = (
            TelemedicineSession.query.filter_by(patient_id=patient_id)
            if patient_id
            else TelemedicineSession.query.filter_by(id=-1)
        )

    sessions = query.order_by(TelemedicineSession.created_at.desc()).all()
    return jsonify({"sessions": [s.to_dict() for s in sessions]})
