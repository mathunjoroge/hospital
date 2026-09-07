"""
Appointments and Queue Management API routes.
"""

from datetime import datetime

from flask import jsonify, request
from flask_login import login_required

from . import bp
from .engine import ScheduleEngine

_engine = ScheduleEngine()


@bp.route("/")
@login_required
def index():
    return "Appointments & Queue Module Active"


@bp.route("/api/book", methods=["POST"])
@login_required
def book_appointment():
    """
    Books a new appointment, enforcing double-booking prevention.
    Body: { "patient_id": 1, "provider_id": 2, "start_time": "2025-01-01T09:00:00Z", "duration_minutes": 30 }
    """
    data = request.get_json(silent=True) or {}

    patient_id = data.get("patient_id")
    provider_id = data.get("provider_id")
    start_time_str = data.get("start_time")

    if not all([patient_id, provider_id, start_time_str]):
        return jsonify(
            {"error": "patient_id, provider_id, and start_time are required"}
        ), 400

    try:
        from datetime import timezone

        start_time = datetime.fromisoformat(start_time_str)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
    except ValueError:
        return jsonify({"error": "Invalid start_time format. Use ISO 8601."}), 400

    appt = _engine.book_appointment(
        patient_id=patient_id,
        provider_id=provider_id,
        start_time=start_time,
        duration_minutes=data.get("duration_minutes", 30),
        appointment_type=data.get("appointment_type", "CONSULTATION"),
        reason=data.get("reason"),
    )

    if not appt:
        return jsonify(
            {
                "error": "Slot Unavailable",
                "message": "The requested time slot is already booked or overlaps with an existing appointment.",
            }
        ), 409  # Conflict

    return jsonify(
        {
            "status": "success",
            "appointment_id": appt.id,
            "scheduled_start": appt.scheduled_start.isoformat(),
            "scheduled_end": appt.scheduled_end.isoformat(),
        }
    ), 201


@bp.route("/api/provider/<int:provider_id>/today", methods=["GET"])
@login_required
def get_today_appointments(provider_id: int):
    """
    Retrieves the full daily schedule for a provider.
    """
    from datetime import timezone

    today = datetime.now(timezone.utc)
    schedule = _engine.get_provider_schedule(provider_id, today)

    return jsonify(
        {
            "provider_id": provider_id,
            "date": today.date().isoformat(),
            "total_appointments": len(schedule),
            "appointments": [
                {
                    "id": a.id,
                    "patient_id": a.patient_id,
                    "start": a.scheduled_start.isoformat(),
                    "end": a.scheduled_end.isoformat(),
                    "status": a.status,
                    "type": a.appointment_type,
                }
                for a in schedule
            ],
        }
    ), 200


@bp.route("/api/check-in/<string:appointment_id>", methods=["POST"])
@login_required
def check_in_patient(appointment_id: str):
    """
    Checks a patient in, moving them to the live waiting room queue.
    """
    appt = _engine.check_in(appointment_id)
    if not appt:
        return jsonify({"error": "Appointment not found or already checked in."}), 400

    return jsonify(
        {
            "status": "success",
            "message": "Patient checked in successfully.",
            "appointment_id": appt.id,
            "appointment_status": appt.status,
        }
    ), 200


@bp.route("/api/queue/<int:provider_id>", methods=["GET"])
@login_required
def get_live_queue(provider_id: int):
    """
    Retrieves the live waiting room queue for the provider's display.
    """
    queue = _engine.get_live_queue(provider_id)

    return jsonify(
        {
            "provider_id": provider_id,
            "waiting_count": len(queue),
            "queue": [
                {
                    "appointment_id": a.id,
                    "patient_id": a.patient_id,
                    "checked_in_at": a.updated_at.isoformat(),
                    "type": a.appointment_type,
                }
                for a in queue
            ],
        }
    ), 200


@bp.route("/api/no-show/<string:appointment_id>", methods=["POST"])
@login_required
def mark_no_show(appointment_id: str):
    """
    Marks an appointment as a no-show.
    """
    appt = _engine.mark_no_show(appointment_id)
    if not appt:
        return jsonify({"error": "Appointment not found."}), 404

    return jsonify(
        {
            "status": "success",
            "message": "Patient marked as no-show.",
            "appointment_id": appt.id,
        }
    ), 200


@bp.route("/api/queue/<int:provider_id>/rows", methods=["GET"])
@login_required
def get_live_queue_rows(provider_id: int):
    """
    Returns HTML table rows for the HTMX live queue dashboard.
    """
    queue = _engine.get_live_queue(provider_id)

    if not queue:
        return (
            '<tr><td colspan="5" class="px-6 py-8 text-center text-gray-400">'
            "Queue is empty. No patients currently waiting.</td></tr>"
        )

    status_colors = {
        "CHECKED_IN": "bg-amber-100 text-amber-800",
        "IN_PROGRESS": "bg-green-100 text-green-800",
        "SCHEDULED": "bg-blue-100 text-blue-800",
    }

    rows = []
    for idx, appt in enumerate(queue, 1):
        checked_in_time = (
            appt.updated_at.strftime("%H:%M") if appt.updated_at else "--:--"
        )
        color = status_colors.get(appt.status, "bg-gray-100 text-gray-800")
        status_label = appt.status.replace("_", " ")
        disabled = (
            'disabled class="opacity-50 cursor-not-allowed"'
            if appt.status != "SCHEDULED"
            else ""
        )

        rows.append(
            f'<tr class="border-b border-gray-100 hover:bg-gray-50 transition">'
            f'<td class="px-6 py-4 text-sm font-medium text-gray-900">{idx}</td>'
            f'<td class="px-6 py-4 text-sm text-gray-500">Patient #{appt.patient_id}</td>'
            f'<td class="px-6 py-4 text-sm text-gray-500">{checked_in_time}</td>'
            f'<td class="px-6 py-4">'
            f'<span class="px-2.5 py-1 inline-flex text-xs leading-5 font-semibold rounded-full {color}">'
            f"{status_label}</span></td>"
            f'<td class="px-6 py-4 text-right text-sm font-medium">'
            f'<button hx-post="/appointments/api/check-in/{appt.id}" '
            f'hx-target="closest tr" hx-swap="outerHTML" '
            f'class="text-blue-600 bg-blue-50 px-3 py-1 rounded-md text-xs font-medium '
            f'hover:bg-blue-100 transition" {disabled}>Start Consultation</button>'
            f"</td></tr>"
        )

    return "\n".join(rows)
