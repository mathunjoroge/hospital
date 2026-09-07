from flask import jsonify, request
from flask_login import login_required

from . import bp


@bp.route("/")
@login_required
def index():
    return "Appointments Module Active - Phase 2 MVP"


@bp.route("/api/provider/<int:provider_id>/today", methods=["GET"])
@login_required
def get_today_appointments(provider_id: int):
    return jsonify(
        {
            "provider_id": provider_id,
            "message": "Today schedule endpoint active. Query filters to be added in Phase 2.1.",
        }
    )


@bp.route("/api/book", methods=["POST"])
@login_required
def book_appointment():
    _data = request.get_json() or {}
    return jsonify(
        {"status": "success", "message": "Appointment booking stub active."}
    ), 201
