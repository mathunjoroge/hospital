from flask import jsonify, request
from flask_login import login_required

from . import bp


@bp.route("/")
@login_required
def index():
    return "Clinical Safety & CDS Module Active - Phase 4 MVP"


@bp.route("/api/check", methods=["POST"])
@login_required
def check_safety():
    """
    Stub for the Clinical Decision Support (CDS) engine.
    Will eventually cross-reference requested drugs against PatientAllergy and interactions.
    """
    _data = request.get_json() or {}
    return jsonify(
        {"status": "success", "message": "Safety check engine stub active."}
    ), 200


@bp.route("/api/override", methods=["POST"])
@login_required
def log_override():
    """
    Stub for logging when a doctor intentionally overrides a critical safety alert.
    """
    _data = request.get_json() or {}
    return jsonify({"status": "success", "message": "Override audit stub active."}), 201
