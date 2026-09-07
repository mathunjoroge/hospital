from flask import jsonify, request
from flask_login import login_required

from . import bp


@bp.route("/")
@login_required
def index():
    return "MCH & Immunization Module Active - Phase 5 MVP"


@bp.route("/api/anc-visit", methods=["POST"])
@login_required
def log_anc_visit():
    _data = request.get_json() or {}
    return jsonify({"status": "success", "message": "ANC visit stub active."}), 201


@bp.route("/api/immunize", methods=["POST"])
@login_required
def record_immunization():
    _data = request.get_json() or {}
    return jsonify({"status": "success", "message": "Immunization stub active."}), 201
