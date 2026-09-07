from flask import jsonify, request
from flask_login import login_required

from . import bp


@bp.route("/")
@login_required
def index():
    return "Consent Module Active - Phase 1 MVP"


@bp.route("/api/patient/<int:patient_id>", methods=["GET"])
@login_required
def get_consents(patient_id: int):
    return jsonify(
        {
            "patient_id": patient_id,
            "message": "Consent retrieval endpoint active. Full query logic in Phase 1.1.",
        }
    )


@bp.route("/api/grant", methods=["POST"])
@login_required
def grant_consent():
    _data = request.get_json() or {}
    return jsonify({"status": "success", "message": "Consent grant stub active."}), 201


@bp.route("/api/revoke", methods=["POST"])
@login_required
def revoke_consent():
    _data = request.get_json() or {}
    return jsonify(
        {"status": "success", "message": "Consent revocation stub active."}
    ), 200
