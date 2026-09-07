from flask import jsonify, request
from flask_login import login_required

from . import bp


@bp.route("/")
@login_required
def index():
    return "Referrals & Continuity of Care Module Active - Phase 3 MVP"


@bp.route("/api/initiate", methods=["POST"])
@login_required
def initiate_referral():
    _data = request.get_json() or {}
    return jsonify(
        {"status": "success", "message": "Referral initiation stub active."}
    ), 201


@bp.route("/api/discharge", methods=["POST"])
@login_required
def create_discharge_summary():
    _data = request.get_json() or {}
    return jsonify(
        {"status": "success", "message": "Discharge summary creation stub active."}
    ), 201
