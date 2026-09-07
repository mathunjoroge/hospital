from flask import jsonify, request
from . import bp
from .models import Referral, DischargeSummary
from flask_login import login_required
from extensions import db

@bp.route('/')
@login_required
def index():
    return "Referrals & Continuity of Care Module Active - Phase 3 MVP"

@bp.route('/api/initiate', methods=['POST'])
@login_required
def initiate_referral():
    # Stub for referral initiation
    data = request.get_json() or {}
    return jsonify({'status': 'success', 'message': 'Referral initiation stub active.'}), 201

@bp.route('/api/discharge', methods=['POST'])
@login_required
def create_discharge_summary():
    # Stub for discharge summary creation
    data = request.get_json() or {}
    return jsonify({'status': 'success', 'message': 'Discharge summary creation stub active.'}), 201
