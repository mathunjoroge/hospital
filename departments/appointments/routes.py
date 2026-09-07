from flask import jsonify, request
from . import bp
from .models import Appointment
from flask_login import login_required
from extensions import db

@bp.route('/')
@login_required
def index():
    return "Appointments Module Active - Phase 2 MVP"

@bp.route('/api/provider/<int:provider_id>/today', methods=['GET'])
@login_required
def get_today_appointments(provider_id):
    # Basic stub for fetching today's schedule
    return jsonify({
        'provider_id': provider_id,
        'message': 'Today schedule endpoint active. Query filters to be added in Phase 2.1.'
    })

@bp.route('/api/book', methods=['POST'])
@login_required
def book_appointment():
    # Stub for booking logic
    data = request.get_json() or {}
    return jsonify({'status': 'success', 'message': 'Appointment booking stub active.'}), 201
