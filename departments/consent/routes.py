from flask import jsonify
from flask_login import login_required

from . import bp
from .models import Consent


@bp.route('/')
@login_required
def index():
    return "Consent Module Active - Phase 1 MVP"


@bp.route('/api/patient/<int:patient_id>', methods=['GET'])
@login_required
def get_consents(patient_id):
    consents = Consent.query.filter_by(patient_id=patient_id).all()
    return jsonify([{
        'id': c.id,
        'type': c.consent_type,
        'status': c.status,
        'granted_at': c.granted_at.isoformat() if c.granted_at else None
    } for c in consents])
