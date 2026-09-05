from flask import request, jsonify
from . import bp  # ✅ Import the API blueprint
from departments.models.records import Patient
from departments.api.auth import jwt_or_session_required
from extensions import limiter

@bp.route('/patients/search', methods=['GET'])
@jwt_or_session_required
@limiter.limit("20 per minute")
def search_patients():
    """API endpoint to search for patients using Select2."""
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify([])

    patients = Patient.query.filter(Patient.name.ilike(f"%{query}%")).limit(10).all()

    return jsonify([
        {"id": patient.patient_id, "text": f"{patient.name} ({patient.patient_id})"}
        for patient in patients
    ])

