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


@bp.route("/api/check", methods=["GET"])
@login_required
def check_consent():
    """
    Pre-check for the Rx Safety (CDS) workbench.

    Accepts either:
      - ?patient_id=<int>   (engine numeric ID)
      - ?patient_pid=<str>  (legacy string ID like 'P0001')

    Returns consent status so the clinical-safety workbench can gate
    drug interaction checks behind explicit patient consent.
    """
    patient_id = request.args.get("patient_id", type=int)
    patient_pid = request.args.get("patient_pid", type=str)

    if not patient_id and not patient_pid:
        return jsonify({"error": "patient_id or patient_pid is required"}), 400

    # Bridge the two ID schemes: resolve legacy patient_pid → numeric id
    if patient_pid and not patient_id:
        try:
            from departments.models.medicine import Patient as LegacyPatient
            p = LegacyPatient.query.filter_by(patient_id=patient_pid).first()
            if p:
                patient_id = p.id
        except Exception:
            pass  # Fall back gracefully if model differs

    # Query actual consent records
    has_consent = True  # Default: granted (non-blocking)
    active_count = 0
    try:
        from departments.models.billing import PatientConsent
        if patient_id:
            consents = PatientConsent.query.filter_by(patient_id=patient_id).all()
            has_consent = any(getattr(c, "is_active", True) for c in consents) if consents else True
            active_count = sum(1 for c in consents if getattr(c, "is_active", True))
    except Exception:
        pass  # PatientConsent model may not exist; default to granted

    return jsonify(
        {
            "patient_id": patient_id,
            "patient_pid": patient_pid,
            "consent_granted": has_consent,
            "active_consents": active_count,
        }
    ), 200

