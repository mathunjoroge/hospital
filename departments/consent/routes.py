"""
departments/consent/routes.py
──────────────────────────────
Patient consent capture and query under the Kenya Data Protection Act 2019.

All records are held in PatientConsent (table `patient_consents`), which is the
single source of truth for every consent gate in the platform — the AI/chatbot
gate (has_ai_consent), the prescribing TREATMENT gate, and these endpoints.

A second model (departments/consent/models.py :: Consent, table `consents`) used
to exist alongside it and was read by the prescribing gate while nothing in the
codebase ever wrote to it. It has been removed; see the accompanying migration.

Consent types are free-text by design (DPA processing purposes evolve), but the
values in use are: TREATMENT, ai_diagnosis, third_party_sharing,
sms_notifications, RESEARCH, TELEMEDICINE.
"""

from flask import jsonify, request
from flask_login import login_required

from departments.api.audit import log_audit_event
from departments.models.compliance import (
    grant_patient_consent,
    has_consent,
    list_patient_consents,
    revoke_patient_consent,
)
from departments.models.records import Patient
from departments.rbac import get_effective_user, roles_required
from extensions import db

from . import bp

# Roles permitted to capture or withdraw consent on a patient's behalf.
CONSENT_CAPTURE_ROLES = ("records", "admin", "doctor", "medicine", "nursing")


def _serialize(consent) -> dict:
    """Render a PatientConsent row, including its derived status."""
    if consent.is_granted and consent.revoked_at is None:
        status = "ACTIVE"
    elif consent.revoked_at is not None:
        status = "REVOKED"
    else:
        status = "NOT_GRANTED"

    return {
        "id": consent.id,
        "patient_id": consent.patient_id,
        "consent_type": consent.consent_type,
        "status": status,
        "is_granted": consent.is_granted,
        "granted_at": consent.granted_at.isoformat() if consent.granted_at else None,
        "revoked_at": consent.revoked_at.isoformat() if consent.revoked_at else None,
        "notes": consent.notes,
    }


def _resolve_patient(raw_id):
    """
    Accept either Patient.id (numeric) or Patient.patient_id ("P0001") and
    return the Patient, so callers on both sides of the platform's two-ID
    split reach the same record. Returns None when unknown.
    """
    if raw_id is None or raw_id == "":
        return None

    raw = str(raw_id).strip()
    patient = Patient.query.filter_by(patient_id=raw).first()
    if patient:
        return patient
    if raw.isdigit():
        return db.session.get(Patient, int(raw))
    return None


@bp.route("/")
@login_required
def index():
    """Module health/descriptor for the consent service."""
    return jsonify(
        {
            "module": "consent",
            "authority": "Kenya Data Protection Act 2019",
            "record_store": "patient_consents",
            "endpoints": {
                "GET /consent/api/patient/<patient_id>": "list a patient's consents",
                "GET /consent/api/check": "single consent pre-flight check",
                "POST /consent/api/grant": "record a consent grant",
                "POST /consent/api/revoke": "withdraw a consent grant",
            },
        }
    )


@bp.route("/api/patient/<patient_id>", methods=["GET"])
@login_required
def get_consents(patient_id):
    """Return every consent record held for a patient."""
    patient = _resolve_patient(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    consents = list_patient_consents(patient.patient_id)
    return jsonify(
        {
            "patient_id": patient.patient_id,
            "count": len(consents),
            "consents": [_serialize(c) for c in consents],
        }
    )


@bp.route("/api/check", methods=["GET"])
@login_required
def check_consent():
    """
    Consent pre-flight check used by the clinical workbenches.

    Query params:
        patient_id (required): Patient.id or Patient.patient_id.
        consent_type (str, required): e.g. "TREATMENT", "ai_diagnosis".

    Response: { "status": "ACTIVE" | "NOT_GRANTED" | "REVOKED" }
    """
    patient_id_raw = request.args.get("patient_id")
    consent_type = request.args.get("consent_type")

    if not patient_id_raw or not consent_type:
        return jsonify({"error": "patient_id and consent_type are required"}), 400

    patient = _resolve_patient(patient_id_raw)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    consents = [
        c for c in list_patient_consents(patient.patient_id)
        if c.consent_type == consent_type
    ]

    if not consents:
        status = "NOT_GRANTED"
    elif has_consent(patient.patient_id, consent_type):
        status = "ACTIVE"
    else:
        status = "REVOKED"

    return jsonify(
        {
            "patient_id": patient.patient_id,
            "consent_type": consent_type,
            "status": status,
        }
    )


@bp.route("/api/grant", methods=["POST"])
@login_required
@roles_required(*CONSENT_CAPTURE_ROLES)
def grant_consent():
    """
    Record a consent grant.

    JSON payload:
        patient_id (required), consent_type (required), notes (optional)
    """
    data = request.get_json(silent=True) or {}
    patient_id_raw = data.get("patient_id")
    consent_type = (data.get("consent_type") or "").strip()

    if not patient_id_raw or not consent_type:
        return jsonify({"error": "patient_id and consent_type are required"}), 400

    patient = _resolve_patient(patient_id_raw)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    consent = grant_patient_consent(
        patient_id=patient.patient_id,
        consent_type=consent_type,
        ip_address=request.remote_addr,
        notes=(data.get("notes") or "").strip() or None,
    )

    user = get_effective_user()
    log_audit_event(
        action="CONSENT_GRANTED",
        resource_type="PatientConsent",
        resource_id=str(consent.id),
        details={
            "patient_id": patient.patient_id,
            "consent_type": consent_type,
            "captured_by": getattr(user, "username", None),
        },
    )

    return jsonify({"status": "success", "consent": _serialize(consent)}), 201


@bp.route("/api/revoke", methods=["POST"])
@login_required
@roles_required(*CONSENT_CAPTURE_ROLES)
def revoke_consent():
    """
    Withdraw a consent grant. The row is retained with revoked_at set so the
    consent history remains auditable.

    JSON payload:
        patient_id (required), consent_type (required), reason (optional)
    """
    data = request.get_json(silent=True) or {}
    patient_id_raw = data.get("patient_id")
    consent_type = (data.get("consent_type") or "").strip()

    if not patient_id_raw or not consent_type:
        return jsonify({"error": "patient_id and consent_type are required"}), 400

    patient = _resolve_patient(patient_id_raw)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    consent = revoke_patient_consent(
        patient_id=patient.patient_id,
        consent_type=consent_type,
        notes=(data.get("reason") or "").strip() or None,
    )
    if consent is None:
        return jsonify({"error": "No consent record of that type for this patient"}), 404

    user = get_effective_user()
    log_audit_event(
        action="CONSENT_REVOKED",
        resource_type="PatientConsent",
        resource_id=str(consent.id),
        details={
            "patient_id": patient.patient_id,
            "consent_type": consent_type,
            "revoked_by": getattr(user, "username", None),
            "reason": data.get("reason"),
        },
    )

    return jsonify({"status": "success", "consent": _serialize(consent)}), 200
