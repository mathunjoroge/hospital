"""
departments/nursing/bcma.py
────────────────────────────
P8-05 — Bar-Code Medication Administration (BCMA) scan-to-MAR workflow.

Clinician scans the patient wristband barcode + drug barcode before marking
a dose as given.  The system cross-checks the scanned drug against the active
MAR (PrescribedMedicine rows) for that patient and blocks administration on
any mismatch or duplicate-dose-within-safe-window condition.

Hardware assumption: any keyboard-wedge USB or mobile-PWA barcode scanner that
inputs the barcode string as plain text (Code-39 / EAN-13 / GS1-128).
The patient barcode IS the patients.patient_id string (no special encoding).
The drug barcode IS the medicines.id cast to string (can be replaced with NDC
once the formulary carries NDC codes — add Medicine.barcode_code column then).

Endpoints
─────────
POST /nursing/bcma/verify          — Check barcodes; does NOT record the dose.
POST /nursing/bcma/administer      — Verify + record MedicationAdmin row.
GET  /nursing/bcma/mar/<patient_id>— Return active MAR for a patient.
"""

import logging
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request

from departments.api.auth import jwt_or_session_required
from departments.models.medicine import Medicine, PrescribedMedicine
from departments.models.nursing import MedicationAdmin
from departments.rbac import roles_required
from extensions import db

logger = logging.getLogger(__name__)

bcma_bp = Blueprint("bcma", __name__, url_prefix="/nursing/bcma")

# Minimum minutes between two administrations of the same drug for the same
# patient before the duplicate-dose guard fires.
SAFE_DOSE_INTERVAL_MINUTES = 60


def _find_patient_by_barcode(barcode: str):
    """Resolve a wristband barcode to a patient_id string.

    The wristband barcode encodes patients.patient_id directly, so this is a
    simple existence check via PrescribedMedicine (avoids importing the full
    Patient model which lives in records).
    """
    from departments.models.records import Patient  # local import — avoids circular
    return Patient.query.filter_by(patient_id=barcode).first()


def _find_medicine_by_barcode(barcode: str):
    """Resolve a drug barcode to a Medicine row.

    Current formulary uses Medicine.id as the barcode value.  When NDC codes
    are added (Medicine.barcode_code column), update this function.
    """
    try:
        medicine_id = int(barcode)
        return Medicine.query.get(medicine_id)
    except (ValueError, TypeError):
        return Medicine.query.filter_by(brand_name=barcode).first()


def _get_active_prescription(patient_id: str, medicine_id: int):
    """Return the most recent active PrescribedMedicine order (status=0)."""
    return (
        PrescribedMedicine.query
        .filter_by(patient_id=patient_id, medicine_id=medicine_id, status=0)
        .order_by(PrescribedMedicine.id.desc())
        .first()
    )


def _check_duplicate_dose(patient_id: str, medicine_id: int) -> bool:
    """Return True if a dose was already given within SAFE_DOSE_INTERVAL_MINUTES."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SAFE_DOSE_INTERVAL_MINUTES)
    # MedicationAdmin.time_administered may be tz-naive — compare naively
    cutoff_naive = cutoff.replace(tzinfo=None)
    recent = (
        MedicationAdmin.query
        .filter(
            MedicationAdmin.patient_id == patient_id,
            MedicationAdmin.medication == str(medicine_id),
            MedicationAdmin.time_administered >= cutoff_naive,
        )
        .first()
    )
    return recent is not None


def _log_audit(event_type: str, detail: str, patient_id: str):
    """Append a BCMA safety event to AuditLog (best-effort; never raises)."""
    try:
        from departments.api.audit import log_audit_event  # noqa: F401
        log_audit_event(event_type, detail, patient_id=patient_id)
    except Exception:  # noqa: BLE001
        logger.warning("BCMA audit log failed for event %s / patient %s", event_type, patient_id)


# ──────────────────────────────────────────────────────────────────────────────
# Verify endpoint — does NOT write any record
# ──────────────────────────────────────────────────────────────────────────────

@bcma_bp.route("/verify", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine", "clinical")
def bcma_verify():
    """
    Verify patient wristband + drug barcode against the active MAR.

    Body (JSON):
        patient_barcode       : str  — wristband barcode (== patient_id)
        drug_barcode          : str  — drug barcode (== Medicine.id or brand name)
        prescribed_medicine_id: int  — (optional) explicit MAR order ID to check against
    """
    data = request.get_json(silent=True) or {}
    patient_barcode = data.get("patient_barcode", "").strip()
    drug_barcode = data.get("drug_barcode", "").strip()

    if not patient_barcode or not drug_barcode:
        return jsonify({"error": "patient_barcode and drug_barcode are required"}), 400

    # 1. Resolve patient
    patient = _find_patient_by_barcode(patient_barcode)
    if not patient:
        _log_audit("BCMA_PATIENT_MISMATCH", f"Unknown wristband: {patient_barcode}", patient_barcode)
        return jsonify({"match": False, "error": "Patient not found for this wristband barcode"}), 409

    # 2. Resolve drug
    medicine = _find_medicine_by_barcode(drug_barcode)
    if not medicine:
        _log_audit("BCMA_DRUG_MISMATCH", f"Unknown drug barcode: {drug_barcode}", patient.patient_id)
        return jsonify({"match": False, "error": "Drug not found for this barcode"}), 409

    # 3. Cross-check against active MAR
    prescription = _get_active_prescription(patient.patient_id, medicine.id)
    if not prescription:
        _log_audit(
            "BCMA_DRUG_MISMATCH",
            f"Drug {medicine.generic_name} not on active MAR for patient {patient.patient_id}",
            patient.patient_id,
        )
        return jsonify({
            "match": False,
            "error": f"{medicine.generic_name} is not on the active medication order for this patient",
        }), 409

    # 4. Duplicate-dose check
    is_duplicate = _check_duplicate_dose(patient.patient_id, medicine.id)

    warnings = []
    if is_duplicate:
        warnings.append(
            f"A dose of {medicine.generic_name} was already recorded within the last "
            f"{SAFE_DOSE_INTERVAL_MINUTES} minutes."
        )

    return jsonify({
        "match": True,
        "patient_id": patient.patient_id,
        "patient_name": patient.name,
        "medicine_id": medicine.id,
        "drug_name": f"{medicine.generic_name} ({medicine.brand_name})",
        "dosage": prescription.dosage,
        "strength": prescription.strength,
        "frequency": prescription.frequency,
        "prescribed_medicine_id": prescription.id,
        "duplicate_dose_warning": is_duplicate,
        "warnings": warnings,
    }), 200


# ──────────────────────────────────────────────────────────────────────────────
# Administer endpoint — records the dose
# ──────────────────────────────────────────────────────────────────────────────

@bcma_bp.route("/administer", methods=["POST"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine", "clinical")
def bcma_administer():
    """
    Verify and record a medication administration event.

    Body (JSON):
        patient_barcode       : str  — wristband barcode
        drug_barcode          : str  — drug barcode
        nurse_id              : int  — administering nurse/clinician user ID
        override_reason       : str  — (optional) required when overriding a mismatch
    """
    data = request.get_json(silent=True) or {}
    patient_barcode = data.get("patient_barcode", "").strip()
    drug_barcode = data.get("drug_barcode", "").strip()
    nurse_id = data.get("nurse_id")
    override_reason = (data.get("override_reason") or "").strip()

    if not patient_barcode or not drug_barcode or not nurse_id:
        return jsonify({"error": "patient_barcode, drug_barcode and nurse_id are required"}), 400

    # ── Re-run the full verification ──────────────────────────────────────────
    patient = _find_patient_by_barcode(patient_barcode)
    if not patient:
        _log_audit("BCMA_PATIENT_MISMATCH", f"Administer: unknown wristband {patient_barcode}", patient_barcode)
        return jsonify({"error": "Patient not found for this wristband barcode"}), 409

    medicine = _find_medicine_by_barcode(drug_barcode)
    if not medicine:
        _log_audit("BCMA_DRUG_MISMATCH", f"Administer: unknown drug barcode {drug_barcode}", patient.patient_id)
        return jsonify({"error": "Drug not found for this barcode"}), 409

    prescription = _get_active_prescription(patient.patient_id, medicine.id)
    if not prescription:
        if not override_reason or len(override_reason) < 20:
            return jsonify({
                "error": (
                    f"{medicine.generic_name} is not on the active MAR. "
                    "Supply override_reason (≥ 20 characters) to proceed."
                )
            }), 409
        # Override allowed — log it
        _log_audit(
            "BCMA_DRUG_MISMATCH_OVERRIDE",
            f"Override: {override_reason}",
            patient.patient_id,
        )
        prescription_id = None
    else:
        prescription_id = prescription.id

    # ── Duplicate-dose check ──────────────────────────────────────────────────
    if _check_duplicate_dose(patient.patient_id, medicine.id):
        if not override_reason or len(override_reason) < 20:
            return jsonify({
                "error": (
                    f"Duplicate dose: {medicine.generic_name} was already given within "
                    f"the last {SAFE_DOSE_INTERVAL_MINUTES} minutes. "
                    "Supply override_reason (≥ 20 characters) to proceed."
                )
            }), 409
        _log_audit("BCMA_DUPLICATE_DOSE_OVERRIDE", f"Override: {override_reason}", patient.patient_id)

    # ── Record administration ─────────────────────────────────────────────────
    admin_record = MedicationAdmin(
        patient_id=patient.patient_id,
        medication=str(medicine.id),          # stored as medicine_id string
        dosage=prescription.dosage if prescription else data.get("dosage", ""),
        recorded_by=nurse_id,
        time_administered=datetime.now(timezone.utc).replace(tzinfo=None),
        scan_verified=True,
        barcode_patient_id=patient_barcode,
        barcode_drug_code=drug_barcode,
        prescribed_medicine_id=prescription_id,
        override_reason=override_reason or None,
    )
    db.session.add(admin_record)
    db.session.commit()

    _log_audit(
        "BCMA_DOSE_ADMINISTERED",
        f"Drug {medicine.generic_name} administered to {patient.patient_id} "
        f"by nurse {nurse_id}. scan_verified=True override={bool(override_reason)}",
        patient.patient_id,
    )

    return jsonify({
        "success": True,
        "record_id": admin_record.id,
        "patient_id": patient.patient_id,
        "drug_name": f"{medicine.generic_name} ({medicine.brand_name})",
        "scan_verified": True,
        "override_applied": bool(override_reason),
    }), 201


# ──────────────────────────────────────────────────────────────────────────────
# MAR view — active medication orders for a patient
# ──────────────────────────────────────────────────────────────────────────────

@bcma_bp.route("/mar/<string:patient_id>", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "nursing", "medicine", "clinical")
def get_patient_mar(patient_id: str):
    """Return the active Medication Administration Record for a patient."""
    orders = (
        PrescribedMedicine.query
        .filter_by(patient_id=patient_id, status=0)
        .order_by(PrescribedMedicine.id.desc())
        .all()
    )

    mar_entries = []
    for order in orders:
        # Last administration
        last_admin = (
            MedicationAdmin.query
            .filter(
                MedicationAdmin.patient_id == patient_id,
                MedicationAdmin.medication == str(order.medicine_id),
            )
            .order_by(MedicationAdmin.time_administered.desc())
            .first()
        )
        mar_entries.append({
            "prescribed_medicine_id": order.id,
            "medicine_id": order.medicine_id,
            "drug_name": (
                f"{order.medicine.generic_name} ({order.medicine.brand_name})"
                if order.medicine else str(order.medicine_id)
            ),
            "dosage": order.dosage,
            "strength": order.strength,
            "frequency": order.frequency,
            "num_days": order.num_days,
            "last_administered": (
                last_admin.time_administered.isoformat() if last_admin else None
            ),
            "next_due_after_minutes": SAFE_DOSE_INTERVAL_MINUTES,
        })

    return jsonify({
        "patient_id": patient_id,
        "active_orders": len(mar_entries),
        "mar": mar_entries,
    }), 200
