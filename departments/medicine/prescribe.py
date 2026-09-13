"""
departments/medicine/prescribe.py
─────────────────────────────────
Task 3.2 — Clinical Consultation & E-Prescribing System

Features:
  - ICD-10 / ICD-11 Diagnosis search & validation engine
  - Drug-Drug Interaction (DDI) & Patient Allergy safety checker
  - Structured SOAP (Subjective, Objective, Assessment, Plan) consultation forms
  - E-Prescribing sign-off with automatic pharmacy billing line item generation
"""

import logging

from flask import Blueprint, jsonify, request

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.medicine.cdss import evaluate_prescription_safety
from departments.models.billing import InvoiceLineItem
from departments.models.medicine import Medicine, PrescribedMedicine, SOAPNote
from departments.models.records import Patient
from departments.shared.drug_safety_rules import (  # noqa: F401
    ALLERGY_GROUPS,
    KNOWN_INTERACTIONS,
)
from departments.shared.encounter_utils import active_encounter

logger = logging.getLogger(__name__)

prescribe_bp = Blueprint("eprescribe", __name__, url_prefix="/medicine/prescribe")




# ---------------------------------------------------------------------------
# Terminology helpers
# ---------------------------------------------------------------------------

# Threshold: once the DB has at least this many ICD-10 codes we consider it
# fully populated and skip the WHO live-search fallback.
_ICD10_POPULATED_THRESHOLD = 100


def _get_icd10_database():
    """Fetch ICD-10 codes from DB, falling back to minimal hardcoded list for tests."""
    try:
        from departments.models.terminology import ICD10Code
        codes = ICD10Code.query.limit(100).all()
        if codes:
            return [{"code": c.code, "description": c.description, "category": c.chapter or "General"} for c in codes]
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to fetch ICD-10 codes from database: %s", e)
    # Minimal fallback for tests/empty DB (preserves existing test behaviour)
    return [
        {"code": "J00", "description": "Acute nasopharyngitis [common cold]", "category": "Respiratory"},
        {"code": "R50.9", "description": "Fever, unspecified", "category": "General"},
        {"code": "I10", "description": "Essential (primary) hypertension", "category": "Cardiovascular"},
        {"code": "J06.9", "description": "Acute upper respiratory infection, unspecified", "category": "Respiratory"},
    ]


def _icd10_db_count() -> int:
    """Return the number of ICD-10 codes currently in the local DB (0 on error)."""
    try:
        from departments.models.terminology import ICD10Code
        return ICD10Code.query.count()
    except Exception:  # noqa: BLE001
        return 0


def _get_snomed_database():
    """Fetch SNOMED codes from DB, falling back to minimal hardcoded list for tests."""
    try:
        from departments.models.terminology import SnomedCode
        codes = SnomedCode.query.limit(100).all()
        if codes:
            return [{"code": c.code, "description": c.description} for c in codes]
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to fetch SNOMED codes from database: %s", e)
    # Minimal fallback for tests/empty DB
    return [
        {"code": "404684003", "description": "Clinical finding"},
        {"code": "22298006", "description": "Myocardial infarction"},
        {"code": "38341003", "description": "Hypertensive disorder"},
    ]


def _get_loinc_database():
    """Fetch LOINC codes from DB, falling back to minimal hardcoded list for tests."""
    try:
        from departments.models.terminology import LoincCode
        codes = LoincCode.query.limit(100).all()
        if codes:
            return [{"code": c.code, "description": c.description} for c in codes]
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to fetch LOINC codes from database: %s", e)
    # Minimal fallback for tests/empty DB
    return [
        {"code": "8302-2", "description": "Body temperature"},
        {"code": "8867-4", "description": "Heart rate"},
        {"code": "8480-6", "description": "Systolic blood pressure"},
        {"code": "8462-4", "description": "Diastolic blood pressure"},
    ]


def search_icd10(query: str) -> list[dict]:
    """
    Search ICD-10 codes by code or description keyword.

    Strategy (in priority order):
      1. If local DB has ≥100 codes (i.e. WHO import ran), query the DB —
         fast, no network, works offline.  Results capped at 20.
      2. If DB is sparse/empty AND a query string was provided, try the WHO
         live search API (requires WHO credentials in .env).
      3. Final fallback: scan the minimal hardcoded list (for tests / no DB).
    """
    q = (query or "").lower().strip()

    # -- Path 1: populated local DB (normal production path) --
    db_count = _icd10_db_count()
    if db_count >= _ICD10_POPULATED_THRESHOLD:
        try:
            from departments.models.terminology import ICD10Code
            if not q:
                rows = ICD10Code.query.limit(20).all()
            else:
                rows = (
                    ICD10Code.query
                    .filter(
                        db.or_(
                            ICD10Code.code.ilike(f"%{q}%"),
                            ICD10Code.description.ilike(f"%{q}%"),
                        )
                    )
                    .limit(20)
                    .all()
                )
            return [
                {"code": r.code, "description": r.description, "category": r.chapter or "General"}
                for r in rows
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning("ICD-10 DB search failed, falling through: %s", e)

    # -- Path 2: WHO live API fallback (DB is sparse) --
    if q:
        try:
            from departments.medicine.who_icd_client import search_icd10_live
            live_results = search_icd10_live(q)
            if live_results:
                return live_results
        except Exception as e:  # noqa: BLE001
            logger.warning("WHO live ICD-10 search unavailable: %s", e)

    # -- Path 3: minimal hardcoded fallback (tests / no connectivity) --
    if not q:
        return _get_icd10_database()[:5]
    return [
        item
        for item in _get_icd10_database()
        if q in item["code"].lower()
        or q in item["description"].lower()
        or q in item.get("category", "").lower()
    ]


def search_snomed(query: str) -> list[dict]:
    """
    Search SNOMED CT codes by code or description keyword.

    Strategy:
      1. Local DB search (fast, offline, capped at 20).
      2. UMLS REST API live search fallback if query provided and DB has few results.
      3. Minimal hardcoded test fallback.
    """
    q = (query or "").lower().strip()

    # -- Path 1: Local DB search --
    try:
        from departments.models.terminology import SnomedCode
        if not q:
            rows = SnomedCode.query.limit(20).all()
        else:
            rows = (
                SnomedCode.query
                .filter(
                    db.or_(
                        SnomedCode.code.ilike(f"%{q}%"),
                        SnomedCode.description.ilike(f"%{q}%"),
                    )
                )
                .limit(20)
                .all()
            )
        if rows:
            return [{"code": r.code, "description": r.description} for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.warning("SNOMED DB search error: %s", e)

    # -- Path 2: UMLS live REST API fallback --
    if q:
        try:
            from departments.medicine.umls_client import search_snomed_live
            live_results = search_snomed_live(q, max_results=20)
            if live_results:
                return live_results
        except Exception as e:  # noqa: BLE001
            logger.warning("UMLS live SNOMED search failed: %s", e)

    # -- Path 3: Minimal hardcoded fallback for tests --
    if not q:
        return _get_snomed_database()[:5]
    return [
        item
        for item in _get_snomed_database()
        if q in item["code"].lower()
        or q in item["description"].lower()
    ]


def search_loinc(query: str) -> list[dict]:
    """
    Search LOINC codes by code or description keyword.

    Strategy:
      1. Local DB search (fast, offline, capped at 20).
      2. UMLS REST API live search fallback if query provided and DB has few results.
      3. Minimal hardcoded test fallback.
    """
    q = (query or "").lower().strip()

    # -- Path 1: Local DB search --
    try:
        from departments.models.terminology import LoincCode
        if not q:
            rows = LoincCode.query.limit(20).all()
        else:
            rows = (
                LoincCode.query
                .filter(
                    db.or_(
                        LoincCode.code.ilike(f"%{q}%"),
                        LoincCode.description.ilike(f"%{q}%"),
                    )
                )
                .limit(20)
                .all()
            )
        if rows:
            return [{"code": r.code, "description": r.description} for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.warning("LOINC DB search error: %s", e)

    # -- Path 2: UMLS live REST API fallback --
    if q:
        try:
            from departments.medicine.umls_client import search_loinc_live
            live_results = search_loinc_live(q, max_results=20)
            if live_results:
                return live_results
        except Exception as e:  # noqa: BLE001
            logger.warning("UMLS live LOINC search failed: %s", e)

    # -- Path 3: Minimal hardcoded fallback for tests --
    if not q:
        return _get_loinc_database()[:5]
    return [
        item
        for item in _get_loinc_database()
        if q in item["code"].lower()
        or q in item["description"].lower()
    ]



def check_drug_safety(patient_id: str, new_medications: list[str]) -> dict:
    """
    Check new prescription list against patient allergies and drug-drug interactions.

    Returns: {"critical_block": bool, "alerts": list[dict]}

    This is a thin wrapper that delegates to ClinicalSafetyEngine.check_by_names().
    The canonical safety logic now lives in departments/clinical_safety/engine.py.
    """
    from departments.clinical_safety.engine import ClinicalSafetyEngine

    engine = ClinicalSafetyEngine()
    return engine.check_by_names(patient_id, new_medications)


@prescribe_bp.route("/icd10", methods=["GET"])
def handle_icd10_search():
    """Search ICD-10 codes."""
    q = request.args.get("q", "")
    results = search_icd10(q)
    return jsonify({"results": results, "count": len(results)}), 200


@prescribe_bp.route("/validate", methods=["POST"])
def handle_safety_validate():
    """Validate drug safety (allergies and DDIs) prior to sign-off."""
    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    medications = data.get("medications", [])
    if not patient_id or not medications:
        return jsonify({"error": "patient_id and medications list required"}), 400

    safety_report = check_drug_safety(patient_id, medications)
    return jsonify(safety_report), 200


@prescribe_bp.route("/cdss/evaluate", methods=["POST"])
def handle_cdss_evaluate():
    """Comprehensive Clinical Decision Support System (CDSS) evaluation endpoint."""

    data = request.get_json() or {}
    report = evaluate_prescription_safety(
        patient_id=data.get("patient_id"),
        drug_name=data.get("drug_name"),
        existing_meds=data.get("existing_meds", []),
        egfr=data.get("egfr"),
    )
    return jsonify(report), 200


@prescribe_bp.route("/soap", methods=["POST"])
def handle_soap_consultation():
    """Save structured SOAP consultation note."""
    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    data.get("doctor_id", 1)

    subjective = data.get("subjective", "")
    objective = data.get("objective", "")
    assessment = data.get("assessment", "")
    icd10_code = data.get("icd10_code", "")
    plan = data.get("plan", "")

    patient = Patient.query.filter(
        db.or_(
            Patient.patient_id.ilike(f"%{patient_id}%"),
            Patient.name.ilike(f"%{patient_id}%"),
        )
    ).first()
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    note = SOAPNote(
        patient_id=patient_id,
        situation=f"Subjective: {subjective}",
        hpi=f"Objective: {objective}",
        assessment=f"[{icd10_code}] {assessment}" if icd10_code else assessment,
        recommendation=plan,
    )
    db.session.add(note)
    db.session.commit()

    return jsonify(
        {
            "success": True,
            "soap_id": note.id,
            "icd10_code": icd10_code,
            "message": "SOAP consultation recorded successfully.",
        }
    ), 201


@prescribe_bp.route("/signoff", methods=["POST"])
def handle_prescription_signoff():
    """Sign off e-prescription with allergy/DDI validation and auto-invoice item generation."""
    import uuid

    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    data.get("doctor_id", 1)
    prescriptions = data.get(
        "prescriptions", []
    )  # list of dicts: {name, dosage, frequency, duration, cost}
    override_warning = data.get("override_warning", False)
    data.get("override_reason", "")

    med_names = [p.get("name") for p in prescriptions if p.get("name")]
    safety_check = check_drug_safety(patient_id, med_names)

    if safety_check["critical_block"] and not override_warning:
        return jsonify(
            {
                "error": "Prescription blocked due to CRITICAL safety warning (Allergy/Severe Interaction).",
                "safety_alerts": safety_check["alerts"],
                "requires_override": True,
            }
        ), 400

    created_meds = []
    total_pharmacy_charge = 0.0
    rx_uuid = str(uuid.uuid4())
    encounter = active_encounter(patient_id)

    for item in prescriptions:
        name = item.get("name")
        dosage = item.get("dosage", "500mg")
        strength = item.get("strength", "Standard")
        frequency = item.get("frequency", "TDS")
        num_days = int(item.get("num_days", 7))
        cost = float(item.get("cost", 150.0))

        # Find or create Medicine master entry
        med_obj = Medicine.query.filter_by(generic_name=name).first()
        if not med_obj:
            med_obj = Medicine(generic_name=name, brand_name=name, dosage=dosage)
            db.session.add(med_obj)
            db.session.flush()

        rx_item = PrescribedMedicine(
            patient_id=patient_id,
            encounter_id=encounter.id if encounter else None,
            medicine_id=med_obj.id,
            dosage=dosage,
            strength=strength,
            frequency=frequency,
            prescription_id=rx_uuid,
            num_days=num_days,
        )
        db.session.add(rx_item)
        created_meds.append(name)
        total_pharmacy_charge += cost

    # Automatically add to patient's active draft Invoice or create new Invoice
    from departments.billing.sync import get_or_create_open_invoice

    inv = get_or_create_open_invoice(patient_id)

    line_item = InvoiceLineItem(
        invoice_id=inv.id,
        description=f"E-Prescription Medications ({', '.join(created_meds)})",
        amount=total_pharmacy_charge,
        category="PHARMACY",
    )
    db.session.add(line_item)
    inv.recalculate()
    db.session.commit()

    return jsonify(
        {
            "success": True,
            "prescribed_count": len(created_meds),
            "prescription_id": rx_uuid,
            "invoice_number": inv.invoice_number,
            "total_charge": total_pharmacy_charge,
            "warnings_logged": safety_check["alerts"],
        }
    ), 201
