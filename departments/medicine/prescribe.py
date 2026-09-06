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

from departments.models.billing import Invoice, InvoiceLineItem
from departments.models.medicine import Medicine, PrescribedMedicine, SOAPNote
from departments.models.nursing import NursingNote
from departments.models.records import Patient

logger = logging.getLogger(__name__)

prescribe_bp = Blueprint('eprescribe', __name__, url_prefix='/medicine/prescribe')

# Standard ICD-10 Reference Catalog
ICD10_DATABASE = [
    {"code": "J06.9", "description": "Acute upper respiratory infection, unspecified", "category": "Respiratory"},
    {"code": "J18.9", "description": "Pneumonia, unspecified organism", "category": "Respiratory"},
    {"code": "J45.909", "description": "Unspecified asthma, uncomplicated", "category": "Respiratory"},
    {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "category": "Endocrine"},
    {"code": "I10", "description": "Essential (primary) hypertension", "category": "Cardiovascular"},
    {"code": "A09", "description": "Infectious gastroenteritis and colitis, unspecified", "category": "Gastrointestinal"},
    {"code": "K29.7", "description": "Gastritis, unspecified", "category": "Gastrointestinal"},
    {"code": "B34.9", "description": "Viral infection, unspecified", "category": "Infectious"},
    {"code": "M54.5", "description": "Low back pain, unspecified", "category": "Musculoskeletal"},
    {"code": "N39.0", "description": "Urinary tract infection, site unspecified", "category": "Genitourinary"},
    {"code": "R50.9", "description": "Fever, unspecified", "category": "General"},
    {"code": "R05", "description": "Cough", "category": "Respiratory"},
]

# Drug Allergy Cross-Reactivity Dictionary
ALLERGY_GROUPS = {
    "penicillin": ["amoxicillin", "ampicillin", "penicillin", "augmentin", "piperacillin", "amoxil"],
    "sulfa": ["bactrim", "cotrimoxazole", "sulfamethoxazole", "septrin"],
    "nsaid": ["ibuprofen", "diclofenac", "naproxen", "aspirin", "indomethacin", "brufen"],
    "macrolide": ["azithromycin", "erythromycin", "clarithromycin"]
}

# Drug-Drug Interaction Warning Rules (pairs -> severity, warning message)
KNOWN_INTERACTIONS = [
    ({"warfarin", "aspirin"}, "CRITICAL", "High risk of major gastrointestinal hemorrhage and severe bleeding."),
    ({"warfarin", "ibuprofen"}, "HIGH", "Increased risk of bleeding and gastric mucosal ulceration."),
    ({"lisinopril", "spironolactone"}, "HIGH", "Severe hyperkalemia risk; requires close serum potassium monitoring."),
    ({"ciprofloxacin", "antacid"}, "MEDIUM", "Chelation reduces ciprofloxacin bioavailability and therapeutic efficacy."),
    ({"metformin", "contrast"}, "HIGH", "Risk of contrast-induced acute renal failure and metformin lactic acidosis.")
]


def search_icd10(query: str) -> list[dict]:
    """Search ICD-10 reference database by code or description keyword."""
    if not query:
        return ICD10_DATABASE[:5]
    q = query.lower().strip()
    return [
        item for item in ICD10_DATABASE
        if q in item["code"].lower() or q in item["description"].lower() or q in item["category"].lower()
    ]


def check_drug_safety(patient_id: str, new_medications: list[str]) -> dict:
    """
    Check new prescription list against patient allergies and drug-drug interactions.
    Returns: {"has_warnings": bool, "alerts": list[dict]}
    """
    alerts = []
    new_meds_lower = [m.lower().strip() for m in new_medications if m]

    # 1. Fetch patient allergy history from NursingNotes or Patient record
    notes = NursingNote.query.filter_by(patient_id=patient_id).all()
    documented_allergies = []
    for n in notes:
        if n.allergies:
            documented_allergies.extend([a.strip().lower() for a in n.allergies.split(',')])

    # Check allergy cross-reactivity
    for drug in new_meds_lower:
        for group_name, drug_list in ALLERGY_GROUPS.items():
            if any(d in drug for d in drug_list):
                # Check if patient is allergic to this group
                if any(group_name in allergy or any(d in allergy for d in drug_list) for allergy in documented_allergies):
                    alerts.append({
                        "type": "ALLERGY_WARNING",
                        "severity": "CRITICAL",
                        "drug": drug,
                        "message": f"PATIENT ALLERGY ALERT: Patient is allergic to {group_name.upper()} group! Drug '{drug.title()}' is contraindicated."
                    })

    # 2. Check Drug-Drug Interactions (DDI)
    # Fetch current active prescribed meds for patient
    active_prescriptions = PrescribedMedicine.query.filter_by(patient_id=patient_id).all()
    current_meds_lower = [p.medicine.generic_name.lower().strip() for p in active_prescriptions if p.medicine and p.medicine.generic_name]
    all_meds = set(new_meds_lower + current_meds_lower)

    for drug_set, severity, msg in KNOWN_INTERACTIONS:
        matched = [d for d in drug_set if any(d in med for med in all_meds)]
        if len(matched) >= 2:
            alerts.append({
                "type": "DRUG_INTERACTION",
                "severity": severity,
                "drugs": matched,
                "message": f"DRUG INTERACTION WARNING [{severity}]: {' + '.join([m.title() for m in matched])} — {msg}"
            })

    return {
        "has_warnings": len(alerts) > 0,
        "critical_block": any(a["severity"] == "CRITICAL" for a in alerts),
        "alerts": alerts
    }


# API Routes
@prescribe_bp.route('/icd10', methods=['GET'])
def handle_icd10_search():
    """Search ICD-10 codes."""
    q = request.args.get('q', '')
    results = search_icd10(q)
    return jsonify({"results": results, "count": len(results)}), 200


@prescribe_bp.route('/validate', methods=['POST'])
def handle_safety_validate():
    """Validate drug safety (allergies and DDIs) prior to sign-off."""
    data = request.get_json() or {}
    patient_id = data.get('patient_id')
    medications = data.get('medications', [])

    if not patient_id or not medications:
        return jsonify({'error': 'patient_id and medications list required'}), 400

    safety_report = check_drug_safety(patient_id, medications)
    return jsonify(safety_report), 200


@prescribe_bp.route('/cdss/evaluate', methods=['POST'])
def handle_cdss_evaluate():
    """Comprehensive Clinical Decision Support System (CDSS) evaluation endpoint."""
    from departments.medicine.cdss import evaluate_prescription_safety

    data = request.get_json() or {}
    report = evaluate_prescription_safety(
        patient_id=data.get('patient_id'),
        drug_name=data.get('drug_name'),
        existing_meds=data.get('existing_meds', []),
        egfr=data.get('egfr'),
    )
    return jsonify(report), 200


@prescribe_bp.route('/soap', methods=['POST'])
def handle_soap_consultation():
    """Save structured SOAP consultation note."""
    data = request.get_json() or {}
    patient_id = data.get('patient_id')
    data.get('doctor_id', 1)

    subjective = data.get('subjective', '')
    objective = data.get('objective', '')
    assessment = data.get('assessment', '')
    icd10_code = data.get('icd10_code', '')
    plan = data.get('plan', '')

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    note = SOAPNote(
        patient_id=patient_id,
        situation=f"Subjective: {subjective}",
        hpi=f"Objective: {objective}",
        assessment=f"[{icd10_code}] {assessment}" if icd10_code else assessment,
        recommendation=plan
    )
    db.session.add(note)
    db.session.commit()

    return jsonify({
        "success": True,
        "soap_id": note.id,
        "icd10_code": icd10_code,
        "message": "SOAP consultation recorded successfully."
    }), 201


@prescribe_bp.route('/signoff', methods=['POST'])
def handle_prescription_signoff():
    """Sign off e-prescription with allergy/DDI validation and auto-invoice item generation."""
    import uuid
    data = request.get_json() or {}
    patient_id = data.get('patient_id')
    data.get('doctor_id', 1)
    prescriptions = data.get('prescriptions', [])  # list of dicts: {name, dosage, frequency, duration, cost}
    override_warning = data.get('override_warning', False)
    data.get('override_reason', '')

    med_names = [p.get('name') for p in prescriptions if p.get('name')]
    safety_check = check_drug_safety(patient_id, med_names)

    if safety_check['critical_block'] and not override_warning:
        return jsonify({
            "error": "Prescription blocked due to CRITICAL safety warning (Allergy/Severe Interaction).",
            "safety_alerts": safety_check['alerts'],
            "requires_override": True
        }), 400

    created_meds = []
    total_pharmacy_charge = 0.0
    rx_uuid = str(uuid.uuid4())

    for item in prescriptions:
        name = item.get('name')
        dosage = item.get('dosage', '500mg')
        strength = item.get('strength', 'Standard')
        frequency = item.get('frequency', 'TDS')
        num_days = int(item.get('num_days', 7))
        cost = float(item.get('cost', 150.0))

        # Find or create Medicine master entry
        med_obj = Medicine.query.filter_by(generic_name=name).first()
        if not med_obj:
            med_obj = Medicine(generic_name=name, brand_name=name, dosage=dosage)
            db.session.add(med_obj)
            db.session.flush()

        rx_item = PrescribedMedicine(
            patient_id=patient_id,
            medicine_id=med_obj.id,
            dosage=dosage,
            strength=strength,
            frequency=frequency,
            prescription_id=rx_uuid,
            num_days=num_days
        )
        db.session.add(rx_item)
        created_meds.append(name)
        total_pharmacy_charge += cost

    # Automatically add to patient's active draft Invoice or create new Invoice
    inv = Invoice.query.filter_by(patient_id=patient_id, status='DRAFT').first()
    if not inv:
        inv = Invoice(
            patient_id=patient_id,
            invoice_number=Invoice.generate_invoice_number(),
            status='DRAFT'
        )
        db.session.add(inv)
        db.session.flush()

    line_item = InvoiceLineItem(
        invoice_id=inv.id,
        description=f"E-Prescription Medications ({', '.join(created_meds)})",
        amount=total_pharmacy_charge,
        category='PHARMACY'
    )
    db.session.add(line_item)
    inv.recalculate()
    db.session.commit()

    return jsonify({
        "success": True,
        "prescribed_count": len(created_meds),
        "prescription_id": rx_uuid,
        "invoice_number": inv.invoice_number,
        "total_charge": total_pharmacy_charge,
        "warnings_logged": safety_check['alerts']
    }), 201
