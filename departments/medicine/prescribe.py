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
from departments.models.records import Patient, PatientAllergy

logger = logging.getLogger(__name__)

prescribe_bp = Blueprint("eprescribe", __name__, url_prefix="/medicine/prescribe")


# Standard ICD-10 Reference Catalog (STOPGAP: Comprehensive Common Clinical Catalog)
# Note: Full ICD-10-CM offline ingestion requires WHO ICD API client registration credentials.
ICD10_DATABASE = [
    # Respiratory & ENT
    {
        "code": "J06.9",
        "description": "Acute upper respiratory infection, unspecified",
        "category": "Respiratory",
    },
    {
        "code": "J18.9",
        "description": "Pneumonia, unspecified organism",
        "category": "Respiratory",
    },
    {
        "code": "J45.909",
        "description": "Unspecified asthma, uncomplicated",
        "category": "Respiratory",
    },
    {
        "code": "J44.9",
        "description": "Chronic obstructive pulmonary disease, unspecified",
        "category": "Respiratory",
    },
    {
        "code": "J01.90",
        "description": "Acute sinusitis, unspecified",
        "category": "Respiratory",
    },
    {
        "code": "J02.9",
        "description": "Acute pharyngitis, unspecified",
        "category": "Respiratory",
    },
    {"code": "R05", "description": "Cough", "category": "Respiratory"},
    # Endocrine & Metabolic
    {
        "code": "E11.9",
        "description": "Type 2 diabetes mellitus without complications",
        "category": "Endocrine",
    },
    {
        "code": "E10.9",
        "description": "Type 1 diabetes mellitus without complications",
        "category": "Endocrine",
    },
    {
        "code": "E03.9",
        "description": "Hypothyroidism, unspecified",
        "category": "Endocrine",
    },
    {"code": "E66.9", "description": "Obesity, unspecified", "category": "Endocrine"},
    {
        "code": "E87.1",
        "description": "Hypo-osmolality and hyponatremia",
        "category": "Endocrine",
    },
    # Cardiovascular
    {
        "code": "I10",
        "description": "Essential (primary) hypertension",
        "category": "Cardiovascular",
    },
    {
        "code": "I50.9",
        "description": "Heart failure, unspecified",
        "category": "Cardiovascular",
    },
    {
        "code": "I25.10",
        "description": "Atherosclerotic heart disease of native coronary artery",
        "category": "Cardiovascular",
    },
    {
        "code": "I48.91",
        "description": "Unspecified atrial fibrillation",
        "category": "Cardiovascular",
    },
    {
        "code": "I21.9",
        "description": "Acute myocardial infarction, unspecified",
        "category": "Cardiovascular",
    },
    # Gastrointestinal & Hepatic
    {
        "code": "A09",
        "description": "Infectious gastroenteritis and colitis, unspecified",
        "category": "Gastrointestinal",
    },
    {
        "code": "K29.7",
        "description": "Gastritis, unspecified",
        "category": "Gastrointestinal",
    },
    {
        "code": "K21.9",
        "description": "Gastro-esophageal reflux disease without esophagitis",
        "category": "Gastrointestinal",
    },
    {
        "code": "K80.20",
        "description": "Calculus of gallbladder without cholecystitis without obstruction",
        "category": "Gastrointestinal",
    },
    {
        "code": "K35.80",
        "description": "Unspecified acute appendicitis",
        "category": "Gastrointestinal",
    },
    # Infectious Diseases & Malaria
    {
        "code": "B34.9",
        "description": "Viral infection, unspecified",
        "category": "Infectious",
    },
    {"code": "B54", "description": "Unspecified malaria", "category": "Infectious"},
    {
        "code": "B20",
        "description": "Human immunodeficiency virus [HIV] disease",
        "category": "Infectious",
    },
    {"code": "A15.0", "description": "Tuberculosis of lung", "category": "Infectious"},
    {
        "code": "A01.00",
        "description": "Typhoid fever, unspecified",
        "category": "Infectious",
    },
    # Musculoskeletal
    {
        "code": "M54.5",
        "description": "Low back pain, unspecified",
        "category": "Musculoskeletal",
    },
    {
        "code": "M17.9",
        "description": "Osteoarthritis of knee, unspecified",
        "category": "Musculoskeletal",
    },
    {"code": "M79.7", "description": "Fibromyalgia", "category": "Musculoskeletal"},
    # Nephrology & Genitourinary
    {
        "code": "N39.0",
        "description": "Urinary tract infection, site unspecified",
        "category": "Genitourinary",
    },
    {
        "code": "N18.9",
        "description": "Chronic kidney disease, unspecified",
        "category": "Genitourinary",
    },
    {"code": "N20.1", "description": "Calculus of ureter", "category": "Genitourinary"},
    # Oncology & Hematology
    {
        "code": "C50.919",
        "description": "Malignant neoplasm of unspecified site of unspecified female breast",
        "category": "Oncology",
    },
    {
        "code": "C61",
        "description": "Malignant neoplasm of prostate",
        "category": "Oncology",
    },
    {
        "code": "C34.90",
        "description": "Malignant neoplasm of unspecified part of unspecified bronchus or lung",
        "category": "Oncology",
    },
    {
        "code": "D50.9",
        "description": "Iron deficiency anemia, unspecified",
        "category": "Hematology",
    },
    {
        "code": "D57.1",
        "description": "Sickle-cell disease without crisis",
        "category": "Hematology",
    },
    # Obstetrics & Gynecology
    {
        "code": "O80",
        "description": "Encounter for full-term uncomplicated delivery",
        "category": "Obstetrics",
    },
    {
        "code": "O14.90",
        "description": "Unspecified pre-eclampsia",
        "category": "Obstetrics",
    },
    {
        "code": "N94.6",
        "description": "Dysmenorrhea, unspecified",
        "category": "Gynecology",
    },
    # Neurology & Psychiatry
    {
        "code": "G43.909",
        "description": "Migraine, unspecified, not intractable",
        "category": "Neurology",
    },
    {
        "code": "G40.909",
        "description": "Epilepsy, unspecified, not intractable",
        "category": "Neurology",
    },
    {
        "code": "F32.9",
        "description": "Major depressive disorder, single episode, unspecified",
        "category": "Psychiatry",
    },
    {
        "code": "F41.1",
        "description": "Generalized anxiety disorder",
        "category": "Psychiatry",
    },
    # General & Symptoms
    {"code": "R50.9", "description": "Fever, unspecified", "category": "General"},
    {"code": "R51.9", "description": "Headache, unspecified", "category": "General"},
    {"code": "R53.83", "description": "Other fatigue", "category": "General"},
]


# Drug Allergy Cross-Reactivity Dictionary
ALLERGY_GROUPS = {
    "penicillin": [
        "amoxicillin",
        "ampicillin",
        "penicillin",
        "augmentin",
        "piperacillin",
        "amoxil",
    ],
    "sulfa": ["bactrim", "cotrimoxazole", "sulfamethoxazole", "septrin"],
    "nsaid": [
        "ibuprofen",
        "diclofenac",
        "naproxen",
        "aspirin",
        "indomethacin",
        "brufen",
    ],
    "macrolide": ["azithromycin", "erythromycin", "clarithromycin"],
}

# Drug-Drug Interaction Warning Rules (pairs -> severity, warning message)
KNOWN_INTERACTIONS = [
    (
        {"warfarin", "aspirin"},
        "CRITICAL",
        "High risk of major gastrointestinal hemorrhage and severe bleeding.",
    ),
    (
        {"warfarin", "ibuprofen"},
        "HIGH",
        "Increased risk of bleeding and gastric mucosal ulceration.",
    ),
    (
        {"lisinopril", "spironolactone"},
        "HIGH",
        "Severe hyperkalemia risk; requires close serum potassium monitoring.",
    ),
    (
        {"ciprofloxacin", "antacid"},
        "MEDIUM",
        "Chelation reduces ciprofloxacin bioavailability and therapeutic efficacy.",
    ),
    (
        {"metformin", "contrast"},
        "HIGH",
        "Risk of contrast-induced acute renal failure and metformin lactic acidosis.",
    ),
]


def search_icd10(query: str) -> list[dict]:
    """Search ICD-10 reference database by code or description keyword."""
    if not query:
        return ICD10_DATABASE[:5]
    q = query.lower().strip()
    return [
        item
        for item in ICD10_DATABASE
        if q in item["code"].lower()
        or q in item["description"].lower()
        or q in item["category"].lower()
    ]


def check_drug_safety(patient_id: str, new_medications: list[str]) -> dict:
    """
    Check new prescription list against patient allergies (structured PatientAllergy + free-text NursingNotes)
    and drug-drug interactions.
    Returns: {"has_warnings": bool, "alerts": list[dict]}
    """
    alerts = []
    new_meds_lower = [m.lower().strip() for m in new_medications if m]

    # 1a. Fetch structured patient allergies from PatientAllergy model
    structured_allergies = PatientAllergy.query.filter_by(patient_id=patient_id).all()
    structured_allergen_names = [
        a.allergen.lower().strip() for a in structured_allergies if a.allergen
    ]

    # 1b. Fetch patient allergy history from NursingNotes free text
    notes = NursingNote.query.filter_by(patient_id=patient_id).all()
    documented_allergies = list(structured_allergen_names)
    for n in notes:
        if n.allergies:
            documented_allergies.extend(
                [a.strip().lower() for a in n.allergies.split(",")]
            )

    # Check allergy cross-reactivity and direct matches
    for drug in new_meds_lower:
        # Direct allergen match from PatientAllergy registry or free-text
        for allergen in documented_allergies:
            if allergen in drug or drug in allergen:
                alerts.append(
                    {
                        "type": "ALLERGY_WARNING",
                        "severity": "CRITICAL",
                        "drug": drug,
                        "message": f"PATIENT ALLERGY ALERT: Patient has documented allergy to '{allergen.title()}'! Drug '{drug.title()}' is contraindicated.",
                    }
                )
                break
        else:
            # Check group cross-reactivity
            for group_name, drug_list in ALLERGY_GROUPS.items():
                if any(d in drug for d in drug_list):
                    if any(
                        group_name in allergy or any(d in allergy for d in drug_list)
                        for allergy in documented_allergies
                    ):
                        alerts.append(
                            {
                                "type": "ALLERGY_WARNING",
                                "severity": "CRITICAL",
                                "drug": drug,
                                "message": f"PATIENT ALLERGY ALERT: Patient is allergic to {group_name.upper()} group! Drug '{drug.title()}' is contraindicated.",
                            }
                        )
                        break

    # 2. Check Drug-Drug Interactions (DDI)
    # Fetch current active prescribed meds for patient
    active_prescriptions = PrescribedMedicine.query.filter_by(
        patient_id=patient_id
    ).all()
    current_meds_lower = [
        p.medicine.generic_name.lower().strip()
        for p in active_prescriptions
        if p.medicine and p.medicine.generic_name
    ]
    all_meds = set(new_meds_lower + current_meds_lower)

    for drug_set, severity, msg in KNOWN_INTERACTIONS:
        matched = [d for d in drug_set if any(d in med for med in all_meds)]
        if len(matched) >= 2:
            alerts.append(
                {
                    "type": "DRUG_INTERACTION",
                    "severity": severity,
                    "drugs": matched,
                    "message": f"DRUG INTERACTION WARNING [{severity}]: {' + '.join([m.title() for m in matched])} — {msg}",
                }
            )

    return {
        "has_warnings": len(alerts) > 0,
        "critical_block": any(a["severity"] == "CRITICAL" for a in alerts),
        "alerts": alerts,
    }


# API Routes
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
    from departments.medicine.cdss import evaluate_prescription_safety

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

    patient = Patient.query.filter_by(patient_id=patient_id).first()
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
    inv = Invoice.query.filter_by(patient_id=patient_id, status="DRAFT").first()
    if not inv:
        inv = Invoice(
            patient_id=patient_id,
            invoice_number=Invoice.generate_invoice_number(),
            status="DRAFT",
        )
        db.session.add(inv)
        db.session.flush()

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
