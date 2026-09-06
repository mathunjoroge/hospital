"""
departments/medicine/cdss.py
─────────────────────────────
Phase G — Clinical Decision Support System (CDSS)

Provides:
  - check_drug_interactions(): dangerous drug-drug pair checking
  - check_patient_allergies(): patient allergy screening
  - calculate_dosing_adjustment(): renal/hepatic dose adjustment guidance
  - evaluate_prescription_safety(): composite CDSS evaluation
"""

import logging

from departments.models.records import Patient

logger = logging.getLogger(__name__)

# Known dangerous drug-drug interaction matrix
KNOWN_INTERACTIONS = [
    {
        'pair': {'warfarin', 'aspirin'},
        'severity': 'HIGH',
        'title': 'Severe Bleeding Risk',
        'mechanism': 'Additive anticoagulant and antiplatelet effects.',
        'recommendation': 'Avoid concurrent use or monitor INR & stool occult blood closely.',
    },
    {
        'pair': {'sildenafil', 'nitroglycerin'},
        'severity': 'HIGH',
        'title': 'Severe Hypotension Hazard',
        'mechanism': 'Potentiation of nitric oxide-mediated vasodilation.',
        'recommendation': 'CONTRAINDICATED. Do not co-administer within 24-48 hours.',
    },
    {
        'pair': {'lisinopril', 'spironolactone'},
        'severity': 'MODERATE',
        'title': 'Hyperkalemia Risk',
        'mechanism': 'Dual inhibition of renin-angiotensin-aldosterone axis.',
        'recommendation': 'Monitor serum potassium and renal function within 1 week of initiation.',
    },
    {
        'pair': {'clopidogrel', 'omeprazole'},
        'severity': 'MODERATE',
        'title': 'Reduced Antiplatelet Efficacy',
        'mechanism': 'CYP2C19 competitive inhibition by omeprazole.',
        'recommendation': 'Consider pantoprazole or H2-receptor antagonist instead.',
    },
    {
        'pair': {'ibuprofen', 'prednisolone'},
        'severity': 'MODERATE',
        'title': 'GI Ulceration Risk',
        'mechanism': 'Synergistic mucosal toxicity and COX inhibition.',
        'recommendation': 'Co-prescribe PPI gastroprotection (e.g. esomeprazole).',
    },
]

# Common allergen drug classes & keywords
ALLERGY_PATTERNS = {
    'penicillin': ['penicillin', 'amoxicillin', 'ampicillin', 'co-amoxiclav', 'augmentin'],
    'sulfa': ['sulfamethoxazole', 'co-trimoxazole', 'bactrim', 'sulfasalazine'],
    'nsaid': ['ibuprofen', 'naproxen', 'diclofenac', 'aspirin', 'indomethacin'],
}

# Narrow therapeutic index drugs requiring renal dose adjustment
RENAL_DOSE_DRUGS = {
    'metformin': {
        'cutoff_egfr': 45.0,
        'guidance': 'Avoid initiating if eGFR 30-45; discontinue if eGFR < 30 mL/min due to Lactic Acidosis risk.',
    },
    'gentamicin': {
        'cutoff_egfr': 60.0,
        'guidance': 'Extend dosing interval (e.g. q24h -> q36h/q48h) and monitor trough levels due to nephrotoxicity risk.',
    },
    'vancomycin': {
        'cutoff_egfr': 50.0,
        'guidance': 'Dose reduction required; check trough levels prior to 4th dose.',
    },
    'enoxaparin': {
        'cutoff_egfr': 30.0,
        'guidance': 'Reduce dose to 1 mg/kg once daily if CrCl < 30 mL/min.',
    },
}


def check_drug_interactions(medications: list[str]) -> list[dict]:
    """
    Check a list of medication names for known dangerous drug-drug interactions.

    Returns:
        List of interaction warning dicts.
    """
    if not medications or len(medications) < 2:
        return []

    normalized = [m.strip().lower() for m in medications if m]
    warnings = []

    for rule in KNOWN_INTERACTIONS:
        pair = rule['pair']
        matched_drugs = []
        for keyword in pair:
            for med in normalized:
                if keyword in med:
                    matched_drugs.append(keyword)
                    break

        if len(matched_drugs) == len(pair):
            warnings.append({
                'severity': rule['severity'],
                'title': rule['title'],
                'interacting_drugs': sorted(matched_drugs),
                'mechanism': rule['mechanism'],
                'recommendation': rule['recommendation'],
            })

    return warnings


def check_patient_allergies(patient: Patient, drug_name: str) -> dict | None:
    """
    Check if patient has recorded allergies matching the drug.

    Returns:
        Warning dict if allergen match found, otherwise None.
    """
    if not patient or not drug_name:
        return None

    drug_lower = drug_name.strip().lower()

    # Search patient notes / next_of_kin / allergy fields
    combined_history = " ".join([
        patient.name or '',
        patient.relationship_with_next_of_kin or '',
    ]).lower()

    for allergen, keywords in ALLERGY_PATTERNS.items():
        if any(kw in drug_lower for kw in keywords):
            if allergen in combined_history or "allergy" in combined_history:
                return {
                    'severity': 'HIGH',
                    'title': f'Possible Allergy Alert ({allergen.title()})',
                    'allergen_class': allergen,
                    'recommendation': f'Verify patient allergy status prior to administering {drug_name}.',
                }

    return None


def calculate_dosing_adjustment(drug_name: str, egfr: float = None) -> dict | None:
    """
    Check if drug requires renal dose adjustment based on patient eGFR.
    """
    if not drug_name or egfr is None:
        return None

    drug_lower = drug_name.strip().lower()
    for drug_key, rule in RENAL_DOSE_DRUGS.items():
        if drug_key in drug_lower:
            if egfr < rule['cutoff_egfr']:
                return {
                    'drug_name': drug_name,
                    'egfr': egfr,
                    'cutoff_egfr': rule['cutoff_egfr'],
                    'severity': 'MODERATE',
                    'guidance': rule['guidance'],
                }

    return None


def evaluate_prescription_safety(
    patient_id: str = None,
    drug_name: str = None,
    existing_meds: list[str] = None,
    egfr: float = None,
) -> dict:
    """
    Comprehensive Clinical Decision Support evaluation.
    Combines drug interaction, allergy contraindication, and renal dosing guidance.
    """
    existing_meds = existing_meds or []
    all_meds = list(existing_meds)
    if drug_name:
        all_meds.append(drug_name)

    interactions = check_drug_interactions(all_meds)

    patient = Patient.query.filter_by(patient_id=patient_id).first() if patient_id else None
    allergy_alert = check_patient_allergies(patient, drug_name) if (patient and drug_name) else None
    dosing_alert = calculate_dosing_adjustment(drug_name, egfr=egfr) if drug_name else None

    warnings = list(interactions)
    if allergy_alert:
        warnings.append(allergy_alert)

    has_high_severity = any(w.get('severity') == 'HIGH' for w in warnings)

    return {
        'has_warnings': len(warnings) > 0 or (dosing_alert is not None),
        'high_risk': has_high_severity,
        'warnings_count': len(warnings),
        'warnings': warnings,
        'dosing_guidance': dosing_alert,
    }
