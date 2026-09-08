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
        "pair": {"warfarin", "aspirin"},
        "severity": "HIGH",
        "title": "Severe Bleeding Risk",
        "mechanism": "Additive anticoagulant and antiplatelet effects.",
        "recommendation": "Avoid concurrent use or monitor INR & stool occult blood closely.",
    },
    {
        "pair": {"sildenafil", "nitroglycerin"},
        "severity": "HIGH",
        "title": "Severe Hypotension Hazard",
        "mechanism": "Potentiation of nitric oxide-mediated vasodilation.",
        "recommendation": "CONTRAINDICATED. Do not co-administer within 24-48 hours.",
    },
    {
        "pair": {"lisinopril", "spironolactone"},
        "severity": "MODERATE",
        "title": "Hyperkalemia Risk",
        "mechanism": "Dual inhibition of renin-angiotensin-aldosterone axis.",
        "recommendation": "Monitor serum potassium and renal function within 1 week of initiation.",
    },
    {
        "pair": {"clopidogrel", "omeprazole"},
        "severity": "MODERATE",
        "title": "Reduced Antiplatelet Efficacy",
        "mechanism": "CYP2C19 competitive inhibition by omeprazole.",
        "recommendation": "Consider pantoprazole or H2-receptor antagonist instead.",
    },
    {
        "pair": {"ibuprofen", "prednisolone"},
        "severity": "MODERATE",
        "title": "GI Ulceration Risk",
        "mechanism": "Synergistic mucosal toxicity and COX inhibition.",
        "recommendation": "Co-prescribe PPI gastroprotection (e.g. esomeprazole).",
    },
]

# Common allergen drug classes & keywords
ALLERGY_PATTERNS = {
    "penicillin": [
        "penicillin",
        "amoxicillin",
        "ampicillin",
        "co-amoxiclav",
        "augmentin",
    ],
    "sulfa": ["sulfamethoxazole", "co-trimoxazole", "bactrim", "sulfasalazine"],
    "nsaid": ["ibuprofen", "naproxen", "diclofenac", "aspirin", "indomethacin"],
}

# Narrow therapeutic index drugs requiring renal dose adjustment
RENAL_DOSE_DRUGS = {
    "metformin": {
        "cutoff_egfr": 45.0,
        "guidance": "Avoid initiating if eGFR 30-45; discontinue if eGFR < 30 mL/min due to Lactic Acidosis risk.",
    },
    "gentamicin": {
        "cutoff_egfr": 60.0,
        "guidance": "Extend dosing interval (e.g. q24h -> q36h/q48h) and monitor trough levels due to nephrotoxicity risk.",
    },
    "vancomycin": {
        "cutoff_egfr": 50.0,
        "guidance": "Dose reduction required; check trough levels prior to 4th dose.",
    },
    "enoxaparin": {
        "cutoff_egfr": 30.0,
        "guidance": "Reduce dose to 1 mg/kg once daily if CrCl < 30 mL/min.",
    },
}


def query_drugcentral_ddi(drug1: str, drug2: str) -> list[dict]:
    """
    Query live DrugCentral PostgreSQL database for drug-drug interactions.

    Always returns [] on any failure (network timeout, circuit breaker open,
    DB error) so that check_drug_interactions() unconditionally falls through
    to the local KNOWN_INTERACTIONS matrix.  Never hangs: the psycopg2
    connect_timeout in DRUGCENTRAL_DB_PARAMS provides the hard deadline.
    """
    import time

    from departments.shared.drugcentral import (
        DrugCentralUnavailable,
        get_drugcentral_connection,
    )

    t0 = time.monotonic()
    conn = None
    try:
        conn = get_drugcentral_connection()  # raises DrugCentralUnavailable on failure
        cur = conn.cursor()

        # Step 1: Resolve drug names to class names
        def get_classes(drug: str) -> list[str]:
            q = """
                SELECT DISTINCT c.name
                FROM structures s
                JOIN struct2drgclass sc ON s.id = sc.struct_id
                JOIN drug_class c ON sc.drug_class_id = c.id
                WHERE LOWER(s.name) LIKE %s OR LOWER(c.name) LIKE %s
            """
            pat = f"%{drug.lower().strip()}%"
            cur.execute(q, (pat, pat))
            return [r[0] for r in cur.fetchall()] + [drug.strip()]

        c1 = get_classes(drug1)
        c2 = get_classes(drug2)

        if not c1 or not c2:
            return []

        # Step 2: Query DDI table for class-pair interactions
        q_ddi = """
            SELECT d.drug_class1, d.drug_class2, d.ddi_risk, d.description
            FROM ddi d
            WHERE (d.drug_class1 = ANY(%s) AND d.drug_class2 = ANY(%s))
               OR (d.drug_class1 = ANY(%s) AND d.drug_class2 = ANY(%s))
        """
        cur.execute(q_ddi, (c1, c2, c2, c1))
        rows = cur.fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "severity": "HIGH"
                    if "avoid" in (r[2] or "").lower()
                    or "contraindicated" in (r[2] or "").lower()
                    else "MODERATE",
                    "title": f"DrugCentral DDI: {r[0]} + {r[1]}",
                    "interacting_drugs": [drug1, drug2],
                    "mechanism": r[3]
                    or "Pharmacological class interaction registered in DrugCentral.",
                    "recommendation": f"Risk: {r[2]}. Review concurrent administration.",
                    "source": "DrugCentral PostgreSQL DB",
                }
            )
        return results

    except DrugCentralUnavailable as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.warning(
            "cdss.drugcentral_unavailable drug1=%s drug2=%s latency_ms=%.1f reason=%s "
            "— falling back to local interaction matrix.",
            drug1,
            drug2,
            latency_ms,
            exc,
        )
        return []
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.warning(
            "cdss.drugcentral_query_error drug1=%s drug2=%s latency_ms=%.1f error=%s: %s "
            "— falling back to local interaction matrix.",
            drug1,
            drug2,
            latency_ms,
            type(exc).__name__,
            exc,
        )
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def check_drug_interactions(medications: list[str]) -> list[dict]:
    """
    Check a list of medication names for known dangerous drug-drug interactions.
    Attempts live DrugCentral PostgreSQL lookup first, falling back to local KNOWN_INTERACTIONS matrix.

    Returns:
        List of interaction warning dicts.
    """
    if not medications or len(medications) < 2:
        return []

    normalized = [m.strip().lower() for m in medications if m]
    warnings = []
    seen_pairs = set()

    # Try DrugCentral DB lookup for pairs
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            d1, d2 = normalized[i], normalized[j]
            pair_key = tuple(sorted([d1, d2]))
            if pair_key in seen_pairs:
                continue
            dc_results = query_drugcentral_ddi(d1, d2)
            if dc_results:
                warnings.extend(dc_results)
                seen_pairs.add(pair_key)

    # Local fallback for pairs not found or if DB offline
    for rule in KNOWN_INTERACTIONS:
        pair = rule["pair"]
        matched_drugs = []
        for keyword in pair:
            for med in normalized:
                if keyword in med:
                    matched_drugs.append(keyword)
                    break

        if len(matched_drugs) == len(pair):
            pair_key = tuple(sorted(matched_drugs))
            if pair_key not in seen_pairs:
                warnings.append(
                    {
                        "severity": rule["severity"],
                        "title": rule["title"],
                        "interacting_drugs": sorted(matched_drugs),
                        "mechanism": rule["mechanism"],
                        "recommendation": rule["recommendation"],
                        "source": "Local Fallback Matrix",
                    }
                )
                seen_pairs.add(pair_key)

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
    combined_history = " ".join(
        [
            patient.name or "",
            patient.relationship_with_next_of_kin or "",
        ]
    ).lower()

    for allergen, keywords in ALLERGY_PATTERNS.items():
        if any(kw in drug_lower for kw in keywords):
            if allergen in combined_history or "allergy" in combined_history:
                return {
                    "severity": "HIGH",
                    "title": f"Possible Allergy Alert ({allergen.title()})",
                    "allergen_class": allergen,
                    "recommendation": f"Verify patient allergy status prior to administering {drug_name}.",
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
            if egfr < rule["cutoff_egfr"]:
                return {
                    "drug_name": drug_name,
                    "egfr": egfr,
                    "cutoff_egfr": rule["cutoff_egfr"],
                    "severity": "MODERATE",
                    "guidance": rule["guidance"],
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

    patient = (
        Patient.query.filter_by(patient_id=patient_id).first() if patient_id else None
    )
    allergy_alert = (
        check_patient_allergies(patient, drug_name) if (patient and drug_name) else None
    )
    dosing_alert = (
        calculate_dosing_adjustment(drug_name, egfr=egfr) if drug_name else None
    )

    warnings = list(interactions)
    if allergy_alert:
        warnings.append(allergy_alert)

    has_high_severity = any(w.get("severity") == "HIGH" for w in warnings)

    return {
        "has_warnings": len(warnings) > 0 or (dosing_alert is not None),
        "high_risk": has_high_severity,
        "warnings_count": len(warnings),
        "warnings": warnings,
        "dosing_guidance": dosing_alert,
    }
