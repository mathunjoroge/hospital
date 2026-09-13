"""
departments/medicine/chemotherapy_engine.py
─────────────────────────────────────────────
Clinical Engine for Oncology Body Surface Area (BSA) calculation,
standard chemotherapy protocol dosing, and cumulative lifetime toxicity tracking.
"""

import math
from typing import Any

from departments.models.oncology_models import ChemotherapyRegimenOrder

# Cumulative Lifetime Toxicity Limits per standard Oncology Guidelines (NCCN/ASCO)
LIFETIME_TOXICITY_CAPS = {
    "Doxorubicin": {"limit_per_m2": 450.0, "unit": "mg/m2", "organ": "Cardiotoxicity (Heart Failure)"},
    "Bleomycin": {"limit_total": 300.0, "unit": "units", "organ": "Pulmonary Toxicity (Pulmonary Fibrosis)"},
    "Vincristine": {"single_dose_cap": 2.0, "unit": "mg", "organ": "Peripheral Neuropathy Cap"},
}

# Standard Chemotherapy Protocols
CHEMO_PROTOCOLS: dict[str, dict[str, Any]] = {
    "FOLFOX6": {
        "description": "Oxaliplatin, Leucovorin, and 5-Fluorouracil (Colorectal Cancer)",
        "cancer_type": "Colorectal Cancer",
        "drugs": [
            {"drug_name": "Oxaliplatin", "dose_per_m2": 85.0, "unit": "mg/m2", "route": "IV Infusion over 2h"},
            {"drug_name": "Leucovorin", "dose_per_m2": 400.0, "unit": "mg/m2", "route": "IV Infusion over 2h"},
            {"drug_name": "5-Fluorouracil (Bolus)", "dose_per_m2": 400.0, "unit": "mg/m2", "route": "IV Push"},
            {"drug_name": "5-Fluorouracil (Infusion)", "dose_per_m2": 2400.0, "unit": "mg/m2", "route": "IV Continuous over 46h"},
        ],
    },
    "AC-T": {
        "description": "Doxorubicin & Cyclophosphamide followed by Paclitaxel (Breast Cancer)",
        "cancer_type": "Breast Cancer",
        "drugs": [
            {"drug_name": "Doxorubicin", "dose_per_m2": 60.0, "unit": "mg/m2", "route": "IV Push"},
            {"drug_name": "Cyclophosphamide", "dose_per_m2": 600.0, "unit": "mg/m2", "route": "IV Infusion over 1h"},
            {"drug_name": "Paclitaxel", "dose_per_m2": 175.0, "unit": "mg/m2", "route": "IV Infusion over 3h"},
        ],
    },
    "ABVD": {
        "description": "Doxorubicin, Bleomycin, Vinblastine, Dacarbazine (Hodgkin Lymphoma)",
        "cancer_type": "Hodgkin Lymphoma",
        "drugs": [
            {"drug_name": "Doxorubicin", "dose_per_m2": 25.0, "unit": "mg/m2", "route": "IV Push"},
            {"drug_name": "Bleomycin", "dose_per_m2": 10.0, "unit": "units/m2", "route": "IV Push"},
            {"drug_name": "Vinblastine", "dose_per_m2": 6.0, "unit": "mg/m2", "route": "IV Push"},
            {"drug_name": "Dacarbazine", "dose_per_m2": 375.0, "unit": "mg/m2", "route": "IV Infusion over 1h"},
        ],
    },
    "CHOP": {
        "description": "Cyclophosphamide, Doxorubicin, Vincristine, Prednisone (Non-Hodgkin Lymphoma)",
        "cancer_type": "Non-Hodgkin Lymphoma",
        "drugs": [
            {"drug_name": "Cyclophosphamide", "dose_per_m2": 750.0, "unit": "mg/m2", "route": "IV Infusion over 1h"},
            {"drug_name": "Doxorubicin", "dose_per_m2": 50.0, "unit": "mg/m2", "route": "IV Push"},
            {"drug_name": "Vincristine", "dose_per_m2": 1.4, "unit": "mg/m2", "route": "IV Push", "cap_max_mg": 2.0},
            {"drug_name": "Prednisone", "fixed_dose": 100.0, "unit": "mg", "route": "PO Daily for 5 days"},
        ],
    },
}


def calculate_bsa(height_cm: float, weight_kg: float, formula: str = "mosteller") -> float:
    """
    Calculate Body Surface Area (BSA) in m².

    Formulas:
      - Mosteller: sqrt((height_cm * weight_kg) / 3600)
      - DuBois: 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)
    """
    try:
        h = float(height_cm)
        w = float(weight_kg)
        if h <= 0 or w <= 0:
            return 1.73  # Standard adult default fallback

        if formula.lower() == "dubois":
            bsa = 0.007184 * (h ** 0.725) * (w ** 0.425)
        else:  # mosteller default
            bsa = math.sqrt((h * w) / 3600.0)

        return round(bsa, 2)
    except (ValueError, TypeError):
        return 1.73


def get_patient_cumulative_doses(patient_id: str) -> dict[str, float]:
    """
    Query all historical chemotherapy orders for patient to sum cumulative lifetime doses.
    Returns dict mapping drug_name -> cumulative_dose.
    """
    totals: dict[str, float] = {}
    try:
        orders = ChemotherapyRegimenOrder.query.filter_by(patient_id=patient_id).all()
        import json
        for order in orders:
            if not order.calculated_doses_json:
                continue
            doses = json.loads(order.calculated_doses_json)
            for d in doses:
                drug_name = d.get("drug_name")
                calc_dose = float(d.get("calculated_dose", 0.0))
                if drug_name:
                    totals[drug_name] = round(totals.get(drug_name, 0.0) + calc_dose, 2)
    except Exception:  # noqa: BLE001
        pass
    return totals


def calculate_regimen_doses(
    patient_id: str,
    protocol_name: str,
    height_cm: float,
    weight_kg: float,
    formula: str = "mosteller",
) -> dict[str, Any]:
    """
    Calculate exact chemotherapy drug doses for patient BSA and verify cumulative toxicity caps.
    """
    proto_key = protocol_name.upper().strip()
    if proto_key not in CHEMO_PROTOCOLS:
        return {
            "error": True,
            "message": f"Protocol '{protocol_name}' is not recognized. Valid options: {list(CHEMO_PROTOCOLS.keys())}",
        }

    protocol = CHEMO_PROTOCOLS[proto_key]
    bsa = calculate_bsa(height_cm, weight_kg, formula=formula)
    prev_cum_doses = get_patient_cumulative_doses(patient_id)

    calculated_drugs = []
    has_toxicity_warning = False
    toxicity_warnings = []

    for drug in protocol["drugs"]:
        drug_name = drug["drug_name"]
        unit = drug["unit"]
        route = drug["route"]

        if "fixed_dose" in drug:
            calc_dose = float(drug["fixed_dose"])
            dose_str = f"{calc_dose} {unit}"
            dose_per_m2 = None
        else:
            d_per_m2 = float(drug["dose_per_m2"])
            dose_per_m2 = d_per_m2
            raw_dose = round(bsa * d_per_m2, 1)

            # Apply single-dose max cap (e.g. Vincristine max 2.0 mg)
            if "cap_max_mg" in drug and raw_dose > drug["cap_max_mg"]:
                calc_dose = float(drug["cap_max_mg"])
                dose_str = f"{calc_dose} {unit} (Capped from {raw_dose} mg)"
            else:
                calc_dose = raw_dose
                dose_str = f"{calc_dose} {unit}"

        # Cumulative lifetime dose evaluation
        prev_tot = prev_cum_doses.get(drug_name, 0.0)
        new_cum_tot = round(prev_tot + calc_dose, 1)
        cap_exceeded = False
        cap_warning = None

        if drug_name in LIFETIME_TOXICITY_CAPS:
            cap_info = LIFETIME_TOXICITY_CAPS[drug_name]
            if "limit_per_m2" in cap_info:
                # Cumulative dose per m2 = new_cum_tot / bsa
                cum_per_m2 = round(new_cum_tot / bsa, 1) if bsa > 0 else 0
                if cum_per_m2 > cap_info["limit_per_m2"]:
                    cap_exceeded = True
                    has_toxicity_warning = True
                    cap_warning = (
                        f"CRITICAL TOXICITY WARNING: Cumulative {drug_name} dose ({cum_per_m2} mg/m²) "
                        f"EXCEEDS lifetime cap of {cap_info['limit_per_m2']} mg/m² ({cap_info['organ']})."
                    )
                    toxicity_warnings.append(cap_warning)
            elif "limit_total" in cap_info:
                if new_cum_tot > cap_info["limit_total"]:
                    cap_exceeded = True
                    has_toxicity_warning = True
                    cap_warning = (
                        f"CRITICAL TOXICITY WARNING: Cumulative {drug_name} dose ({new_cum_tot} {unit}) "
                        f"EXCEEDS lifetime cap of {cap_info['limit_total']} {unit} ({cap_info['organ']})."
                    )
                    toxicity_warnings.append(cap_warning)

        calculated_drugs.append({
            "drug_name": drug_name,
            "dose_per_m2": dose_per_m2,
            "calculated_dose": calc_dose,
            "dose_display": dose_str,
            "unit": unit,
            "route": route,
            "prev_cumulative_dose": prev_tot,
            "new_cumulative_dose": new_cum_tot,
            "cap_exceeded": cap_exceeded,
            "cap_warning": cap_warning,
        })

    return {
        "patient_id": patient_id,
        "protocol_name": proto_key,
        "description": protocol["description"],
        "cancer_type": protocol["cancer_type"],
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bsa_m2": bsa,
        "bsa_formula": formula,
        "drugs": calculated_drugs,
        "has_toxicity_warning": has_toxicity_warning,
        "toxicity_warnings": toxicity_warnings,
    }
