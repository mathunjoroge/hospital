"""
departments/medicine/chemotherapy_engine.py
─────────────────────────────────────────────
Clinical Engine for Oncology Body Surface Area (BSA) calculation,
standard chemotherapy protocol dosing, and cumulative lifetime toxicity tracking.

Safety posture (this module drives real drug doses):
  * Never invent patient data. Missing / non-numeric / non-finite / implausible
    height or weight is rejected — there is no silent "standard adult" fallback.
  * Fail CLOSED. If the cumulative-dose history cannot be read, the calculation
    is refused rather than reporting "no prior exposure" (which would silently
    disable the lifetime-cap check).
  * Cancelled orders never count toward lifetime totals.
  * Lifetime mg/m² exposure is accumulated per order using the BSA in effect when
    that order was written, not by dividing a milligram total by today's BSA.
"""

import json
import math
from typing import Any

from departments.models.oncology_models import ChemotherapyRegimenOrder
from extensions import db

# Plausibility envelope for BSA inputs (adult + paediatric). Outside it, the
# value is almost certainly a unit or data-entry error, so refuse to dose.
HEIGHT_CM_RANGE = (30.0, 250.0)
WEIGHT_KG_RANGE = (1.0, 350.0)
BSA_FORMULAS = ("mosteller", "dubois")
MAX_CYCLES = 30


class ChemoInputError(ValueError):
    """Invalid clinical input (biometrics, formula, cycle...). Safe to show to the prescriber."""

    code = "INVALID_INPUT"


class ChemoCycleError(ChemoInputError):
    code = "CYCLE_OUT_OF_RANGE"


class CumulativeHistoryUnavailable(RuntimeError):
    """Prior chemotherapy exposure could not be read reliably; lifetime caps cannot be verified."""


# Cumulative Lifetime Toxicity Limits per standard Oncology Guidelines (NCCN/ASCO)
LIFETIME_TOXICITY_CAPS = {
    "Doxorubicin": {
        "limit_per_m2": 450.0,
        "unit": "mg/m2",
        "organ": "Cardiotoxicity (Heart Failure)",
    },
    "Bleomycin": {
        "limit_total": 300.0,
        "unit": "units",
        "organ": "Pulmonary Toxicity (Pulmonary Fibrosis)",
    },
    "Vincristine": {
        "single_dose_cap": 2.0,
        "unit": "mg",
        "organ": "Peripheral Neuropathy Cap",
    },
}

# Standard Chemotherapy Protocols
CHEMO_PROTOCOLS: dict[str, dict[str, Any]] = {
    "FOLFOX6": {
        "description": "Oxaliplatin, Leucovorin, and 5-Fluorouracil (Colorectal Cancer)",
        "cancer_type": "Colorectal Cancer",
        "default_total_cycles": 12,
        "drugs": [
            {
                "drug_name": "Oxaliplatin",
                "dose_per_m2": 85.0,
                "unit": "mg/m2",
                "route": "IV Infusion over 2h",
            },
            {
                "drug_name": "Leucovorin",
                "dose_per_m2": 400.0,
                "unit": "mg/m2",
                "route": "IV Infusion over 2h",
            },
            {
                "drug_name": "5-Fluorouracil (Bolus)",
                "dose_per_m2": 400.0,
                "unit": "mg/m2",
                "route": "IV Push",
            },
            {
                "drug_name": "5-Fluorouracil (Infusion)",
                "dose_per_m2": 2400.0,
                "unit": "mg/m2",
                "route": "IV Continuous over 46h",
            },
        ],
    },
    "AC-T": {
        "description": "Doxorubicin & Cyclophosphamide followed by Paclitaxel (Breast Cancer)",
        "cancer_type": "Breast Cancer",
        "default_total_cycles": 8,
        "drugs": [
            {
                "drug_name": "Doxorubicin",
                "dose_per_m2": 60.0,
                "unit": "mg/m2",
                "route": "IV Push",
                "cycles": (1, 4),  # AC phase
            },
            {
                "drug_name": "Cyclophosphamide",
                "dose_per_m2": 600.0,
                "unit": "mg/m2",
                "route": "IV Infusion over 1h",
                "cycles": (1, 4),  # AC phase
            },
            {
                "drug_name": "Paclitaxel",
                "dose_per_m2": 175.0,
                "unit": "mg/m2",
                "route": "IV Infusion over 3h",
                "cycles": (5, 8),  # T phase (sequential, never concurrent with AC)
            },
        ],
    },
    "ABVD": {
        "description": "Doxorubicin, Bleomycin, Vinblastine, Dacarbazine (Hodgkin Lymphoma)",
        "cancer_type": "Hodgkin Lymphoma",
        "default_total_cycles": 6,
        "drugs": [
            {
                "drug_name": "Doxorubicin",
                "dose_per_m2": 25.0,
                "unit": "mg/m2",
                "route": "IV Push",
            },
            {
                "drug_name": "Bleomycin",
                "dose_per_m2": 10.0,
                "unit": "units/m2",
                "route": "IV Push",
            },
            {
                "drug_name": "Vinblastine",
                "dose_per_m2": 6.0,
                "unit": "mg/m2",
                "route": "IV Push",
            },
            {
                "drug_name": "Dacarbazine",
                "dose_per_m2": 375.0,
                "unit": "mg/m2",
                "route": "IV Infusion over 1h",
            },
        ],
    },
    "CHOP": {
        "description": "Cyclophosphamide, Doxorubicin, Vincristine, Prednisone (Non-Hodgkin Lymphoma)",
        "cancer_type": "Non-Hodgkin Lymphoma",
        "default_total_cycles": 6,
        "drugs": [
            {
                "drug_name": "Cyclophosphamide",
                "dose_per_m2": 750.0,
                "unit": "mg/m2",
                "route": "IV Infusion over 1h",
            },
            {
                "drug_name": "Doxorubicin",
                "dose_per_m2": 50.0,
                "unit": "mg/m2",
                "route": "IV Push",
            },
            {
                "drug_name": "Vincristine",
                "dose_per_m2": 1.4,
                "unit": "mg/m2",
                "route": "IV Push",
                "cap_max_mg": 2.0,
            },
            {
                "drug_name": "Prednisone",
                "fixed_dose": 100.0,
                "unit": "mg",
                "route": "PO Daily for 5 days",
            },
        ],
    },
}


def _finite_number(value: Any, name: str) -> float:
    """Parse a required, finite number. Never substitutes a default."""
    if value is None or isinstance(value, bool):
        raise ChemoInputError(f"{name} is required and must be a number.")
    if isinstance(value, str) and not value.strip():
        raise ChemoInputError(f"{name} is required and must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ChemoInputError(f"{name} must be a number.") from None
    if not math.isfinite(number):
        raise ChemoInputError(f"{name} must be a finite number.")
    return number


def normalize_bsa_formula(formula: Any) -> str:
    """Return 'mosteller' or 'dubois'; reject anything else (no silent fallback)."""
    if formula is None or (isinstance(formula, str) and not formula.strip()):
        return "mosteller"
    normalized = str(formula).strip().lower()
    if normalized not in BSA_FORMULAS:
        raise ChemoInputError(
            f"Unsupported BSA formula '{formula}'. Valid options: {list(BSA_FORMULAS)}."
        )
    return normalized


def validate_biometrics(height_cm: Any, weight_kg: Any) -> tuple[float, float]:
    """Validate and return (height_cm, weight_kg) as finite, plausible floats."""
    h = _finite_number(height_cm, "height (cm)")
    w = _finite_number(weight_kg, "weight (kg)")
    if not (HEIGHT_CM_RANGE[0] <= h <= HEIGHT_CM_RANGE[1]):
        raise ChemoInputError(
            f"Height {h:g} cm is outside the plausible range "
            f"{HEIGHT_CM_RANGE[0]:g}-{HEIGHT_CM_RANGE[1]:g} cm. Check units."
        )
    if not (WEIGHT_KG_RANGE[0] <= w <= WEIGHT_KG_RANGE[1]):
        raise ChemoInputError(
            f"Weight {w:g} kg is outside the plausible range "
            f"{WEIGHT_KG_RANGE[0]:g}-{WEIGHT_KG_RANGE[1]:g} kg. Check units."
        )
    return h, w


def calculate_bsa(
    height_cm: float, weight_kg: float, formula: str = "mosteller"
) -> float:
    """
    Calculate Body Surface Area (BSA) in m².

    Formulas:
      - Mosteller: sqrt((height_cm * weight_kg) / 3600)
      - DuBois: 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)

    Raises ChemoInputError for missing, non-numeric, non-finite or implausible
    inputs and for an unknown formula. There is deliberately NO default BSA.
    """
    h, w = validate_biometrics(height_cm, weight_kg)
    formula_key = normalize_bsa_formula(formula)

    if formula_key == "dubois":
        bsa = 0.007184 * (h**0.725) * (w**0.425)
    else:
        bsa = math.sqrt((h * w) / 3600.0)

    return round(bsa, 2)


def get_protocol_max_cycle(protocol: dict[str, Any]) -> int | None:
    """Highest cycle any phase-gated drug is scheduled for, or None if not phase-gated."""
    ends = [d["cycles"][1] for d in protocol["drugs"] if "cycles" in d]
    return max(ends) if ends else None


def validate_cycle(cycle_number: Any, protocol: dict[str, Any]) -> int:
    """Validate a cycle number against global limits and the protocol's phases."""
    if isinstance(cycle_number, bool):
        raise ChemoCycleError("cycle number must be a whole number.")
    try:
        as_float = float(cycle_number)
    except (TypeError, ValueError):
        raise ChemoCycleError("cycle number must be a whole number.") from None
    if not math.isfinite(as_float) or as_float != int(as_float):
        raise ChemoCycleError("cycle number must be a whole number.")
    cycle = int(as_float)
    if not (1 <= cycle <= MAX_CYCLES):
        raise ChemoCycleError(f"cycle number must be between 1 and {MAX_CYCLES}.")
    max_cycle = get_protocol_max_cycle(protocol)
    if max_cycle is not None and cycle > max_cycle:
        raise ChemoCycleError(
            f"cycle {cycle} is beyond the last cycle ({max_cycle}) of this protocol."
        )
    return cycle


def _absolute_unit(basis_unit: str) -> str:
    """'mg/m2' -> 'mg', 'units/m2' -> 'units', 'mg' -> 'mg'."""
    return basis_unit.split("/")[0]


def get_patient_cumulative_exposure(patient_id: str) -> dict[str, dict[str, float]]:
    """
    Sum every non-cancelled chemotherapy order for the patient.

    Returns {drug_name: {"total": <absolute dose>, "per_m2": <mg/m2 exposure>}}.
    ``per_m2`` is accumulated per order using that order's own recorded BSA.

    Raises CumulativeHistoryUnavailable if the history cannot be read or is
    malformed — callers must NOT treat that as "no prior exposure".
    """
    totals: dict[str, float] = {}
    per_m2: dict[str, float] = {}
    try:
        orders = ChemotherapyRegimenOrder.query.filter(
            ChemotherapyRegimenOrder.patient_id == patient_id,
            ChemotherapyRegimenOrder.status != "CANCELLED",
        ).all()
        for order in orders:
            if not order.calculated_doses_json:
                continue
            bsa = float(order.bsa_m2 or 0.0)
            if not math.isfinite(bsa) or bsa <= 0:
                raise CumulativeHistoryUnavailable(
                    f"Order #{order.id} has an invalid recorded BSA ({order.bsa_m2})."
                )
            for dose in json.loads(order.calculated_doses_json):
                drug_name = dose.get("drug_name")
                if not drug_name:
                    continue
                amount = float(dose.get("calculated_dose", 0.0))
                if not math.isfinite(amount):
                    raise CumulativeHistoryUnavailable(
                        f"Order #{order.id} has a non-numeric dose for {drug_name}."
                    )
                totals[drug_name] = totals.get(drug_name, 0.0) + amount
                per_m2[drug_name] = per_m2.get(drug_name, 0.0) + amount / bsa
    except CumulativeHistoryUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately broad: ANY failure must fail closed
        db.session.rollback()
        raise CumulativeHistoryUnavailable(
            "Prior chemotherapy history could not be read; lifetime toxicity "
            "caps cannot be verified."
        ) from exc

    return {
        name: {"total": round(totals[name], 2), "per_m2": round(per_m2[name], 2)}
        for name in totals
    }


def get_patient_cumulative_doses(patient_id: str) -> dict[str, float]:
    """
    Cumulative absolute lifetime dose per drug (cancelled orders excluded).
    Raises CumulativeHistoryUnavailable instead of silently returning {}.
    """
    return {
        name: values["total"]
        for name, values in get_patient_cumulative_exposure(patient_id).items()
    }


def _error(code: str, message: str) -> dict[str, Any]:
    return {"error": True, "code": code, "message": message}


def calculate_regimen_doses(
    patient_id: str,
    protocol_name: str,
    height_cm: float,
    weight_kg: float,
    formula: str = "mosteller",
    cycle_number: int = 1,
) -> dict[str, Any]:
    """
    Calculate exact chemotherapy drug doses for patient BSA and verify cumulative toxicity caps.

    On any problem returns ``{"error": True, "code": ..., "message": ...}`` — it never
    falls back to assumed biometrics or to an assumed-empty dose history.
    """
    proto_key = str(protocol_name or "").upper().strip()
    if proto_key not in CHEMO_PROTOCOLS:
        return _error(
            "UNKNOWN_PROTOCOL",
            f"Protocol '{protocol_name}' is not recognized. Valid options: {list(CHEMO_PROTOCOLS.keys())}",
        )

    protocol = CHEMO_PROTOCOLS[proto_key]

    try:
        formula_key = normalize_bsa_formula(formula)
        bsa = calculate_bsa(height_cm, weight_kg, formula=formula_key)
        cycle = validate_cycle(cycle_number, protocol)
    except ChemoInputError as exc:
        return _error(exc.code, str(exc))

    try:
        prev_exposure = get_patient_cumulative_exposure(patient_id)
    except CumulativeHistoryUnavailable as exc:
        return _error("CUMULATIVE_HISTORY_UNAVAILABLE", str(exc))

    calculated_drugs = []
    has_toxicity_warning = False
    toxicity_warnings = []

    for drug in protocol["drugs"]:
        # Phase gating (e.g. AC-T: doxorubicin/cyclophosphamide cycles 1-4, paclitaxel 5-8)
        if "cycles" in drug and not (drug["cycles"][0] <= cycle <= drug["cycles"][1]):
            continue

        drug_name = drug["drug_name"]
        unit = drug["unit"]  # basis unit, e.g. "mg/m2"
        dose_unit = _absolute_unit(unit)  # unit of the calculated dose, e.g. "mg"
        route = drug["route"]

        if "fixed_dose" in drug:
            calc_dose = float(drug["fixed_dose"])
            dose_str = f"{calc_dose} {dose_unit}"
            dose_per_m2 = None
        else:
            d_per_m2 = float(drug["dose_per_m2"])
            dose_per_m2 = d_per_m2
            raw_dose = round(bsa * d_per_m2, 1)

            # Single-dose max cap (e.g. Vincristine max 2.0 mg). Protocol-level
            # cap wins; otherwise fall back to the drug's global single-dose cap
            # so a future protocol containing the drug cannot bypass it.
            single_cap = drug.get(
                "cap_max_mg",
                LIFETIME_TOXICITY_CAPS.get(drug_name, {}).get("single_dose_cap"),
            )
            if single_cap is not None and raw_dose > single_cap:
                calc_dose = float(single_cap)
                dose_str = (
                    f"{calc_dose} {dose_unit} (Capped from {raw_dose} {dose_unit})"
                )
            else:
                calc_dose = raw_dose
                dose_str = f"{calc_dose} {dose_unit}"

        # Cumulative lifetime dose evaluation
        prev = prev_exposure.get(drug_name, {"total": 0.0, "per_m2": 0.0})
        prev_tot = prev["total"]
        new_cum_tot = round(prev_tot + calc_dose, 1)
        prev_per_m2 = prev["per_m2"]
        new_per_m2 = round(prev_per_m2 + calc_dose / bsa, 1)
        cap_exceeded = False
        cap_warning = None

        cap_info = LIFETIME_TOXICITY_CAPS.get(drug_name, {})
        if "limit_per_m2" in cap_info:
            if new_per_m2 > cap_info["limit_per_m2"]:
                cap_exceeded = True
                cap_warning = (
                    f"CRITICAL TOXICITY WARNING: Cumulative {drug_name} dose ({new_per_m2} mg/m²) "
                    f"EXCEEDS lifetime cap of {cap_info['limit_per_m2']} mg/m² ({cap_info['organ']})."
                )
        elif "limit_total" in cap_info:
            if new_cum_tot > cap_info["limit_total"]:
                cap_exceeded = True
                cap_warning = (
                    f"CRITICAL TOXICITY WARNING: Cumulative {drug_name} dose ({new_cum_tot} {dose_unit}) "
                    f"EXCEEDS lifetime cap of {cap_info['limit_total']} {dose_unit} ({cap_info['organ']})."
                )
        if cap_exceeded:
            has_toxicity_warning = True
            toxicity_warnings.append(cap_warning)

        calculated_drugs.append(
            {
                "drug_name": drug_name,
                "dose_per_m2": dose_per_m2,
                "calculated_dose": calc_dose,
                "dose_display": dose_str,
                "unit": unit,  # basis unit (per m2) — kept for API compatibility
                "dose_unit": dose_unit,  # unit of calculated_dose and the cumulative totals
                "route": route,
                "prev_cumulative_dose": prev_tot,
                "new_cumulative_dose": new_cum_tot,
                "prev_cumulative_per_m2": prev_per_m2,
                "new_cumulative_per_m2": new_per_m2,
                "cap_exceeded": cap_exceeded,
                "cap_warning": cap_warning,
            }
        )

    return {
        "patient_id": patient_id,
        "protocol_name": proto_key,
        "description": protocol["description"],
        "cancer_type": protocol["cancer_type"],
        "cycle_number": cycle,
        "default_total_cycles": protocol.get("default_total_cycles"),
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
        "bsa_m2": bsa,
        "bsa_formula": formula_key,
        "drugs": calculated_drugs,
        "has_toxicity_warning": has_toxicity_warning,
        "toxicity_warnings": toxicity_warnings,
    }
