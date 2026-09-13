"""
departments/clinical_safety/cdss_advanced.py
─────────────────────────────────────────────
Johns Hopkins–Grade Advanced Clinical Decision Support (CDSS) Engine.

Includes:
1. RenalDosingEngine       — eGFR (CKD-EPI 2021) & CrCl (Cockcroft-Gault) calculators, dose adjustments & contraindication blocks.
2. HepaticDosingEngine     — Child-Pugh / hepatic impairment dosing caps & contraindications.
3. PediatricDosingEngine   — Weight-based mg/kg dosing, adult dose cap enforcement, & pediatric age contraindications.
4. PregnancySafetyEngine   — FDA Category D & X pregnancy & lactation contraindication checks.
5. AlertFatigueManager     — Tiered severity triage, 24h duplicate alert suppression, and override audit trail.
"""

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. RENAL DOSING ENGINE (eGFR & CrCl Calculators + Adjustment Rules)
# ---------------------------------------------------------------------------

def calculate_egfr(creatinine_mg_dl: float, age_years: int, is_female: bool) -> float:
    """
    Calculate eGFR using the CKD-EPI 2021 Race-Free Equation.
    eGFR = 142 * min(Scr/kappa, 1)^alpha * max(Scr/kappa, 1)^-1.200 * 0.9938^Age * (1.012 if female)
    """
    if creatinine_mg_dl <= 0 or age_years <= 0:
        return 90.0  # Default to normal eGFR if invalid input

    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    scr_ratio = creatinine_mg_dl / kappa

    min_part = min(scr_ratio, 1.0) ** alpha
    max_part = max(scr_ratio, 1.0) ** -1.200
    age_part = 0.9938 ** age_years
    female_part = 1.012 if is_female else 1.0

    egfr = 142.0 * min_part * max_part * age_part * female_part
    return round(egfr, 1)


def calculate_crcl(creatinine_mg_dl: float, age_years: int, weight_kg: float, is_female: bool) -> float:
    """
    Calculate Creatinine Clearance (CrCl) using the Cockcroft-Gault Equation.
    CrCl = ((140 - Age) * Weight_kg) / (72 * Scr) [* 0.85 if female]
    """
    if creatinine_mg_dl <= 0 or age_years <= 0 or weight_kg <= 0:
        return 100.0

    crcl = ((140.0 - age_years) * weight_kg) / (72.0 * creatinine_mg_dl)
    if is_female:
        crcl *= 0.85
    return round(crcl, 1)


# Renal Adjustment Guidelines Table (Drug -> threshold eGFR -> action/warning)
RENAL_DRUG_RULES = {
    "metformin": [
        {"max_egfr": 30, "severity": "CRITICAL", "action": "BLOCK", "message": "METFORMIN CONTRAINDICATED: eGFR < 30 mL/min/1.73m². High risk of fatal lactic acidosis."},
        {"max_egfr": 45, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING WARNING: eGFR 30–45 mL/min/1.73m². Reduce maximum dose to 1000 mg/day; monitor eGFR every 3 months."},
    ],
    "ciprofloxacin": [
        {"max_egfr": 30, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING WARNING: eGFR < 30 mL/min/1.73m². Reduce Ciprofloxacin dose by 50% or extend interval to q24h."},
    ],
    "enoxaparin": [
        {"max_egfr": 30, "severity": "CRITICAL", "action": "WARN", "message": "RENAL DOSING ALERT: Severe renal impairment (eGFR < 30 mL/min). Reduce therapeutic dose to 1 mg/kg ONCE daily (q24h)."},
    ],
    "gentamicin": [
        {"max_egfr": 30, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING ALERT: Severe renal impairment. Therapeutic drug monitoring (TDM) mandatory; extend dosing interval (q48h) or dose reduce."},
    ],
    "vancomycin": [
        {"max_egfr": 45, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING ALERT: Renal impairment (eGFR < 45 mL/min). Monitor serum trough levels (target 15-20 mcg/mL); adjust dosing interval."},
    ],
    "allopurinol": [
        {"max_egfr": 30, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING ALERT: eGFR < 30 mL/min. Reduce Allopurinol max dose to 100 mg/day to prevent Steven-Johnson / hypersensitivity syndrome."},
    ],
    "digoxin": [
        {"max_egfr": 30, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING ALERT: eGFR < 30 mL/min. Reduce Digoxin dose by 50% (0.0625 mg/day or QOD); high risk of fatal digitalis toxicity."},
    ],
    "fluconazole": [
        {"max_egfr": 50, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING ALERT: eGFR < 50 mL/min. Administer standard loading dose, then reduce maintenance dose by 50%."},
    ],
    "acyclovir": [
        {"max_egfr": 25, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING ALERT: eGFR < 25 mL/min. Reduce IV Acyclovir dose to 5 mg/kg q24h; ensure adequate hydration to prevent renal precipitation."},
    ],
    "atenolol": [
        {"max_egfr": 35, "severity": "HIGH", "action": "WARN", "message": "RENAL DOSING ALERT: eGFR < 35 mL/min. Reduce Atenolol max dose to 50 mg/day (or 25 mg/day if eGFR < 15)."},
    ],
}


class RenalDosingEngine:
    """Evaluates renal function against medication dosing guidelines."""

    @staticmethod
    def evaluate(drug_name: str, egfr: float) -> list[dict]:
        alerts = []
        d_lower = (drug_name or "").lower().strip()

        for drug_key, rules in RENAL_DRUG_RULES.items():
            if drug_key in d_lower:
                for rule in rules:
                    if egfr < rule["max_egfr"]:
                        alerts.append({
                            "type": "RENAL_DOSING_ALERT",
                            "severity": rule["severity"],
                            "drug": drug_name,
                            "egfr": egfr,
                            "action": rule["action"],
                            "message": rule["message"],
                        })
                        break  # Match most severe applicable rule
        return alerts


# ---------------------------------------------------------------------------
# 2. HEPATIC IMPAIRMENT DOSING ENGINE
# ---------------------------------------------------------------------------

HEPATIC_DRUG_RULES = {
    "paracetamol": {
        "max_daily_dose_g": 2.0,
        "severity": "HIGH",
        "message": "HEPATIC DOSING WARNING: Max Paracetamol dose is 2.0 g/day in patients with hepatic impairment/cirrhosis (standard max 4.0 g/day)."
    },
    "acetaminophen": {
        "max_daily_dose_g": 2.0,
        "severity": "HIGH",
        "message": "HEPATIC DOSING WARNING: Max Acetaminophen dose is 2.0 g/day in patients with hepatic impairment."
    },
    "methotrexate": {
        "severity": "CRITICAL",
        "action": "BLOCK",
        "message": "HEPATIC CONTRAINDICATION: Methotrexate is CONTRAINDICATED in active liver disease/cirrhosis due to severe hepatotoxicity."
    },
    "simvastatin": {
        "severity": "HIGH",
        "action": "BLOCK",
        "message": "HEPATIC CONTRAINDICATION: Active liver disease or unexplained elevated transaminases contraindicates Simvastatin."
    },
    "voriconazole": {
        "severity": "HIGH",
        "message": "HEPATIC DOSING WARNING: Standard loading dose, then reduce maintenance dose by 50% in Child-Pugh Class A or B cirrhosis."
    },
    "rifampicin": {
        "severity": "HIGH",
        "message": "HEPATIC DOSING ALERT: Hepatotoxic agent. Monitor ALT/AST baseline and every 2 weeks in hepatic impairment."
    }
}


class HepaticDosingEngine:
    """Evaluates hepatic function and cirrhosis status against medication safety rules."""

    @staticmethod
    def evaluate(drug_name: str, has_hepatic_impairment: bool = False, is_cirrhotic: bool = False) -> list[dict]:
        alerts = []
        if not (has_hepatic_impairment or is_cirrhotic):
            return alerts

        d_lower = (drug_name or "").lower().strip()
        for drug_key, rule in HEPATIC_DRUG_RULES.items():
            if drug_key in d_lower:
                alerts.append({
                    "type": "HEPATIC_DOSING_ALERT",
                    "severity": rule.get("severity", "HIGH"),
                    "drug": drug_name,
                    "message": rule["message"],
                })
        return alerts


# ---------------------------------------------------------------------------
# 3. PEDIATRIC WEIGHT-BASED & AGE CONTRAINDICATION ENGINE
# ---------------------------------------------------------------------------

PEDIATRIC_AGE_CONTRAINDICATIONS = [
    {
        "keywords": ["doxycycline", "tetracycline", "minocycline"],
        "max_age_years": 8,
        "severity": "HIGH",
        "message": "PEDIATRIC AGE CONTRAINDICATION: Tetracyclines (Doxycycline) in children < 8 years can cause permanent tooth discoloration and enamel hypoplasia."
    },
    {
        "keywords": ["aspirin", "acetylsalicylic"],
        "max_age_years": 16,
        "severity": "CRITICAL",
        "message": "PEDIATRIC CONTRAINDICATION: Aspirin in children/adolescents < 16 years with viral illness causes Reye's Syndrome (encephalopathy and acute liver failure)."
    },
    {
        "keywords": ["ciprofloxacin", "levofloxacin", "ofloxacin"],
        "max_age_years": 18,
        "severity": "HIGH",
        "message": "PEDIATRIC WARNING: Fluoroquinolones in pediatric patients < 18 years increase risk of arthropathy and tendon rupture (reserve for complicated infections)."
    },
    {
        "keywords": ["codeine", "tramadol"],
        "max_age_years": 12,
        "severity": "CRITICAL",
        "message": "PEDIATRIC CONTRAINDICATION: Codeine/Tramadol in children < 12 years is CONTRAINDICATED due to risk of ultra-rapid CYP2D6 metabolism and fatal respiratory depression."
    }
]

# Standard pediatric weight-based dosing guidelines (drug -> mg_per_kg_per_dose or mg_per_kg_per_day, adult_max_dose_mg)
PEDIATRIC_DOSING_RULES = {
    "paracetamol": {"mg_per_kg_dose": 15.0, "adult_max_dose_mg": 1000.0, "max_daily_mg_per_kg": 60.0},
    "amoxicillin": {"mg_per_kg_day": 45.0, "adult_max_dose_mg": 1000.0},
    "ibuprofen": {"mg_per_kg_dose": 10.0, "adult_max_dose_mg": 400.0},
    "ceftriaxone": {"mg_per_kg_day": 50.0, "adult_max_dose_mg": 2000.0},
    "azithromycin": {"mg_per_kg_day": 10.0, "adult_max_dose_mg": 500.0},
}


class PediatricDosingEngine:
    """Evaluates pediatric weight-based dosing and age-based contraindications."""

    @staticmethod
    def evaluate(drug_name: str, dose_mg: float | None, weight_kg: float | None, age_years: int) -> list[dict]:
        alerts = []
        d_lower = (drug_name or "").lower().strip()

        # 1. Pediatric Age Contraindications
        for rule in PEDIATRIC_AGE_CONTRAINDICATIONS:
            if age_years < rule["max_age_years"]:
                if any(kw in d_lower for kw in rule["keywords"]):
                    alerts.append({
                        "type": "PEDIATRIC_AGE_ALERT",
                        "severity": rule["severity"],
                        "drug": drug_name,
                        "age_years": age_years,
                        "message": rule["message"],
                    })

        # 2. Pediatric Weight-Based Dose Calculation & Adult Cap Check
        if age_years < 18 and weight_kg and weight_kg > 0 and dose_mg and dose_mg > 0:
            for drug_key, rule in PEDIATRIC_DOSING_RULES.items():
                if drug_key in d_lower:
                    adult_cap = rule.get("adult_max_dose_mg", 1000.0)
                    if "mg_per_kg_dose" in rule:
                        rec_dose = round(weight_kg * rule["mg_per_kg_dose"], 1)
                        # Cap at adult max
                        capped_rec_dose = min(rec_dose, adult_cap)
                        if dose_mg > (capped_rec_dose * 1.25):  # >25% higher than recommended
                            alerts.append({
                                "type": "PEDIATRIC_DOSE_HIGH",
                                "severity": "HIGH",
                                "drug": drug_name,
                                "prescribed_mg": dose_mg,
                                "recommended_mg": capped_rec_dose,
                                "message": f"PEDIATRIC DOSE ALERT: Prescribed dose ({dose_mg} mg) exceeds recommended weight-based dose ({capped_rec_dose} mg based on {rule['mg_per_kg_dose']} mg/kg for {weight_kg} kg)."
                            })
        return alerts


# ---------------------------------------------------------------------------
# 4. PREGNANCY & LACTATION CONTRAINDICATION ENGINE
# ---------------------------------------------------------------------------

PREGNANCY_CATEGORY_RULES = [
    # Category X — Absolute Contraindications
    {
        "keywords": ["methotrexate"],
        "category": "X",
        "severity": "CRITICAL",
        "message": "PREGNANCY CONTRAINDICATION (Category X): Methotrexate causes severe fetal death and congenital malformations. ABSOLUTE CONTRAINDICATION."
    },
    {
        "keywords": ["warfarin", "coumadin"],
        "category": "X",
        "severity": "CRITICAL",
        "message": "PREGNANCY CONTRAINDICATION (Category X): Warfarin causes fetal warfarin syndrome (nasal hypoplasia, CNS defects, hemorrhage). Use LMWH instead."
    },
    {
        "keywords": ["statins", "simvastatin", "atorvastatin", "rosuvastatin"],
        "category": "X",
        "severity": "CRITICAL",
        "message": "PREGNANCY CONTRAINDICATION (Category X): Statins disrupt cholesterol synthesis essential for fetal development. Discontinue immediately."
    },
    {
        "keywords": ["isotretinoin", "roaccutane"],
        "category": "X",
        "severity": "CRITICAL",
        "message": "PREGNANCY CONTRAINDICATION (Category X): Isotretinoin is extremely teratogenic (craniofacial, cardiac, CNS defects). ABSOLUTELY CONTRAINDICATED."
    },
    {
        "keywords": ["misoprostol"],
        "category": "X",
        "severity": "CRITICAL",
        "message": "PREGNANCY CONTRAINDICATION (Category X): Misoprostol induces uterine contractions and abortion. Contraindicated unless used for labor induction under protocol."
    },
    {
        "keywords": ["valproate", "valproic acid", "epilim"],
        "category": "X",
        "severity": "CRITICAL",
        "message": "PREGNANCY CONTRAINDICATION (Category X): High risk of neural tube defects (spina bifida) and neurodevelopmental impairment."
    },

    # Category D — High Risk Warnings / Trimester-Specific
    {
        "keywords": ["lisinopril", "enalapril", "losartan", "valsartan", "ace inhibitor", "arb"],
        "category": "D",
        "severity": "CRITICAL",
        "message": "PREGNANCY CONTRAINDICATION (Category D): ACE Inhibitors / ARBs in 2nd/3rd trimester cause fetal renal failure, oligohydramnios, and skull hypoplasia."
    },
    {
        "keywords": ["ibuprofen", "diclofenac", "naproxen", "indomethacin"],
        "trimester_min": 3,
        "category": "D",
        "severity": "HIGH",
        "message": "PREGNANCY WARNING (3rd Trimester NSAID): NSAIDs in 3rd trimester cause premature closure of ductus arteriosus and neonatal pulmonary hypertension."
    },
    {
        "keywords": ["phenytoin", "carbamazepine"],
        "category": "D",
        "severity": "HIGH",
        "message": "PREGNANCY WARNING (Category D): Anticonvulsants increase risk of major congenital malformations and fetal hydantoin syndrome. Requires high-dose folate supplementation."
    }
]


class PregnancySafetyEngine:
    """Evaluates pregnancy & lactation safety contraindications."""

    @staticmethod
    def evaluate(drug_name: str, is_pregnant: bool = False, trimester: int | None = None, is_lactating: bool = False) -> list[dict]:
        alerts = []
        if not (is_pregnant or is_lactating):
            return alerts

        d_lower = (drug_name or "").lower().strip()

        if is_pregnant:
            for rule in PREGNANCY_CATEGORY_RULES:
                if any(kw in d_lower for kw in rule["keywords"]):
                    req_trim = rule.get("trimester_min")
                    if req_trim is None or (trimester and trimester >= req_trim):
                        alerts.append({
                            "type": "PREGNANCY_CONTRAINDICATION",
                            "severity": rule["severity"],
                            "drug": drug_name,
                            "category": rule.get("category", "D"),
                            "trimester": trimester,
                            "message": rule["message"],
                        })
        return alerts


# ---------------------------------------------------------------------------
# 5. ALERT FATIGUE CONTROL & CLINICAL TRIAGE MANAGER
# ---------------------------------------------------------------------------

class AlertFatigueManager:
    """
    Tiered Severity Triage & Duplicate Suppression Filter.

    Rules:
      1. CRITICAL alerts (Allergy block, Pregnancy Cat X, Severe DDI) are NEVER suppressed.
      2. MODERATE / LOW alerts occurring within a 24-hour window for the same patient
         session are suppressed to avoid clinician alert fatigue.
    """

    def __init__(self, suppression_window_hours: int = 24):
        self.suppression_window_hours = suppression_window_hours

    def process_and_filter(self, alerts: list[dict], patient_id: str, recent_overrides: list[dict] | None = None) -> list[dict]:
        """Filter alerts to remove duplicate low/moderate alerts suppressed by recent clinician overrides."""
        if not alerts:
            return []

        recent_overrides = recent_overrides or []
        overridden_keys = set()

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.suppression_window_hours)
        for ov in recent_overrides:
            ov_time = ov.get("created_at")
            if ov_time and ov_time >= cutoff_time:
                # Key by (alert_type, drug/message)
                overridden_keys.add((ov.get("alert_type"), ov.get("drug", "").lower()))

        filtered_alerts = []
        for alert in alerts:
            severity = alert.get("severity", "MODERATE")
            alert_type = alert.get("type")
            drug_name = alert.get("drug", "").lower()

            # Rule 1: CRITICAL alerts are NEVER suppressed
            if severity == "CRITICAL":
                filtered_alerts.append(alert)
                continue

            # Rule 2: Suppress if clinician overrode same moderate/low alert recently
            if (alert_type, drug_name) in overridden_keys:
                logger.info("Alert fatigue suppression active for patient %s: suppressed %s for %s", patient_id, alert_type, drug_name)
                continue

            filtered_alerts.append(alert)

        return filtered_alerts
