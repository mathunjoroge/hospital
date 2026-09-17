"""
Kenya MOH-645 & MOH-743 Malaria Commodity & Weight Band Reporting Engine.

MOH-645: Health Facility Daily Activity Register for Malaria Commodities
MOH-743: Health Facility Monthly Summary Report for Malaria Commodities
"""
from datetime import date, datetime, timedelta, timezone
from sqlalchemy import func, extract, or_, and_

from extensions import db
from departments.models.pharmacy import DispensedDrug, Drug, Batch
from departments.models.records import Patient
from departments.malaria.models import MalariaCase, MalariaLabResult, MalariaTreatment


ANTIMALARIAL_KEYWORDS = {
    "al": ["artemether", "lumefantrine", "coartem", "al 6", "al 12", "al 18", "al 24", "al6", "al12", "al18", "al24"],
    "artesunate": ["artesunate"],
    "quinine": ["quinine"],
    "sp": ["sulfadoxine", "pyrimethamine", "fansidar"],
}


def estimate_weight_from_age(age_years: float) -> float:
    """
    Estimate patient weight (in kg) from age (in years) based on standard WHO growth approximations.
    Used as fallback when measured weight is not recorded.
    """
    if age_years < 3:
        return 10.0  # 5-14 kg band (AL6)
    elif age_years < 8:
        return 20.0  # 15-24 kg band (AL12)
    elif age_years < 12:
        return 30.0  # 25-34 kg band (AL18)
    else:
        return 45.0  # ≥35 kg band (AL24)


def classify_al_weight_band(weight_kg: float) -> str:
    """
    Classify weight (kg) into standard Kenya MOH-645/743 AL weight bands:
    - AL6  (5 kg to <15 kg)
    - AL12 (15 kg to <25 kg)
    - AL18 (25 kg to <35 kg)
    - AL24 (≥35 kg)
    """
    if weight_kg is None:
        return "AL24"  # Default adult band if completely unknown

    if weight_kg < 15:
        return "AL6"
    elif weight_kg < 25:
        return "AL12"
    elif weight_kg < 35:
        return "AL18"
    else:
        return "AL24"


def classify_drug_category(generic_name: str) -> str:
    """
    Identify antimalarial commodity category from generic name.
    Returns: 'al', 'artesunate', 'quinine', 'sp', or None.
    """
    if not generic_name:
        return None

    name_lower = generic_name.lower()

    for cat, keywords in ANTIMALARIAL_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return cat
    return None


def get_patient_weight_or_estimate(patient_id: str, ref_date: date = None) -> tuple[float, str]:
    """
    Determine patient weight in kg.
    First checks MalariaCase.weight_kg, falls back to Patient.date_of_birth age estimation.
    Returns (weight_kg, source) where source is 'measured' or 'estimated'.
    """
    if ref_date is None:
        ref_date = date.today()

    # 1. Check MalariaCase
    case = (
        MalariaCase.query.filter(MalariaCase.patient_id == str(patient_id))
        .filter(MalariaCase.weight_kg.isnot(None))
        .order_by(MalariaCase.diagnosis_date.desc())
        .first()
    )
    if case and case.weight_kg:
        return (float(case.weight_kg), "measured")

    # 2. Fall back to Patient date_of_birth
    patient = Patient.query.filter_by(patient_id=str(patient_id)).first()
    if patient and patient.date_of_birth:
        dob = patient.date_of_birth
        age_years = (ref_date - dob).days / 365.25
        return (estimate_weight_from_age(age_years), "estimated")

    # Default to adult estimate
    return (45.0, "estimated")


def aggregate_moh645_daily(target_date: date = None) -> dict:
    """
    Aggregate daily activity register tallies for MOH-645 for a given date.
    Returns counts of antimalarial commodities dispensed and patients treated by weight band.
    """
    if target_date is None:
        target_date = date.today()

    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = datetime.combine(target_date, datetime.max.time())

    dispensed_records = (
        db.session.query(DispensedDrug, Drug)
        .join(Drug, DispensedDrug.drug_id == Drug.id)
        .filter(DispensedDrug.date_dispensed >= start_dt)
        .filter(DispensedDrug.date_dispensed <= end_dt)
        .filter(or_(DispensedDrug.status == "0", DispensedDrug.status == "COMPLETED", DispensedDrug.status == "DISPENSED", DispensedDrug.status.is_(None)))
        .all()
    )

    tallies = {
        "al_6": 0,
        "al_12": 0,
        "al_18": 0,
        "al_24": 0,
        "artesunate_inj": 0,
        "quinine": 0,
        "sp": 0,
        "patients_5_14kg": 0,
        "patients_15_24kg": 0,
        "patients_25_34kg": 0,
        "patients_35pluskg": 0,
        "total_antimalarial_dispenses": 0,
    }

    patient_seen_bands = set()

    for disp, drug in dispensed_records:
        cat = classify_drug_category(drug.generic_name)
        if not cat:
            continue

        tallies["total_antimalarial_dispenses"] += 1
        qty = disp.quantity_dispensed or 1

        weight_kg, _ = get_patient_weight_or_estimate(disp.patient_id, target_date)
        band = classify_al_weight_band(weight_kg)

        if cat == "al":
            name_lower = (drug.generic_name + " " + (drug.strength or "")).lower()
            if "6" in name_lower and "al" in name_lower:
                tallies["al_6"] += qty
            elif "12" in name_lower and "al" in name_lower:
                tallies["al_12"] += qty
            elif "18" in name_lower and "al" in name_lower:
                tallies["al_18"] += qty
            elif "24" in name_lower and "al" in name_lower:
                tallies["al_24"] += qty
            else:
                if band == "AL6":
                    tallies["al_6"] += qty
                elif band == "AL12":
                    tallies["al_12"] += qty
                elif band == "AL18":
                    tallies["al_18"] += qty
                else:
                    tallies["al_24"] += qty

        elif cat == "artesunate":
            tallies["artesunate_inj"] += qty
        elif cat == "quinine":
            tallies["quinine"] += qty
        elif cat == "sp":
            tallies["sp"] += qty

        p_key = (disp.patient_id, band)
        if p_key not in patient_seen_bands:
            patient_seen_bands.add(p_key)
            if band == "AL6":
                tallies["patients_5_14kg"] += 1
            elif band == "AL12":
                tallies["patients_15_24kg"] += 1
            elif band == "AL18":
                tallies["patients_25_34kg"] += 1
            else:
                tallies["patients_35pluskg"] += 1

    rdt_count = (
        MalariaLabResult.query.filter(
            func.date(MalariaLabResult.test_date) == target_date
        )
        .filter(MalariaLabResult.test_type.ilike("%RDT%"))
        .count()
    )
    tallies["rdts_used"] = rdt_count

    return tallies


def aggregate_moh743_monthly(year: int = None, month: int = None) -> dict:
    """
    Aggregate monthly summary report tallies for MOH-743.
    """
    today = date.today()
    if year is None:
        year = today.year
    if month is None:
        month = today.month

    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    dispensed_records = (
        db.session.query(DispensedDrug, Drug)
        .join(Drug, DispensedDrug.drug_id == Drug.id)
        .filter(DispensedDrug.date_dispensed >= start_dt)
        .filter(DispensedDrug.date_dispensed <= end_dt)
        .filter(or_(DispensedDrug.status == "0", DispensedDrug.status == "COMPLETED", DispensedDrug.status == "DISPENSED", DispensedDrug.status.is_(None)))
        .all()
    )

    monthly_data = {
        "year": year,
        "month": month,
        "al_6_dispensed": 0,
        "al_12_dispensed": 0,
        "al_18_dispensed": 0,
        "al_24_dispensed": 0,
        "artesunate_inj_dispensed": 0,
        "quinine_dispensed": 0,
        "sp_dispensed": 0,
        "rdts_used": 0,
        "patients_5_14kg": 0,
        "patients_15_24kg": 0,
        "patients_25_34kg": 0,
        "patients_35pluskg": 0,
        "total_patients_treated": 0,
    }

    patient_seen_bands = set()

    for disp, drug in dispensed_records:
        cat = classify_drug_category(drug.generic_name)
        if not cat:
            continue

        qty = disp.quantity_dispensed or 1
        weight_kg, _ = get_patient_weight_or_estimate(disp.patient_id, start_date)
        band = classify_al_weight_band(weight_kg)

        if cat == "al":
            name_lower = (drug.generic_name + " " + (drug.strength or "")).lower()
            if "6" in name_lower and "al" in name_lower:
                monthly_data["al_6_dispensed"] += qty
            elif "12" in name_lower and "al" in name_lower:
                monthly_data["al_12_dispensed"] += qty
            elif "18" in name_lower and "al" in name_lower:
                monthly_data["al_18_dispensed"] += qty
            elif "24" in name_lower and "al" in name_lower:
                monthly_data["al_24_dispensed"] += qty
            else:
                if band == "AL6":
                    monthly_data["al_6_dispensed"] += qty
                elif band == "AL12":
                    monthly_data["al_12_dispensed"] += qty
                elif band == "AL18":
                    monthly_data["al_18_dispensed"] += qty
                else:
                    monthly_data["al_24_dispensed"] += qty

        elif cat == "artesunate":
            monthly_data["artesunate_inj_dispensed"] += qty
        elif cat == "quinine":
            monthly_data["quinine_dispensed"] += qty
        elif cat == "sp":
            monthly_data["sp_dispensed"] += qty

        p_key = (disp.patient_id, band)
        if p_key not in patient_seen_bands:
            patient_seen_bands.add(p_key)
            if band == "AL6":
                monthly_data["patients_5_14kg"] += 1
            elif band == "AL12":
                monthly_data["patients_15_24kg"] += 1
            elif band == "AL18":
                monthly_data["patients_25_34kg"] += 1
            else:
                monthly_data["patients_35pluskg"] += 1

    monthly_data["total_patients_treated"] = len(patient_seen_bands)

    rdt_count = (
        MalariaLabResult.query.filter(
            MalariaLabResult.test_date >= start_dt
        )
        .filter(MalariaLabResult.test_date <= end_dt)
        .filter(MalariaLabResult.test_type.ilike("%RDT%"))
        .count()
    )
    monthly_data["rdts_used"] = rdt_count

    stock_summary = {}
    antimalarial_drugs = Drug.query.all()
    for drug in antimalarial_drugs:
        cat = classify_drug_category(drug.generic_name)
        if cat:
            stock_summary[drug.generic_name] = {
                "category": cat,
                "current_stock": drug.quantity_in_stock,
                "reorder_level": drug.reorder_level or 0,
            }
    monthly_data["stock_summary"] = stock_summary

    return monthly_data
