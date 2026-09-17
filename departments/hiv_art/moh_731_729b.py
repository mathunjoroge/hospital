"""
Kenya MOH 731 HIV/AIDS Summary & MOH 729B ARV FCDRR Reporting Engine.

MOH 731: HIV/AIDS Monthly Summary (ARV Regimen Section)
MOH 729B: ARV FCDRR (Facility Consumption Data Report and Request)
"""
from datetime import date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, or_

from departments.hiv_art.models import (
    ARTEnrollment,
    ARTRegimen,
    MOH729BARVFCDRRMonthly,
    MOH731ARVRegimenPatientMonthly,
)
from departments.models.pharmacy import DispensedDrug, Drug, Expiry, Purchase
from departments.models.records import Patient
from extensions import db

NASCOP_REGIMEN_CATALOG = [
    {
        "regimen_code": "AF1A",
        "regimen_line": "Adult_1st",
        "target_pop": "Adults / Adolescents (≥30kg)",
        "regimen_name": "TLD (Tenofovir 300mg + Lamivudine 300mg + Dolutegravir 50mg)",
        "arv_drug_code": "ARV_TLD_300_300_50",
        "unit_pack_size": "Bottle_30s",
        "keywords": ["tld", "tenofovir/lamivudine/dolutegravir", "3tc/tdf/dtg"],
    },
    {
        "regimen_code": "AF1B",
        "regimen_line": "Adult_1st",
        "target_pop": "Adults >60yrs / Renal / Osteopenia",
        "regimen_name": "TAFLD (Tenofovir Alafenamide 25mg + Lamivudine 300mg + Dolutegravir 50mg)",
        "arv_drug_code": "ARV_TAFLD_25_300_50",
        "unit_pack_size": "Bottle_30s",
        "keywords": ["tafld", "tenofovir alafenamide", "taf/3tc/dtg"],
    },
    {
        "regimen_code": "AF1C",
        "regimen_line": "Adult_1st",
        "target_pop": "Adult / Legacy DTG Intolerant",
        "regimen_name": "TLE (Tenofovir 300mg + Lamivudine 300mg + Efavirenz 400mg)",
        "arv_drug_code": "ARV_TLE_300_300_400",
        "unit_pack_size": "Bottle_30s",
        "keywords": ["tle", "tenofovir/lamivudine/efavirenz", "3tc/tdf/efv"],
    },
    {
        "regimen_code": "AF2A",
        "regimen_line": "Adult_1st",
        "target_pop": "Adults (TDF Toxicity)",
        "regimen_name": "ALD (Abacavir 600mg + Lamivudine 300mg + Dolutegravir 50mg)",
        "arv_drug_code": "ARV_ALD_600_300_50",
        "unit_pack_size": "Bottle_30s",
        "keywords": ["ald", "abacavir/lamivudine/dolutegravir", "abc/3tc/dtg"],
    },
    {
        "regimen_code": "AF2B",
        "regimen_line": "Adult_1st",
        "target_pop": "Adults (Renal Failure TDF/TAF Unsafe)",
        "regimen_name": "ZLD (Zidovudine 300mg + Lamivudine 150mg + Dolutegravir 50mg)",
        "arv_drug_code": "ARV_ZLD_300_150_50",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["zld", "zidovudine/lamivudine/dolutegravir", "azt/3tc/dtg"],
    },
    {
        "regimen_code": "AS1A",
        "regimen_line": "Adult_2nd",
        "target_pop": "Adult / Adolescent",
        "regimen_name": "AZT/3TC (300/150mg) + ATV/r (300/100mg)",
        "arv_drug_code": "ARV_AZT_3TC_ATVR",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["azt/3tc+atv/r", "atazanavir", "atv/r", "azt+3tc+atv/r"],
    },
    {
        "regimen_code": "AS1B",
        "regimen_line": "Adult_2nd",
        "target_pop": "Adult / Adolescent",
        "regimen_name": "AZT/3TC (300/150mg) + LPV/r (200/50mg)",
        "arv_drug_code": "ARV_AZT_3TC_LPVR",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["azt/3tc+lpv/r", "lopinavir", "lpv/r", "kaletra", "azt+3tc+lpv/r"],
    },
    {
        "regimen_code": "AS2A",
        "regimen_line": "Adult_2nd",
        "target_pop": "Adult / Adolescent (AZT 1st Failure)",
        "regimen_name": "TDF/3TC (300/300mg) + ATV/r (300/100mg)",
        "arv_drug_code": "ARV_TDF_3TC_ATVR",
        "unit_pack_size": "Bottle_30s",
        "keywords": ["tdf/3tc+atv/r", "tdf+3tc+atv/r"],
    },
    {
        "regimen_code": "AS5A",
        "regimen_line": "Adult_2nd",
        "target_pop": "Adult / Adolescent (ATV/r Failure)",
        "regimen_name": "AZT/3TC (300/150mg) + DRV/r (600/100mg BD)",
        "arv_drug_code": "ARV_AZT_3TC_DRVR",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["azt/3tc+drv/r", "darunavir", "drv/r"],
    },
    {
        "regimen_code": "AS6A",
        "regimen_line": "Adult_2nd",
        "target_pop": "Adult / Adolescent (Non-DTG 1st Failure)",
        "regimen_name": "AZT/3TC (300/150mg) + DTG (50mg)",
        "arv_drug_code": "ARV_AZT_3TC_DTG",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["azt/3tc+dtg", "azt+3tc+dtg"],
    },
    {
        "regimen_code": "TL3A",
        "regimen_line": "3rd_Line",
        "target_pop": "Adult 3rd-Line Failure",
        "regimen_name": "TLD + DRV/r (TDF/3TC/DTG FDC + DRV/r 600/100mg BD)",
        "arv_drug_code": "ARV_TLD_DRVR_3RD",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["tld+drv/r", "salvage tld", "3rd line drv/r"],
    },
    {
        "regimen_code": "TL3B",
        "regimen_line": "3rd_Line",
        "target_pop": "3rd-Line Renal / Geriatric (>60yrs)",
        "regimen_name": "TAFLD + DRV/r (TAF/3TC/DTG + DRV/r 600/100mg BD)",
        "arv_drug_code": "ARV_TAFLD_DRVR_3RD",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["tafld+drv/r", "3rd line tafld"],
    },
    {
        "regimen_code": "TL3C",
        "regimen_line": "3rd_Line",
        "target_pop": "Complex Multi-Class Salvage",
        "regimen_name": "ETR (200mg BD) + DRV/r (600/100mg BD) + DTG (50mg BD)",
        "arv_drug_code": "ARV_ETR_DRVR_DTG_SALVAGE",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["etravirine", "etr", "complex salvage", "etr+drv/r+dtg"],
    },
    {
        "regimen_code": "PF1A",
        "regimen_line": "Ped_1st",
        "target_pop": "Pediatric (≥4wks & ≥3kg)",
        "regimen_name": "pALD (ABC/3TC 120/60mg Dispersible + DTG 10mg DT)",
        "arv_drug_code": "ARV_PABC_3TC_DTG_120_60_10",
        "unit_pack_size": "Bottle_90s",
        "keywords": ["pald", "pabc", "dtg 10mg", "pediatric dtg"],
    },
    {
        "regimen_code": "PF1B",
        "regimen_line": "Ped_1st",
        "target_pop": "Pediatric (<4wks / DTG Contraindicated)",
        "regimen_name": "ABC/3TC + LPV/r (Pellets / Granules / Tabs)",
        "arv_drug_code": "ARV_ABC_3TC_LPVR_PED",
        "unit_pack_size": "Bottle_90s",
        "keywords": ["abc/3tc+lpv/r", "lpv/r pellets", "kaletra pellets"],
    },
    {
        "regimen_code": "PF2A",
        "regimen_line": "Ped_2nd",
        "target_pop": "Pediatric (ABC Failure)",
        "regimen_name": "AZT/3TC + DTG (10mg DT)",
        "arv_drug_code": "ARV_AZT_3TC_DTG_PED",
        "unit_pack_size": "Bottle_60s",
        "keywords": ["azt/3tc+dtg ped", "pediatric azt/3tc+dtg", "pf2a"],
    },
]


def classify_nascop_regimen(regimen_code: str = None, arv_drugs_text: str = None) -> dict | None:
    """
    Classify a regimen code or drug description string into the standard NASCOP master catalog.
    """
    if regimen_code:
        code_upper = regimen_code.strip().upper()
        for item in NASCOP_REGIMEN_CATALOG:
            if item["regimen_code"] == code_upper:
                return item

    if arv_drugs_text:
        text_lower = arv_drugs_text.lower()
        for item in NASCOP_REGIMEN_CATALOG:
            for kw in item["keywords"]:
                if kw in text_lower:
                    return item

    # Default to AF1A (TLD) if unknown adult 1st line
    return NASCOP_REGIMEN_CATALOG[0]


def aggregate_moh731_arv_monthly(year: int = None, month: int = None) -> dict:
    """
    Aggregate MOH 731 HIV/AIDS Monthly Summary Regimen Section.
    Calculates TX_CURR (active patients) by gender and regimen code, plus TX_NEW (new initiations).
    """
    today = date.today()
    if year is None:
        year = today.year
    if month is None:
        month = today.month

    period_str = f"{year}{month:02d}"
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    # Get active ART enrollments
    enrollments = (
        db.session.query(ARTEnrollment, Patient, ARTRegimen)
        .outerjoin(Patient, ARTEnrollment.patient_id == Patient.patient_id)
        .outerjoin(ARTRegimen, ARTEnrollment.current_regimen_id == ARTRegimen.id)
        .filter(ARTEnrollment.enrollment_date <= end_dt)
        .all()
    )

    regimen_counts = {item["regimen_code"]: {
        "regimen_code": item["regimen_code"],
        "regimen_line": item["regimen_line"],
        "regimen_name": item["regimen_name"],
        "target_pop": item["target_pop"],
        "active_patients_male": 0,
        "active_patients_female": 0,
        "total_active_patients": 0,
        "new_patients_started": 0,
    } for item in NASCOP_REGIMEN_CATALOG}

    tx_curr_total = 0
    tx_new_total = 0

    for enr, patient, regimen in enrollments:
        reg_code = (regimen.regimen_code if regimen else None) or "AF1A"
        nascop_match = classify_nascop_regimen(reg_code, regimen.arv_drugs if regimen else None)
        rcode = nascop_match["regimen_code"]

        is_male = (patient.sex in ["M", "Male"]) if patient else True
        is_new = (enr.art_start_date and start_dt <= enr.art_start_date <= end_dt) or (start_dt <= enr.enrollment_date <= end_dt)

        reg_data = regimen_counts[rcode]
        if is_male:
            reg_data["active_patients_male"] += 1
        else:
            reg_data["active_patients_female"] += 1

        reg_data["total_active_patients"] += 1
        tx_curr_total += 1

        if is_new:
            reg_data["new_patients_started"] += 1
            tx_new_total += 1

    # Persist or update database records
    for rcode, item in regimen_counts.items():
        db_rec = MOH731ARVRegimenPatientMonthly.query.filter_by(
            period=period_str, regimen_code=rcode
        ).first()

        if not db_rec:
            db_rec = MOH731ARVRegimenPatientMonthly(
                facility_id="KE_MOH_HOSPITAL_001",
                period=period_str,
                regimen_code=rcode,
                regimen_line=item["regimen_line"],
                active_patients_male=item["active_patients_male"],
                active_patients_female=item["active_patients_female"],
                total_active_patients=item["total_active_patients"],
                new_patients_started=item["new_patients_started"],
            )
            db.session.add(db_rec)
        else:
            db_rec.active_patients_male = item["active_patients_male"]
            db_rec.active_patients_female = item["active_patients_female"]
            db_rec.total_active_patients = item["total_active_patients"]
            db_rec.new_patients_started = item["new_patients_started"]

    db.session.commit()

    return {
        "period": period_str,
        "year": year,
        "month": month,
        "tx_curr_total": tx_curr_total,
        "tx_new_total": tx_new_total,
        "regimen_details": sorted(list(regimen_counts.values()), key=lambda x: x["regimen_code"]),
    }


def aggregate_moh729b_fcdrr_monthly(year: int = None, month: int = None) -> dict:
    """
    Aggregate MOH 729B ARV FCDRR Monthly Commodity & Patient Load Report.
    Links active patients per regimen to stock balance and pack consumption metrics.
    """
    today = date.today()
    if year is None:
        year = today.year
    if month is None:
        month = today.month

    period_str = f"{year}{month:02d}"
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    # Get MOH 731 patient load metrics
    moh731 = aggregate_moh731_arv_monthly(year, month)
    patient_map = {r["regimen_code"]: r["total_active_patients"] for r in moh731["regimen_details"]}

    fcdrr_items = []
    total_requested = 0

    for catalog_item in NASCOP_REGIMEN_CATALOG:
        rcode = catalog_item["regimen_code"]
        arv_code = catalog_item["arv_drug_code"]
        pack_size = catalog_item["unit_pack_size"]
        patients = patient_map.get(rcode, 0)

        # Match drug from inventory
        drug = None
        for kw in catalog_item["keywords"]:
            drug = Drug.query.filter(Drug.generic_name.ilike(f"%{kw}%")).first()
            if drug:
                break

        if drug:
            received = (
                db.session.query(func.coalesce(func.sum(Purchase.quantity_purchased), 0))
                .filter(Purchase.drug_id == drug.id)
                .filter(Purchase.purchase_date >= start_date)
                .filter(Purchase.purchase_date <= end_date)
                .scalar()
            )

            dispensed = (
                db.session.query(func.coalesce(func.sum(DispensedDrug.quantity_dispensed), 0))
                .filter(DispensedDrug.drug_id == drug.id)
                .filter(DispensedDrug.date_dispensed >= start_dt)
                .filter(DispensedDrug.date_dispensed <= end_dt)
                .filter(or_(DispensedDrug.status == "0", DispensedDrug.status == "COMPLETED", DispensedDrug.status == "DISPENSED", DispensedDrug.status.is_(None)))
                .scalar()
            )

            losses = (
                db.session.query(func.coalesce(func.sum(Expiry.quantity_removed), 0))
                .filter(Expiry.drug_id == drug.id)
                .filter(Expiry.removal_date >= start_date)
                .filter(Expiry.removal_date <= end_date)
                .scalar()
            )

            ending_balance = drug.quantity_in_stock or 0
            beginning_balance = max(0, ending_balance + dispensed + losses - received)
        else:
            received = 0
            dispensed = 0
            losses = 0
            ending_balance = 0
            beginning_balance = 0

        days_out = 0 if ending_balance > 0 else 31
        monthly_cons = max(dispensed, patients * 1)  # Estimate 1 bottle per patient per month minimum
        mos = round(Decimal(ending_balance) / Decimal(max(monthly_cons, 1)), 2)

        # Quantity requested for 3-month supply target
        target_stock = max(patients * 3, 30)
        quantity_requested = max(0, target_stock - ending_balance)
        total_requested += quantity_requested

        fcdrr_entry = {
            "regimen_code": rcode,
            "arv_drug_code": arv_code,
            "drug_description": catalog_item["regimen_name"],
            "unit_pack_size": pack_size,
            "patients_on_regimen": patients,
            "beginning_balance": int(beginning_balance),
            "quantity_received": int(received),
            "quantity_dispensed": int(dispensed),
            "losses_adjustments": int(losses),
            "ending_balance": int(ending_balance),
            "days_stocked_out": days_out,
            "months_of_stock": float(mos),
            "quantity_requested": int(quantity_requested),
        }
        fcdrr_items.append(fcdrr_entry)

        # Persist or update MOH 729B DB record
        db_fcdrr = MOH729BARVFCDRRMonthly.query.filter_by(
            period=period_str, arv_drug_code=arv_code
        ).first()

        if not db_fcdrr:
            db_fcdrr = MOH729BARVFCDRRMonthly(
                facility_id="KE_MOH_HOSPITAL_001",
                period=period_str,
                arv_drug_code=arv_code,
                unit_pack_size=pack_size,
                patients_on_regimen=patients,
                beginning_balance=int(beginning_balance),
                quantity_received=int(received),
                quantity_dispensed=int(dispensed),
                losses_adjustments=int(losses),
                ending_balance=int(ending_balance),
                days_stocked_out=days_out,
                months_of_stock=Decimal(str(mos)),
                quantity_requested=int(quantity_requested),
            )
            db.session.add(db_fcdrr)
        else:
            db_fcdrr.patients_on_regimen = patients
            db_fcdrr.beginning_balance = int(beginning_balance)
            db_fcdrr.quantity_received = int(received)
            db_fcdrr.quantity_dispensed = int(dispensed)
            db_fcdrr.losses_adjustments = int(losses)
            db_fcdrr.ending_balance = int(ending_balance)
            db_fcdrr.days_stocked_out = days_out
            db_fcdrr.months_of_stock = Decimal(str(mos))
            db_fcdrr.quantity_requested = int(quantity_requested)

    db.session.commit()

    return {
        "period": period_str,
        "year": year,
        "month": month,
        "total_requested_packs": total_requested,
        "fcdrr_details": fcdrr_items,
    }


def aggregate_pmtct_tx_pvls_monthly(year: int = None, month: int = None) -> dict:
    """
    Aggregate PMTCT (Maternal ART, HEI Prophylaxis & EID) and Viral Load Suppression (TX_PVLS) Metrics.
    """
    today = date.today()
    if year is None:
        year = today.year
    if month is None:
        month = today.month

    period_str = f"{year}{month:02d}"
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    end_dt = datetime.combine(end_date, datetime.max.time())
    six_months_ago = end_dt - timedelta(days=180)

    # 1. PMTCT Metrics
    pmtct_art_count = ARTEnrollment.query.filter(
        or_(ARTEnrollment.is_pregnant.is_(True), ARTEnrollment.is_breastfeeding.is_(True)),
        ARTEnrollment.enrollment_date <= end_dt,
    ).count()

    hei_prophylaxis_count = ARTEnrollment.query.filter(
        ARTEnrollment.hei_infant_prophylaxis.isnot(None),
        ARTEnrollment.enrollment_date <= end_dt,
    ).count()

    eid_6wk_pcr_count = ARTEnrollment.query.filter(
        ARTEnrollment.eid_dna_pcr_6wk_result.isnot(None),
        ARTEnrollment.enrollment_date <= end_dt,
    ).count()

    # 2. Viral Load Suppression (TX_PVLS) Metrics
    tx_pvls_eligible = ARTEnrollment.query.filter(
        ARTEnrollment.art_start_date <= six_months_ago,
    ).count()

    from departments.hiv_art.models import ViralLoad
    vl_records = (
        db.session.query(ViralLoad)
        .filter(ViralLoad.test_date <= end_dt)
        .order_by(ViralLoad.test_date.desc())
        .all()
    )

    # Latest VL per patient
    latest_vl = {}
    for vl in vl_records:
        if vl.patient_id not in latest_vl:
            latest_vl[vl.patient_id] = vl

    tx_pvls_tested = len(latest_vl)
    tx_pvls_suppressed = sum(1 for vl in latest_vl.values() if vl.viral_load_copies is not None and vl.viral_load_copies < 50)
    tx_pvls_unsuppressed = sum(1 for vl in latest_vl.values() if vl.viral_load_copies is not None and vl.viral_load_copies >= 1000)

    suppression_rate = round((tx_pvls_suppressed / tx_pvls_tested * 100), 1) if tx_pvls_tested > 0 else 100.0

    return {
        "period": period_str,
        "year": year,
        "month": month,
        "pmtct_art_count": pmtct_art_count,
        "hei_prophylaxis_count": hei_prophylaxis_count,
        "eid_6wk_pcr_count": eid_6wk_pcr_count,
        "tx_pvls_eligible": tx_pvls_eligible,
        "tx_pvls_tested": tx_pvls_tested,
        "tx_pvls_suppressed": tx_pvls_suppressed,
        "tx_pvls_unsuppressed": tx_pvls_unsuppressed,
        "suppression_rate_pct": suppression_rate,
    }

