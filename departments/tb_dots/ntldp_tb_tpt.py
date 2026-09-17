"""
Kenya NTLD-P Tuberculosis (TB) & TPT (TB Preventive Therapy) Reporting Engine.

Covers:
- Drug-Susceptible TB (DS-TB): Adult 2RHZE/4RH, Pediatric 2RHZ/4RH
- Drug-Resistant TB (DR-TB): All-Oral BPaLM, BPaL, and Individualized 18-20 month regimens
- TB Preventive Therapy (TPT): 3HP, 1HP, 3RH, 6H (+ Vit B6), 6LFX
"""
from datetime import date, datetime, timedelta, timezone
from sqlalchemy import func, or_, and_

from extensions import db
from departments.models.records import Patient
from departments.tb_dots.models import TBEnrollment, TBRegimen
from departments.models.clinical_regimens import MasterClinicalRegimen
from departments.hiv_art.moh_731_729b import NASCOP_REGIMEN_CATALOG


NTLDP_TB_TPT_CATALOG = [
    # 1. Drug-Susceptible TB (DS-TB)
    {
        "nascop_ntldp_code": "DS-TB-ADULT",
        "program_domain": "TB",
        "regimen_acronym": "2RHZE/4RH",
        "line_tier": "1st_Line",
        "target_population": "DS-TB Adults & Adolescents (≥15 yrs)",
        "drug_components": "Intensive: 2RHZE (R150/H75/Z400/E275 FDC) x 2 Months -> Continuation: 4RH (R150/H75 FDC) x 4 Months",
        "keywords": ["2rhze", "4rh", "ds-tb adult", "rhze"],
    },
    {
        "nascop_ntldp_code": "DS-TB-PED",
        "program_domain": "TB",
        "regimen_acronym": "2RHZ/4RH Dispersible",
        "line_tier": "1st_Line",
        "target_population": "DS-TB Children (<15 yrs)",
        "drug_components": "Intensive: 2RHZ Dispersible (R75/H50/Z150) x 2 Months -> Continuation: 4RH Dispersible (R75/H50) x 4 Months",
        "keywords": ["2rhz", "dispersible rhz", "pediatric tb", "ped tb"],
    },

    # 2. Drug-Resistant TB (DR-TB) All-Oral Short Regimens
    {
        "nascop_ntldp_code": "DR-BPaLM",
        "program_domain": "TB",
        "regimen_acronym": "BPaLM",
        "line_tier": "2nd_Line",
        "target_population": "MDR/RR-TB Fluoroquinolone-Susceptible (≥14 yrs)",
        "drug_components": "Bedaquiline (BDQ) + Pretomanid (Pa) + Linezolid (LZD) + Moxifloxacin (MFX) x 26 Weeks (6 Months)",
        "keywords": ["bpalm", "bedaquiline", "pretomanid", "moxifloxacin"],
    },
    {
        "nascop_ntldp_code": "DR-BPaL",
        "program_domain": "TB",
        "regimen_acronym": "BPaL",
        "line_tier": "2nd_Line",
        "target_population": "Pre-XDR TB Fluoroquinolone-Resistant",
        "drug_components": "Bedaquiline (BDQ) + Pretomanid (Pa) + Linezolid (LZD) x 26 Weeks (6 Months)",
        "keywords": ["bpal", "pre-xdr", "fq resistant"],
    },
    {
        "nascop_ntldp_code": "DR-INDIVIDUALIZED",
        "program_domain": "TB",
        "regimen_acronym": "Longer Oral DR-TB",
        "line_tier": "2nd_Line",
        "target_population": "Contraindicated BPaLM/BPaL / Extensive Resistance",
        "drug_components": "Group A (BDQ, LFX/MFX, LZD) + Group B (CFZ, CS/TRD) + Group C (DLM, Z, E) x 18-20 Months",
        "keywords": ["individualized dr-tb", "longer dr-tb", "group a group b"],
    },

    # 3. TB Preventive Therapy (TPT)
    {
        "nascop_ntldp_code": "TPT-3HP",
        "program_domain": "TPT",
        "regimen_acronym": "3HP",
        "line_tier": "TPT",
        "target_population": "PLHIV / Household Contacts (≥2 yrs)",
        "drug_components": "Rifapentine 300mg + Isoniazid 300mg Co-administered Weekly x 12 Doses (3 Months)",
        "keywords": ["3hp", "rifapentine", "weekly tpt"],
    },
    {
        "nascop_ntldp_code": "TPT-1HP",
        "program_domain": "TPT",
        "regimen_acronym": "1HP",
        "line_tier": "TPT",
        "target_population": "PLHIV / Short Daily TPT",
        "drug_components": "Rifapentine 300mg + Isoniazid 300mg Daily x 28 Doses (1 Month)",
        "keywords": ["1hp", "daily rifapentine"],
    },
    {
        "nascop_ntldp_code": "TPT-3RH",
        "program_domain": "TPT",
        "regimen_acronym": "3RH",
        "line_tier": "TPT",
        "target_population": "Pediatric Contacts (<2 yrs)",
        "drug_components": "Rifampicin 75mg + Isoniazid 50mg Dispersible FDC Daily x 3 Months",
        "keywords": ["3rh", "pediatric tpt"],
    },
    {
        "nascop_ntldp_code": "TPT-6H",
        "program_domain": "TPT",
        "regimen_acronym": "6H",
        "line_tier": "TPT",
        "target_population": "PLHIV / Rifamycin Contraindicated",
        "drug_components": "Isoniazid 300mg Daily + Pyridoxine (Vit B6 25-50mg) x 6 Months",
        "keywords": ["6h", "isoniazid monotherapy", "inh monotherapy"],
    },
    {
        "nascop_ntldp_code": "TPT-6LFX",
        "program_domain": "TPT",
        "regimen_acronym": "6LFX",
        "line_tier": "TPT",
        "target_population": "Household Contacts of MDR-TB Cases",
        "drug_components": "Levofloxacin 500mg Daily x 6 Months",
        "keywords": ["6lfx", "levofloxacin tpt", "mdr tpt"],
    },
]


def seed_master_clinical_regimens_catalog():
    """
    Seed or sync MasterClinicalRegimen database table with official Kenya NASCOP & NTLD-P master catalogs.
    """
    # Seed NASCOP HIV Regimens
    for item in NASCOP_REGIMEN_CATALOG:
        code = item["regimen_code"]
        rec = MasterClinicalRegimen.query.filter_by(nascop_ntldp_code=code).first()
        if not rec:
            rec = MasterClinicalRegimen(
                program_domain="HIV",
                nascop_ntldp_code=code,
                regimen_acronym=item["regimen_name"].split(" ")[0],
                line_tier=item["regimen_line"],
                target_population=item["target_pop"],
                drug_components=item["regimen_name"],
                is_active=True,
            )
            db.session.add(rec)
        else:
            rec.line_tier = item["regimen_line"]
            rec.target_population = item["target_pop"]
            rec.drug_components = item["regimen_name"]

    # Seed NTLD-P TB & TPT Regimens
    for item in NTLDP_TB_TPT_CATALOG:
        code = item["nascop_ntldp_code"]
        rec = MasterClinicalRegimen.query.filter_by(nascop_ntldp_code=code).first()
        if not rec:
            rec = MasterClinicalRegimen(
                program_domain=item["program_domain"],
                nascop_ntldp_code=code,
                regimen_acronym=item["regimen_acronym"],
                line_tier=item["line_tier"],
                target_population=item["target_population"],
                drug_components=item["drug_components"],
                is_active=True,
            )
            db.session.add(rec)
        else:
            rec.program_domain = item["program_domain"]
            rec.regimen_acronym = item["regimen_acronym"]
            rec.line_tier = item["line_tier"]
            rec.target_population = item["target_population"]
            rec.drug_components = item["drug_components"]

    db.session.commit()


def classify_ntldp_regimen(regimen_code: str = None, drugs_text: str = None) -> dict:
    """
    Classify a regimen code or drug description string into NTLD-P TB/TPT master catalog.
    """
    if regimen_code:
        code_upper = regimen_code.strip().upper()
        for item in NTLDP_TB_TPT_CATALOG:
            if item["nascop_ntldp_code"].upper() == code_upper or item["regimen_acronym"].upper() == code_upper:
                return item

    if drugs_text:
        text_lower = drugs_text.lower()
        for item in NTLDP_TB_TPT_CATALOG:
            for kw in item["keywords"]:
                if kw in text_lower:
                    return item

    # Default to DS-TB-ADULT if unknown
    return NTLDP_TB_TPT_CATALOG[0]


def aggregate_ntldp_tb_tpt_monthly(year: int = None, month: int = None) -> dict:
    """
    Aggregate NTLD-P TB & TPT Monthly Reporting Statistics.
    Counts active patients under treatment for DS-TB, DR-TB (BPaLM/BPaL/Individualized), and TPT regimens.
    """
    seed_master_clinical_regimens_catalog()

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

    # Get active TB enrollments
    enrollments = (
        db.session.query(TBEnrollment, Patient, TBRegimen)
        .outerjoin(Patient, TBEnrollment.patient_id == Patient.patient_id)
        .outerjoin(TBRegimen, TBEnrollment.current_regimen_id == TBRegimen.id)
        .filter(TBEnrollment.enrollment_date <= end_dt)
        .all()
    )

    regimen_counts = {item["nascop_ntldp_code"]: {
        "nascop_ntldp_code": item["nascop_ntldp_code"],
        "program_domain": item["program_domain"],
        "regimen_acronym": item["regimen_acronym"],
        "line_tier": item["line_tier"],
        "target_population": item["target_population"],
        "drug_components": item["drug_components"],
        "active_patients_male": 0,
        "active_patients_female": 0,
        "total_active_patients": 0,
        "new_patients_started": 0,
    } for item in NTLDP_TB_TPT_CATALOG}

    total_tb_active = 0
    total_tpt_active = 0

    for enr, patient, regimen in enrollments:
        code = (regimen.regimen_code if regimen else None) or "DS-TB-ADULT"
        matched = classify_ntldp_regimen(code, regimen.drugs if regimen else None)
        mcode = matched["nascop_ntldp_code"]

        is_male = (patient.sex in ["M", "Male"]) if patient else True
        is_new = (enr.treatment_start_date and start_dt <= enr.treatment_start_date <= end_dt) or (start_dt <= enr.enrollment_date <= end_dt)

        reg_data = regimen_counts[mcode]
        if is_male:
            reg_data["active_patients_male"] += 1
        else:
            reg_data["active_patients_female"] += 1

        reg_data["total_active_patients"] += 1

        if matched["program_domain"] == "TB":
            total_tb_active += 1
        else:
            total_tpt_active += 1

        if is_new:
            reg_data["new_patients_started"] += 1

    return {
        "period": period_str,
        "year": year,
        "month": month,
        "total_tb_active_patients": total_tb_active,
        "total_tpt_active_patients": total_tpt_active,
        "regimen_details": sorted(list(regimen_counts.values()), key=lambda x: x["nascop_ntldp_code"]),
    }
