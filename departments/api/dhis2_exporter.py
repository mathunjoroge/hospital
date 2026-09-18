"""
departments/api/dhis2_exporter.py
───────────────────────────────────
DHIS2 / KHIS Monthly Aggregate Data Exporter for Kenya Ministry of Health
(MOH 705A/B, 711, 731) plus the National Malaria Control Programme (NMCP)
malaria case return, which is submitted through the same MOH 705A/B channel.
Generates standard DHIS2 dataValueSets JSON and downloadable CSV formats.

Blueprint endpoints (registered under /api/khis):
  - GET /api/khis/reports/monthly?year=YYYY&month=MM
  - GET /api/khis/export/dhis2_json?year=YYYY&month=MM
  - GET /api/khis/export/csv?year=YYYY&month=MM

NOTE ON DATA ELEMENT / ORG UNIT IDENTIFIERS:
The `dataElement` and `orgUnit` values below are human-readable internal
codes, not the real DHIS2 UIDs used by the live KHIS instance. Before this
export is submitted to production KHIS, each code must be mapped to its
real UID via KHIS_DATA_ELEMENT_UID_MAP / KHIS_ORG_UNIT_UID (defined just
below) — pulled from the facility's KHIS metadata (Maintenance app or
`/api/dataElements.json` on the KHIS instance). The dataValueSet shape
itself (dataElement, period, orgUnit, categoryOptionCombo, value) already
matches the DHIS2 import spec, so once real UIDs are filled in, the JSON
output can be POSTed directly to KHIS's `/api/dataValueSets` endpoint.
Every export response also carries a `khisReady` flag (JSON) /
`khis_upload_readiness` (internal dict) so the exporter never silently
implies an unmapped export is safe to upload.
"""

import csv
import io
import logging
from calendar import monthrange
from datetime import date, datetime, timezone

from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
)

from departments.api.auth import jwt_or_session_required
from departments.malaria.models import MalariaCase
from departments.mch.models import AncVisit, ImmunizationRecord
from departments.models.laboratory import LabResult
from departments.models.medicine import AdmittedPatient, PrescribedMedicine, SOAPNote
from departments.models.records import Patient
from departments.rbac import roles_required

logger = logging.getLogger(__name__)

khis_bp = Blueprint("khis", __name__)

DEFAULT_ORG_UNIT_ID = "KE_MOH_HOSPITAL_001"

# ---------------------------------------------------------------------------
# KHIS live-instance identifier mapping
# ---------------------------------------------------------------------------
# Real DHIS2 dataElement / orgUnit / categoryOptionCombo UIDs are specific to
# the facility's KHIS account and are NOT known at build time, so they are
# intentionally left blank here rather than guessed. To go live:
#   1. Log into the facility's KHIS instance → Maintenance app → Data Element,
#      (or GET https://hiskenya.org/api/dataElements.json?filter=name:like:<name>
#      with facility API credentials) and look up the UID for each code below.
#   2. Fill in KHIS_DATA_ELEMENT_UID_MAP, KHIS_ORG_UNIT_UID and
#      KHIS_CATEGORY_OPTION_COMBO_UID (the facility's "default" COC UID).
# Until a code is mapped, the export falls back to the internal human-readable
# code so nothing is silently dropped — but `khisReady` will be False and the
# unmapped codes are listed, so this is never accidentally treated as
# submit-ready.
KHIS_DATA_ELEMENT_UID_MAP: dict[str, str] = {
    # "MOH705A_UNDER5_OPD_MALE": "<real-dhis2-uid>",
}
KHIS_ORG_UNIT_UID: str | None = None
KHIS_CATEGORY_OPTION_COMBO_UID: str | None = None


def _resolve_data_element(code: str) -> str:
    """Map an internal data element code to its real KHIS UID if configured."""
    return KHIS_DATA_ELEMENT_UID_MAP.get(code, code)


def _resolve_org_unit(code: str) -> str:
    return KHIS_ORG_UNIT_UID or code


def _resolve_category_option_combo() -> str:
    return KHIS_CATEGORY_OPTION_COMBO_UID or "default"


def _khis_upload_readiness(data_elements: list[dict]) -> dict:
    """
    Report whether this export is mapped to real KHIS UIDs and thus safe to
    POST to the live instance. Codes that lack an entry in
    KHIS_DATA_ELEMENT_UID_MAP are listed so it's obvious what's left to map.
    """
    unmapped = sorted(
        {
            elem["dataElement"]
            for elem in data_elements
            if elem["dataElement"] not in KHIS_DATA_ELEMENT_UID_MAP
        }
    )
    return {
        "ready": not unmapped and bool(KHIS_ORG_UNIT_UID),
        "org_unit_mapped": bool(KHIS_ORG_UNIT_UID),
        "unmapped_data_elements": unmapped,
    }


def calculate_age(dob: date | None) -> int:
    """Calculate age in years from date of birth."""
    if not dob:
        return 25  # default to adult if unknown
    today = datetime.now(timezone.utc).date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def aggregate_monthly_khis_data(year: int, month: int) -> dict:
    """
    Query database models and aggregate monthly stats for MOH reporting.
    Returns structured data dictionary.
    """
    _, last_day = monthrange(year, month)
    start_date = datetime(year, month, 1, 0, 0, 0)  # noqa: DTZ001
    end_date = datetime(year, month, last_day, 23, 59, 59)  # noqa: DTZ001

    # 1. Total Patients Registered in Month
    new_patients_count = Patient.query.filter(
        Patient.date_registered >= start_date, Patient.date_registered <= end_date
    ).count()

    total_patients_count = Patient.query.count()

    # 2. Outpatient Consultations (SOAP Notes)
    soap_notes = SOAPNote.query.filter(
        SOAPNote.created_at >= start_date, SOAPNote.created_at <= end_date
    ).all()

    opd_under_5_male = 0
    opd_under_5_female = 0
    opd_over_5_male = 0
    opd_over_5_female = 0

    diagnosis_counts = {}

    for note in soap_notes:
        patient = Patient.query.filter_by(patient_id=note.patient_id).first()
        age = calculate_age(patient.date_of_birth) if patient else 25
        is_male = (patient.sex in ["M", "Male"]) if patient else True

        if age < 5:
            if is_male:
                opd_under_5_male += 1
            else:
                opd_under_5_female += 1
        else:
            if is_male:
                opd_over_5_male += 1
            else:
                opd_over_5_female += 1

        diag = note.assessment or "Unspecified"
        diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1

    # 3. Admissions & Discharges
    admissions_count = AdmittedPatient.query.filter(
        AdmittedPatient.admitted_on >= start_date,
        AdmittedPatient.admitted_on <= end_date,
    ).count()

    discharges_count = AdmittedPatient.query.filter(
        AdmittedPatient.discharged_on >= start_date,
        AdmittedPatient.discharged_on <= end_date,
    ).count()

    # 4. Lab Tests Performed
    lab_tests_count = LabResult.query.filter(
        LabResult.test_date >= start_date, LabResult.test_date <= end_date
    ).count()

    # 5. Prescriptions Issued (no timestamp column - count all)
    prescriptions_count = PrescribedMedicine.query.count()

    # 6. MCH Antenatal Care (ANC) Visits (MOH 731)
    anc_visits_count = AncVisit.query.filter(
        AncVisit.visit_date >= start_date, AncVisit.visit_date <= end_date
    ).count()

    # 7. Child Immunizations Administered (MOH 710)
    immunizations_count = ImmunizationRecord.query.filter(
        ImmunizationRecord.administered_at >= start_date,
        ImmunizationRecord.administered_at <= end_date,
    ).count()

    # 8. Malaria Cases (MOH 705A/B — National Malaria Control Programme return)
    # Every MalariaCase row is a parasitologically confirmed case (microscopy,
    # RDT or PCR) — the module has no "suspected but not tested"/negative
    # record, so we report confirmed cases and their standard NMCP
    # disaggregations only. We deliberately do NOT compute a test-positivity
    # rate here: doing so from confirmed-only records would always read
    # 100% and misrepresent the real KHIS indicator, which needs a
    # separate "total tested" register this module doesn't capture yet.
    malaria_cases = MalariaCase.query.filter(
        MalariaCase.diagnosis_date >= start_date,
        MalariaCase.diagnosis_date <= end_date,
    ).all()

    malaria_confirmed_microscopy_u5 = 0
    malaria_confirmed_microscopy_o5 = 0
    malaria_confirmed_rdt_u5 = 0
    malaria_confirmed_rdt_o5 = 0
    malaria_confirmed_other_u5 = 0  # PCR / unspecified method
    malaria_confirmed_other_o5 = 0
    malaria_severe_count = 0
    malaria_in_pregnancy_count = 0
    malaria_species_breakdown: dict[str, int] = {}

    for case in malaria_cases:
        patient = Patient.query.filter_by(patient_id=case.patient_id).first()
        age = calculate_age(patient.date_of_birth) if patient else 25
        method = (case.diagnosis_method or "").strip().lower()

        if method == "microscopy":
            if age < 5:
                malaria_confirmed_microscopy_u5 += 1
            else:
                malaria_confirmed_microscopy_o5 += 1
        elif method == "rdt":
            if age < 5:
                malaria_confirmed_rdt_u5 += 1
            else:
                malaria_confirmed_rdt_o5 += 1
        else:
            if age < 5:
                malaria_confirmed_other_u5 += 1
            else:
                malaria_confirmed_other_o5 += 1

        if (case.severity or "").strip().lower() == "severe":
            malaria_severe_count += 1

        pregnancy_status = (case.pregnancy_status or "").strip().lower()
        if pregnancy_status and pregnancy_status not in ("not_pregnant", ""):
            malaria_in_pregnancy_count += 1

        species = (case.malaria_species or "unspecified").strip().lower()
        malaria_species_breakdown[species] = (
            malaria_species_breakdown.get(species, 0) + 1
        )

    malaria_confirmed_total = (
        malaria_confirmed_microscopy_u5
        + malaria_confirmed_microscopy_o5
        + malaria_confirmed_rdt_u5
        + malaria_confirmed_rdt_o5
        + malaria_confirmed_other_u5
        + malaria_confirmed_other_o5
    )

    # Cases where antimalarial treatment was started within the period
    # (NMCP "cases treated" indicator).
    malaria_treated_count = MalariaCase.query.filter(
        MalariaCase.treatment_start_date >= start_date,
        MalariaCase.treatment_start_date <= end_date,
    ).count()

    period_str = f"{year}{month:02d}"

    data_elements = [
        {
            "dataElement": "MOH705A_UNDER5_OPD_MALE",
            "category": "MOH 705A",
            "value": opd_under_5_male,
        },
        {
            "dataElement": "MOH705A_UNDER5_OPD_FEMALE",
            "category": "MOH 705A",
            "value": opd_under_5_female,
        },
        {
            "dataElement": "MOH705B_OVER5_OPD_MALE",
            "category": "MOH 705B",
            "value": opd_over_5_male,
        },
        {
            "dataElement": "MOH705B_OVER5_OPD_FEMALE",
            "category": "MOH 705B",
            "value": opd_over_5_female,
        },
        {
            "dataElement": "MOH711_TOTAL_NEW_PATIENTS",
            "category": "MOH 711",
            "value": new_patients_count,
        },
        {
            "dataElement": "MOH711_TOTAL_ADMISSIONS",
            "category": "MOH 711",
            "value": admissions_count,
        },
        {
            "dataElement": "MOH711_TOTAL_DISCHARGES",
            "category": "MOH 711",
            "value": discharges_count,
        },
        {
            "dataElement": "MOH711_LAB_TESTS_CONDUCTED",
            "category": "MOH 711",
            "value": lab_tests_count,
        },
        {
            "dataElement": "MOH711_PRESCRIPTIONS_ISSUED",
            "category": "MOH 711",
            "value": prescriptions_count,
        },
        {
            "dataElement": "MOH731_ANC_VISITS_TOTAL",
            "category": "MOH 731",
            "value": anc_visits_count,
        },
        {
            "dataElement": "MOH710_IMMUNIZATIONS_ADMINISTERED",
            "category": "MOH 710",
            "value": immunizations_count,
        },
        {
            "dataElement": "MOH705_MALARIA_CONFIRMED_MICROSCOPY_UNDER5",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_confirmed_microscopy_u5,
        },
        {
            "dataElement": "MOH705_MALARIA_CONFIRMED_MICROSCOPY_OVER5",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_confirmed_microscopy_o5,
        },
        {
            "dataElement": "MOH705_MALARIA_CONFIRMED_RDT_UNDER5",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_confirmed_rdt_u5,
        },
        {
            "dataElement": "MOH705_MALARIA_CONFIRMED_RDT_OVER5",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_confirmed_rdt_o5,
        },
        {
            "dataElement": "MOH705_MALARIA_CONFIRMED_OTHER_METHOD_UNDER5",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_confirmed_other_u5,
        },
        {
            "dataElement": "MOH705_MALARIA_CONFIRMED_OTHER_METHOD_OVER5",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_confirmed_other_o5,
        },
        {
            "dataElement": "MOH705_MALARIA_CONFIRMED_TOTAL",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_confirmed_total,
        },
        {
            "dataElement": "MOH705_MALARIA_SEVERE_CASES",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_severe_count,
        },
        {
            "dataElement": "MOH705_MALARIA_CASES_IN_PREGNANCY",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_in_pregnancy_count,
        },
        {
            "dataElement": "MOH705_MALARIA_CASES_TREATED",
            "category": "MOH 705A/B (Malaria)",
            "value": malaria_treated_count,
        },
    ]

    # 9. MOH 645/743 Antimalarial Commodities and Weight Band Disaggregation
    from departments.malaria.moh_645_743 import aggregate_moh743_monthly

    moh743_data = aggregate_moh743_monthly(year, month)

    data_elements.extend(
        [
            {
                "dataElement": "MOH645_AL6_DISPENSED",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["al_6_dispensed"],
            },
            {
                "dataElement": "MOH645_AL12_DISPENSED",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["al_12_dispensed"],
            },
            {
                "dataElement": "MOH645_AL18_DISPENSED",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["al_18_dispensed"],
            },
            {
                "dataElement": "MOH645_AL24_DISPENSED",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["al_24_dispensed"],
            },
            {
                "dataElement": "MOH645_ARTESUNATE_INJ_DISPENSED",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["artesunate_inj_dispensed"],
            },
            {
                "dataElement": "MOH645_QUININE_DISPENSED",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["quinine_dispensed"],
            },
            {
                "dataElement": "MOH645_SP_DISPENSED",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["sp_dispensed"],
            },
            {
                "dataElement": "MOH645_RDTS_USED",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["rdts_used"],
            },
            {
                "dataElement": "MOH645_PATIENTS_TREATED_BY_WBAND_5_14",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["patients_5_14kg"],
            },
            {
                "dataElement": "MOH645_PATIENTS_TREATED_BY_WBAND_15_24",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["patients_15_24kg"],
            },
            {
                "dataElement": "MOH645_PATIENTS_TREATED_BY_WBAND_25_34",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["patients_25_34kg"],
            },
            {
                "dataElement": "MOH645_PATIENTS_TREATED_BY_WBAND_35PLUS",
                "category": "MOH 645/743 (Malaria Commodities)",
                "value": moh743_data["patients_35pluskg"],
            },
        ]
    )

    # 10. MOH 647 Tracer Health Products and Technologies (HPT)
    from departments.pharmacy.moh_647 import aggregate_moh647_monthly

    moh647_data = aggregate_moh647_monthly(year, month)

    data_elements.extend(
        [
            {
                "dataElement": "MOH647_TOTAL_TRACER_ITEMS_MONITORED",
                "category": "MOH 647 (Tracer HPT)",
                "value": moh647_data["total_monitored"],
            },
            {
                "dataElement": "MOH647_TRACER_ITEMS_IN_STOCK",
                "category": "MOH 647 (Tracer HPT)",
                "value": moh647_data["in_stock_count"],
            },
            {
                "dataElement": "MOH647_TRACER_ITEMS_STOCKOUT_COUNT",
                "category": "MOH 647 (Tracer HPT)",
                "value": moh647_data["stockout_count"],
            },
            {
                "dataElement": "MOH647_TRACER_ITEMS_LOW_STOCK_COUNT",
                "category": "MOH 647 (Tracer HPT)",
                "value": moh647_data["low_stock_count"],
            },
        ]
    )

    for item in moh647_data["tracer_items"]:
        code_clean = (
            "MOH647_"
            + item["generic_name"]
            .upper()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            + "_ISSUED"
        )
        data_elements.append(
            {
                "dataElement": code_clean,
                "category": f"MOH 647 ({item['category']})",
                "value": item["issued"],
            }
        )

    # 11. MOH 731 HIV/AIDS Summary (ARV Regimen Patient Counts) & MOH 729B ARV FCDRR
    from departments.hiv_art.moh_731_729b import (
        aggregate_moh729b_fcdrr_monthly,
        aggregate_moh731_arv_monthly,
    )

    moh731_arv_data = aggregate_moh731_arv_monthly(year, month)
    moh729b_fcdrr_data = aggregate_moh729b_fcdrr_monthly(year, month)

    data_elements.extend(
        [
            {
                "dataElement": "MOH731_TX_CURR_TOTAL",
                "category": "MOH 731 (HIV ART)",
                "value": moh731_arv_data["tx_curr_total"],
            },
            {
                "dataElement": "MOH731_TX_NEW_TOTAL",
                "category": "MOH 731 (HIV ART)",
                "value": moh731_arv_data["tx_new_total"],
            },
        ]
    )

    for reg in moh731_arv_data["regimen_details"]:
        data_elements.append(
            {
                "dataElement": f"MOH731_TX_CURR_{reg['regimen_code']}",
                "category": f"MOH 731 (Regimen: {reg['regimen_line']})",
                "value": reg["total_active_patients"],
            }
        )

    for fcdrr in moh729b_fcdrr_data["fcdrr_details"]:
        data_elements.append(
            {
                "dataElement": f"MOH729B_{fcdrr['arv_drug_code']}_DISPENSED",
                "category": "MOH 729B ARV FCDRR",
                "value": fcdrr["quantity_dispensed"],
            }
        )
        data_elements.append(
            {
                "dataElement": f"MOH729B_{fcdrr['arv_drug_code']}_ENDING_STOCK",
                "category": "MOH 729B ARV FCDRR",
                "value": fcdrr["ending_balance"],
            }
        )

    # 12. NTLD-P TB & TPT (TB Preventive Therapy) Master Clinical Regimens
    from departments.tb_dots.ntldp_tb_tpt import aggregate_ntldp_tb_tpt_monthly

    ntldp_tb_data = aggregate_ntldp_tb_tpt_monthly(year, month)

    data_elements.extend(
        [
            {
                "dataElement": "NTLDP_TB_ACTIVE_TOTAL",
                "category": "NTLD-P TB",
                "value": ntldp_tb_data["total_tb_active_patients"],
            },
            {
                "dataElement": "NTLDP_TPT_ACTIVE_TOTAL",
                "category": "NTLD-P TPT",
                "value": ntldp_tb_data["total_tpt_active_patients"],
            },
        ]
    )

    for reg in ntldp_tb_data["regimen_details"]:
        data_elements.append(
            {
                "dataElement": f"NTLDP_{reg['nascop_ntldp_code'].upper().replace('-', '_')}_ACTIVE",
                "category": f"NTLD-P ({reg['program_domain']}: {reg['regimen_acronym']})",
                "value": reg["total_active_patients"],
            }
        )

    # 13. PMTCT & Viral Load Suppression (TX_PVLS)
    from departments.hiv_art.moh_731_729b import aggregate_pmtct_tx_pvls_monthly

    pmtct_pvls_data = aggregate_pmtct_tx_pvls_monthly(year, month)

    data_elements.extend(
        [
            {
                "dataElement": "MOH731_PMTCT_ART_COUNT",
                "category": "MOH 731 PMTCT",
                "value": pmtct_pvls_data["pmtct_art_count"],
            },
            {
                "dataElement": "MOH731_HEI_PROPHYLAXIS_COUNT",
                "category": "MOH 731 PMTCT",
                "value": pmtct_pvls_data["hei_prophylaxis_count"],
            },
            {
                "dataElement": "MOH731_EID_6WK_PCR_COUNT",
                "category": "MOH 731 PMTCT",
                "value": pmtct_pvls_data["eid_6wk_pcr_count"],
            },
            {
                "dataElement": "MOH731_TX_PVLS_ELIGIBLE",
                "category": "MOH 731 (TX_PVLS)",
                "value": pmtct_pvls_data["tx_pvls_eligible"],
            },
            {
                "dataElement": "MOH731_TX_PVLS_TESTED",
                "category": "MOH 731 (TX_PVLS)",
                "value": pmtct_pvls_data["tx_pvls_tested"],
            },
            {
                "dataElement": "MOH731_TX_PVLS_SUPPRESSED",
                "category": "MOH 731 (TX_PVLS)",
                "value": pmtct_pvls_data["tx_pvls_suppressed"],
            },
            {
                "dataElement": "MOH731_TX_PVLS_UNSUPPRESSED",
                "category": "MOH 731 (TX_PVLS)",
                "value": pmtct_pvls_data["tx_pvls_unsuppressed"],
            },
        ]
    )

    return {
        "period": period_str,
        "year": year,
        "month": month,
        "orgUnit": DEFAULT_ORG_UNIT_ID,
        "summary": {
            "new_patients": new_patients_count,
            "total_patients": total_patients_count,
            "opd_under_5": opd_under_5_male + opd_under_5_female,
            "opd_over_5": opd_over_5_male + opd_over_5_female,
            "total_opd_attendances": len(soap_notes),
            "admissions": admissions_count,
            "discharges": discharges_count,
            "lab_tests": lab_tests_count,
            "prescriptions": prescriptions_count,
            "anc_visits": anc_visits_count,
            "immunizations": immunizations_count,
            "malaria_confirmed_total": malaria_confirmed_total,
            "malaria_confirmed_under5": malaria_confirmed_microscopy_u5
            + malaria_confirmed_rdt_u5
            + malaria_confirmed_other_u5,
            "malaria_confirmed_over5": malaria_confirmed_microscopy_o5
            + malaria_confirmed_rdt_o5
            + malaria_confirmed_other_o5,
            "malaria_severe": malaria_severe_count,
            "malaria_in_pregnancy": malaria_in_pregnancy_count,
            "malaria_treated": malaria_treated_count,
            "malaria_pct_of_opd": round(
                (malaria_confirmed_total / len(soap_notes) * 100), 1
            )
            if soap_notes
            else 0.0,
        },
        "malaria_species_breakdown": sorted(
            [{"species": k, "count": v} for k, v in malaria_species_breakdown.items()],
            key=lambda x: x["count"],
            reverse=True,
        ),
        "top_diagnoses": sorted(
            [{"diagnosis": k, "count": v} for k, v in diagnosis_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10],
        "malaria_commodities": moh743_data,
        "tracer_hpt": moh647_data,
        "hiv_arv_regimens": moh731_arv_data,
        "arv_fcdrr": moh729b_fcdrr_data,
        "tb_tpt_regimens": ntldp_tb_data,
        "pmtct_pvls": pmtct_pvls_data,
        "data_elements": data_elements,
        "khis_upload_readiness": _khis_upload_readiness(data_elements),
    }


@khis_bp.route("/export/dhis2_json", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "records", "medicine", "hr", "api")
def export_dhis2_json():
    """Export monthly aggregated values formatted as standard DHIS2 dataValueSets JSON."""
    now = datetime.now(timezone.utc)
    year = request.args.get("year", default=now.year, type=int)
    month = request.args.get("month", default=now.month, type=int)

    aggregated = aggregate_monthly_khis_data(year, month)

    data_values = []
    for elem in aggregated["data_elements"]:
        data_values.append(
            {
                "dataElement": _resolve_data_element(elem["dataElement"]),
                "period": aggregated["period"],
                "orgUnit": _resolve_org_unit(aggregated["orgUnit"]),
                "categoryOptionCombo": _resolve_category_option_combo(),
                "value": str(elem["value"]),
            }
        )

    payload = {
        "dataSet": "MOH_MONTHLY_SUMMARY_V2",
        "completeDate": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "period": aggregated["period"],
        "orgUnit": _resolve_org_unit(aggregated["orgUnit"]),
        "dataValues": data_values,
        # True only once every code below is mapped to a real KHIS UID via
        # KHIS_DATA_ELEMENT_UID_MAP / KHIS_ORG_UNIT_UID — see module header.
        "khisReady": aggregated["khis_upload_readiness"]["ready"],
        "unmappedDataElements": aggregated["khis_upload_readiness"][
            "unmapped_data_elements"
        ],
    }

    return jsonify(payload)


@khis_bp.route("/export/csv", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "records", "medicine", "hr", "api")
def export_dhis2_csv():
    """Export monthly aggregated values formatted as downloadable CSV for KHIS upload."""
    now = datetime.now(timezone.utc)
    year = request.args.get("year", default=now.year, type=int)
    month = request.args.get("month", default=now.month, type=int)

    aggregated = aggregate_monthly_khis_data(year, month)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["dataElement", "period", "orgUnit", "categoryOptionCombo", "value"]
    )

    for elem in aggregated["data_elements"]:
        writer.writerow(
            [
                _resolve_data_element(elem["dataElement"]),
                aggregated["period"],
                _resolve_org_unit(aggregated["orgUnit"]),
                _resolve_category_option_combo(),
                elem["value"],
            ]
        )

    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = (
        f"attachment; filename=khis_dhis2_export_{aggregated['period']}.csv"
    )
    return response


@khis_bp.route("/reports/monthly", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "records", "medicine", "hr", "api")
def khis_monthly_report_dashboard():
    """Render preview dashboard for KHIS / DHIS2 monthly reporting metrics."""
    now = datetime.now(timezone.utc)
    year = request.args.get("year", default=now.year, type=int)
    month = request.args.get("month", default=now.month, type=int)

    aggregated = aggregate_monthly_khis_data(year, month)
    return jsonify(aggregated)
