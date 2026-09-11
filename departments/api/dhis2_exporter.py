"""
departments/api/dhis2_exporter.py
───────────────────────────────────
DHIS2 / KHIS Monthly Aggregate Data Exporter for Kenya Ministry of Health (MOH 705A/B, 711, 731).
Generates standard DHIS2 dataValueSets JSON and downloadable CSV formats.

Blueprint endpoints (registered under /api/khis):
  - GET /api/khis/reports/monthly?year=YYYY&month=MM
  - GET /api/khis/export/dhis2_json?year=YYYY&month=MM
  - GET /api/khis/export/csv?year=YYYY&month=MM
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
from departments.mch.models import AncVisit, ImmunizationRecord
from departments.models.laboratory import LabResult
from departments.models.medicine import AdmittedPatient, PrescribedMedicine, SOAPNote
from departments.models.records import Patient
from departments.rbac import roles_required

logger = logging.getLogger(__name__)

khis_bp = Blueprint("khis", __name__)

DEFAULT_ORG_UNIT_ID = "KE_MOH_HOSPITAL_001"


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
    ]

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
        },
        "top_diagnoses": sorted(
            [{"diagnosis": k, "count": v} for k, v in diagnosis_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10],
        "data_elements": data_elements,
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
                "dataElement": elem["dataElement"],
                "period": aggregated["period"],
                "orgUnit": aggregated["orgUnit"],
                "value": str(elem["value"]),
            }
        )

    payload = {
        "dataSet": "MOH_MONTHLY_SUMMARY_V2",
        "completeDate": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "period": aggregated["period"],
        "orgUnit": aggregated["orgUnit"],
        "dataValues": data_values,
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
                elem["dataElement"],
                aggregated["period"],
                aggregated["orgUnit"],
                "default",
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
