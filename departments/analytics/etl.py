"""
departments/analytics/etl.py
─────────────────────────────
P8-09 — Analytics read-optimized database ETL pipeline.

Aggregates transactional EHR data (encounters, lab tests, admissions,
revenue, disease program visits) into `DailyKpiSnapshot` records for
read-heavy executive reporting and MoH DHIS2 / KHIS reporting.
"""

import logging
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Optional

from flask import Blueprint, jsonify, request
from flask_login import login_required

from departments.analytics.models import DailyKpiSnapshot
from departments.models.encounter import Encounter
from departments.models.laboratory import LabResult
from departments.rbac import roles_required
from extensions import db

logger = logging.getLogger(__name__)

etl_bp = Blueprint("analytics_etl", __name__)


def run_daily_kpi_etl(target_date: Optional[date] = None) -> DailyKpiSnapshot:
    """
    Execute read-optimized ETL aggregation for specified target date.
    Defaults to today if target_date is None.
    """
    if target_date is None:
        target_date = date.today()

    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=timezone.utc)

    # 1. Outpatient visits (OPD encounters started today)
    total_opd = Encounter.query.filter(
        Encounter.encounter_type == "OPD",
        db.func.date(Encounter.started_at) == target_date,
    ).count()

    # 2. Admissions (IPD encounters started today)
    total_adm = Encounter.query.filter(
        Encounter.encounter_type == "IPD",
        db.func.date(Encounter.started_at) == target_date,
    ).count()

    # 3. Discharges (IPD encounters ended today)
    total_dis = Encounter.query.filter(
        Encounter.encounter_type == "IPD",
        db.func.date(Encounter.ended_at) == target_date,
    ).count()

    # 4. Emergency cases (EMERGENCY encounters started today)
    total_emer = Encounter.query.filter(
        Encounter.encounter_type == "EMERGENCY",
        db.func.date(Encounter.started_at) == target_date,
    ).count()

    # 5. ANC visits (MCH/ANC encounters started today)
    total_anc = Encounter.query.filter(
        Encounter.encounter_type.in_(["MCH", "ANC"]),
        db.func.date(Encounter.started_at) == target_date,
    ).count()

    # 6. Immunizations (Immunization records or encounters today)
    total_imm = Encounter.query.filter(
        Encounter.encounter_type == "IMMUNIZATION",
        db.func.date(Encounter.started_at) == target_date,
    ).count()

    # 7. Lab tests ordered today
    total_labs = LabResult.query.filter(
        db.func.date(LabResult.test_date) == target_date,
    ).count()

    # 8. Revenue collected today
    total_rev = Decimal("0.00")
    try:
        from departments.models.billing import Invoice
        invoices = Invoice.query.filter(
            db.func.date(Invoice.created_at) == target_date,
        ).all()
        for inv in invoices:
            total_rev += Decimal(str(getattr(inv, "amount_paid", 0) or getattr(inv, "total_amount", 0) or 0))
    except Exception as e:
        logger.warning(f"Could not compute revenue for ETL snapshot: {e}")

    # 9. Average Length of Stay (ALOS) for IPD discharges today
    discharged_ipd = Encounter.query.filter(
        Encounter.encounter_type == "IPD",
        db.func.date(Encounter.ended_at) == target_date,
    ).all()

    total_stay_days = 0.0
    if discharged_ipd:
        for enc in discharged_ipd:
            if enc.started_at and enc.ended_at:
                diff = (enc.ended_at - enc.started_at).total_seconds() / 86400.0
                total_stay_days += max(0.5, diff)
        avg_alos = round(total_stay_days / len(discharged_ipd), 2)
    else:
        avg_alos = 3.5  # Standard clinical baseline

    # 10. Bed Occupancy Rate
    active_ipd = Encounter.query.filter(
        Encounter.encounter_type == "IPD",
        db.func.date(Encounter.started_at) <= target_date,
        db.or_(Encounter.ended_at.is_(None), db.func.date(Encounter.ended_at) > target_date),
    ).count()
    total_capacity = 100
    occupancy_rate = min(100.0, round((active_ipd / total_capacity) * 100.0, 1))

    # 11. 30-Day Readmissions
    admitted_today = Encounter.query.filter(
        Encounter.encounter_type == "IPD",
        db.func.date(Encounter.started_at) == target_date,
    ).all()

    readmission_count = 0
    from datetime import timedelta
    for enc in admitted_today:
        thirty_days_prior = target_date - timedelta(days=30)
        prior_discharge = Encounter.query.filter(
            Encounter.patient_id == enc.patient_id,
            Encounter.encounter_type == "IPD",
            db.func.date(Encounter.ended_at) >= thirty_days_prior,
            db.func.date(Encounter.ended_at) < target_date,
        ).first()
        if prior_discharge:
            readmission_count += 1

    # 12. Top Diagnoses & Disease Surveillance
    from collections import Counter

    from departments.models.medicine import SOAPNote

    notes = SOAPNote.query.filter(
        db.func.date(SOAPNote.created_at) == target_date,
    ).all()

    diag_counter = Counter()
    surveillance_counter = {
        "Malaria": 0,
        "Tuberculosis": 0,
        "HIV/ART": 0,
        "Hypertension": 0,
        "Diabetes": 0,
        "Pneumonia": 0,
    }

    for n in notes:
        if n.assessment:
            diag = n.assessment.strip()
            diag_counter[diag] += 1
            for k in surveillance_counter:
                if k.lower() in diag.lower():
                    surveillance_counter[k] += 1

    top_diagnoses = dict(diag_counter.most_common(10))

    # Upsert snapshot record
    snapshot = DailyKpiSnapshot.query.filter_by(snapshot_date=target_date).first()
    if not snapshot:
        snapshot = DailyKpiSnapshot(snapshot_date=target_date)
        db.session.add(snapshot)

    snapshot.total_outpatient_visits = total_opd
    snapshot.total_admissions = total_adm
    snapshot.total_discharges = total_dis
    snapshot.total_emergency_cases = total_emer
    snapshot.total_anc_visits = total_anc
    snapshot.total_immunizations = total_imm
    snapshot.total_lab_tests_ordered = total_labs
    snapshot.total_revenue_collected = total_rev
    snapshot.avg_length_of_stay = avg_alos
    snapshot.bed_occupancy_rate = occupancy_rate
    snapshot.thirty_day_readmission_count = readmission_count
    snapshot.top_diagnoses_json = top_diagnoses
    snapshot.disease_surveillance_json = surveillance_counter

    db.session.commit()
    logger.info(f"Daily KPI Snapshot created/updated for date {target_date}")
    return snapshot


@etl_bp.route("/api/analytics/etl/trigger", methods=["POST"])
@login_required
@roles_required("admin")
def trigger_etl():
    """Manual trigger for daily KPI ETL pipeline."""
    data = request.get_json(silent=True) or {}
    date_str = data.get("date")
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        target_date = date.today()

    snapshot = run_daily_kpi_etl(target_date)
    return jsonify({
        "status": "success",
        "message": f"ETL snapshot created for {target_date.isoformat()}",
        "snapshot": {
            "snapshot_date": snapshot.snapshot_date.isoformat(),
            "outpatient_visits": snapshot.total_outpatient_visits,
            "admissions": snapshot.total_admissions,
            "discharges": snapshot.total_discharges,
            "emergency_cases": snapshot.total_emergency_cases,
            "anc_visits": snapshot.total_anc_visits,
            "immunizations": snapshot.total_immunizations,
            "lab_tests": snapshot.total_lab_tests_ordered,
            "revenue": float(snapshot.total_revenue_collected or 0.0),
        }
    })
