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

from extensions import db
from departments.analytics.models import DailyKpiSnapshot
from departments.models.encounter import Encounter
from departments.models.laboratory import LabResult
from departments.rbac import roles_required


# Phase 7 Disease Program Models
from departments.hiv_art.models import ARTEnrollment
from departments.tb_dots.models import TBEnrollment
from departments.malaria.models import MalariaCase
logger = logging.getLogger(__name__)

etl_bp = Blueprint("analytics_etl", __name__)


def run_daily_kpi_etl(target_date: Optional[date] = None) -> DailyKpiSnapshot:
    """
    Execute read-optimized ETL aggregation for specified target date.
    Defaults to today if target_date is None.
    """
    if target_date is None:
        target_date = date.today()

    start_dt = datetime.combine(target_date, time.min)
    end_dt = datetime.combine(target_date, time.max)

    # 1. Outpatient visits (OPD encounters started today)
    total_opd = Encounter.query.filter(
        Encounter.encounter_type == "OPD",
        Encounter.started_at >= start_dt,
        Encounter.started_at <= end_dt,
    ).count()

    # 2. Admissions (IPD encounters started today)
    total_adm = Encounter.query.filter(
        Encounter.encounter_type == "IPD",
        Encounter.started_at >= start_dt,
        Encounter.started_at <= end_dt,
    ).count()

    # 3. Discharges (IPD encounters ended today)
    total_dis = Encounter.query.filter(
        Encounter.encounter_type == "IPD",
        Encounter.ended_at >= start_dt,
        Encounter.ended_at <= end_dt,
    ).count()

    # 4. Emergency cases (EMERGENCY encounters started today)
    total_emer = Encounter.query.filter(
        Encounter.encounter_type == "EMERGENCY",
        Encounter.started_at >= start_dt,
        Encounter.started_at <= end_dt,
    ).count()

    # 5. ANC visits (MCH/ANC encounters started today)
    total_anc = Encounter.query.filter(
        Encounter.encounter_type.in_(["MCH", "ANC"]),
        Encounter.started_at >= start_dt,
        Encounter.started_at <= end_dt,
    ).count()

    # 6. Immunizations (Immunization records or encounters today)
    total_imm = Encounter.query.filter(
        Encounter.encounter_type == "IMMUNIZATION",
        Encounter.started_at >= start_dt,
        Encounter.started_at <= end_dt,
    ).count()

    # 7. Lab tests ordered today
    total_labs = LabResult.query.filter(
        LabResult.test_date >= start_dt,
        LabResult.test_date <= end_dt,
    ).count()

    # 8. Revenue collected today
    total_rev = Decimal("0.00")
    try:
        from departments.models.billing import Invoice
        invoices = Invoice.query.filter(
            Invoice.created_at >= start_dt,
            Invoice.created_at <= end_dt,
        ).all()
        for inv in invoices:
            total_rev += Decimal(str(getattr(inv, "amount_paid", 0) or getattr(inv, "total_amount", 0) or 0))
    except Exception as e:
        logger.warning(f"Could not compute revenue for ETL snapshot: {e}")

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
