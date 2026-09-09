from datetime import datetime, timedelta, timezone

from sqlalchemy import func

from departments.models.billing import PaidBill
from departments.models.insurance import Claim
from departments.models.medicine import AdmittedPatient, Bed, Ward
from extensions import db


def get_bed_occupancy_stats():
    """
    Calculate hospital-wide and per-ward bed occupancy statistics.
    """
    total_beds = (
        db.session.query(func.sum(Ward.number_of_beds)).scalar()
        or Bed.query.count()
        or 0
    )

    # Active admissions (discharged_on is None)
    active_admissions = AdmittedPatient.query.filter(
        AdmittedPatient.discharged_on.is_(None)
    ).all()
    occupied_beds = len(active_admissions)
    available_beds = max(total_beds - occupied_beds, 0)
    occupancy_rate = (
        round((occupied_beds / total_beds) * 100, 1) if total_beds > 0 else 0.0
    )

    # Breakdown per ward
    wards = Ward.query.all()
    ward_breakdown = []
    for ward in wards:
        ward_bed_count = ward.number_of_beds or 0
        ward_occupied = (
            db.session.query(AdmittedPatient)
            .filter(
                AdmittedPatient.ward_id == ward.id,
                AdmittedPatient.discharged_on.is_(None),
            )
            .count()
        )
        ward_avail = max(ward_bed_count - ward_occupied, 0)
        ward_pct = (
            round((ward_occupied / ward_bed_count) * 100, 1)
            if ward_bed_count > 0
            else 0.0
        )

        ward_breakdown.append(
            {
                "ward_id": ward.id,
                "ward_name": ward.name,
                "capacity": ward_bed_count,
                "occupied": ward_occupied,
                "available": ward_avail,
                "occupancy_pct": ward_pct,
            }
        )

    return {
        "total_beds": total_beds,
        "occupied_beds": occupied_beds,
        "available_beds": available_beds,
        "occupancy_rate": occupancy_rate,
        "ward_breakdown": ward_breakdown,
    }


def get_inpatient_admission_trends(days=30):
    """
    Returns daily inpatient admission counts for the past `days` days.
    """
    start_date = datetime.now(timezone.utc).date() - timedelta(days=days - 1)

    # Group by date of admission
    results = (
        db.session.query(
            func.date(AdmittedPatient.admitted_on).label("adm_date"),
            func.count(AdmittedPatient.id).label("cnt"),
        )
        .filter(func.date(AdmittedPatient.admitted_on) >= start_date)
        .group_by(func.date(AdmittedPatient.admitted_on))
        .all()
    )

    date_counts = {str(res.adm_date): res.cnt for res in results}

    # Generate full date series for continuous charting
    trend_series = []
    for i in range(days):
        dt_str = str(start_date + timedelta(days=i))
        trend_series.append(
            {
                "date": dt_str,
                "admissions": date_counts.get(dt_str, 0),
            }
        )

    return trend_series


def get_revenue_summary():
    """
    Summarize total revenue collected by payment channel.
    """
    results = (
        db.session.query(
            PaidBill.payment_method,
            func.sum(PaidBill.amount_paid).label("total_amount"),
        )
        .group_by(PaidBill.payment_method)
        .all()
    )

    by_method = {}
    total_collected = 0.0

    for res in results:
        method_str = str(res.payment_method or "Other").upper()
        amount = float(res.total_amount or 0.0)
        by_method[method_str] = amount
        total_collected += amount

    return {
        "total_collected": round(total_collected, 2),
        "by_method": by_method,
    }


def get_insurance_claims_stats():
    """
    Calculate insurance claims lifecycle status metrics.
    """
    results = (
        db.session.query(Claim.status, func.count(Claim.id).label("cnt"))
        .group_by(Claim.status)
        .all()
    )

    by_status = {}
    total_claims = 0
    approved_claims = 0

    for res in results:
        status_str = str(
            res.status.value if hasattr(res.status, "value") else res.status
        ).upper()
        cnt = res.cnt
        by_status[status_str] = cnt
        total_claims += cnt
        if status_str in ("APPROVED", "PAID"):
            approved_claims += cnt

    approval_rate = (
        round((approved_claims / total_claims) * 100, 1) if total_claims > 0 else 0.0
    )

    return {
        "total_claims": total_claims,
        "approved_claims": approved_claims,
        "approval_rate": approval_rate,
        "by_status": by_status,
    }


def get_executive_kpi_summary():
    """
    Consolidated executive KPI payload.
    """
    bed_stats = get_bed_occupancy_stats()
    admission_trends = get_inpatient_admission_trends(days=30)
    revenue_summary = get_revenue_summary()
    claims_stats = get_insurance_claims_stats()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bed_occupancy": bed_stats,
        "admission_trends": admission_trends,
        "revenue_summary": revenue_summary,
        "claims_stats": claims_stats,
    }
