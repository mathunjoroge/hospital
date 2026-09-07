"""
Billing reconciliation endpoints - compare legacy billing totals with unified Invoice system.
"""

from datetime import datetime, timedelta

from flask import Blueprint, jsonify, render_template, request
from flask_login import login_required
from sqlalchemy import func

from departments.models.billing import (
    Billing,
    ClinicBill,
    DrugsBill,
    ImagingBill,
    Invoice,
    InvoiceLineItem,
    LabBill,
    TheatreBill,
    WardBill,
)
from extensions import db

reconciliation_bp = Blueprint(
    "reconciliation", __name__, url_prefix="/admin/billing-reconciliation"
)


@reconciliation_bp.route("/")
@login_required
def reconciliation_dashboard():
    """Dashboard showing billing sync status and discrepancies."""
    # Get date range from query params (default: last 30 days)
    days = request.args.get("days", 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)

    # Calculate legacy billing totals by category
    legacy_totals = {}

    # Billing (consultations)
    legacy_totals["consult"] = (
        db.session.query(func.coalesce(func.sum(Billing.amount), 0))
        .filter(Billing.created_at >= start_date)
        .scalar()
    )

    # DrugsBill
    legacy_totals["drug"] = (
        db.session.query(func.coalesce(func.sum(DrugsBill.amount), 0))
        .filter(DrugsBill.created_at >= start_date)
        .scalar()
    )

    # LabBill
    legacy_totals["lab"] = (
        db.session.query(func.coalesce(func.sum(LabBill.amount), 0))
        .filter(LabBill.created_at >= start_date)
        .scalar()
    )

    # ClinicBill
    legacy_totals["clinic"] = (
        db.session.query(func.coalesce(func.sum(ClinicBill.amount), 0))
        .filter(ClinicBill.created_at >= start_date)
        .scalar()
    )

    # TheatreBill
    legacy_totals["theatre"] = (
        db.session.query(func.coalesce(func.sum(TheatreBill.amount), 0))
        .filter(TheatreBill.created_at >= start_date)
        .scalar()
    )

    # WardBill
    legacy_totals["ward"] = (
        db.session.query(func.coalesce(func.sum(WardBill.amount), 0))
        .filter(WardBill.created_at >= start_date)
        .scalar()
    )

    # ImagingBill
    legacy_totals["imaging"] = (
        db.session.query(func.coalesce(func.sum(ImagingBill.amount), 0))
        .filter(ImagingBill.created_at >= start_date)
        .scalar()
    )

    # Calculate unified Invoice totals by category
    invoice_totals = {}
    categories = ["consult", "drug", "lab", "clinic", "theatre", "ward", "imaging"]

    for category in categories:
        invoice_totals[category] = (
            db.session.query(func.coalesce(func.sum(InvoiceLineItem.total_price), 0))
            .join(Invoice)
            .filter(
                InvoiceLineItem.category == category,
                InvoiceLineItem.created_at >= start_date,
            )
            .scalar()
        )

    # Calculate discrepancies
    discrepancies = {}
    for category in categories:
        legacy = float(legacy_totals.get(category, 0))
        unified = float(invoice_totals.get(category, 0))
        discrepancies[category] = {
            "legacy": legacy,
            "unified": unified,
            "difference": legacy - unified,
            "status": "✅ Synced" if abs(legacy - unified) < 0.01 else "⚠️ Discrepancy",
        }

    # Summary stats
    total_legacy = sum(legacy_totals.values())
    total_unified = sum(invoice_totals.values())
    total_discrepancy = total_legacy - total_unified

    return render_template(
        "billing/reconciliation.html",
        discrepancies=discrepancies,
        total_legacy=total_legacy,
        total_unified=total_unified,
        total_discrepancy=total_discrepancy,
        days=days,
        start_date=start_date,
    )


@reconciliation_bp.route("/api/status")
@login_required
def sync_status():
    """API endpoint returning JSON status of billing sync."""
    from departments.billing.sync import check_billing_sync_enabled

    enabled = check_billing_sync_enabled()

    # Count synced vs unsynced items
    synced_invoices = Invoice.query.count()
    synced_line_items = InvoiceLineItem.query.count()

    return jsonify(
        {
            "sync_enabled": enabled,
            "synced_invoices": synced_invoices,
            "synced_line_items": synced_line_items,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
