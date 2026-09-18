"""
departments/nursing/mar.py
───────────────────────────
Task 3.6 — Inpatient ADT & Medication Administration Record (MAR)
Features:
  - Ward & Bed Occupancy Dashboard endpoints (ADT workflow)
  - Medication Administration Record (MAR) charting for nurses
  - Daily room rate & nursing care auto-billing engine
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal

from flask import Blueprint, jsonify, request
from flask_login import login_required

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.billing import InvoiceLineItem
from departments.models.medicine import AdmittedPatient, Ward
from departments.models.nursing import MedicationAdmin
from departments.rbac import roles_required

logger = logging.getLogger(__name__)

mar_bp = Blueprint("mar", __name__, url_prefix="/nursing/mar")


@mar_bp.route("/occupancy", methods=["GET"])
@login_required
@roles_required("nursing", "admin", "doctor")
def get_ward_occupancy():
    """Real-time Ward & Bed Occupancy Dashboard data."""
    wards = Ward.query.all()
    occupancy_data = []

    for ward in wards:
        occupancy_data.append(
            {
                "ward_id": ward.id,
                "name": ward.name,
                "total_beds": ward.number_of_beds,
                "occupied_beds": ward.occupied_beds,
                "available_beds": ward.available_beds(),
                "daily_charge": float(ward.daily_charge),
            }
        )

    return jsonify({"occupancy": occupancy_data, "count": len(occupancy_data)}), 200


@mar_bp.route("/chart", methods=["POST"])
@login_required
@roles_required("nursing", "nurse", "admin")
def chart_medication():
    """MAR charting endpoint for nurses to record medication administration.

    nurse_id is always derived from the authenticated session — never accepted
    from the request body — to prevent identity spoofing.
    """
    from flask_login import current_user

    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    medication = data.get("medication")
    dosage = data.get("dosage")

    # P0-FIX: nurse_id comes from the authenticated session, not the request body.
    # Accepting nurse_id from the caller would allow any user to forge another
    # nurse's identity in the medication administration record.
    nurse_id = current_user.id

    if not all([patient_id, medication, dosage]):
        return jsonify(
            {"error": "Missing required fields: patient_id, medication, dosage"}
        ), 400

    admin_record = MedicationAdmin(
        patient_id=patient_id,
        medication=medication,
        dosage=dosage,
        recorded_by=nurse_id,
        time_administered=datetime.now(timezone.utc),
    )
    db.session.add(admin_record)
    db.session.commit()

    logger.info(
        "MAR chart: nurse=%s recorded %s %s for patient=%s",
        nurse_id,
        medication,
        dosage,
        patient_id,
    )

    return jsonify(
        {
            "success": True,
            "message": "Medication administration recorded successfully.",
            "record_id": admin_record.id,
        }
    ), 201


@mar_bp.route("/auto_bill", methods=["POST"])
@login_required
@roles_required("admin", "billing")
def trigger_daily_billing():
    """Trigger daily room rate and nursing care auto-billing for admitted patients.

    Restricted to admin/billing roles — this endpoint modifies every active
    inpatient invoice and must not be callable by unauthenticated requests or
    fired multiple times inadvertently.
    """
    admitted = AdmittedPatient.query.filter(
        AdmittedPatient.discharged_on.is_(None)
    ).all()
    billed_count = 0
    total_amount = Decimal(0)

    for admission in admitted:
        ward = db.session.get(Ward, admission.ward_id)
        if not ward:
            continue

        from departments.billing.sync import get_or_create_open_invoice

        invoice = get_or_create_open_invoice(admission.patient_id)

        daily_charge = ward.daily_charge

        # Add daily ward charge line item
        line_item = InvoiceLineItem(
            invoice_id=invoice.id,
            description=f"Daily Ward Charge - {ward.name}",
            category="ward",
            quantity=1,
            unit_price=daily_charge,
            total=daily_charge,
        )
        # Use Decimal arithmetic — never float() — on Numeric invoice fields
        invoice.subtotal = Decimal(str(invoice.subtotal or 0)) + daily_charge
        invoice.grand_total = Decimal(str(invoice.grand_total or 0)) + daily_charge
        invoice.balance = Decimal(str(invoice.balance or 0)) + daily_charge
        db.session.add(line_item)

        billed_count += 1
        total_amount += daily_charge

    db.session.commit()

    logger.info("auto_bill: billed %d patients, total=%s", billed_count, total_amount)

    return jsonify(
        {
            "success": True,
            "patients_billed": billed_count,
            "total_amount_billed": str(total_amount),
        }
    ), 200
