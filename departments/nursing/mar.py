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
from datetime import datetime
from flask import Blueprint, request, jsonify

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.medicine import Ward, AdmittedPatient, WardBedHistory
from departments.models.nursing import MedicationAdmin
from departments.models.billing import Invoice, InvoiceLineItem, InvoiceStatus

logger = logging.getLogger(__name__)

mar_bp = Blueprint('mar', __name__, url_prefix='/nursing/mar')

@mar_bp.route('/occupancy', methods=['GET'])
def get_ward_occupancy():
    """Real-time Ward & Bed Occupancy Dashboard data."""
    wards = Ward.query.all()
    occupancy_data = []
    
    for ward in wards:
        occupancy_data.append({
            "ward_id": ward.id,
            "name": ward.name,
            "total_beds": ward.number_of_beds,
            "occupied_beds": ward.occupied_beds,
            "available_beds": ward.available_beds(),
            "daily_charge": float(ward.daily_charge)
        })
        
    return jsonify({"occupancy": occupancy_data, "count": len(occupancy_data)}), 200


@mar_bp.route('/chart', methods=['POST'])
def chart_medication():
    """MAR charting endpoint for nurses to record medication administration."""
    data = request.get_json() or {}
    patient_id = data.get('patient_id')
    medication = data.get('medication')
    dosage = data.get('dosage')
    nurse_id = data.get('nurse_id')
    
    if not all([patient_id, medication, dosage, nurse_id]):
        return jsonify({"error": "Missing required fields"}), 400
        
    admin_record = MedicationAdmin(
        patient_id=patient_id,
        medication=medication,
        dosage=dosage,
        recorded_by=nurse_id,
        time_administered=datetime.utcnow()
    )
    db.session.add(admin_record)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": "Medication administration recorded successfully.",
        "record_id": admin_record.id
    }), 201


@mar_bp.route('/auto_bill', methods=['POST'])
def trigger_daily_billing():
    """Trigger daily room rate and nursing care auto-billing for admitted patients."""
    admitted = AdmittedPatient.query.filter(AdmittedPatient.discharged_on.is_(None)).all()
    billed_count = 0
    total_amount = 0.0
    
    for admission in admitted:
        ward = Ward.query.get(admission.ward_id)
        if not ward:
            continue
            
        # Check if an invoice already exists for this patient, otherwise create one
        invoice = Invoice.query.filter_by(patient_id=admission.patient_id, status=InvoiceStatus.DRAFT).first()
        if not invoice:
            invoice = Invoice(
                invoice_number=f"INV-{int(datetime.utcnow().timestamp())}-{admission.patient_id[:5]}",
                patient_id=admission.patient_id,
                subtotal=0.0,
                grand_total=0.0,
                amount_paid=0.0,
                balance=0.0,
                status=InvoiceStatus.DRAFT
            )
            db.session.add(invoice)
            db.session.flush() # To get invoice ID
            
        # Add daily ward charge
        line_item = InvoiceLineItem(
            invoice_id=invoice.id,
            description=f"Daily Ward Charge - {ward.name}",
            category='ward',
            quantity=1,
            unit_price=float(ward.daily_charge),
            total=float(ward.daily_charge)
        )
        invoice.subtotal = float(invoice.subtotal) + float(ward.daily_charge)
        invoice.grand_total = float(invoice.grand_total) + float(ward.daily_charge)
        invoice.balance = float(invoice.balance) + float(ward.daily_charge)
        db.session.add(line_item)
        
        billed_count += 1
        total_amount += float(ward.daily_charge)
        
    db.session.commit()
    
    return jsonify({
        "success": True,
        "patients_billed": billed_count,
        "total_amount_billed": total_amount
    }), 200
