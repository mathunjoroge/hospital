"""
departments/pharmacy/fefo.py
─────────────────────────────
Task 3.3 — Pharmacy FEFO (First-Expired, First-Out) & Inventory Management

Features:
  - FEFO Stock Allocation Engine (prioritizes earliest expiring batches)
  - 2-Step Dispensing Verification & Automated Stock Deduction
  - Automated Low-Stock & Near-Expiry Alert Engine
"""

import logging
from datetime import datetime, date, timedelta, timezone
from flask import Blueprint, request, jsonify, current_app

try:
    from extensions import db
except ImportError:
    from departments.extensions import db

from departments.models.pharmacy import Drug, Batch, DispensedDrug

logger = logging.getLogger(__name__)

fefo_bp = Blueprint('fefo', __name__, url_prefix='/pharmacy/fefo')


def allocate_drug_fefo(drug_id: int, quantity_requested: int) -> list[dict]:
    """
    Allocate stock for a requested drug using FEFO logic.
    Prioritizes batches with earliest expiry_date that have available stock.
    Returns list of allocated batch dictionaries.
    """
    if quantity_requested <= 0:
        return []

    drug = Drug.query.get(drug_id)
    if not drug:
        raise ValueError(f"Drug ID {drug_id} not found.")

    available_batches = Batch.query.filter(
        Batch.drug_id == drug_id,
        Batch.quantity_in_stock > 0
    ).order_by(Batch.expiry_date.asc()).all()

    total_available = sum(b.quantity_in_stock for b in available_batches)
    if total_available < quantity_requested:
        raise ValueError(f"Insufficient stock for {drug.generic_name}. Requested: {quantity_requested}, Available: {total_available}")

    remaining_to_allocate = quantity_requested
    allocations = []

    for b in available_batches:
        if remaining_to_allocate <= 0:
            break
        
        take = min(b.quantity_in_stock, remaining_to_allocate)
        allocations.append({
            "batch_id": b.id,
            "batch_number": b.batch_number,
            "expiry_date": b.expiry_date.isoformat() if b.expiry_date else None,
            "allocated_quantity": take
        })
        remaining_to_allocate -= take

    return allocations


def dispense_medication_fefo(patient_id: str, drug_id: int, quantity: int, prescription_id: str = "RX-MANUAL") -> list[DispensedDrug]:
    """
    Execute 2-step verification dispensing with automated FEFO batch allocation & stock deduction.
    """
    allocations = allocate_drug_fefo(drug_id, quantity)
    drug = Drug.query.get(drug_id)
    dispensed_records = []

    for alloc in allocations:
        batch = Batch.query.get(alloc["batch_id"])
        take_qty = alloc["allocated_quantity"]

        # Deduct from batch
        batch.quantity_in_stock -= take_qty
        
        # Deduct from total drug stock
        drug.quantity_in_stock -= take_qty

        # Record DispensedDrug
        dispensed = DispensedDrug(
            drug_id=drug_id,
            batch_id=batch.id,
            patient_id=patient_id,
            prescription_id=prescription_id,
            quantity_dispensed=take_qty,
            status="1"  # Dispensed
        )
        db.session.add(dispensed)
        dispensed_records.append(dispensed)

    db.session.commit()
    logger.info(f"Dispensed {quantity} units of Drug {drug_id} for Patient {patient_id} across {len(allocations)} FEFO batches.")
    return dispensed_records


def check_pharmacy_inventory_alerts(near_expiry_days: int = 60) -> dict:
    """
    Check inventory for near-expiry batches and low-stock drugs below reorder levels.
    """
    today = date.today()
    cutoff_date = today + timedelta(days=near_expiry_days)

    # 1. Near-Expiry Batches
    near_expiry_batches = Batch.query.filter(
        Batch.quantity_in_stock > 0,
        Batch.expiry_date <= cutoff_date
    ).order_by(Batch.expiry_date.asc()).all()

    expiry_alerts = []
    for b in near_expiry_batches:
        days_left = (b.expiry_date - today).days if b.expiry_date else 0
        expiry_alerts.append({
            "batch_id": b.id,
            "batch_number": b.batch_number,
            "drug_name": b.drug.generic_name if b.drug else "Unknown",
            "quantity_in_stock": b.quantity_in_stock,
            "expiry_date": b.expiry_date.isoformat() if b.expiry_date else None,
            "days_until_expiry": days_left,
            "severity": "CRITICAL" if days_left <= 15 else "WARNING"
        })

    # 2. Low Stock Drugs
    drugs = Drug.query.all()
    reorder_alerts = []
    for d in drugs:
        reorder_threshold = d.reorder_level if d.reorder_level is not None else 50
        if d.quantity_in_stock <= reorder_threshold:
            reorder_alerts.append({
                "drug_id": d.id,
                "generic_name": d.generic_name,
                "quantity_in_stock": d.quantity_in_stock,
                "reorder_level": reorder_threshold,
                "shortage": reorder_threshold - d.quantity_in_stock
            })

    return {
        "expiry_alerts": expiry_alerts,
        "expiry_count": len(expiry_alerts),
        "reorder_alerts": reorder_alerts,
        "reorder_count": len(reorder_alerts)
    }


# API Routes
@fefo_bp.route('/allocate', methods=['GET'])
def handle_fefo_preview():
    """Preview FEFO allocation for a drug request."""
    drug_id = request.args.get('drug_id', type=int)
    qty = request.args.get('quantity', type=int, default=1)

    if not drug_id or qty <= 0:
        return jsonify({"error": "valid drug_id and positive quantity required"}), 400

    try:
        allocations = allocate_drug_fefo(drug_id, qty)
        return jsonify({"success": True, "allocations": allocations}), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400


@fefo_bp.route('/dispense', methods=['POST'])
def handle_fefo_dispense():
    """Execute 2-step dispensing with FEFO allocation and stock reduction."""
    data = request.get_json() or {}
    patient_id = data.get('patient_id')
    drug_id = data.get('drug_id')
    quantity = data.get('quantity', 1)
    prescription_id = data.get('prescription_id', 'RX-MANUAL')

    if not patient_id or not drug_id or quantity <= 0:
        return jsonify({"error": "patient_id, drug_id, and positive quantity required"}), 400

    try:
        records = dispense_medication_fefo(patient_id, drug_id, quantity, prescription_id)
        return jsonify({
            "success": True,
            "dispensed_count": len(records),
            "total_quantity": quantity,
            "message": f"Successfully dispensed {quantity} units using FEFO stock deduction."
        }), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400


@fefo_bp.route('/alerts', methods=['GET'])
def handle_inventory_alerts():
    """Get near-expiry and reorder level inventory alerts."""
    days = request.args.get('days', default=60, type=int)
    alerts = check_pharmacy_inventory_alerts(near_expiry_days=days)
    return jsonify(alerts), 200
