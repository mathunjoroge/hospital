"""
departments/pharmacy/po_routes.py
──────────────────────────────────
Phase F — Automated Pharmacy Inventory & Supplier Purchase Orders

Blueprint & Routes:
  - GET/POST /pharmacy/suppliers            — list & create suppliers
  - GET      /pharmacy/low-stock            — scan low stock drugs
  - POST     /pharmacy/po/auto-generate      — auto-generate draft POs for low stock drugs
  - GET      /pharmacy/po/list               — list purchase orders
  - GET      /pharmacy/po/<id>               — inspect purchase order details
  - POST     /pharmacy/po/<id>/order         — mark PO as ORDERED
  - POST     /pharmacy/po/<id>/receive       — receive shipment & auto-create FEFO batches
"""

import secrets
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request
from flask_login import login_required

from departments.api.audit import log_audit_event
from departments.models.pharmacy import Batch, Drug
from departments.models.supplier import PurchaseOrder, PurchaseOrderItem, Supplier
from departments.rbac import roles_required
from extensions import db

po_bp = Blueprint('pharmacy_po', __name__)


@po_bp.route('/pharmacy/suppliers', methods=['GET', 'POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores')
def manage_suppliers():
    """GET lists suppliers; POST creates a new supplier."""
    if request.method == 'POST':
        data = request.get_json() or {}
        name = data.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Supplier name is required'}), 400

        supplier = Supplier(
            name=name,
            contact_email=data.get('contact_email'),
            phone=data.get('phone'),
            address=data.get('address'),
            lead_time_days=int(data.get('lead_time_days', 3)),
        )
        db.session.add(supplier)
        db.session.commit()
        return jsonify({'message': 'Supplier created', 'supplier': supplier.to_dict()}), 201

    suppliers = Supplier.query.filter_by(is_active=True).all()
    return jsonify({'suppliers': [s.to_dict() for s in suppliers]})


@po_bp.route('/pharmacy/low-stock', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin', 'stores')
def low_stock_inventory():
    """Scan pharmacy drugs where quantity_in_stock <= reorder_level."""
    low_drugs = Drug.query.filter(Drug.quantity_in_stock <= Drug.reorder_level).all()
    return jsonify({
        'low_stock_count': len(low_drugs),
        'drugs': [
            {
                'id': d.id,
                'name': d.generic_name,
                'current_stock': d.quantity_in_stock,
                'reorder_level': d.reorder_level,
                'price': float(d.selling_price) if d.selling_price else 0.0,
            }
            for d in low_drugs
        ],
    })


@po_bp.route('/pharmacy/po/auto-generate', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores')
def auto_generate_pos():
    """
    Auto-detect drugs with quantity_in_stock <= reorder_level and create
    draft purchase orders grouped by available suppliers.
    """
    supplier_id = (request.get_json() or {}).get('supplier_id')
    supplier = None
    if supplier_id:
        supplier = db.session.get(Supplier, supplier_id)
    if not supplier:
        supplier = Supplier.query.filter_by(is_active=True).first()

    if not supplier:
        return jsonify({'error': 'No active supplier found. Please create a supplier first.'}), 400

    low_drugs = Drug.query.filter(Drug.quantity_in_stock <= Drug.reorder_level).all()
    if not low_drugs:
        return jsonify({'message': 'No low stock drugs found.', 'created_pos': []})

    po_num = f"PO-{secrets.token_hex(4).upper()}"
    po = PurchaseOrder(
        po_number=po_num,
        supplier_id=supplier.id,
        status='DRAFT',
        notes='Auto-generated from low-stock threshold trigger',
    )
    db.session.add(po)
    db.session.flush()

    for drug in low_drugs:
        reorder_qty = max(50, (drug.reorder_level * 2) - drug.quantity_in_stock)
        unit_cost = float(drug.buying_price) if drug.buying_price else 10.0
        item = PurchaseOrderItem(
            po_id=po.id,
            drug_id=drug.id,
            quantity_ordered=reorder_qty,
            unit_cost=unit_cost,
        )
        db.session.add(item)

    db.session.flush()
    po.recalculate_total()
    db.session.commit()

    log_audit_event(
        action='AUTO_GENERATE_PO',
        resource_type='PurchaseOrder',
        resource_id=po.po_number,
        details={'items_count': len(low_drugs), 'total_cost': float(po.total_cost)},
    )

    return jsonify({
        'message': f'Auto-generated purchase order {po.po_number}',
        'purchase_order': po.to_dict(),
    }), 201


@po_bp.route('/pharmacy/po/list', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin', 'stores')
def list_pos():
    """List purchase orders."""
    pos = PurchaseOrder.query.order_by(PurchaseOrder.created_at.desc()).all()
    return jsonify({'purchase_orders': [p.to_dict() for p in pos]})


@po_bp.route('/pharmacy/po/<int:po_id>', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin', 'stores')
def get_po_details(po_id):
    """Retrieve details for a specific purchase order."""
    po = db.session.get(PurchaseOrder, po_id)
    if not po:
        return jsonify({'error': 'Purchase order not found'}), 404
    return jsonify({'purchase_order': po.to_dict()})


@po_bp.route('/pharmacy/po/<int:po_id>/order', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores')
def submit_po_order(po_id):
    """Transition purchase order status from DRAFT to ORDERED."""
    po = db.session.get(PurchaseOrder, po_id)
    if not po:
        return jsonify({'error': 'Purchase order not found'}), 404

    po.status = 'ORDERED'
    po.ordered_at = datetime.now(timezone.utc)
    db.session.commit()

    return jsonify({'message': f'PO {po.po_number} marked as ORDERED', 'purchase_order': po.to_dict()})


@po_bp.route('/pharmacy/po/<int:po_id>/receive', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores')
def receive_po_shipment(po_id):
    """
    Receive shipment for a purchase order.
    Creates a new FEFO Batch entry for each drug item and updates total stock.
    """
    po = db.session.get(PurchaseOrder, po_id)
    if not po:
        return jsonify({'error': 'Purchase order not found'}), 404

    if po.status == 'RECEIVED':
        return jsonify({'error': 'Purchase order has already been received'}), 400

    now = datetime.now(timezone.utc)
    expiry_date = (now + timedelta(days=365)).date()

    for item in po.items:
        item.quantity_received = item.quantity_ordered
        drug = item.drug

        # Create FEFO batch
        batch_num = f"B-PO-{po.id}-{item.drug_id}"
        batch = Batch(
            drug_id=drug.id,
            batch_number=batch_num,
            quantity_in_stock=item.quantity_received,
            expiry_date=expiry_date,
        )
        db.session.add(batch)

        # Update drug quantity_in_stock
        drug.quantity_in_stock += item.quantity_received

    po.status = 'RECEIVED'
    po.received_at = now
    db.session.commit()

    log_audit_event(
        action='RECEIVE_PURCHASE_ORDER',
        resource_type='PurchaseOrder',
        resource_id=po.po_number,
        details={'items_count': len(po.items)},
    )

    return jsonify({
        'message': f'Shipment for PO {po.po_number} received successfully and inventory updated.',
        'purchase_order': po.to_dict(),
    })
