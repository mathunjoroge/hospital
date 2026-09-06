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
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required

from departments.api.audit import log_audit_event
from departments.models.budget import VoteHead
from departments.models.pharmacy import Batch, Drug
from departments.models.stock_movement import record_movement
from departments.models.stores import NonPharmItem
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
    user_id = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None
    po = PurchaseOrder(
        po_number=po_num,
        supplier_id=supplier.id,
        status='DRAFT',
        created_by_id=user_id,
        notes='Auto-generated from low-stock threshold trigger',
    )
    db.session.add(po)
    db.session.flush()

    for drug in low_drugs:
        reorder_qty = max(50, (drug.reorder_level * 2) - drug.quantity_in_stock)
        unit_cost = float(drug.buying_price) if drug.buying_price else 10.0
        item = PurchaseOrderItem(
            po_id=po.id,
            item_type='DRUG',
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
    """Transition purchase order status from DRAFT to ORDERED with Segregation of Duties check."""
    po = db.session.get(PurchaseOrder, po_id)
    if not po:
        return jsonify({'error': 'Purchase order not found'}), 404

    current_uid = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None

    # Segregation of duties: Creator cannot approve/order their own PO
    if current_uid and po.created_by_id and current_uid == po.created_by_id:
        return jsonify({
            'error': 'Segregation of duties constraint: Creator cannot approve or issue their own Purchase Order.'
        }), 403

    # Phase D — Budget / Vote-Head Validation & Encumbrance
    vote_head_totals = {}
    for item in po.items:
        if item.vote_head_id:
            cost = float(item.quantity_ordered) * float(item.unit_cost)
            vote_head_totals[item.vote_head_id] = vote_head_totals.get(item.vote_head_id, 0.0) + cost

    for vh_id, cost in vote_head_totals.items():
        vh = db.session.get(VoteHead, vh_id)
        if vh:
            if not vh.can_encumber(cost):
                return jsonify({
                    'error': f'Budget vote-head cap exceeded for {vh.code}. Available: {vh.available_amount:.2f}, Required: {cost:.2f}'
                }), 400
            vh.encumber(cost)

    po.status = 'ORDERED'
    po.approved_by_id = current_uid
    po.ordered_at = datetime.now(timezone.utc)
    db.session.commit()

    return jsonify({'message': f'PO {po.po_number} marked as ORDERED', 'purchase_order': po.to_dict()})



@po_bp.route('/pharmacy/po/<int:po_id>/receive', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores')
def receive_po_shipment(po_id):
    """
    Receive shipment for a purchase order.
    Requires explicit itemized receiving data including batch_number and expiry_date.
    No longer fabricates placeholder expiry dates.
    Records StockMovement ledger entries and flags SOD warnings if receiver == approver.
    """
    po = db.session.get(PurchaseOrder, po_id)
    if not po:
        return jsonify({'error': 'Purchase order not found'}), 404

    if po.status == 'RECEIVED':
        return jsonify({'error': 'Purchase order has already been received'}), 400

    data = request.get_json(silent=True) or {}
    items_input = data.get('items', [])

    # Create mapping of item receipts by line item id or drug_id / non_pharm_item_id
    items_map = {}
    if isinstance(items_input, list):
        for item_data in items_input:
            key = item_data.get('item_id') or item_data.get('drug_id') or item_data.get('non_pharm_item_id')
            if key:
                items_map[int(key)] = item_data
    elif isinstance(items_input, dict):
        items_map = {int(k): v for k, v in items_input.items()}

    # If no items provided or missing expiry dates, reject immediately
    if not items_map:
        return jsonify({
            'error': 'Receiving shipment requires explicit item details with valid expiry_date (YYYY-MM-DD) and batch_number.'
        }), 400

    # Validate that every line item being received has an explicit expiry date
    now = datetime.now(timezone.utc)
    for po_item in po.items:
        key = po_item.id if po_item.id in items_map else (po_item.drug_id or po_item.non_pharm_item_id)
        receipt_info = items_map.get(key)

        if not receipt_info or not receipt_info.get('expiry_date'):
            return jsonify({
                'error': f'Explicit expiry_date (YYYY-MM-DD) is required for item {po_item.item_name}.'
            }), 400

        expiry_str = str(receipt_info.get('expiry_date')).strip()
        try:
            exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({
                'error': f'Invalid expiry_date format for {po_item.item_name}. Use YYYY-MM-DD.'
            }), 400

        qty_rcvd = int(receipt_info.get('quantity_received', po_item.quantity_ordered))
        batch_num = receipt_info.get('batch_number') or f"B-PO-{po.id}-{po_item.id}"

        po_item.quantity_received = (po_item.quantity_received or 0) + qty_rcvd
        drug = po_item.drug
        non_pharm = po_item.non_pharm_item

        if drug:
            # Create FEFO batch with real receiving expiry date
            batch = Batch(
                drug_id=drug.id,
                batch_number=batch_num,
                quantity_in_stock=qty_rcvd,
                expiry_date=exp_date,
            )
            db.session.add(batch)
            db.session.flush()
            drug.quantity_in_stock += qty_rcvd

            # Append to immutable StockMovement ledger
            current_uid = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None
            record_movement(
                item_type='DRUG',
                item_id=drug.id,
                batch_id=batch.id,
                movement_type='RECEIVED',
                quantity_delta=qty_rcvd,
                balance_after=drug.quantity_in_stock,
                reference_type='PURCHASE_ORDER',
                reference_id=po.po_number,
                user_id=current_uid,
            )
        elif non_pharm:
            non_pharm.stock_level += qty_rcvd
            current_uid = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None
            record_movement(
                item_type='NON_PHARM',
                item_id=non_pharm.id,
                batch_id=None,
                movement_type='RECEIVED',
                quantity_delta=qty_rcvd,
                balance_after=non_pharm.stock_level,
                reference_type='PURCHASE_ORDER',
                reference_id=po.po_number,
                user_id=current_uid,
            )

    # SOD Audit flag
    current_uid = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None
    po.received_by_id = current_uid
    if po.received_by_id and po.approved_by_id and po.received_by_id == po.approved_by_id:
        po.sod_warning = True

    # Determine status
    all_fulfilled = all(item.quantity_received >= item.quantity_ordered for item in po.items)
    po.status = 'RECEIVED' if all_fulfilled else 'PARTIALLY_RECEIVED'
    po.received_at = now
    db.session.commit()

    log_audit_event(
        action='RECEIVE_PURCHASE_ORDER',
        resource_type='PurchaseOrder',
        resource_id=po.po_number,
        details={'items_count': len(po.items), 'status': po.status, 'sod_warning': po.sod_warning},
    )

    return jsonify({
        'message': f'Shipment for PO {po.po_number} processed successfully.',
        'purchase_order': po.to_dict(),
    })


@po_bp.route('/pharmacy/receipt/direct', methods=['POST'])
@login_required
@roles_required('Admin', 'Pharmacist', 'Storekeeper')
def record_direct_receipt():
    """
    Record direct receipt of supplies from a supplier without a prior Purchase Order.
    Auto-creates a RECEIVED PurchaseOrder record for auditing and inventory updates.
    """
    data = request.get_json() or {}
    supplier_id = data.get('supplier_id')
    items_data = data.get('items', [])
    notes = data.get('notes', 'Direct Receipt (No PO)')

    if not supplier_id or not items_data:
        return jsonify({'error': 'supplier_id and at least one item are required'}), 400

    supplier = Supplier.query.get(supplier_id)
    if not supplier:
        return jsonify({'error': 'Supplier not found'}), 404

    now = datetime.now(timezone.utc)
    po_num = f"PO-DIR-{now.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(2).upper()}"

    current_uid = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None

    po = PurchaseOrder(
        po_number=po_num,
        supplier_id=supplier.id,
        status='RECEIVED',
        created_by_id=current_uid,
        received_by_id=current_uid,
        notes=notes,
        ordered_at=now,
        received_at=now,
    )
    db.session.add(po)
    db.session.flush()

    total_cost = 0.0

    for item in items_data:
        item_type = str(item.get('item_type', 'DRUG')).upper()
        quantity = int(item.get('quantity', 0))
        unit_cost = float(item.get('unit_cost', 0.0))
        batch_num = item.get('batch_number') or f"B-DIR-{po.id}-{secrets.token_hex(2).upper()}"
        expiry_str = item.get('expiry_date')

        if quantity <= 0:
            return jsonify({'error': 'Quantity must be > 0'}), 400

        if not expiry_str:
            return jsonify({'error': 'Expiry date (YYYY-MM-DD) is required for all items'}), 400

        try:
            exp_date = datetime.strptime(str(expiry_str).strip(), "%Y-%m-%d").date()
        except ValueError:
            return jsonify({'error': 'Invalid expiry_date format. Use YYYY-MM-DD'}), 400

        drug_id = item.get('drug_id') if item_type == 'DRUG' else None
        non_pharm_id = item.get('non_pharm_item_id') if item_type == 'NON_PHARM' else None

        po_item = PurchaseOrderItem(
            po_id=po.id,
            item_type=item_type,
            drug_id=drug_id,
            non_pharm_item_id=non_pharm_id,
            quantity_ordered=quantity,
            quantity_received=quantity,
            unit_cost=unit_cost,
        )
        db.session.add(po_item)
        total_cost += quantity * unit_cost

        if item_type == 'DRUG':
            drug = Drug.query.get(drug_id) if drug_id else None
            if not drug:
                return jsonify({'error': f'Drug ID {drug_id} not found'}), 404
            batch = Batch(
                drug_id=drug.id,
                batch_number=batch_num,
                quantity_in_stock=quantity,
                expiry_date=exp_date,
            )
            db.session.add(batch)
            db.session.flush()
            drug.quantity_in_stock += quantity

            record_movement(
                item_type='DRUG',
                item_id=drug.id,
                batch_id=batch.id,
                movement_type='RECEIVED',
                quantity_delta=quantity,
                balance_after=drug.quantity_in_stock,
                reference_type='PURCHASE_ORDER',
                reference_id=po.po_number,
                user_id=current_uid,
            )
        elif item_type == 'NON_PHARM':
            non_pharm = NonPharmItem.query.get(non_pharm_id) if non_pharm_id else None
            if not non_pharm:
                return jsonify({'error': f'Non-pharm item ID {non_pharm_id} not found'}), 404
            non_pharm.stock_level += quantity

            record_movement(
                item_type='NON_PHARM',
                item_id=non_pharm.id,
                batch_id=None,
                movement_type='RECEIVED',
                quantity_delta=quantity,
                balance_after=non_pharm.stock_level,
                reference_type='PURCHASE_ORDER',
                reference_id=po.po_number,
                user_id=current_uid,
            )

    po.total_cost = round(total_cost, 2)
    db.session.commit()

    log_audit_event(
        action='RECORD_DIRECT_RECEIPT',
        resource_type='PurchaseOrder',
        resource_id=po.po_number,
        details={'items_count': len(items_data), 'total_cost': po.total_cost},
    )

    return jsonify({
        'message': f'Direct receipt recorded under PO {po.po_number}.',
        'purchase_order': po.to_dict(),
    }), 201



