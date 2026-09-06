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
from departments.models.stock_movement import StockMovement, record_movement
from departments.models.stores import NonPharmItem
from departments.models.supplier import (
    PurchaseOrder,
    PurchaseOrderItem,
    Supplier,
    SupplierReturn,
    SupplierReturnItem,
)
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


@po_bp.route('/pharmacy/po/create', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores', 'Storekeeper', 'Admin', 'Pharmacist')
def create_manual_po():
    """
    Manually create a draft Purchase Order for a specified supplier and line items.
    """
    data = request.get_json() or {}
    supplier_id = data.get('supplier_id')
    items_data = data.get('items', [])
    notes = data.get('notes', 'Manual PO Creation')

    if not supplier_id or not items_data:
        return jsonify({'error': 'supplier_id and at least one item are required'}), 400

    supplier = db.session.get(Supplier, supplier_id)
    if not supplier:
        return jsonify({'error': 'Supplier not found'}), 404

    now = datetime.now(timezone.utc)
    po_num = f"PO-MAN-{now.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(2).upper()}"
    current_uid = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None

    po = PurchaseOrder(
        po_number=po_num,
        supplier_id=supplier.id,
        status='DRAFT',
        created_by_id=current_uid,
        notes=notes,
    )
    db.session.add(po)
    db.session.flush()

    for item in items_data:
        item_type = str(item.get('item_type', 'DRUG')).upper()
        drug_id = item.get('drug_id') if item_type == 'DRUG' else None
        non_pharm_id = item.get('non_pharm_item_id') if item_type == 'NON_PHARM' else None
        qty_ordered = int(item.get('quantity_ordered', 0))
        unit_cost = float(item.get('unit_cost', 0.0))
        vh_id = item.get('vote_head_id')

        if qty_ordered <= 0:
            return jsonify({'error': 'quantity_ordered must be > 0'}), 400

        po_item = PurchaseOrderItem(
            po_id=po.id,
            item_type=item_type,
            drug_id=drug_id,
            non_pharm_item_id=non_pharm_id,
            vote_head_id=vh_id,
            quantity_ordered=qty_ordered,
            unit_cost=unit_cost,
        )
        db.session.add(po_item)

    db.session.commit()
    log_audit_event(
        action='CREATE_MANUAL_PURCHASE_ORDER',
        resource_type='PurchaseOrder',
        resource_id=po.po_number,
        details={'supplier_id': supplier.id, 'items_count': len(items_data)},
    )
    return jsonify({'message': f'Manual PO {po.po_number} created', 'purchase_order': po.to_dict()}), 201


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
    Calculates quantity receiving variances (SHORT_RECEIPT / OVER_RECEIPT).
    """
    po = db.session.get(PurchaseOrder, po_id)
    if not po:
        return jsonify({'error': 'Purchase order not found'}), 404

    if po.status in ('RECEIVED', 'RECEIVED_WITH_DISCREPANCY'):
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
    discrepancies = []

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

        if po_item.quantity_received != po_item.quantity_ordered:
            variance = po_item.quantity_received - po_item.quantity_ordered
            discrepancies.append({
                'item_id': po_item.id,
                'item_name': po_item.item_name,
                'quantity_ordered': po_item.quantity_ordered,
                'quantity_received': po_item.quantity_received,
                'variance': variance,
                'kind': 'OVER_RECEIPT' if variance > 0 else 'SHORT_RECEIPT',
            })

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
    if not all_fulfilled:
        po.status = 'PARTIALLY_RECEIVED'
    elif discrepancies:
        po.status = 'RECEIVED_WITH_DISCREPANCY'
    else:
        po.status = 'RECEIVED'

    po.received_at = now
    if discrepancies:
        variance_note = "; ".join(
            f"{d['item_name']}: ordered {d['quantity_ordered']}, received {d['quantity_received']} ({d['kind']})"
            for d in discrepancies
        )
        po.notes = f"{po.notes or ''}\n[DISCREPANCY] {variance_note}".strip()

    db.session.commit()

    log_audit_event(
        action='RECEIVE_PURCHASE_ORDER',
        resource_type='PurchaseOrder',
        resource_id=po.po_number,
        details={
            'items_count': len(po.items),
            'status': po.status,
            'sod_warning': po.sod_warning,
            'discrepancies': discrepancies,
        },
    )

    response = {
        'message': f'Shipment received for PO {po.po_number}',
        'purchase_order': po.to_dict(),
    }
    if discrepancies:
        response['discrepancies'] = discrepancies
        response['warning'] = 'Quantity variances detected on received PO line items.'
    return jsonify(response)


@po_bp.route('/pharmacy/receipt/direct', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores', 'Storekeeper', 'Admin', 'Pharmacist')
def record_direct_receipt():
    """
    Record direct receipt of supplies from a supplier without a prior Purchase Order.
    Auto-creates a RECEIVED PurchaseOrder record for auditing and inventory updates.
    Supports optional VoteHead budget encumbrance if vote_head_id is supplied.
    """
    data = request.get_json() or {}
    supplier_id = data.get('supplier_id')
    items_data = data.get('items', [])
    notes = data.get('notes', 'Direct Receipt (No PO)')

    if not supplier_id or not items_data:
        return jsonify({'error': 'supplier_id and at least one item are required'}), 400

    supplier = db.session.get(Supplier, supplier_id)
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
        vh_id = item.get('vote_head_id') or data.get('vote_head_id')

        if quantity <= 0:
            return jsonify({'error': 'Quantity must be > 0'}), 400

        if not expiry_str:
            return jsonify({'error': 'Expiry date (YYYY-MM-DD) is required for all items'}), 400

        try:
            exp_date = datetime.strptime(str(expiry_str).strip(), "%Y-%m-%d").date()
        except ValueError:
            return jsonify({'error': 'Invalid expiry_date format. Use YYYY-MM-DD'}), 400

        # Optional VoteHead Budget Encumbrance
        if vh_id:
            vh = db.session.get(VoteHead, vh_id)
            if vh:
                line_cost = quantity * unit_cost
                if not vh.can_encumber(line_cost):
                    return jsonify({
                        'error': f'Budget vote-head cap exceeded for {vh.code}. Available: {vh.available_amount:.2f}, Required: {line_cost:.2f}'
                    }), 400
                vh.encumber(line_cost)

        drug_id = item.get('drug_id') if item_type == 'DRUG' else None
        non_pharm_id = item.get('non_pharm_item_id') if item_type == 'NON_PHARM' else None

        po_item = PurchaseOrderItem(
            po_id=po.id,
            item_type=item_type,
            drug_id=drug_id,
            non_pharm_item_id=non_pharm_id,
            vote_head_id=vh_id,
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


@po_bp.route('/pharmacy/rtv/create', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores', 'Storekeeper', 'Admin', 'Pharmacist')
def create_supplier_return():
    """Create a draft Return to Vendor (RTV) record."""
    data = request.get_json() or {}
    supplier_id = data.get('supplier_id')
    po_id = data.get('po_id')
    reason = data.get('reason', 'Return to Vendor')
    items_data = data.get('items', [])

    if not supplier_id or not items_data:
        return jsonify({'error': 'supplier_id and items are required'}), 400

    supplier = Supplier.query.get(supplier_id)
    if not supplier:
        return jsonify({'error': 'Supplier not found'}), 404

    now = datetime.now(timezone.utc)
    rtv_num = f"RTV-{now.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(2).upper()}"
    current_uid = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None

    rtv = SupplierReturn(
        rtv_number=rtv_num,
        supplier_id=supplier.id,
        po_id=po_id,
        status='DRAFT',
        reason=reason,
        created_by_id=current_uid,
    )
    db.session.add(rtv)
    db.session.flush()

    total_credit = 0.0
    for item in items_data:
        item_type = str(item.get('item_type', 'DRUG')).upper()
        quantity = int(item.get('quantity', 0))
        unit_cost = float(item.get('unit_cost', 0.0))
        batch_num = item.get('batch_number')
        item_reason = item.get('reason', reason)

        if quantity <= 0:
            continue

        rtv_item = SupplierReturnItem(
            supplier_return_id=rtv.id,
            item_type=item_type,
            drug_id=item.get('drug_id') if item_type == 'DRUG' else None,
            non_pharm_item_id=item.get('non_pharm_item_id') if item_type == 'NON_PHARM' else None,
            batch_number=batch_num,
            quantity_returned=quantity,
            unit_cost=unit_cost,
            reason=item_reason,
        )
        db.session.add(rtv_item)
        total_credit += quantity * unit_cost

    rtv.total_credit_amount = round(total_credit, 2)
    db.session.commit()

    log_audit_event(
        action='CREATE_SUPPLIER_RETURN',
        resource_type='SupplierReturn',
        resource_id=rtv.rtv_number,
        details={'supplier_id': supplier_id, 'total_credit': rtv.total_credit_amount},
    )

    return jsonify({
        'message': f'Draft RTV {rtv.rtv_number} created.',
        'supplier_return': rtv.to_dict(),
    }), 201


@po_bp.route('/pharmacy/rtv/<int:rtv_id>/dispatch', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin', 'stores', 'Storekeeper', 'Admin', 'Pharmacist')
def dispatch_supplier_return(rtv_id):
    """Dispatch Return to Vendor (RTV), deduct stock, and record RETURN_TO_VENDOR movements."""
    rtv = db.session.get(SupplierReturn, rtv_id)
    if not rtv:
        return jsonify({'error': 'RTV record not found'}), 404

    if rtv.status != 'DRAFT':
        return jsonify({'error': f'RTV is in {rtv.status} status and cannot be dispatched'}), 400

    current_uid = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None

    for item in rtv.items:
        qty = item.quantity_returned
        if item.item_type == 'DRUG' and item.drug:
            drug = item.drug
            drug.quantity_in_stock = max(0, drug.quantity_in_stock - qty)
            record_movement(
                item_type='DRUG',
                item_id=drug.id,
                movement_type='RETURN_TO_VENDOR',
                quantity_delta=-qty,
                balance_after=drug.quantity_in_stock,
                reference_type='SUPPLIER_RETURN',
                reference_id=rtv.rtv_number,
                user_id=current_uid,
                notes=f"Return to vendor {rtv.supplier.name if rtv.supplier else ''}: {item.reason or rtv.reason}",
            )
        elif item.item_type == 'NON_PHARM' and item.non_pharm_item:
            np = item.non_pharm_item
            np.stock_level = max(0, np.stock_level - qty)
            record_movement(
                item_type='NON_PHARM',
                item_id=np.id,
                movement_type='RETURN_TO_VENDOR',
                quantity_delta=-qty,
                balance_after=np.stock_level,
                reference_type='SUPPLIER_RETURN',
                reference_id=rtv.rtv_number,
                user_id=current_uid,
                notes=f"Return to vendor {rtv.supplier.name if rtv.supplier else ''}: {item.reason or rtv.reason}",
            )

    rtv.status = 'DISPATCHED'
    rtv.dispatched_at = datetime.now(timezone.utc)
    db.session.commit()

    log_audit_event(
        action='DISPATCH_SUPPLIER_RETURN',
        resource_type='SupplierReturn',
        resource_id=rtv.rtv_number,
        details={'status': rtv.status},
    )

    return jsonify({
        'message': f'RTV {rtv.rtv_number} dispatched to vendor.',
        'supplier_return': rtv.to_dict(),
    })


@po_bp.route('/pharmacy/suppliers/<int:supplier_id>/metrics', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin', 'stores', 'Storekeeper', 'Admin', 'Pharmacist')
def get_supplier_otif_metrics(supplier_id):
    """Calculate Supplier On-Time In-Full (OTIF) performance scorecards."""
    supplier = db.session.get(Supplier, supplier_id)
    if not supplier:
        return jsonify({'error': 'Supplier not found'}), 404

    pos = PurchaseOrder.query.filter_by(supplier_id=supplier.id).all()

    total_orders = len(pos)
    received_pos = [po for po in pos if po.status == 'RECEIVED' and po.ordered_at and po.received_at]

    lead_times = []
    on_time_count = 0
    total_ordered_items = 0
    total_received_items = 0

    for po in received_pos:
        days = (po.received_at - po.ordered_at).days
        lead_times.append(days)
        if days <= supplier.lead_time_days:
            on_time_count += 1

        for item in po.items:
            total_ordered_items += item.quantity_ordered
            total_received_items += item.quantity_received

    avg_actual_lead_time = round(sum(lead_times) / len(lead_times), 1) if lead_times else float(supplier.lead_time_days)
    on_time_rate = round((on_time_count / len(received_pos)) * 100, 1) if received_pos else 100.0
    fill_rate = round((total_received_items / total_ordered_items) * 100, 1) if total_ordered_items > 0 else 100.0

    return jsonify({
        'supplier_id': supplier.id,
        'supplier_name': supplier.name,
        'promised_lead_time_days': supplier.lead_time_days,
        'actual_avg_lead_time_days': avg_actual_lead_time,
        'total_orders': total_orders,
        'received_orders': len(received_pos),
        'on_time_delivery_rate': on_time_rate,
        'fill_rate_percentage': fill_rate,
        'total_spend': round(sum(float(po.total_cost) for po in pos), 2),
    })


@po_bp.route('/pharmacy/smart-reorder', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin', 'stores', 'Storekeeper', 'Admin', 'Pharmacist')
def calculate_smart_reorder():
    """Calculate 30-day Average Daily Consumption (ADC) and dynamic Reorder Points (ROP)."""
    import math
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)

    # Fetch 30-day outward movements
    movements = StockMovement.query.filter(
        StockMovement.created_at >= cutoff,
        StockMovement.movement_type.in_(['DISPENSED', 'ISSUED', 'TRANSFER_OUT'])
    ).all()

    drug_consumption = {}
    non_pharm_consumption = {}

    for m in movements:
        qty = abs(m.quantity_delta)
        if m.item_type == 'DRUG':
            drug_consumption[m.item_id] = drug_consumption.get(m.item_id, 0) + qty
        elif m.item_type == 'NON_PHARM':
            non_pharm_consumption[m.item_id] = non_pharm_consumption.get(m.item_id, 0) + qty

    reorder_proposals = []

    # Evaluate Drugs
    drugs = Drug.query.all()
    for d in drugs:
        consumed_30d = drug_consumption.get(d.id, 0)
        adc = round(consumed_30d / 30.0, 2)
        lead_days = 3
        safety_stock = math.ceil(adc * 2)
        dynamic_rop = math.ceil(adc * lead_days) + safety_stock

        current_stock = d.quantity_in_stock
        effective_rop = max(d.reorder_level or 0, dynamic_rop)

        needs_reorder = current_stock <= effective_rop
        suggested_qty = max(50, effective_rop * 2) if needs_reorder else 0

        reorder_proposals.append({
            'item_type': 'DRUG',
            'id': d.id,
            'name': d.generic_name,
            'current_stock': current_stock,
            'static_reorder_level': d.reorder_level or 0,
            'adc': adc,
            'dynamic_rop': dynamic_rop,
            'effective_rop': effective_rop,
            'storage_condition': d.storage_condition,
            'needs_reorder': needs_reorder,
            'suggested_reorder_qty': suggested_qty,
        })

    # Evaluate NonPharmItems
    non_pharms = NonPharmItem.query.all()
    for np in non_pharms:
        consumed_30d = non_pharm_consumption.get(np.id, 0)
        adc = round(consumed_30d / 30.0, 2)
        lead_days = 3
        safety_stock = math.ceil(adc * 2)
        dynamic_rop = math.ceil(adc * lead_days) + safety_stock

        current_stock = np.stock_level
        needs_reorder = current_stock <= dynamic_rop
        suggested_qty = max(20, dynamic_rop * 2) if needs_reorder else 0

        reorder_proposals.append({
            'item_type': 'NON_PHARM',
            'id': np.id,
            'name': np.name,
            'current_stock': current_stock,
            'static_reorder_level': 0,
            'adc': adc,
            'dynamic_rop': dynamic_rop,
            'effective_rop': dynamic_rop,
            'storage_condition': np.storage_condition,
            'needs_reorder': needs_reorder,
            'suggested_reorder_qty': suggested_qty,
        })

    items_to_reorder = [p for p in reorder_proposals if p['needs_reorder']]

    return jsonify({
        'total_items_analyzed': len(reorder_proposals),
        'items_needing_reorder_count': len(items_to_reorder),
        'proposals': reorder_proposals,
    })




