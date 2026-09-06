"""
departments/stores/transfer_routes.py
──────────────────────────────────────
Phase E — Inter-Facility Stock Transfer Routes & Controller

Blueprint & Routes:
  - GET  /stores/transfers/list            — list inter-facility transfer orders
  - GET  /stores/transfers/<id>            — inspect transfer details
  - POST /stores/transfers/create          — create draft inter-facility transfer
  - POST /stores/transfers/<id>/dispatch   — dispatch outbound stock (TRANSFER_OUT)
  - POST /stores/transfers/<id>/receive    — receive inbound stock (TRANSFER_IN)
"""

import secrets
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required

from departments.api.audit import log_audit_event
from departments.models.facility import Facility, get_home_facility
from departments.models.pharmacy import Batch
from departments.models.stock_movement import record_movement
from departments.models.transfer import TransferOrder, TransferOrderItem
from departments.rbac import roles_required
from extensions import db

transfer_bp = Blueprint('stores_transfers', __name__)


@transfer_bp.route('/stores/transfers/list', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin')
def list_transfers():
    """List inter-facility transfer orders."""
    transfers = TransferOrder.query.order_by(TransferOrder.created_at.desc()).all()
    return jsonify({'transfers': [t.to_dict() for t in transfers]})


@transfer_bp.route('/stores/transfers/<int:transfer_id>', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin')
def get_transfer_details(transfer_id):
    """Retrieve details for a specific transfer order."""
    t = db.session.get(TransferOrder, transfer_id)
    if not t:
        return jsonify({'error': 'Transfer order not found'}), 404
    return jsonify({'transfer': t.to_dict()})


@transfer_bp.route('/stores/transfers/create', methods=['POST'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin')
def create_transfer():
    """Create a new draft inter-facility transfer order."""
    data = request.get_json() or {}
    target_facility_id = data.get('target_facility_id')
    if not target_facility_id:
        return jsonify({'error': 'target_facility_id is required'}), 400

    target_facility = db.session.get(Facility, target_facility_id)
    if not target_facility:
        return jsonify({'error': 'Target facility not found'}), 404

    home_facility = get_home_facility()
    user_id = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None

    transfer_num = f"TR-{secrets.token_hex(4).upper()}"
    transfer = TransferOrder(
        transfer_number=transfer_num,
        source_facility_id=home_facility.id,
        target_facility_id=target_facility.id,
        status='DRAFT',
        created_by_id=user_id,
        notes=data.get('notes', 'Inter-facility commodity stock transfer'),
    )
    db.session.add(transfer)
    db.session.flush()

    items_data = data.get('items', [])
    for item_info in items_data:
        item_type = (item_info.get('item_type') or 'DRUG').upper()
        qty = int(item_info.get('quantity_requested', 0))
        if qty <= 0:
            continue

        toi = TransferOrderItem(
            transfer_id=transfer.id,
            item_type=item_type,
            drug_id=item_info.get('drug_id') if item_type == 'DRUG' else None,
            non_pharm_item_id=item_info.get('non_pharm_item_id') if item_type == 'NON_PHARM' else None,
            quantity_requested=qty,
        )
        db.session.add(toi)

    db.session.commit()

    log_audit_event(
        action='CREATE_INTER_FACILITY_TRANSFER',
        resource_type='TransferOrder',
        resource_id=transfer.transfer_number,
        details={'target_facility': target_facility.name, 'items_count': len(transfer.items)},
    )

    return jsonify({
        'message': f'Draft transfer order {transfer.transfer_number} created',
        'transfer': transfer.to_dict(),
    }), 201


@transfer_bp.route('/stores/transfers/<int:transfer_id>/dispatch', methods=['POST'])
@login_required
@roles_required('store', 'stores', 'admin')
def dispatch_transfer(transfer_id):
    """
    Dispatch stock for an inter-facility transfer.
    Deducts stock from local inventory and appends TRANSFER_OUT rows to StockMovement ledger.
    """
    transfer = db.session.get(TransferOrder, transfer_id)
    if not transfer:
        return jsonify({'error': 'Transfer order not found'}), 404

    home_facility = get_home_facility()
    if transfer.source_facility_id != home_facility.id:
        return jsonify({
            'error': 'Facility boundary violation: only the source facility that owns this '
                     'transfer may dispatch it.'
        }), 403

    if transfer.status != 'DRAFT':
        return jsonify({'error': f'Transfer is in {transfer.status} status and cannot be dispatched'}), 400

    data = request.get_json(silent=True) or {}
    items_input = data.get('items', [])

    items_map = {}
    if isinstance(items_input, list):
        for item_data in items_input:
            key = item_data.get('item_id') or item_data.get('drug_id') or item_data.get('non_pharm_item_id')
            if key:
                items_map[int(key)] = item_data

    user_id = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None

    for toi in transfer.items:
        key = toi.id if toi.id in items_map else (toi.drug_id or toi.non_pharm_item_id)
        dispatch_info = items_map.get(key, {})
        qty_dispatched = int(dispatch_info.get('quantity_dispatched', toi.quantity_requested))

        toi.quantity_dispatched = qty_dispatched
        toi.batch_number = dispatch_info.get('batch_number') or f"B-TR-{transfer.id}-{toi.id}"
        expiry_str = dispatch_info.get('expiry_date')
        if expiry_str:
            try:
                toi.expiry_date = datetime.strptime(str(expiry_str).strip(), "%Y-%m-%d").date()
            except ValueError:
                pass

        if toi.item_type == 'DRUG' and toi.drug:
            drug = toi.drug
            drug.quantity_in_stock = max(0, drug.quantity_in_stock - qty_dispatched)

            record_movement(
                item_type='DRUG',
                item_id=drug.id,
                movement_type='TRANSFER_OUT',
                quantity_delta=-qty_dispatched,
                balance_after=drug.quantity_in_stock,
                reference_type='TRANSFER',
                reference_id=transfer.transfer_number,
                user_id=user_id,
                facility_id=transfer.source_facility_id,
                notes=f"Outbound dispatch to facility #{transfer.target_facility_id}",
            )

        elif toi.item_type == 'NON_PHARM' and toi.non_pharm_item:
            item = toi.non_pharm_item
            item.stock_level = max(0, item.stock_level - qty_dispatched)

            record_movement(
                item_type='NON_PHARM',
                item_id=item.id,
                movement_type='TRANSFER_OUT',
                quantity_delta=-qty_dispatched,
                balance_after=item.stock_level,
                reference_type='TRANSFER',
                reference_id=transfer.transfer_number,
                user_id=user_id,
                facility_id=transfer.source_facility_id,
                notes=f"Outbound dispatch to facility #{transfer.target_facility_id}",
            )

    transfer.status = 'DISPATCHED'
    transfer.dispatched_by_id = user_id
    transfer.dispatched_at = datetime.now(timezone.utc)
    db.session.commit()

    log_audit_event(
        action='DISPATCH_INTER_FACILITY_TRANSFER',
        resource_type='TransferOrder',
        resource_id=transfer.transfer_number,
        details={'status': transfer.status},
    )

    return jsonify({
        'message': f'Transfer {transfer.transfer_number} dispatched successfully',
        'transfer': transfer.to_dict(),
    })


@transfer_bp.route('/stores/transfers/<int:transfer_id>/receive', methods=['POST'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin')
def receive_transfer(transfer_id):
    """
    Receive stock for an inbound inter-facility transfer.
    Increments local inventory and appends TRANSFER_IN rows to StockMovement ledger.
    """
    transfer = db.session.get(TransferOrder, transfer_id)
    if not transfer:
        return jsonify({'error': 'Transfer order not found'}), 404

    home_facility = get_home_facility()
    if transfer.target_facility_id != home_facility.id:
        return jsonify({
            'error': 'Facility boundary violation: only the target facility this transfer '
                     'was dispatched to may receive it.'
        }), 403

    if transfer.status == 'RECEIVED':
        return jsonify({'error': 'Transfer has already been received'}), 400

    if transfer.status not in ('DISPATCHED',):
        return jsonify({
            'error': f'Transfer is in {transfer.status} status and cannot be received '
                     '(it must be DISPATCHED first).'
        }), 400

    data = request.get_json(silent=True) or {}
    items_input = data.get('items', [])

    items_map = {}
    if isinstance(items_input, list):
        for item_data in items_input:
            key = item_data.get('item_id') or item_data.get('drug_id') or item_data.get('non_pharm_item_id')
            if key:
                items_map[int(key)] = item_data

    user_id = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None

    discrepancies = []

    for toi in transfer.items:
        key = toi.id if toi.id in items_map else (toi.drug_id or toi.non_pharm_item_id)
        receive_info = items_map.get(key, {})
        qty_rcvd = int(receive_info.get('quantity_received', toi.quantity_dispatched or toi.quantity_requested))

        toi.quantity_received = (toi.quantity_received or 0) + qty_rcvd

        # Phase E.2 — variance detection: flag under- or over-receipt against what was
        # actually dispatched rather than silently accepting whatever quantity is submitted.
        if toi.quantity_received != toi.quantity_dispatched:
            variance = toi.quantity_received - toi.quantity_dispatched
            discrepancies.append({
                'item_id': toi.id,
                'item_name': toi.item_name,
                'quantity_dispatched': toi.quantity_dispatched,
                'quantity_received': toi.quantity_received,
                'variance': variance,
                'kind': 'OVER_RECEIPT' if variance > 0 else 'SHORT_RECEIPT',
            })

        if toi.item_type == 'DRUG' and toi.drug:
            drug = toi.drug
            batch_num = receive_info.get('batch_number') or toi.batch_number or f"B-TR-{transfer.id}-{toi.id}"
            exp_date = toi.expiry_date

            expiry_str = receive_info.get('expiry_date')
            if expiry_str:
                try:
                    exp_date = datetime.strptime(str(expiry_str).strip(), "%Y-%m-%d").date()
                except ValueError:
                    pass

            if exp_date:
                batch = Batch(
                    drug_id=drug.id,
                    batch_number=batch_num,
                    quantity_in_stock=qty_rcvd,
                    expiry_date=exp_date,
                )
                db.session.add(batch)
                db.session.flush()
                batch_id = batch.id
            else:
                batch_id = None

            drug.quantity_in_stock += qty_rcvd

            record_movement(
                item_type='DRUG',
                item_id=drug.id,
                batch_id=batch_id,
                movement_type='TRANSFER_IN',
                quantity_delta=qty_rcvd,
                balance_after=drug.quantity_in_stock,
                reference_type='TRANSFER',
                reference_id=transfer.transfer_number,
                user_id=user_id,
                facility_id=transfer.target_facility_id,
                notes=f"Inbound transfer from facility #{transfer.source_facility_id}",
            )

        elif toi.item_type == 'NON_PHARM' and toi.non_pharm_item:
            item = toi.non_pharm_item
            item.stock_level += qty_rcvd

            record_movement(
                item_type='NON_PHARM',
                item_id=item.id,
                movement_type='TRANSFER_IN',
                quantity_delta=qty_rcvd,
                balance_after=item.stock_level,
                reference_type='TRANSFER',
                reference_id=transfer.transfer_number,
                user_id=user_id,
                facility_id=transfer.target_facility_id,
                notes=f"Inbound transfer from facility #{transfer.source_facility_id}",
            )

    all_fulfilled = all(toi.quantity_received >= toi.quantity_dispatched for toi in transfer.items)
    if not all_fulfilled:
        transfer.status = 'DISPATCHED'
    elif discrepancies:
        transfer.status = 'RECEIVED_WITH_DISCREPANCY'
    else:
        transfer.status = 'RECEIVED'
    transfer.received_by_id = user_id
    transfer.received_at = datetime.now(timezone.utc)
    if discrepancies:
        variance_note = "; ".join(
            f"{d['item_name']}: dispatched {d['quantity_dispatched']}, received "
            f"{d['quantity_received']} ({d['kind']})"
            for d in discrepancies
        )
        transfer.notes = f"{transfer.notes or ''}\n[DISCREPANCY] {variance_note}".strip()
    db.session.commit()

    log_audit_event(
        action='RECEIVE_INTER_FACILITY_TRANSFER',
        resource_type='TransferOrder',
        resource_id=transfer.transfer_number,
        details={'status': transfer.status, 'discrepancies': discrepancies},
    )

    response = {
        'message': f'Inbound transfer {transfer.transfer_number} processed successfully',
        'transfer': transfer.to_dict(),
    }
    if discrepancies:
        response['discrepancies'] = discrepancies
        response['warning'] = (
            'Received quantities differ from dispatched quantities for one or more items — '
            'review before closing this transfer.'
        )
    return jsonify(response)
