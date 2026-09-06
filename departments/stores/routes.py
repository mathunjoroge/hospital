import os
from datetime import datetime

from flask import (
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload

from departments.models.pharmacy import Batch, Drug, DrugRequest, RequestItem
from departments.models.stock_movement import (
    StockMovement,
    reconcile_stock_balance,
    record_movement,
)
from departments.models.stores import NonPharmCategory, NonPharmItem, OtherOrder
from departments.models.supplier import PurchaseOrder, Supplier
from departments.models.user import User  # Import User model
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

FILE_NAME = os.path.basename(__file__)



@bp.route('/', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'admin')
def index():
    try:
        # Fetch submitted and pending drug requests
        pending_requests = DrugRequest.query.filter(
            DrugRequest.status.in_(['Submitted', 'Pending'])
        ).order_by(DrugRequest.request_date.desc()).all()

        user_ids = [req.requested_by for req in pending_requests]
        users = User.query.filter(User.id.in_(user_ids)).all()
        user_name_map = {user.id: user.username for user in users}

        current_app.logger.debug("[%s -> index()] Found %d pending requests", FILE_NAME, len(pending_requests))
        return render_template('stores/index.html',
                             pending_requests=pending_requests,
                             user_name_map=user_name_map)
    except SQLAlchemyError as e:
        current_app.logger.error("[%s -> index()] Database error: %s", FILE_NAME, str(e), exc_info=True)
        flash("Database error occurred while loading dashboard", 'error')
        return redirect(url_for('stores.index'))
    except Exception as e:
        current_app.logger.error("[%s -> index()] Unexpected error: %s", FILE_NAME, str(e), exc_info=True)
        flash("An unexpected error occurred loading dashboard", 'error')
        return redirect(url_for('stores.index'))


@bp.route('/inventory', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'admin')
def inventory():
    try:
        drugs = Drug.query.order_by(Drug.generic_name).all()
        return render_template('stores/inventory.html', drugs=drugs)
    except SQLAlchemyError as e:
        current_app.logger.error("[%s -> inventory()] Database error: %s", FILE_NAME, str(e), exc_info=True)
        flash("Database error fetching inventory", 'error')
        return redirect(url_for('stores.index'))
    except Exception as e:
        current_app.logger.error("[%s -> inventory()] Error fetching inventory: %s", FILE_NAME, str(e), exc_info=True)
        flash("Unexpected error fetching inventory", 'error')
        return redirect(url_for('stores.index'))


@bp.route('/issue_request', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'admin')
def list_issue_requests():
    try:
        pending_requests = (
            DrugRequest.query
            .filter(DrugRequest.status.in_(['Submitted', 'Pending']))
            .order_by(DrugRequest.request_date.desc())
            .all()
        )
        return render_template(
            'stores/issue_request_list.html',
            pending_requests=pending_requests,
            title="Pending Drug Requests"
        )
    except SQLAlchemyError as e:
        current_app.logger.error("[%s -> list_issue_requests()] Database error: %s", FILE_NAME, str(e), exc_info=True)
        flash("Database error occurred while fetching requests", "error")
        return redirect(url_for('stores.index')), 500
    except Exception as e:
        current_app.logger.error("[%s -> list_issue_requests()] Unexpected error: %s", FILE_NAME, str(e), exc_info=True)
        flash("Unexpected error occurred", "error")
        return redirect(url_for('stores.index')), 500


@bp.route('/issue_request/<int:request_id>', methods=['GET', 'POST'])
@login_required
@roles_required('store', 'stores', 'admin')
def issue_request(request_id):
    try:
        drug_request = (
            db.session.query(DrugRequest)
            .options(
                joinedload(DrugRequest.items)
                .joinedload(RequestItem.drug)
            )
            .filter(DrugRequest.id == request_id)
            .first_or_404(description="Drug request not found")
        )

        if request.method == 'POST':
            try:
                # Process submitted quantities, expiry dates, and batch numbers
                for item in drug_request.items:
                    field_name = f"quantity_issued_{item.id}"
                    quantity_issued = request.form.get(field_name, type=int)
                    if quantity_issued is None or quantity_issued < 0 or quantity_issued > item.quantity_requested:
                        flash(f"Invalid quantity for {item.drug.generic_name}", "error")
                        return render_template(
                            'stores/issue_request.html',
                            drug_request=drug_request,
                            title=f"Issue Request #{request_id}"
                        )

                    item.quantity_issued = quantity_issued

                    if quantity_issued > 0:
                        expiry_date_str = request.form.get(f"expiry_date_{item.id}") or request.form.get("expiry_date")
                        batch_number_str = request.form.get(f"batch_number_{item.id}") or request.form.get("batch_number") or f"REQ-{drug_request.id}"

                        if not expiry_date_str:
                            flash(f"Expiry date (YYYY-MM-DD) is required for issued drug {item.drug.generic_name}", "error")
                            return render_template(
                                'stores/issue_request.html',
                                drug_request=drug_request,
                                title=f"Issue Request #{request_id}"
                            ), 400

                        try:
                            expiry_date = datetime.strptime(expiry_date_str.strip(), "%Y-%m-%d").date()
                        except ValueError:
                            flash(f"Invalid expiry date format for {item.drug.generic_name}. Use YYYY-MM-DD.", "error")
                            return render_template(
                                'stores/issue_request.html',
                                drug_request=drug_request,
                                title=f"Issue Request #{request_id}"
                            ), 400

                        # Deduct from Store stock
                        item.drug.quantity_in_stock = max(0, item.drug.quantity_in_stock - quantity_issued)

                        # Create/update Pharmacy Batch so Pharmacy receives the issued stock with actual expiry date
                        batch = Batch.query.filter_by(drug_id=item.drug_id, batch_number=batch_number_str).first()
                        if not batch:
                            batch = Batch(
                                drug_id=item.drug_id,
                                batch_number=batch_number_str,
                                quantity_in_stock=quantity_issued,
                                expiry_date=expiry_date
                            )
                            db.session.add(batch)
                        else:
                            batch.quantity_in_stock += quantity_issued
                            batch.expiry_date = expiry_date

                        db.session.flush()
                        user_id = current_user.id if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated else None
                        record_movement(
                            item_type='DRUG',
                            item_id=item.drug_id,
                            batch_id=batch.id,
                            movement_type='ISSUED',
                            quantity_delta=-quantity_issued,
                            balance_after=item.drug.quantity_in_stock,
                            reference_type='DRUG_REQUEST',
                            reference_id=str(drug_request.id),
                            user_id=user_id,
                        )


                if all(item.quantity_issued is not None for item in drug_request.items):
                    drug_request.status = 'Completed'
                db.session.commit()
                flash("Drug request issued successfully and stock released to pharmacy", "success")
                return redirect(url_for('stores.list_issue_requests'))

            except SQLAlchemyError as e:
                db.session.rollback()
                current_app.logger.error("[%s -> issue_request()] Database error for request %d: %s", FILE_NAME, request_id, str(e), exc_info=True)
                flash("Error updating request quantities", "error")
                return render_template(
                    'stores/issue_request.html',
                    drug_request=drug_request,
                    title=f"Issue Request #{request_id}"
                ), 500

        return render_template(
            'stores/issue_request.html',
            drug_request=drug_request,
            title=f"Issue Request #{request_id}"
        )

    except SQLAlchemyError as e:
        current_app.logger.error("[%s -> issue_request()] Database query error for request %d: %s", FILE_NAME, request_id, str(e), exc_info=True)
        flash("Database error occurred while fetching request", "error")
        return redirect(url_for('stores.list_issue_requests')), 500
    except Exception as e:
        current_app.logger.error("[%s -> issue_request()] Unexpected error for request %d: %s", FILE_NAME, request_id, str(e), exc_info=True)
        flash("Unexpected error occurred", "error")
        return redirect(url_for('stores.list_issue_requests')), 500


@bp.route('/non_pharms', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'nursing', 'kitchen', 'laundry', 'admin')
def non_pharms():
    try:
        categories = NonPharmCategory.query.order_by(NonPharmCategory.name).all()
        items = NonPharmItem.query.order_by(NonPharmItem.category_id, NonPharmItem.name).all()

        items_by_category = {}
        for item in items:
            category_id = item.category_id
            if category_id not in items_by_category:
                items_by_category[category_id] = []
            items_by_category[category_id].append(item)

        category_name_map = {cat.id: cat.name for cat in categories}
        current_app.logger.debug("[%s -> non_pharms()] Loaded %d non-pharm items across %d categories", FILE_NAME, len(items), len(categories))

        return render_template('stores/non_pharms.html',
                             items_by_category=items_by_category,
                             category_name_map=category_name_map)
    except SQLAlchemyError as e:
        current_app.logger.error("[%s -> non_pharms()] Database error: %s", FILE_NAME, str(e), exc_info=True)
        flash("Database error loading items", 'error')
        return redirect(url_for('stores.index'))
    except Exception as e:
        current_app.logger.error("[%s -> non_pharms()] Error loading items: %s", FILE_NAME, str(e), exc_info=True)
        flash("Error loading non-pharmaceutical items", 'error')
        return redirect(url_for('stores.index'))


@bp.route('/manage_reagent_requests', methods=['GET', 'POST'])
@login_required
@roles_required('store', 'stores', 'admin')
def manage_reagent_requests():
    try:
        if request.method == 'POST':
            request_id = request.form.get('request_id')
            action = request.form.get('action')

            reagent_request = OtherOrder.query.get_or_404(request_id)

            if action == "approve":
                reagent = NonPharmItem.query.get(reagent_request.item_id)
                if reagent:
                    reagent.stock_level = max(0, reagent.stock_level - reagent_request.quantity_requested)
                    reagent.in_dispensing += reagent_request.quantity_requested
                reagent_request.status = "Approved"
                reagent_request.quantity_issued = reagent_request.quantity_requested
                db.session.commit()
                flash(f"Commodity request approved! Stock released for {reagent.name if reagent else 'item'}.", "success")
            else:
                reagent_request.status = "Rejected"
                db.session.commit()
                flash("Commodity request rejected!", "warning")

            return redirect(url_for('stores.manage_reagent_requests'))

        requests = OtherOrder.query.filter_by(status="Pending").options(
            joinedload(OtherOrder.item)
        ).all()

        return render_template('stores/manage_reagent_requests.html', requests=requests)

    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error("[%s -> manage_reagent_requests()] DB error: %s", FILE_NAME, str(e), exc_info=True)
        flash('Database error handling request.', 'error')
        return redirect(url_for('stores.index'))
    except Exception as e:
        db.session.rollback()
        current_app.logger.error("[%s -> manage_reagent_requests()] Unexpected error: %s", FILE_NAME, str(e), exc_info=True)
        flash('Something went wrong. Please try again.', 'error')
        return redirect(url_for('stores.index'))


@bp.route('/bin-card/<string:item_type>/<int:item_id>', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin')
def get_bin_card(item_type, item_id):
    """Phase F — Bin Card ledger view for a specific drug or non-pharm item."""
    item_type_upper = item_type.upper()
    if item_type_upper not in ('DRUG', 'NON_PHARM'):
        return jsonify({'error': 'Invalid item_type. Must be DRUG or NON_PHARM.'}), 400

    movements = StockMovement.query.filter_by(
        item_type=item_type_upper,
        item_id=item_id
    ).order_by(StockMovement.created_at.asc()).all()

    reconciliation = reconcile_stock_balance(item_type_upper, item_id)

    if request.headers.get('Accept') == 'application/json' or request.is_json:
        return jsonify({
            'item_type': item_type_upper,
            'item_id': item_id,
            'movements': [m.to_dict() for m in movements],
            'reconciliation': reconciliation,
        })

    return render_template(
        'stores/bin_card.html',
        item_type=item_type_upper,
        item_id=item_id,
        movements=movements,
        reconciliation=reconciliation
    )


@bp.route('/reconciliation-report', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin')
def get_reconciliation_report():
    """Phase F — Discrepancy report comparing stock balances against append-only ledger sum."""
    drug_ids = [d.id for d in Drug.query.with_entities(Drug.id).all()]
    non_pharm_ids = [n.id for n in NonPharmItem.query.with_entities(NonPharmItem.id).all()]

    report = {
        'drugs': [reconcile_stock_balance('DRUG', did) for did in drug_ids],
        'non_pharm': [reconcile_stock_balance('NON_PHARM', nid) for nid in non_pharm_ids],
    }

    discrepancies = [r for r in report['drugs'] + report['non_pharm'] if not r['match']]
    report['has_discrepancies'] = len(discrepancies) > 0
    report['discrepancy_count'] = len(discrepancies)

    if request.headers.get('Accept') == 'application/json' or request.is_json:
        return jsonify(report)

    return render_template('stores/reconciliation_report.html', report=report)


@bp.route('/purchase-orders', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin', 'Storekeeper', 'Admin', 'Pharmacist')
def list_purchase_orders():
    """List purchase orders for stores view."""
    status_filter = request.args.get('status', '').strip().upper()
    query = PurchaseOrder.query.order_by(PurchaseOrder.created_at.desc())
    if status_filter:
        query = query.filter(PurchaseOrder.status == status_filter)
    purchase_orders = query.all()
    return render_template('stores/po_list.html', purchase_orders=purchase_orders, current_status=status_filter)


@bp.route('/purchase-orders/<int:po_id>/receive', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin', 'Storekeeper', 'Admin', 'Pharmacist')
def view_po_receive(po_id):
    """Render PO shipment receiving form."""
    po = PurchaseOrder.query.get_or_404(po_id)
    return render_template('stores/po_receive.html', po=po)


@bp.route('/receipt/direct', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin', 'Storekeeper', 'Admin', 'Pharmacist')
def view_direct_receipt():
    """Render form to record direct receipt of goods (no PO)."""
    suppliers = Supplier.query.filter_by(is_active=True).order_by(Supplier.name).all()
    drugs = Drug.query.order_by(Drug.generic_name).all()
    non_pharms = NonPharmItem.query.order_by(NonPharmItem.name).all()
    return render_template(
        'stores/direct_receipt.html',
        suppliers=suppliers,
        drugs=drugs,
        non_pharms=non_pharms,
    )


@bp.route('/receipt-history', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'pharmacy', 'admin', 'Storekeeper', 'Admin', 'Pharmacist')
def receipt_history():
    """Render history of received shipments & direct receipts."""
    received_pos = PurchaseOrder.query.filter(
        PurchaseOrder.status.in_(['RECEIVED', 'PARTIALLY_RECEIVED'])
    ).order_by(PurchaseOrder.received_at.desc()).all()
    return render_template('stores/receipt_history.html', received_pos=received_pos)




