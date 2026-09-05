from flask import render_template, redirect, url_for, request, flash, jsonify, abort
from flask_login import login_required, current_user
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload
from datetime import datetime
import os

from extensions import db
from . import bp  # Import the blueprint
from departments.models.pharmacy import Drug, Batch, DrugRequest, RequestItem
from departments.models.stores import NonPharmCategory, NonPharmItem, OtherOrder
from departments.models.user import User  # Import User model

# Get the filename for error reporting
FILE_NAME = os.path.basename(__file__)

from departments.rbac import roles_required

@bp.route('/', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'admin')
def index():  

    try:
        # Fetch submitted and pending drug requests
        pending_requests = DrugRequest.query.filter(
            DrugRequest.status.in_(['Submitted', 'Pending'])
        ).order_by(DrugRequest.request_date.desc()).all()

        # Get user IDs from the requests
        user_ids = [req.requested_by for req in pending_requests]  # List of requested_by IDs
        users = User.query.filter(User.id.in_(user_ids)).all()  # Fetch users in one query
        user_name_map = {user.id: user.username for user in users}  # Map user IDs to names

        # Debugging output
        print(f"Debug: Found {len(pending_requests)} pending requests")
        for req in pending_requests:
            requester_name = user_name_map.get(req.requested_by, 'Unknown')
            print(f"Debug: Request ID {req.id}, Requested by {requester_name}")

        return render_template('stores/index.html', 
                             pending_requests=pending_requests, 
                             user_name_map=user_name_map)
    except Exception as e:
        error_message = f"[{FILE_NAME} -> index()] Error loading dashboard: {e}"
        flash(error_message, 'error')
        print(f"Debug: {error_message}")
        return redirect(url_for('stores.index'))


@bp.route('/inventory', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'admin')
def inventory():

    try:
        drugs = Drug.query.order_by(Drug.generic_name).all()
        return render_template('stores/inventory.html', drugs=drugs)
    except Exception as e:
        error_message = f"[{FILE_NAME} -> inventory()] Error fetching inventory: {e}"
        flash(error_message, 'error')
        print(f"Debug: {error_message}")
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
        flash("Database error occurred while fetching requests", "error")
        print(f"[list_issue_requests] Database error: {str(e)}")
        return redirect(url_for('stores.index')), 500
    except Exception as e:
        flash("Unexpected error occurred", "error")
        print(f"[list_issue_requests] Unexpected error: {str(e)}")
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
                # Process submitted quantities
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
                        # Deduct from Store stock
                        item.drug.quantity_in_stock = max(0, item.drug.quantity_in_stock - quantity_issued)

                        # Create/update Pharmacy Batch so Pharmacy receives the issued stock
                        batch = Batch.query.filter_by(drug_id=item.drug_id, batch_number=f"REQ-{drug_request.id}").first()
                        if not batch:
                            batch = Batch(
                                drug_id=item.drug_id,
                                batch_number=f"REQ-{drug_request.id}",
                                quantity_in_stock=quantity_issued,
                                expiry_date=datetime.today().date().replace(year=datetime.today().year + 2)
                            )
                            db.session.add(batch)
                        else:
                            batch.quantity_in_stock += quantity_issued

                # Update status if all items have quantities set
                if all(item.quantity_issued is not None for item in drug_request.items):
                    drug_request.status = 'Completed'
                db.session.commit()
                flash("Drug request issued successfully and stock released to pharmacy", "success")
                return redirect(url_for('stores.list_issue_requests'))

            except SQLAlchemyError as e:
                db.session.rollback()
                flash("Error updating request quantities", "error")
                print(f"[issue_request] Database error for request {request_id}: {str(e)}")
                return render_template(
                    'stores/issue_request.html',
                    drug_request=drug_request,
                    title=f"Issue Request #{request_id}"
                ), 500

        # GET request - show the form
        return render_template(
            'stores/issue_request.html',
            drug_request=drug_request,
            title=f"Issue Request #{request_id}"
        )

    except SQLAlchemyError as e:
        flash("Database error occurred while fetching request", "error")
        print(f"[issue_request] Database error for request {request_id}: {str(e)}")
        return redirect(url_for('stores.list_issue_requests')), 500
    except Exception as e:
        flash("Unexpected error occurred", "error")
        print(f"[issue_request] Unexpected error for request {request_id}: {str(e)}")
        return redirect(url_for('stores.list_issue_requests')), 500
    
@bp.route('/non_pharms', methods=['GET'])
@login_required
@roles_required('store', 'stores', 'nursing', 'kitchen', 'laundry', 'admin')
def non_pharms():

    try:
        # Fetch all categories and items
        categories = NonPharmCategory.query.order_by(NonPharmCategory.name).all()
        items = NonPharmItem.query.order_by(NonPharmItem.category_id, NonPharmItem.name).all()

        # Group items by category_id
        items_by_category = {}
        for item in items:
            category_id = item.category_id
            if category_id not in items_by_category:
                items_by_category[category_id] = []
            items_by_category[category_id].append(item)

        # Map category IDs to names for easier template use
        category_name_map = {cat.id: cat.name for cat in categories}

        # Debugging
        print(f"Debug: Found {len(items)} non-pharm items across {len(categories)} categories")
        for cat_id, cat_items in items_by_category.items():
            cat_name = category_name_map.get(cat_id, 'Unknown')
            print(f"Debug: Category {cat_name} has {len(cat_items)} items")
            for item in cat_items:
                print(f"  - {item.name}, Unit: {item.unit}, Cost: ${item.unit_cost}, Stock: {item.stock_level}")

        return render_template('stores/non_pharms.html',
                             items_by_category=items_by_category,
                             category_name_map=category_name_map)
    except Exception as e:
        error_message = f"[{FILE_NAME} -> non_pharms()] Error loading items: {e}"
        flash(error_message, 'error')
        print(f"Debug: {error_message}")
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
                    # Deduct from Store stock and issue to requesting department
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

        # Fetch all pending restock requests
        requests = OtherOrder.query.filter_by(status="Pending").options(
            joinedload(OtherOrder.item)
        ).all()

        return render_template('stores/manage_reagent_requests.html', requests=requests)

    except Exception as e:
        db.session.rollback()
        flash(f"Error managing commodity requests: {e}", "error")
        print(f"Debug: Error in stores.manage_reagent_requests: {e}")
        return redirect(url_for('stores.index'))

