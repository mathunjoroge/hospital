from flask import render_template, redirect, url_for, request, flash, jsonify, abort
from flask_login import login_required, current_user
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload
from datetime import datetime
import os

from extensions import db
from . import bp  # Import the blueprint
from departments.models.pharmacy import Drug, DrugRequest, RequestItem
from departments.models.stores import NonPharmCategory, NonPharmItem, OtherOrder
from departments.models.user import User  # Import User model

# Get the filename for error reporting
FILE_NAME = os.path.basename(__file__)

@bp.route('/', methods=['GET'])
@login_required
def index():
    """Store dashboard showing pending drug requests."""
    FILE_NAME = "stores/routes.py"  # Ensure this is defined

    if current_user.role.lower() not in ['store', 'admin']:
        flash('Unauthorized access. Store staff only.', 'error')
        return redirect(url_for('login'))  

    try:
        # Fetch pending drug requests
        pending_requests = DrugRequest.query.filter_by(status='Pending').order_by(DrugRequest.request_date.desc()).all()

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
def inventory():
    """Displays the store's drug inventory."""
    if current_user.role not in ['stores', 'admin']:
        flash('Unauthorized access. Store staff only.', 'error')
        return redirect(url_for('login'))

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
def list_issue_requests():
    """
    Lists all pending drug requests for issuance.
    Accessible only to store staff.
    """
    # Check authorization first
    if current_user.role not in ['stores', 'admin']: 
        abort(403, description="Access restricted to store staff only")

    try:
        pending_requests = (
            DrugRequest.query
            .filter_by(status='Pending')
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
def issue_request(request_id):
    """
    Allows store staff to view and issue drugs for a specific request.
    Handles quantity issuance input and updates.
    """
    if current_user.role not in ['stores', 'admin']:
        flash("Access restricted to store staff only", "error")
        return redirect(url_for('login')), 403

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

                # Update status if all items have quantities set
                if all(item.quantity_issued is not None for item in drug_request.items):
                    drug_request.status = 'Issued'
                db.session.commit()
                flash("Drug request issued successfully", "success")
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
def non_pharms():
    """Display all non-pharmaceutical items grouped by category."""
    FILE_NAME = "stores/routes.py"

    if current_user.role.lower() not in ['store', 'stores', 'nursing', 'kitchen', 'laundry']:
        flash('Unauthorized access. Authorized staff only.', 'error')
        return redirect(url_for('login'))

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
def manage_reagent_requests():
    """Handles approving or rejecting reagent restock requests."""
    if current_user.role != 'admin':
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        if request.method == 'POST':
            request_id = request.form.get('request_id')
            action = request.form.get('action')

            reagent_request = OtherOrder.query.get_or_404(request_id)

            if action == "approve":
                reagent = NonPharmItem.query.get(reagent_request.item_id)
                reagent.stock_level += reagent_request.quantity_requested
                reagent_request.status = "Approved"
                db.session.commit()
                flash(f"Reagent restock approved! Stock updated for {reagent.name}.", "success")
            else:
                db.session.delete(reagent_request)
                db.session.commit()
                flash("Reagent restock request rejected!", "warning")

            return redirect(url_for('stores.manage_reagent_requests'))

        # Fetch all pending restock requests
        requests = OtherOrder.query.filter_by(status="Pending").options(
            joinedload(OtherOrder.item)
        ).all()

        return render_template('stores/manage_reagent_requests.html', requests=requests)

    except Exception as e:
        flash(f"Error managing reagent requests: {e}", "error")
        print(f"Debug: Error in stores.manage_reagent_requests: {e}")
        return redirect(url_for('stores.index'))

