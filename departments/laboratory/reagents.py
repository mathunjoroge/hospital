from datetime import date, datetime

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from flask_socketio import SocketIO
from sqlalchemy.orm import joinedload

from departments.models.stores import NonPharmItem, OtherOrder
from departments.models.user import User
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

socketio = SocketIO()
 # Generate a UUID and convert it to a string

# Display the lab waiting list

    #order for reagets
@bp.route('/reagents_order', methods=['GET', 'POST'])
@login_required
@roles_required('laboratory', 'admin')
def reagents_order():
    """Handles ordering of lab reagents."""

    try:
        if request.method == 'POST':
            # Extract form data
            item_id = request.form.get('item_id', type=int)
            quantity = request.form.get('quantity', type=int)
            notes = request.form.get('notes', '')

            # Validation: Ensure item selection and valid quantity
            if not item_id or not quantity or quantity <= 0:
                flash('Please select a valid reagent and enter a quantity greater than zero.', 'error')
                return redirect(url_for('laboratory.reagents_order'))

            # Fetch the reagent from the inventory (Category ID = 6 for Lab Reagents)
            reagent = NonPharmItem.query.filter_by(id=item_id, category_id=6).first()

            if not reagent:
                flash('Selected item is not a valid lab reagent.', 'error')
                return redirect(url_for('laboratory.reagents_order'))

            # Check stock availability
            if reagent.stock_level < quantity:
                flash(f'Insufficient stock for {reagent.name}. Available: {reagent.stock_level}', 'error')
                return redirect(url_for('laboratory.reagents_order'))

            # Create a new reagent order
            new_order = OtherOrder(
                request_date=date.today(),
                status='Pending',
                requested_by=current_user.id,
                item_id=item_id,
                quantity_requested=quantity,
                quantity_issued=0,
                notes=notes
            )
            db.session.add(new_order)
            db.session.commit()

            flash(f'Order for {reagent.name} added successfully!', 'success')
            return redirect(url_for('laboratory.reagents_order'))

        # Fetch all pending reagent orders
        pending_orders = OtherOrder.query.filter_by(status='Pending')\
                                        .options(joinedload(OtherOrder.item))\
                                        .order_by(OtherOrder.request_date.desc())\
                                        .all()

        # Fetch only reagents (category_id=6)
        lab_reagents = NonPharmItem.query.filter_by(category_id=6).order_by(NonPharmItem.name).all()

        # Fetch user details for pending orders
        user_ids = [order.requested_by for order in pending_orders]
        users = User.query.filter(User.id.in_(user_ids)).all()

        # Use `username` instead of `name`
        user_name_map = {user.id: user.username for user in users}

        # Debugging Logs
        print(f"Debug: Found {len(pending_orders)} pending orders")
        for order in pending_orders:
            requester_name = user_name_map.get(order.requested_by, 'Unknown')
            item_name = order.item.name if order.item else 'Unknown'
            print(f"Debug: Order ID {order.id}, Requested by {requester_name}, Item: {item_name}, Quantity: {order.quantity_requested}")

        return render_template(
            'laboratory/reagents_order.html',
            pending_orders=pending_orders,
            user_name_map=user_name_map,
            lab_reagents=lab_reagents
        )

    except Exception as e:
        db.session.rollback()
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.reagents_order: {e}")
        return redirect(url_for('laboratory.index'))



#inventory, to be edited later
@bp.route('/lab_reagent_inventory')
@login_required
@roles_required('laboratory', 'admin')
def lab_reagent_inventory():
    """Displays available lab reagents and stock levels."""

    try:
        # Fetch all reagents where category_id = 6
        reagents = NonPharmItem.query.filter_by(category_id=6).order_by(NonPharmItem.name).all()

        return render_template(
            'laboratory/lab_reagent_inventory.html',
            reagents=reagents
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.lab_reagent_inventory: {e}")
        return redirect(url_for('laboratory.index'))
@bp.route('/request_reagent_restock', methods=['POST'])
@login_required
@roles_required('laboratory', 'admin')
def request_reagent_restock():
    """Handles reagent restock requests."""

    try:
        item_id = request.form.get('item_id')
        quantity = request.form.get('quantity')

        if not item_id or not quantity:
            flash("Item and quantity required!", "error")
            return redirect(url_for('laboratory.lab_reagent_inventory'))

        new_request = OtherOrder(
            item_id=item_id,
            quantity_requested=int(quantity),
            request_date=datetime.utcnow(),
            status="Pending",
            requested_by=current_user.id
        )

        db.session.add(new_request)
        db.session.commit()

        flash("Reagent restock request submitted!", "success")
        return redirect(url_for('laboratory.lab_reagent_inventory'))

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.request_reagent_restock: {e}")
        return redirect(url_for('laboratory.lab_reagent_inventory'))
#########
