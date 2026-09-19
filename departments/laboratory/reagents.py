import logging
from datetime import datetime, timezone

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload

from departments.models.stores import NonPharmItem, OtherOrder
from departments.models.user import User
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

logger = logging.getLogger(__name__)


# order for reagets
@bp.route("/reagents_order", methods=["GET", "POST"])
@login_required
@roles_required("laboratory", "admin")
def reagents_order():
    """Handles ordering of lab reagents."""

    try:
        if request.method == "POST":
            # Extract form data
            item_id = request.form.get("item_id", type=int)
            quantity = request.form.get("quantity", type=int)
            notes = request.form.get("notes", "")

            # Validation: Ensure item selection and valid quantity
            if not item_id or not quantity or quantity <= 0:
                flash(
                    "Please select a valid reagent and enter a quantity greater than zero.",
                    "error",
                )
                return redirect(url_for("laboratory.reagents_order"))

            # Fetch the reagent from the inventory (Category ID = 6 for Lab Reagents)
            reagent = NonPharmItem.query.filter_by(id=item_id, category_id=6).first()

            if not reagent:
                flash("Selected item is not a valid lab reagent.", "error")
                return redirect(url_for("laboratory.reagents_order"))

            # Check stock availability
            if reagent.stock_level < quantity:
                flash(
                    f"Insufficient stock for {reagent.name}. Available: {reagent.stock_level}",
                    "error",
                )
                return redirect(url_for("laboratory.reagents_order"))

            # Create a new reagent order
            new_order = OtherOrder(
                request_date=datetime.now(timezone.utc).date(),
                status="Pending",
                requested_by=current_user.id,
                item_id=item_id,
                quantity_requested=quantity,
                quantity_issued=0,
                notes=notes,
            )
            db.session.add(new_order)
            db.session.commit()

            flash(f"Order for {reagent.name} added successfully!", "success")
            return redirect(url_for("laboratory.reagents_order"))

        # Fetch all pending reagent orders
        pending_orders = (
            OtherOrder.query.filter_by(status="Pending")
            .options(joinedload(OtherOrder.item))
            .order_by(OtherOrder.request_date.desc())
            .all()
        )

        # Fetch only reagents (category_id=6)
        lab_reagents = (
            NonPharmItem.query.filter_by(category_id=6)
            .order_by(NonPharmItem.name)
            .all()
        )

        # Fetch user details for pending orders
        user_ids = [order.requested_by for order in pending_orders]
        users = User.query.filter(User.id.in_(user_ids)).all()

        # Use `username` instead of `name`
        user_name_map = {user.id: user.username for user in users}

        return render_template(
            "laboratory/reagents_order.html",
            pending_orders=pending_orders,
            user_name_map=user_name_map,
            lab_reagents=lab_reagents,
        )

    except (SQLAlchemyError, ValueError) as e:
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.reagents_order")
        return redirect(url_for("laboratory.index"))


# inventory, to be edited later
@bp.route("/lab_reagent_inventory")
@login_required
@roles_required("laboratory", "admin")
def lab_reagent_inventory():
    """Displays available lab reagents and stock levels."""

    try:
        # Fetch all reagents where category_id = 6
        reagents = (
            NonPharmItem.query.filter_by(category_id=6)
            .order_by(NonPharmItem.name)
            .all()
        )

        return render_template(
            "laboratory/lab_reagent_inventory.html", reagents=reagents
        )

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.lab_reagent_inventory")
        return redirect(url_for("laboratory.index"))


@bp.route("/request_reagent_restock", methods=["POST"])
@login_required
@roles_required("laboratory", "admin")
def request_reagent_restock():
    """Handles reagent restock requests."""

    try:
        item_id = request.form.get("item_id")
        quantity = request.form.get("quantity")

        if not item_id or not quantity:
            flash("Item and quantity required!", "error")
            return redirect(url_for("laboratory.lab_reagent_inventory"))

        # P2-17: validate the item is actually a lab reagent (category 6) —
        # a tampered item_id could otherwise file restock orders for any
        # non-pharm inventory item.
        reagent = NonPharmItem.query.filter_by(id=item_id, category_id=6).first()
        if not reagent:
            flash("Selected item is not a valid lab reagent.", "error")
            return redirect(url_for("laboratory.lab_reagent_inventory"))

        new_request = OtherOrder(
            item_id=item_id,
            quantity_requested=int(quantity),
            request_date=datetime.now(timezone.utc),
            status="Pending",
            requested_by=current_user.id,
        )

        db.session.add(new_request)
        db.session.commit()

        flash("Reagent restock request submitted!", "success")
        return redirect(url_for("laboratory.lab_reagent_inventory"))

    except (SQLAlchemyError, ValueError) as e:
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in laboratory.request_reagent_restock")
        return redirect(url_for("laboratory.lab_reagent_inventory"))


#########
