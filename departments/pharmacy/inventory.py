import logging
import uuid  # Import the uuid module
from datetime import datetime, timedelta, timezone

from flask import (
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy.sql import (
    func,  # Import the text function
)

from departments.models.medicine import PrescribedMedicine
from departments.models.pharmacy import (  # Import PatientWaitingList and Patient models
    Batch,
    Drug,
    DrugRequest,
    Expiry,
    Purchase,
    RequestItem,
)
from departments.models.records import Patient
from departments.models.user import User
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@bp.route("/")
@login_required
@roles_required("pharmacy", "admin")
def index():
    try:
        # Fetch pending prescriptions (status=0) and join with Patient
        pending_prescriptions = (
            db.session.query(PrescribedMedicine, Patient)
            .join(Patient, PrescribedMedicine.patient_id == Patient.patient_id)
            .filter(PrescribedMedicine.status == 0)
            .order_by(PrescribedMedicine.id.desc())
            .all()
        )
        return render_template(
            "pharmacy/index.html",
            prescriptions=[(p, pt) for p, pt in pending_prescriptions],
        )
    except Exception:
        logger.exception("Error fetching pending prescriptions: ")
        flash("Unable to load prescriptions. Please try again.", "error")
        return redirect(url_for("pharmacy.index"))


# fech expiries
@bp.route("/expiries", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def expiries():
    """Displays only expired medications in the inventory."""
    try:
        today = datetime.now(timezone.utc).date()

        expired_drugs = (
            db.session.query(
                Drug.generic_name,
                Drug.brand_name,
                Drug.dosage_form,
                Drug.strength,
                Batch.id.label("batch_id"),
                Batch.batch_number,
                Batch.quantity_in_stock.label("batch_quantity"),
                Batch.expiry_date,
                func.sum(Batch.quantity_in_stock)
                .over(partition_by=Drug.id)
                .label("total_stock"),
            )
            .join(Batch, Drug.id == Batch.drug_id)
            .filter(Batch.expiry_date < today)
            .order_by(Batch.expiry_date.desc())
            .all()
        )

        return render_template("pharmacy/expiries.html", expired_drugs=expired_drugs)

    except Exception as e:  # noqa: BLE001
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.expiries: {e}")
        return redirect(url_for("home"))


@bp.route("/remove_batch/<int:batch_id>", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def remove_batch(batch_id):
    """Removes a batch from inventory and logs it in expiries."""
    try:
        batch = Batch.query.get_or_404(batch_id)
        drug = Drug.query.get(batch.drug_id)

        # Log the batch in expiries table
        expiry_record = Expiry(
            drug_id=batch.drug_id,
            batch_number=batch.batch_number,
            quantity_removed=batch.quantity_in_stock,
            expiry_date=batch.expiry_date,
            removal_date=datetime.now(timezone.utc).date(),
        )
        db.session.add(expiry_record)

        # Update drugs.quantity_in_stock
        drug.quantity_in_stock -= batch.quantity_in_stock
        drug.quantity_in_stock = max(drug.quantity_in_stock, 0)

        # Remove the batch
        db.session.delete(batch)
        db.session.commit()

        flash(f"Batch {batch.batch_number} removed from inventory.", "success")
    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.remove_batch: {e}")
    return redirect(url_for("pharmacy.expiries"))


@bp.route("/remove_all_expiries", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def remove_all_expiries():
    """Removes all expired batches from inventory and logs them in expiries."""
    try:
        today = datetime.now(timezone.utc).date()

        # Fetch all expired batches
        expired_batches = Batch.query.filter(Batch.expiry_date < today).all()

        if not expired_batches:
            flash("No expired batches to remove.", "info")
            return redirect(url_for("pharmacy.expiries"))

        for batch in expired_batches:
            drug = Drug.query.get(batch.drug_id)

            # Log each batch in expiries table
            expiry_record = Expiry(
                drug_id=batch.drug_id,
                batch_number=batch.batch_number,
                quantity_removed=batch.quantity_in_stock,
                expiry_date=batch.expiry_date,
                removal_date=today,
            )
            db.session.add(expiry_record)

            # Update drugs.quantity_in_stock
            drug.quantity_in_stock -= batch.quantity_in_stock
            drug.quantity_in_stock = max(drug.quantity_in_stock, 0)

            # Remove the batch
            db.session.delete(batch)

        db.session.commit()
        flash(
            f"Removed {len(expired_batches)} expired batches from inventory.", "success"
        )
    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.remove_all_expiries: {e}")
    return redirect(url_for("pharmacy.expiries"))


# expires report


@bp.route("/inventory", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def inventory():
    """Displays the full inventory with expiry categories."""
    try:
        # Define a threshold for near expiry (e.g., within 30 days)
        today = datetime.now(timezone.utc).date()
        near_expiry_threshold = today + timedelta(days=30)

        # Query the database with total stock as sum of batch quantities
        inventory_data = (
            db.session.query(
                Drug.generic_name,
                Drug.brand_name,
                Drug.dosage_form,
                Drug.strength,
                Batch.batch_number,
                Batch.quantity_in_stock.label("batch_quantity"),
                Batch.expiry_date,
                func.sum(Batch.quantity_in_stock)
                .over(partition_by=Drug.id)
                .label("total_stock"),
            )
            .join(Batch, Drug.id == Batch.drug_id)
            .order_by(
                Batch.expiry_date.desc()  # Order by expiry date descending
            )
            .all()
        )

        # Separate drugs into categories: expired, near expiry, and normal stock
        expired_drugs = []
        near_expiry_drugs = []
        normal_stock_drugs = []

        for item in inventory_data:
            if item.expiry_date and item.expiry_date < today:
                expired_drugs.append(item)
            elif item.expiry_date and item.expiry_date <= near_expiry_threshold:
                near_expiry_drugs.append(item)
            else:
                normal_stock_drugs.append(item)

        return render_template(
            "pharmacy/inventory.html",
            expired_drugs=expired_drugs,
            near_expiry_drugs=near_expiry_drugs,
            normal_stock_drugs=normal_stock_drugs,
        )

    except Exception as e:  # noqa: BLE001
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.inventory: {e}")
        return redirect(url_for("home"))

    # prescriptions


@bp.route("/record_purchase", methods=["GET", "POST"])
@login_required
@roles_required("pharmacy", "admin")
def record_purchase():
    """Handles recording a new purchase of medications."""
    try:
        if request.method == "POST":
            # Extract form data
            drug_ids = request.form.getlist("drug_ids[]")
            batch_numbers = request.form.getlist("batch_numbers[]")
            quantities = request.form.getlist("quantities[]")
            unit_costs = request.form.getlist("unit_costs[]")

            # Validate input
            if not all([drug_ids, batch_numbers, quantities, unit_costs]):
                flash("All fields are required!", "error")
                return redirect(url_for("pharmacy.record_purchase"))

            # Record each purchase
            for drug_id, batch_number, quantity, unit_cost in zip(
                drug_ids, batch_numbers, quantities, unit_costs
            ):
                if not quantity.strip() or not unit_cost.strip():
                    continue  # Skip empty entries

                # Fetch the drug and batch
                drug = Drug.query.get_or_404(int(drug_id))
                batch = Batch.query.filter_by(
                    drug_id=drug.id, batch_number=batch_number
                ).first()

                if not batch:
                    # Create a new batch if it doesn't exist
                    batch = Batch(
                        drug_id=drug.id,
                        batch_number=batch_number,
                        expiry_date=None,  # Expiry date can be added later
                        quantity_in_stock=0,
                    )
                    db.session.add(batch)
                    db.session.commit()

                # Update batch stock level
                batch.quantity_in_stock += int(quantity)

                # Record the purchase
                new_purchase = Purchase(
                    drug_id=drug.id,
                    batch_id=batch.id,
                    purchase_date=datetime.now(timezone.utc).date(),
                    quantity_purchased=int(quantity),
                    unit_cost=float(unit_cost),
                    total_cost=float(unit_cost) * int(quantity),
                )
                db.session.add(new_purchase)

            db.session.commit()
            flash("Purchase recorded successfully!", "success")
            return redirect(url_for("pharmacy.inventory"))

        # Fetch all drugs for the purchase form
        drugs = Drug.query.all()

        return render_template("pharmacy/record_purchase.html", drugs=drugs)

    except Exception as e:  # noqa: BLE001
        flash("Something went wrong. Please try again.", "error")
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in pharmacy.record_purchase: {e}")  # Debugging
        return redirect(url_for("pharmacy.index"))
    # view prescription


@bp.route("/low_stock", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def low_stock():
    """Displays drugs with stock levels below their reorder threshold."""
    try:
        # Subquery to identify drugs with total stock below reorder_level
        total_stock_subquery = (
            db.session.query(
                Batch.drug_id, func.sum(Batch.quantity_in_stock).label("total_stock")
            )
            .group_by(Batch.drug_id)
            .having(
                func.sum(Batch.quantity_in_stock)
                < db.session.query(Drug.reorder_level)
                .filter(Drug.id == Batch.drug_id)
                .scalar_subquery()
            )
            .subquery()
        )

        # Main query to fetch drug details with total stock
        low_stock_drugs = (
            db.session.query(
                Drug.generic_name,
                Drug.brand_name,
                Drug.dosage_form,
                Drug.strength,
                Drug.reorder_level,
                total_stock_subquery.c.total_stock.label("current_stock"),
            )
            .join(total_stock_subquery, Drug.id == total_stock_subquery.c.drug_id)
            .all()
        )

        # Debug: Print results to verify data
        if not low_stock_drugs:
            print("Debug: No low stock drugs found.")
        for drug in low_stock_drugs:
            print(
                f"Drug: {drug.generic_name}, Stock: {drug.current_stock}, Reorder: {drug.reorder_level}"
            )

        return render_template(
            "pharmacy/low_stock.html", low_stock_drugs=low_stock_drugs
        )

    except Exception as e:  # noqa: BLE001
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.low_stock: {e}")
        return redirect(url_for("pharmacy.index"))


@bp.route("/drug-requests", methods=["GET", "POST"])
@login_required
@roles_required("pharmacy", "admin")
def drug_requests():
    """Handles drug requests from the store (only latest request shown)."""
    try:
        if request.method == "POST":
            drug_id = request.form.get("drug_id")
            quantity = request.form.get("quantity")

            if not drug_id or not quantity or int(quantity) <= 0:
                flash("Please select a drug and enter a valid quantity.", "error")
                return redirect(url_for("pharmacy.drug_requests"))

            # Check if an open request exists for the user
            existing_request = DrugRequest.query.filter_by(
                requested_by=current_user.id, status="Submitted"
            ).first()

            if not existing_request:
                # Create a new request with a unique UUID
                existing_request = DrugRequest(
                    request_uuid=str(uuid.uuid4()),  # Generate a unique UUID
                    request_date=datetime.now(timezone.utc).date(),
                    status="Pending",
                    requested_by=current_user.id,
                )
                db.session.add(existing_request)
                db.session.flush()  # Get request ID before commit

            # Add drug to request
            item = RequestItem(
                request_id=existing_request.id,
                drug_id=int(drug_id),
                quantity_requested=int(quantity),
                quantity_issued=0,  # Default
            )
            db.session.add(item)
            db.session.commit()

            flash("Drug request added successfully.", "success")
            return redirect(url_for("pharmacy.drug_requests"))

        # Fetch only the latest request for the current user
        latest_request = (
            DrugRequest.query.filter_by(requested_by=current_user.id, status="Pending")
            .order_by(DrugRequest.request_date.desc())
            .first()
        )

        all_drugs = Drug.query.all()  # Get available drugs

        return render_template(
            "pharmacy/drug_requests.html",
            all_drugs=all_drugs,
            latest_request=latest_request,
        )

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.drug_requests: {e}")
        return redirect(url_for("pharmacy.index"))


# save order
@bp.route("/save-order", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def save_order():
    """Finalizes the current drug request and redirects to the dashboard."""
    try:
        # Get the latest pending request for the user
        latest_request = DrugRequest.query.filter_by(
            requested_by=current_user.id, status="Pending"
        ).first()

        if latest_request:
            latest_request.status = "Submitted"  # Mark the request as submitted
            db.session.commit()
            flash("Order saved successfully.", "success")

        return redirect(url_for("pharmacy.index"))  # Redirect to pharmacy dashboard

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.save_order: {e}")
        return redirect(url_for("pharmacy.index"))


@bp.route("/pending-requests", methods=["GET", "POST"])
@login_required
@roles_required("pharmacy", "admin")
def pending_requests():
    """Displays pending drug requests with date range filtering."""
    try:
        # Query with User's username instead of ID
        query = (
            db.session.query(
                DrugRequest.id,
                DrugRequest.request_date,
                DrugRequest.status,
                User.username.label("requested_by"),  # Fetch `username`
            )
            .join(User, User.id == DrugRequest.requested_by)
            .filter(DrugRequest.status == "Submitted")
            .order_by(DrugRequest.request_date.desc())
        )

        # Date filtering
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        if start_date and end_date:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()  # noqa: DTZ007
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()  # noqa: DTZ007
            query = query.filter(DrugRequest.request_date.between(start_date, end_date))

        pending_requests = query.limit(5).all()

        return render_template(
            "pharmacy/pending_requests.html", pending_requests=pending_requests
        )

    except Exception:  # noqa: BLE001
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("pharmacy.index"))


# pending requests details
@bp.route("/pending-request-details/<int:request_id>", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def pending_request_details(request_id):
    """Displays the details of a pending drug request."""
    # Fetch request details with username
    request_details = (
        db.session.query(DrugRequest, User.username.label("requested_by"))
        .join(User, User.id == DrugRequest.requested_by)
        .filter(DrugRequest.id == request_id)
        .first_or_404()
    )

    # Extract the DrugRequest instance
    drug_request = request_details.DrugRequest  # This is the actual model instance
    requested_by = request_details.requested_by  # Extract the username

    return render_template(
        "pharmacy/pending_request_details.html",
        drug_request=drug_request,
        requested_by=requested_by,
    )


@bp.route("/served-requests", methods=["GET", "POST"])
@login_required
@roles_required("pharmacy", "admin")
def served_requests():
    """Displays served drug requests with date range filtering."""
    try:
        # Query with User's username instead of ID
        query = (
            db.session.query(
                DrugRequest.id,
                DrugRequest.request_date,
                DrugRequest.status,
                User.username.label("requested_by"),  # Fetch `username`
            )
            .join(User, User.id == DrugRequest.requested_by)
            .filter(DrugRequest.status == "Completed")
            .order_by(DrugRequest.request_date.desc())
        )

        # Date filtering
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        if start_date and end_date:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()  # noqa: DTZ007
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()  # noqa: DTZ007
            query = query.filter(DrugRequest.request_date.between(start_date, end_date))

        served_requests = query.limit(5).all()  # Fetch last 5 served requests

        return render_template(
            "pharmacy/served_requests.html", served_requests=served_requests
        )

    except Exception:  # noqa: BLE001
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("pharmacy.index"))


# Served Request Details Route
@bp.route("/served-request-details/<int:request_id>", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def served_requests_details(request_id):
    """Displays the details of a served drug request."""
    # Fetch request details with username
    request_details = (
        db.session.query(DrugRequest, User.username.label("requested_by"))
        .join(User, User.id == DrugRequest.requested_by)
        .filter(DrugRequest.id == request_id)
        .first_or_404()
    )

    # Extract variables properly
    drug_request, requested_by = request_details  # Correct unpacking

    return render_template(
        "pharmacy/served_request_details.html",
        drug_request=drug_request,
        requested_by=requested_by,
    )


@bp.route("/get_all_batches", methods=["GET"])
@login_required
def get_all_batches():
    """Fetch all available drugs with unique batches, ordered by expiry date."""
    try:
        batches = (
            db.session.query(Batch)
            .join(Drug, Batch.drug_id == Drug.id)
            .filter(Batch.quantity_in_stock > 0)
            .order_by(Batch.expiry_date.asc())
            .all()
        )

        if not batches:
            print("📌 DEBUG: No available drugs found")
            return jsonify({"error": "No available drugs"}), 200

        # Convert to JSON format
        batch_list = [
            {
                "drug_id": batch.drug_id,
                "generic_name": batch.drug.generic_name,
                "brand_name": batch.drug.brand_name,
                "dosage_form": batch.drug.dosage_form,
                "strength": batch.drug.strength,
                "selling_price": batch.drug.selling_price,
                "batch_qty": batch.quantity_in_stock,
                "batch_id": batch.id,
            }
            for batch in batches
        ]

        print(f"📌 DEBUG: API Response: {len(batch_list)} unique batches returned")
        return jsonify(batch_list), 200

    except Exception:
        logger.exception("Error in get_all_batches: ")
        return jsonify(
            {"error": "Failed to fetch drug batches. Please try again."}
        ), 500
