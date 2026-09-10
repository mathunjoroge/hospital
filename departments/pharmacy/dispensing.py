import logging
from collections import defaultdict
from datetime import datetime

from flask import current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.orm import joinedload

from departments.models.admin import Log
from departments.models.medicine import PrescribedMedicine
from departments.models.pharmacy import (  # Import PatientWaitingList and Patient models
    Batch,
    DispensedDrug,
    Drug,
)
from departments.models.records import Patient
from departments.rbac import roles_required
from departments.shared.encounter_utils import is_encounter_open_for_dispensing
from departments.shared.payment_gate import has_unpaid_charges
from extensions import db

from . import bp  # Import the blueprint

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@bp.route("/prescriptions", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def prescriptions():
    """Displays all active prescriptions."""
    try:
        # Fetch all active prescriptions
        active_prescriptions = (
            PrescribedMedicine.query.filter(PrescribedMedicine.num_days > 0)
            .options(
                joinedload(PrescribedMedicine.medicine),
                joinedload(PrescribedMedicine.patient),
            )
            .all()
        )

        return render_template(
            "pharmacy/prescriptions.html", prescriptions=active_prescriptions
        )

    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.prescriptions: {e}")
        return redirect(url_for("pharmacy.index"))


# record purchase


@bp.route("/view_prescriptions/<string:patient_id>", methods=["GET"])
@login_required
@roles_required("medicine", "pharmacy", "admin")
def view_prescriptions(patient_id):
    """Displays all prescribed medicines for a specific patient, grouped by prescription."""
    try:
        # Debug: Print the patient_id being fetched
        print(f"Debug: Fetching prescriptions for patient_id: {patient_id}")

        # Fetch the patient from the database
        patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()

        # Debug: Print patient details
        print(f"Debug: Fetched patient: {patient.name} (ID: {patient.patient_id})")

        # Fetch all prescribed medicines for the patient
        prescribed_medicines = (
            PrescribedMedicine.query.filter_by(patient_id=patient_id)
            .options(joinedload(PrescribedMedicine.medicine))
            .all()
        )

        # Group medicines by prescription_id
        prescriptions = defaultdict(list)
        for medicine in prescribed_medicines:
            prescriptions[medicine.prescription_id].append(medicine)

        # Convert defaultdict to a list of dictionaries for easier template handling
        prescription_list = [
            {"prescription_id": pres_id, "medicines": pres_data}
            for pres_id, pres_data in prescriptions.items()
        ]

        # Debug: Print grouped prescriptions
        print(f"Debug: Grouped Prescriptions for Patient {patient.patient_id}:")
        for pres in prescription_list:
            print(
                f"  - Prescription ID: {pres['prescription_id']}, Medicines: {len(pres['medicines'])}"
            )

        return render_template(
            "pharmacy/view_prescriptions.html",
            patient=patient,
            prescription_list=prescription_list,
        )

    except Exception as e:
        # Debug: Log the exception details
        print(f"Debug: Error fetching prescriptions: {e}")
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("/"))


@bp.route("/dispense/<string:prescription_id>", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def dispense_prescription(prescription_id):
    """Displays prescribed medicines with status=0 and available stock for dispensing."""
    try:
        # Filter prescribed medicines by prescription_id and status='0'
        prescribed_medicines = (
            PrescribedMedicine.query.filter_by(
                prescription_id=prescription_id,
                status=0,  # Add status='0' filter
            )
            .options(joinedload(PrescribedMedicine.medicine))
            .all()
        )

        if not prescribed_medicines:
            flash(
                f'Prescription with ID {prescription_id} has no medicines with status "0"!',
                "info",
            )
            return redirect(url_for("pharmacy.index"))

        if has_unpaid_charges(prescribed_medicines[0].patient_id):
            flash(
                "Warning: this patient has unsettled charges — dispensing on credit."
                " Set PHARMACY_REQUIRE_PAID=True to enforce payment first.",
                "warning",
            )

        drugs = Drug.query.all()
        drug_batches = {
            medicine.medicine.id: Batch.query.filter_by(drug_id=medicine.medicine.id)
            .order_by(Batch.expiry_date.asc())
            .first()
            for medicine in prescribed_medicines
        }

        dispensed_drugs = DispensedDrug.query.filter_by(
            prescription_id=prescription_id
        ).all()

        return render_template(
            "pharmacy/dispense_prescription.html",
            prescribed_medicines=prescribed_medicines,
            prescription_id=prescription_id,
            drug_batches=drug_batches,
            drugs=drugs,
            dispensed_drugs=dispensed_drugs,
        )

    except Exception:
        flash("Something went wrong. Please try again.", "error")

        return redirect(url_for("pharmacy.index"))


@bp.route("/delete_dispensed_drug/<int:dispensed_drug_id>", methods=["GET"])
@login_required
def delete_dispensed_drug(dispensed_drug_id):
    """
    Deletes a dispensed drug entry and restores stock.
    """
    try:
        dispensed_drug = DispensedDrug.query.get(dispensed_drug_id)
        if not dispensed_drug:
            flash("Dispensed drug not found!", "error")
            return redirect(request.referrer)

        # Restore stock to the batch
        batch = Batch.query.filter_by(batch_number=dispensed_drug.batch_no).first()
        if batch:
            batch.quantity_in_stock += dispensed_drug.quantity_dispensed

        db.session.delete(dispensed_drug)
        db.session.commit()

        flash("Dispensed drug deleted successfully!", "success")

    except Exception:
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")

    return redirect(request.referrer)  # ✅ Redirects to the previous page


@bp.route("/save_dispensed_drugs", methods=["POST"])
@login_required
def save_dispensed_drugs():
    """Saves dispensed drugs, updates stock from multiple batches if needed, and marks prescription as completed."""
    try:
        print("DEBUG: Entering save_dispensed_drugs route")

        # Get form data
        prescription_id = request.form.get("prescription_id")
        print(f"DEBUG: Received prescription_id from form: '{prescription_id}'")

        if not prescription_id:
            print("DEBUG: Prescription ID is missing or empty")
            flash("Prescription ID is required!", "error")
            return redirect(url_for("pharmacy.index"))

        # Parse updated drugs from form data
        print("DEBUG: Parsing updated drugs from form data")
        updated_drugs = []
        for key in request.form:
            if key.startswith("updatedDrugs["):
                drug_id = key[len("updatedDrugs[") : -1]
                quantity = request.form.get(key)
                updated_drugs.append({"id": drug_id, "quantity": quantity})
                print(
                    f"DEBUG: Found updated drug - ID: {drug_id}, Quantity: {quantity}"
                )

        if not updated_drugs:
            print("DEBUG: No updated drugs found in form data")
            flash("No drugs to update!", "error")
            return redirect(url_for("pharmacy.index"))

        print(f"DEBUG: Processing {len(updated_drugs)} updated drugs")
        for drug_data in updated_drugs:
            drug_id = drug_data.get("id")
            new_quantity = int(drug_data.get("quantity", 0))
            print(f"DEBUG: Processing drug ID: {drug_id}, New Quantity: {new_quantity}")

            if new_quantity <= 0:
                print(f"DEBUG: Invalid quantity ({new_quantity}) for drug ID {drug_id}")
                flash("Quantity must be greater than 0!", "error")
                return redirect(url_for("pharmacy.index"))

            dispensed_drug = DispensedDrug.query.get(drug_id)
            if not dispensed_drug:
                print(f"DEBUG: Dispensed drug ID {drug_id} not found")
                flash(f"Dispensed drug {drug_id} not found!", "error")
                return redirect(url_for("pharmacy.index"))

            drug = Drug.query.get(dispensed_drug.drug_id)
            if not drug:
                print(f"DEBUG: Drug ID {dispensed_drug.drug_id} not found")
                flash("Drug not found!", "error")
                return redirect(url_for("pharmacy.index"))

            quantity_difference = new_quantity - dispensed_drug.quantity_dispensed
            print(
                f"DEBUG: Quantity difference for drug ID {drug_id}: {quantity_difference}"
            )

            if quantity_difference > 0:
                # Get all batches for this drug, ordered by expiry date
                batches = (
                    Batch.query.filter_by(drug_id=dispensed_drug.drug_id)
                    .order_by(Batch.expiry_date.asc())
                    .all()
                )
                if not batches:
                    print(
                        f"DEBUG: No batches found for drug ID {dispensed_drug.drug_id}"
                    )
                    flash("No batches available for this drug!", "error")
                    return redirect(url_for("pharmacy.index"))

                # Calculate total available stock across all batches
                total_available = sum(batch.quantity_in_stock for batch in batches)
                print(
                    f"DEBUG: Total available stock for drug ID {dispensed_drug.drug_id}: {total_available}"
                )

                if total_available < quantity_difference:
                    # Dispense all available stock and report shortfall
                    remaining_quantity = quantity_difference
                    total_dispensed = 0

                    for batch in batches:
                        if remaining_quantity <= 0:
                            break
                        qty_to_dispense = min(
                            remaining_quantity, batch.quantity_in_stock
                        )
                        if qty_to_dispense > 0:
                            if batch.id == dispensed_drug.batch_id:
                                # Update original dispensed_drug
                                dispensed_drug.quantity_dispensed += qty_to_dispense
                                db.session.add(dispensed_drug)
                            else:
                                # Create new DispensedDrug for additional batch
                                new_dispensed_drug = DispensedDrug(
                                    drug_id=dispensed_drug.drug_id,
                                    batch_id=batch.id,
                                    patient_id=dispensed_drug.patient_id,
                                    prescription_id=prescription_id,
                                    quantity_dispensed=qty_to_dispense,
                                    date_dispensed=dispensed_drug.date_dispensed,
                                    status=dispensed_drug.status,
                                )
                                db.session.add(new_dispensed_drug)
                            batch.quantity_in_stock -= qty_to_dispense
                            drug.quantity_in_stock -= qty_to_dispense
                            total_dispensed += qty_to_dispense
                            remaining_quantity -= qty_to_dispense
                            print(
                                f"DEBUG: Dispensed {qty_to_dispense} from batch ID {batch.id}, Remaining: {remaining_quantity}"
                            )

                    shortfall = quantity_difference - total_dispensed
                    print(
                        f"DEBUG: Insufficient stock. Dispensed: {total_dispensed}, Shortfall: {shortfall}"
                    )
                    flash(
                        f"Insufficient stock for {drug.generic_name}. Dispensed: {total_dispensed}, Shortfall: {shortfall}",
                        "warning",
                    )

                else:
                    # Enough stock available, dispense as before
                    remaining_quantity = quantity_difference
                    batch_index = 0
                    original_batch = Batch.query.filter_by(
                        id=dispensed_drug.batch_id
                    ).first()

                    if original_batch and original_batch.quantity_in_stock > 0:
                        qty_from_original = min(
                            remaining_quantity, original_batch.quantity_in_stock
                        )
                        original_batch.quantity_in_stock -= qty_from_original
                        drug.quantity_in_stock -= qty_from_original
                        dispensed_drug.quantity_dispensed += qty_from_original
                        remaining_quantity -= qty_from_original
                        print(
                            f"DEBUG: Dispensed {qty_from_original} from original batch ID {original_batch.id}, Remaining: {remaining_quantity}"
                        )
                        db.session.add(original_batch)
                        db.session.add(drug)
                        db.session.add(dispensed_drug)

                    while remaining_quantity > 0 and batch_index < len(batches):
                        next_batch = batches[batch_index]
                        if next_batch.id == dispensed_drug.batch_id:
                            batch_index += 1
                            continue
                        qty_from_next = min(
                            remaining_quantity, next_batch.quantity_in_stock
                        )
                        if qty_from_next > 0:
                            next_batch.quantity_in_stock -= qty_from_next
                            drug.quantity_in_stock -= qty_from_next
                            new_dispensed_drug = DispensedDrug(
                                drug_id=dispensed_drug.drug_id,
                                batch_id=next_batch.id,
                                patient_id=dispensed_drug.patient_id,
                                prescription_id=prescription_id,
                                quantity_dispensed=qty_from_next,
                                date_dispensed=dispensed_drug.date_dispensed,
                                status=dispensed_drug.status,
                            )
                            remaining_quantity -= qty_from_next
                            print(
                                f"DEBUG: Dispensed {qty_from_next} from next batch ID {next_batch.id}, Remaining: {remaining_quantity}"
                            )
                            db.session.add(next_batch)
                            db.session.add(drug)
                            db.session.add(new_dispensed_drug)
                        batch_index += 1

            elif quantity_difference < 0:
                # Reduce quantity in original batch
                batch = Batch.query.filter_by(id=dispensed_drug.batch_id).first()
                batch.quantity_in_stock -= (
                    quantity_difference  # Adds back since difference is negative
                )
                drug.quantity_in_stock -= quantity_difference
                dispensed_drug.quantity_dispensed = new_quantity
                print(
                    f"DEBUG: Reduced quantity - Batch stock: {batch.quantity_in_stock}, Drug stock: {drug.quantity_in_stock}, Dispensed qty: {dispensed_drug.quantity_dispensed}"
                )
                db.session.add(batch)
                db.session.add(drug)
                db.session.add(dispensed_drug)

        # Debug: Check prescribed medicines
        print(
            f"DEBUG: Querying prescribed medicines for prescription_id '{prescription_id}'"
        )
        prescribed_meds = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .all()
        )

        # --- T3.7: Encounter Scoping Check ---
        if prescribed_meds and not is_encounter_open_for_dispensing(prescribed_meds[0].encounter_id):
            flash('Cannot dispense: The associated encounter is closed or the patient has been discharged.', 'danger')
            return redirect(request.referrer or url_for('pharmacy.index'))
        # -------------------------------------
        print(f"DEBUG: Found {len(prescribed_meds)} prescribed medicines")
        for med in prescribed_meds:
            print(
                f"DEBUG: PrescribedMedicine ID: {med.id}, Status: {med.status}, Prescription ID: '{med.prescription_id}'"
            )

        if not prescribed_meds:
            print(
                f"DEBUG: No prescribed medicines found for prescription_id '{prescription_id}'"
            )
            flash("Prescription not found!", "error")
            return redirect(url_for("pharmacy.index"))

        # Update status of all matching prescribed medicines
        print(f"DEBUG: Updating status to 1 for prescription_id '{prescription_id}'")
        updated_rows = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .update({"status": 1})
        )
        print(f"DEBUG: Updated {updated_rows} prescribed medicine rows to status=1")

        # FIX 4: Advance encounter stage after dispensing completion
        from departments.shared.visit_closure import advance_after_completion
        _patient_id = prescribed_meds[0].patient_id if prescribed_meds else None
        if _patient_id:
            advance_after_completion(_patient_id)

        # Verify before commit
        pre_commit_meds = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .all()
        )
        for med in pre_commit_meds:
            print(
                f"DEBUG PRE-COMMIT: PrescribedMedicine ID: {med.id}, Status: {med.status}"
            )

        db.session.commit()
                # FIX 4: Advance encounter stage after dispensing completion
        # FIX 4: Advance encounter stage after dispensing
        # FIX 4: Advance encounter stage after dispensing completion
        from departments.shared.visit_closure import advance_after_completion
        patient_id = prescribed_meds[0].patient_id if prescribed_meds else None
        if patient_id:
            advance_after_completion(patient_id)
        print("DEBUG: Database commit successful")

        # Verify after commit
        post_commit_meds = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .all()
        )
        for med in post_commit_meds:
            print(
                f"DEBUG POST-COMMIT: PrescribedMedicine ID: {med.id}, Status: {med.status}"
            )

        flash("Dispensed drugs updated successfully!", "success")
        print("DEBUG: Redirecting to pharmacy.index with success message")
        return redirect(url_for("pharmacy.index"))

    except Exception as e:
        db.session.rollback()
        print(f"DEBUG: Exception occurred: {str(e)}")
        flash("Something went wrong. Please try again.", "error")
        print("DEBUG: Redirecting to pharmacy.index with error message")
        return redirect(url_for("pharmacy.index"))


@bp.route("/save_prescription/<string:prescription_id>", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def save_prescription(prescription_id):
    """Save dispensed drugs for a specific prescription."""
    try:
        # Extract form data
        drug_ids = request.form.getlist("drugs[]")  # List of selected drug IDs
        quantities = request.form.getlist("quantity[]")  # List of quantities
        batch_numbers = request.form.getlist("batch_number[]")  # List of batch numbers

        # Validate input
        if not drug_ids or not quantities or not batch_numbers:
            flash(
                "Invalid input! Please select drugs and specify their quantities.",
                "error",
            )
            return redirect(
                url_for(
                    "pharmacy.view_prescriptions",
                    patient_id=request.form.get("patient_id"),
                )
            )

        if len(drug_ids) != len(quantities) or len(drug_ids) != len(batch_numbers):
            flash(
                "Mismatched input! Ensure each drug has a corresponding quantity and batch number.",
                "error",
            )
            return redirect(
                url_for(
                    "pharmacy.view_prescriptions",
                    patient_id=request.form.get("patient_id"),
                )
            )

        # Fetch the first prescribed medicine to get patient ID
        prescribed_medicines = PrescribedMedicine.query.filter_by(
            prescription_id=prescription_id
        ).all()
        if not prescribed_medicines:
            flash(f"Prescription with ID {prescription_id} does not exist!", "error")
            return redirect(url_for("pharmacy.index"))

        patient_id = prescribed_medicines[0].patient_id

        # Phase-1 payment gate
        unpaid = has_unpaid_charges(patient_id)
        if unpaid and current_app.config.get("PHARMACY_REQUIRE_PAID", False):
            flash(
                "Dispensing blocked: patient has unsettled charges. "
                "Complete billing first (or disable PHARMACY_REQUIRE_PAID).",
                "error",
            )
            return redirect(url_for("pharmacy.view_prescriptions", patient_id=patient_id))
        if unpaid:
            db.session.add(
                Log(
                    level="WARNING",
                    message=f"Pharmacy dispensing on credit for patient {patient_id} "
                    f"(prescription {prescription_id}) with unsettled charges.",
                    user_id=current_user.id,
                    source="pharmacy",
                )
            )

        # Process each drug in the form
        for i, drug_id in enumerate(drug_ids):
            try:
                quantity_dispensed = int(quantities[i])
                if quantity_dispensed <= 0:
                    raise ValueError("Quantity must be greater than zero.")
            except (ValueError, IndexError):
                flash(f"Invalid quantity for drug with ID {drug_id}!", "error")
                return redirect(
                    url_for("pharmacy.view_prescriptions", patient_id=patient_id)
                )

            batch_number = batch_numbers[i]

            # Check if the drug exists
            drug = Drug.query.get(drug_id)
            if not drug:
                flash(f"Drug with ID {drug_id} does not exist!", "error")
                return redirect(
                    url_for("pharmacy.view_prescriptions", patient_id=patient_id)
                )

            # Check if the batch exists and has sufficient stock
            batch = Batch.query.filter_by(
                batch_number=batch_number, drug_id=drug.id
            ).first()
            if not batch or batch.quantity_in_stock < quantity_dispensed:
                flash(
                    f'Insufficient stock for {drug.generic_name} ({drug.brand_name}). Available: {batch.quantity_in_stock if batch else "N/A"}',
                    "error",
                )
                return redirect(
                    url_for("pharmacy.view_prescriptions", patient_id=patient_id)
                )

            # Create a new dispensed drug entry
            new_dispensed_drug = DispensedDrug(
                drug_id=drug.id,
                batch_id=batch.id,
                patient_id=patient_id,
                prescription_id=prescription_id,
                quantity_dispensed=quantity_dispensed,
                date_dispensed=datetime.today().date(),
            )
            db.session.add(new_dispensed_drug)

            # Update batch stock
            batch.quantity_in_stock -= quantity_dispensed
            db.session.add(batch)

        # Commit changes to the database
        db.session.commit()

        flash("Drugs dispensed successfully!", "success")
        return redirect(url_for("pharmacy.view_prescriptions", patient_id=patient_id))

    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in pharmacy.save_prescription: {e}")
        db.session.rollback()  # Rollback changes in case of error
        return redirect(
            url_for(
                "pharmacy.view_prescriptions", patient_id=request.form.get("patient_id")
            )
        )
