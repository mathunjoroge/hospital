import logging
from collections import defaultdict
from datetime import datetime, timezone

from flask import current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload

from departments.models.admin import Log
from departments.models.medicine import PrescribedMedicine
from departments.models.pharmacy import (  # Import PatientWaitingList and Patient models
    Batch,
    DispensedDrug,
    Drug,
)
from departments.models.records import Patient
from departments.models.stock_movement import record_movement
from departments.rbac import roles_required
from departments.shared.encounter_utils import is_encounter_open_for_dispensing
from departments.shared.payment_gate import has_unpaid_charges
from extensions import db

from . import bp  # Import the blueprint

# Set up logging
logger = logging.getLogger(__name__)


@bp.route("/prescriptions", methods=["GET"])
@login_required
@roles_required("pharmacy", "admin")
def prescriptions():
    """Displays pharmacy waiting list using unified patient flow queue."""
    from departments.shared import queue_service

    # Show pharmacy queue using the unified patient flow system
    # This will show patients waiting for medication dispensing (AWAITING_PHARMACY stage)
    pending_prescriptions = queue_service.queue_for("pharmacy")

    return render_template(
        "pharmacy/prescriptions.html",
        pending_prescriptions=pending_prescriptions or [],
    )


# record purchase


@bp.route("/view_prescriptions/<string:patient_id>", methods=["GET"])
@login_required
@roles_required("medicine", "pharmacy", "admin")
def view_prescriptions(patient_id):
    """Displays all prescribed medicines for a specific patient, grouped by prescription."""
    try:
        logger.debug(f"Fetching prescriptions for patient_id: {patient_id}")

        # Fetch the patient from the database
        patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()

        logger.debug(f"Fetched patient: {patient.name} (ID: {patient.patient_id})")

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

        logger.debug(f"Grouped Prescriptions for Patient {patient.patient_id}:")
        for pres in prescription_list:
            logger.debug(
                f"  - Prescription ID: {pres['prescription_id']}, Medicines: {len(pres['medicines'])}"
            )

        return render_template(
            "pharmacy/view_prescriptions.html",
            patient=patient,
            prescription_list=prescription_list,
        )

    except SQLAlchemyError as e:
        logger.debug(f"Error fetching prescriptions: {e}")
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
            if medicine.medicine
        }

        dispensed_drugs_raw = (
            DispensedDrug.query.filter_by(prescription_id=prescription_id)
            .options(joinedload(DispensedDrug.drug))
            .all()
        )
        dispensed_drugs = [
            {
                "generic_name": d.drug.generic_name if d.drug else "N/A",
                "id": d.id,
                "brand_name": d.drug.brand_name if d.drug else "N/A",
                "dosage_form": d.drug.dosage_form if d.drug else "N/A",
                "strength": d.drug.strength if d.drug else "N/A",
                "selling_price": d.drug.selling_price if d.drug else 0,
                "batch_id": d.batch_id,
                "quantity_dispensed": d.quantity_dispensed,
                "total": (d.drug.selling_price * d.quantity_dispensed) if d.drug else 0,
            }
            for d in dispensed_drugs_raw
        ]

        return render_template(
            "pharmacy/dispense_prescription.html",
            prescribed_medicines=prescribed_medicines,
            prescription_id=prescription_id,
            drug_batches=drug_batches,
            drugs=drugs,
            dispensed_drugs=dispensed_drugs,
        )

    except SQLAlchemyError:
        flash("Something went wrong. Please try again.", "error")

        return redirect(url_for("pharmacy.index"))


@bp.route("/delete_dispensed_drug/<int:dispensed_drug_id>", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def delete_dispensed_drug(dispensed_drug_id):
    """
    P0-05: Void a dispensed drug entry and restore stock transactionally.
    Delegates to the shared void_service — see void_dispensed_drug().
    """
    from departments.pharmacy.void_service import VoidError, void_dispensed_drug

    void_reason = request.form.get("void_reason", "").strip()
    if not void_reason:
        flash("A void reason is required to reverse a dispensing record.", "error")
        return redirect(request.referrer or url_for("pharmacy.index"))

    try:
        dispensed_drug = void_dispensed_drug(
            dispensed_drug_id, void_reason, current_user.id
        )
        db.session.commit()
        flash("Dispensing record voided and stock restored successfully.", "success")

    except VoidError as exc:
        db.session.rollback()
        flash(str(exc), "warning")

    except (SQLAlchemyError, ValueError) as exc:
        db.session.rollback()
        logger.error("Error voiding dispensed drug %s: %s", dispensed_drug_id, exc)
        flash("Something went wrong. Please try again.", "error")

    return redirect(request.referrer or url_for("pharmacy.index"))


@bp.route("/save_dispensed_drugs", methods=["POST"])
@login_required
@roles_required("pharmacy", "admin")
def save_dispensed_drugs():
    """Saves dispensed drugs, updates stock from multiple batches if needed, and marks prescription as completed."""
    try:
        logger.debug("Entering save_dispensed_drugs route")

        # Get form data
        prescription_id = request.form.get("prescription_id")
        logger.debug(f"Received prescription_id from form: '{prescription_id}'")

        if not prescription_id:
            logger.debug("Prescription ID is missing or empty")
            flash("Prescription ID is required!", "error")
            return redirect(url_for("pharmacy.index"))

        # Parse updated drugs from form data
        logger.debug("Parsing updated drugs from form data")
        updated_drugs = []
        for key in request.form:
            if key.startswith("updatedDrugs["):
                drug_id = key[len("updatedDrugs[") : -1]
                quantity = request.form.get(key)
                updated_drugs.append({"id": drug_id, "quantity": quantity})
                logger.debug(
                    f"Found updated drug - ID: {drug_id}, Quantity: {quantity}"
                )

        if not updated_drugs:
            logger.debug("No updated drugs found in form data")
            flash("No drugs to update!", "error")
            return redirect(url_for("pharmacy.index"))

        # --- T3.7: Encounter Scoping Check ---
        # Must run BEFORE any stock deduction so a closed encounter can never
        # consume stock that then has to be manually reversed.
        first_med = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .first()
        )
        if first_med and not is_encounter_open_for_dispensing(first_med.encounter_id):
            flash(
                "Cannot dispense: The associated encounter is closed or the patient has been discharged.",
                "danger",
            )
            return redirect(request.referrer or url_for("pharmacy.index"))
        # -------------------------------------

        logger.debug(f"Processing {len(updated_drugs)} updated drugs")
        for drug_data in updated_drugs:
            drug_id = drug_data.get("id")
            new_quantity = int(drug_data.get("quantity", 0))
            logger.debug(f"Processing drug ID: {drug_id}, New Quantity: {new_quantity}")

            if new_quantity <= 0:
                logger.debug(f"Invalid quantity ({new_quantity}) for drug ID {drug_id}")
                flash("Quantity must be greater than 0!", "error")
                return redirect(url_for("pharmacy.index"))

            dispensed_drug = db.session.get(DispensedDrug, drug_id)
            if not dispensed_drug:
                logger.debug(f"Dispensed drug ID {drug_id} not found")
                flash(f"Dispensed drug {drug_id} not found!", "error")
                return redirect(url_for("pharmacy.index"))

            drug = db.session.get(Drug, dispensed_drug.drug_id)
            if not drug:
                logger.debug(f"Drug ID {dispensed_drug.drug_id} not found")
                flash("Drug not found!", "error")
                return redirect(url_for("pharmacy.index"))

            quantity_difference = new_quantity - dispensed_drug.quantity_dispensed
            logger.debug(
                f"Quantity difference for drug ID {drug_id}: {quantity_difference}"
            )

            if quantity_difference > 0:
                # Get all batches for this drug, ordered by expiry date
                batches = (
                    Batch.query.filter_by(drug_id=dispensed_drug.drug_id)
                    .order_by(Batch.expiry_date.asc())
                    .all()
                )
                if not batches:
                    logger.debug(
                        f"No batches found for drug ID {dispensed_drug.drug_id}"
                    )
                    flash("No batches available for this drug!", "error")
                    return redirect(url_for("pharmacy.index"))

                # Calculate total available stock across all batches
                total_available = sum(batch.quantity_in_stock for batch in batches)
                logger.debug(
                    f"Total available stock for drug ID {dispensed_drug.drug_id}: {total_available}"
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
                            # Finding A: ledger entry for each batch segment
                            record_movement(
                                item_type="DRUG",
                                item_id=drug.id,
                                movement_type="DISPENSED",
                                quantity_delta=-qty_to_dispense,
                                balance_after=drug.quantity_in_stock,
                                reference_type="PRESCRIPTION",
                                reference_id=prescription_id,
                                user_id=current_user.id,
                                batch_id=batch.id,
                                notes=f"Partial FEFO dispense (insufficient stock) to patient {dispensed_drug.patient_id}",
                            )
                            total_dispensed += qty_to_dispense
                            remaining_quantity -= qty_to_dispense
                            logger.debug(
                                f"Dispensed {qty_to_dispense} from batch ID {batch.id}, Remaining: {remaining_quantity}"
                            )

                    shortfall = quantity_difference - total_dispensed
                    logger.debug(
                        f"Insufficient stock. Dispensed: {total_dispensed}, Shortfall: {shortfall}"
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
                        logger.debug(
                            f"Dispensed {qty_from_original} from original batch ID {original_batch.id}, Remaining: {remaining_quantity}"
                        )
                        db.session.add(original_batch)
                        db.session.add(drug)
                        db.session.add(dispensed_drug)
                        # Finding A: ledger entry for original-batch segment
                        record_movement(
                            item_type="DRUG",
                            item_id=drug.id,
                            movement_type="DISPENSED",
                            quantity_delta=-qty_from_original,
                            balance_after=drug.quantity_in_stock,
                            reference_type="PRESCRIPTION",
                            reference_id=prescription_id,
                            user_id=current_user.id,
                            batch_id=original_batch.id,
                            notes=f"Updated dispense qty (original batch) for patient {dispensed_drug.patient_id}",
                        )

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
                            logger.debug(
                                f"Dispensed {qty_from_next} from next batch ID {next_batch.id}, Remaining: {remaining_quantity}"
                            )
                            db.session.add(next_batch)
                            db.session.add(drug)
                            db.session.add(new_dispensed_drug)
                            # Finding A: ledger entry for each overflow batch segment
                            record_movement(
                                item_type="DRUG",
                                item_id=drug.id,
                                movement_type="DISPENSED",
                                quantity_delta=-qty_from_next,
                                balance_after=drug.quantity_in_stock,
                                reference_type="PRESCRIPTION",
                                reference_id=prescription_id,
                                user_id=current_user.id,
                                batch_id=next_batch.id,
                                notes=f"Updated dispense qty (overflow batch) for patient {dispensed_drug.patient_id}",
                            )
                        batch_index += 1

            elif quantity_difference < 0:
                # Reduce quantity in original batch (stock is returned — delta is positive)
                batch = Batch.query.filter_by(id=dispensed_drug.batch_id).first()
                qty_returned = -quantity_difference  # positive amount returned to stock
                batch.quantity_in_stock += qty_returned
                drug.quantity_in_stock += qty_returned
                dispensed_drug.quantity_dispensed = new_quantity
                logger.debug(
                    f"Reduced quantity - Batch stock: {batch.quantity_in_stock}, Drug stock: {drug.quantity_in_stock}, Dispensed qty: {dispensed_drug.quantity_dispensed}"
                )
                db.session.add(batch)
                db.session.add(drug)
                db.session.add(dispensed_drug)
                # Finding A: ledger entry for stock-return portion
                record_movement(
                    item_type="DRUG",
                    item_id=drug.id,
                    movement_type="VOID_RETURN",
                    quantity_delta=qty_returned,
                    balance_after=drug.quantity_in_stock,
                    reference_type="PRESCRIPTION",
                    reference_id=prescription_id,
                    user_id=current_user.id,
                    batch_id=dispensed_drug.batch_id,
                    notes=f"Quantity reduced on dispense record for patient {dispensed_drug.patient_id}",
                )

        logger.debug(
            f"Querying prescribed medicines for prescription_id '{prescription_id}'"
        )
        prescribed_meds = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .all()
        )

        logger.debug(f"Found {len(prescribed_meds)} prescribed medicines")
        for med in prescribed_meds:
            logger.debug(
                f"PrescribedMedicine ID: {med.id}, Status: {med.status}, Prescription ID: '{med.prescription_id}'"
            )

        if not prescribed_meds:
            logger.debug(
                f"No prescribed medicines found for prescription_id '{prescription_id}'"
            )
            flash("Prescription not found!", "error")
            return redirect(url_for("pharmacy.index"))

        # Update status of all matching prescribed medicines
        logger.debug(f"Updating status to 1 for prescription_id '{prescription_id}'")
        updated_rows = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .update({"status": 1})
        )
        logger.debug(f"Updated {updated_rows} prescribed medicine rows to status=1")

        # Advance encounter stage: prescription fully dispensed → let visit_closure
        # determine the correct next stage (AWAITING_FINAL_BILLING or DISCHARGED)
        from departments.shared.visit_closure import advance_after_completion

        patient_id = prescribed_meds[0].patient_id if prescribed_meds else None
        if patient_id:
            advance_after_completion(patient_id)

        # Verify before commit
        pre_commit_meds = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .all()
        )
        for med in pre_commit_meds:
            logger.debug(
                f"PRE-COMMIT: PrescribedMedicine ID: {med.id}, Status: {med.status}"
            )

        db.session.commit()
        logger.debug("Database commit successful")

        # Verify after commit
        post_commit_meds = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .all()
        )
        for med in post_commit_meds:
            logger.debug(
                f"POST-COMMIT: PrescribedMedicine ID: {med.id}, Status: {med.status}"
            )

        flash("Dispensed drugs updated successfully!", "success")
        logger.debug("Redirecting to pharmacy.index with success message")
        return redirect(url_for("pharmacy.index"))

    except (SQLAlchemyError, ValueError, KeyError) as e:
        db.session.rollback()
        logger.debug(f"Exception occurred: {e!s}")
        flash("Something went wrong. Please try again.", "error")
        logger.debug("Redirecting to pharmacy.index with error message")
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

        # Finding C: build a set of authorised generic names from the prescription so
        # we can warn (non-blocking) if the pharmacist selects an unrelated drug.
        prescribed_names = {
            pm.medicine.generic_name.strip().lower()
            for pm in prescribed_medicines
            if pm.medicine and pm.medicine.generic_name
        }

        # Phase-1 payment gate
        unpaid = has_unpaid_charges(patient_id)
        if unpaid and current_app.config.get("PHARMACY_REQUIRE_PAID", False):
            flash(
                "Dispensing blocked: patient has unsettled charges. "
                "Complete billing first (or disable PHARMACY_REQUIRE_PAID).",
                "error",
            )
            return redirect(
                url_for("pharmacy.view_prescriptions", patient_id=patient_id)
            )
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
            drug = db.session.get(Drug, drug_id)
            if not drug:
                flash(f"Drug with ID {drug_id} does not exist!", "error")
                return redirect(
                    url_for("pharmacy.view_prescriptions", patient_id=patient_id)
                )

            # Finding C: warn if dispensed drug name does not appear in the prescription.
            # Non-blocking — pharmacist may be substituting a generic/brand equivalent,
            # but the discrepancy must be visible and logged.
            if (
                prescribed_names
                and drug.generic_name.strip().lower() not in prescribed_names
            ):
                flash(
                    f"Warning: '{drug.generic_name}' is not listed in this prescription. "
                    "Verify with the prescribing clinician before dispensing.",
                    "warning",
                )
                db.session.add(
                    Log(
                        level="WARNING",
                        message=(
                            f"Pharmacy name-mismatch: drug '{drug.generic_name}' (id={drug.id}) "
                            f"dispensed against prescription {prescription_id} "
                            f"(prescribed: {', '.join(prescribed_names) or 'unknown'}) "
                            f"for patient {patient_id}."
                        ),
                        user_id=current_user.id,
                        source="pharmacy",
                    )
                )

            # Check if the batch exists and has sufficient stock
            # with_for_update() locks the batch row so two concurrent dispenses
            # cannot both pass the stock check and over-dispense.
            batch = (
                Batch.query.filter_by(batch_number=batch_number, drug_id=drug.id)
                .with_for_update()
                .first()
            )
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
                date_dispensed=datetime.now(timezone.utc).date(),
            )
            db.session.add(new_dispensed_drug)

            # Finding B: keep drug-level stock cache in sync with batch deduction.
            # Finding A: append a DISPENSED row to the stock-movement ledger.
            batch.quantity_in_stock -= quantity_dispensed
            drug.quantity_in_stock = max(0, drug.quantity_in_stock - quantity_dispensed)
            db.session.add(batch)
            db.session.add(drug)
            record_movement(
                item_type="DRUG",
                item_id=drug.id,
                movement_type="DISPENSED",
                quantity_delta=-quantity_dispensed,
                balance_after=drug.quantity_in_stock,
                reference_type="PRESCRIPTION",
                reference_id=prescription_id,
                user_id=current_user.id,
                batch_id=batch.id,
                notes=f"Dispensed to patient {patient_id} via save_prescription",
            )

        # Commit changes to the database
        db.session.commit()

        flash("Drugs dispensed successfully!", "success")
        return redirect(url_for("pharmacy.view_prescriptions", patient_id=patient_id))

    except (SQLAlchemyError, ValueError, KeyError) as e:
        flash("Something went wrong. Please try again.", "error")
        logger.debug(f"Error in pharmacy.save_prescription: {e}")
        db.session.rollback()  # Rollback changes in case of error
        return redirect(
            url_for(
                "pharmacy.view_prescriptions", patient_id=request.form.get("patient_id")
            )
        )
