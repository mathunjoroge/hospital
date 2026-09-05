from flask import render_template, redirect, url_for, request, flash, jsonify, Response, session
from sqlalchemy import text
from extensions import db 
import logging
from sqlalchemy.sql import func
from flask_login import login_required, current_user
from departments.models.user import User 
from . import bp  # Import the blueprint
from departments.rbac import roles_required
from departments.models.records import PatientWaitingList,Patient
from departments.models.medicine import PrescribedMedicine
from sqlalchemy.orm import joinedload
from sqlalchemy.sql import text  # Import the text function
from datetime import timedelta,datetime
from flask import render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from departments.models.pharmacy import Drug,Batch,Purchase,DispensedDrug, Expiry,DrugRequest, RequestItem # Import PatientWaitingList and Patient models
from departments.models.billing import DrugsBill
from sqlalchemy.orm import joinedload
import os
import csv
from io import StringIO
import uuid  # Import the uuid module
from collections import defaultdict, Counter
import json
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@bp.route('/')
@login_required
@roles_required('pharmacy', 'admin')
def index():

    try:
        # Fetch pending prescriptions (status=0) and join with Patient
        pending_prescriptions = db.session.query(PrescribedMedicine, Patient)\
            .join(Patient, PrescribedMedicine.patient_id == Patient.patient_id)\
            .filter(PrescribedMedicine.status == 0)\
            .order_by(PrescribedMedicine.id.desc())\
            .all()
        return render_template('pharmacy/index.html', prescriptions=[(p, pt) for p, pt in pending_prescriptions])
    except Exception as e:
        flash(f'Error fetching prescriptions: {str(e)}', 'error')
        return redirect(url_for('pharmacy.index'))
    
#fech expiries
@bp.route('/expiries', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def expiries():
    """Displays only expired medications in the inventory."""
    try:
        today = datetime.today().date()

        expired_drugs = db.session.query(
            Drug.generic_name,
            Drug.brand_name,
            Drug.dosage_form,
            Drug.strength,
            Batch.id.label('batch_id'),
            Batch.batch_number,
            Batch.quantity_in_stock.label('batch_quantity'),
            Batch.expiry_date,
            func.sum(Batch.quantity_in_stock).over(partition_by=Drug.id).label('total_stock')
        ).join(
            Batch, Drug.id == Batch.drug_id
        ).filter(
            Batch.expiry_date < today
        ).order_by(
            Batch.expiry_date.desc()
        ).all()

        return render_template(
            'pharmacy/expiries.html',
            expired_drugs=expired_drugs
        )

    except Exception as e:
        flash(f'Error fetching expired drugs: {e}', 'error')
        print(f"Debug: Error in pharmacy.expiries: {e}")
        return redirect(url_for('home'))

@bp.route('/remove_batch/<int:batch_id>', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
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
            removal_date=datetime.today().date()
        )
        db.session.add(expiry_record)

        # Update drugs.quantity_in_stock
        drug.quantity_in_stock -= batch.quantity_in_stock
        if drug.quantity_in_stock < 0:  # Prevent negative stock
            drug.quantity_in_stock = 0

        # Remove the batch
        db.session.delete(batch)
        db.session.commit()

        flash(f'Batch {batch.batch_number} removed from inventory.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error removing batch: {e}', 'error')
        print(f"Debug: Error in pharmacy.remove_batch: {e}")
    return redirect(url_for('pharmacy.expiries'))

@bp.route('/remove_all_expiries', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def remove_all_expiries():
    """Removes all expired batches from inventory and logs them in expiries."""
    try:
        today = datetime.today().date()

        # Fetch all expired batches
        expired_batches = Batch.query.filter(Batch.expiry_date < today).all()

        if not expired_batches:
            flash('No expired batches to remove.', 'info')
            return redirect(url_for('pharmacy.expiries'))

        for batch in expired_batches:
            drug = Drug.query.get(batch.drug_id)

            # Log each batch in expiries table
            expiry_record = Expiry(
                drug_id=batch.drug_id,
                batch_number=batch.batch_number,
                quantity_removed=batch.quantity_in_stock,
                expiry_date=batch.expiry_date,
                removal_date=today
            )
            db.session.add(expiry_record)

            # Update drugs.quantity_in_stock
            drug.quantity_in_stock -= batch.quantity_in_stock
            if drug.quantity_in_stock < 0:  # Prevent negative stock
                drug.quantity_in_stock = 0

            # Remove the batch
            db.session.delete(batch)

        db.session.commit()
        flash(f'Removed {len(expired_batches)} expired batches from inventory.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error removing all expired batches: {e}', 'error')
        print(f"Debug: Error in pharmacy.remove_all_expiries: {e}")
    return redirect(url_for('pharmacy.expiries'))
#expires report
@bp.route('/expiries_report', methods=['GET', 'POST'])
@login_required
@roles_required('pharmacy', 'admin')
def expiries_report():
    """Generates a report of expired batches removed within a date range."""
    try:
        # Default date range: last 30 days
        default_end = datetime.today().date()
        default_start = default_end - timedelta(days=30)

        start_date = request.form.get('start_date', default_start.strftime('%Y-%m-%d'))
        end_date = request.form.get('end_date', default_end.strftime('%Y-%m-%d'))

        # Convert string dates to datetime objects
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format. Use YYYY-MM-DD.', 'error')
            start_date, end_date = default_start, default_end

        # Query expiries within the date range, joining with drugs for name
        report_data = db.session.query(
            Drug.generic_name,
            Drug.brand_name,
            Expiry.batch_number,
            Expiry.quantity_removed,
            Expiry.expiry_date,
            Expiry.removal_date
        ).join(
            Drug, Expiry.drug_id == Drug.id
        ).filter(
            Expiry.removal_date >= start_date,
            Expiry.removal_date <= end_date
        ).order_by(
            Expiry.removal_date.desc()
        ).all()

        # Calculate total quantity removed
        total_removed = sum(item.quantity_removed for item in report_data)

        return render_template(
            'pharmacy/expiries_report.html',
            report_data=report_data,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            total_removed=total_removed
        )

    except Exception as e:
        flash(f'Error generating expiries report: {e}', 'error')
        print(f"Debug: Error in pharmacy.expiries_report: {e}")
        return redirect(url_for('pharmacy.index'))

@bp.route('/inventory', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def inventory():
    """Displays the full inventory with expiry categories."""
    try:
        # Define a threshold for near expiry (e.g., within 30 days)
        today = datetime.today().date()
        near_expiry_threshold = today + timedelta(days=30)

        # Query the database with total stock as sum of batch quantities
        inventory_data = db.session.query(
            Drug.generic_name,
            Drug.brand_name,
            Drug.dosage_form,
            Drug.strength,
            Batch.batch_number,
            Batch.quantity_in_stock.label('batch_quantity'),
            Batch.expiry_date,
            func.sum(Batch.quantity_in_stock).over(partition_by=Drug.id).label('total_stock')
        ).join(
            Batch, Drug.id == Batch.drug_id
        ).order_by(
            Batch.expiry_date.desc()  # Order by expiry date descending
        ).all()

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
            'pharmacy/inventory.html',
            expired_drugs=expired_drugs,
            near_expiry_drugs=near_expiry_drugs,
            normal_stock_drugs=normal_stock_drugs
        )

    except Exception as e:
        flash(f'Error fetching inventory: {e}', 'error')
        print(f"Debug: Error in pharmacy.inventory: {e}")
        return redirect(url_for('home'))

    #prescriptions
@bp.route('/prescriptions', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def prescriptions():
    """Displays all active prescriptions."""
    try:
        # Fetch all active prescriptions
        active_prescriptions = PrescribedMedicine.query.filter(PrescribedMedicine.num_days > 0).options(
            joinedload(PrescribedMedicine.medicine),
            joinedload(PrescribedMedicine.patient)
        ).all()

        return render_template(
            'pharmacy/prescriptions.html',
            prescriptions=active_prescriptions
        )

    except Exception as e:
        flash(f'Error fetching prescriptions: {e}', 'error')
        print(f"Debug: Error in pharmacy.prescriptions: {e}")
        return redirect(url_for('pharmacy.index'))    
#record purchase
from datetime import datetime

@bp.route('/record_purchase', methods=['GET', 'POST'])
@login_required
@roles_required('pharmacy', 'admin')
def record_purchase():
    """Handles recording a new purchase of medications."""
    try:
        if request.method == 'POST':
            # Extract form data
            drug_ids = request.form.getlist('drug_ids[]')
            batch_numbers = request.form.getlist('batch_numbers[]')
            quantities = request.form.getlist('quantities[]')
            unit_costs = request.form.getlist('unit_costs[]')

            # Validate input
            if not all([drug_ids, batch_numbers, quantities, unit_costs]):
                flash('All fields are required!', 'error')
                return redirect(url_for('pharmacy.record_purchase'))

            # Record each purchase
            for drug_id, batch_number, quantity, unit_cost in zip(drug_ids, batch_numbers, quantities, unit_costs):
                if not quantity.strip() or not unit_cost.strip():
                    continue  # Skip empty entries

                # Fetch the drug and batch
                drug = Drug.query.get_or_404(int(drug_id))
                batch = Batch.query.filter_by(drug_id=drug.id, batch_number=batch_number).first()

                if not batch:
                    # Create a new batch if it doesn't exist
                    batch = Batch(
                        drug_id=drug.id,
                        batch_number=batch_number,
                        expiry_date=None,  # Expiry date can be added later
                        quantity_in_stock=0
                    )
                    db.session.add(batch)
                    db.session.commit()

                # Update batch stock level
                batch.quantity_in_stock += int(quantity)

                # Record the purchase
                new_purchase = Purchase(
                    drug_id=drug.id,
                    batch_id=batch.id,
                    purchase_date=datetime.today().date(),
                    quantity_purchased=int(quantity),
                    unit_cost=float(unit_cost),
                    total_cost=float(unit_cost) * int(quantity)
                )
                db.session.add(new_purchase)

            db.session.commit()
            flash('Purchase recorded successfully!', 'success')
            return redirect(url_for('pharmacy.inventory'))

        # Fetch all drugs for the purchase form
        drugs = Drug.query.all()

        return render_template(
            'pharmacy/record_purchase.html',
            drugs=drugs
        )

    except Exception as e:
        flash(f'Error recording purchase: {e}', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in pharmacy.record_purchase: {e}")  # Debugging
        return redirect(url_for('pharmacy.index'))   
    #view prescription


@bp.route('/view_prescriptions/<string:patient_id>', methods=['GET'])
@login_required
@roles_required('medicine', 'pharmacy', 'admin')
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
        prescribed_medicines = PrescribedMedicine.query.filter_by(patient_id=patient_id).options(
            joinedload(PrescribedMedicine.medicine)
        ).all()

        # Group medicines by prescription_id
        prescriptions = defaultdict(list)
        for medicine in prescribed_medicines:
            prescriptions[medicine.prescription_id].append(medicine)

        # Convert defaultdict to a list of dictionaries for easier template handling
        prescription_list = [
            {
                'prescription_id': pres_id,
                'medicines': pres_data
            }
            for pres_id, pres_data in prescriptions.items()
        ]

        # Debug: Print grouped prescriptions
        print(f"Debug: Grouped Prescriptions for Patient {patient.patient_id}:")
        for pres in prescription_list:
            print(f"  - Prescription ID: {pres['prescription_id']}, Medicines: {len(pres['medicines'])}")

        return render_template(
            'pharmacy/view_prescriptions.html',
            patient=patient,
            prescription_list=prescription_list
        )

    except Exception as e:
        # Debug: Log the exception details
        print(f"Debug: Error fetching prescriptions: {e}")
        flash(f'Error fetching prescriptions: {e}', 'error')
        return redirect(url_for('/'))
    
@bp.route('/dispense/<string:prescription_id>', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def dispense_prescription(prescription_id):
    """Displays prescribed medicines with status='0' and available stock for dispensing."""
    try:
        # Filter prescribed medicines by prescription_id and status='0'
        prescribed_medicines = PrescribedMedicine.query.filter_by(
            prescription_id=prescription_id,
            status='0'  # Add status='0' filter
        ).options(
            joinedload(PrescribedMedicine.medicine)
        ).all()

        if not prescribed_medicines:
            flash(f'Prescription with ID {prescription_id} has no medicines with status "0"!', 'info')
            return redirect(url_for('pharmacy.index'))

        drugs = Drug.query.all()
        drug_batches = {
            medicine.medicine.id: Batch.query.filter_by(drug_id=medicine.medicine.id)
                .order_by(Batch.expiry_date.asc()).first()
            for medicine in prescribed_medicines
        }

        dispensed_drugs = DispensedDrug.query.filter_by(prescription_id=prescription_id).all()

        return render_template(
            'pharmacy/dispense_prescription.html',
            prescribed_medicines=prescribed_medicines,
            prescription_id=prescription_id,
            drug_batches=drug_batches,
            drugs=drugs,
            dispensed_drugs=dispensed_drugs
        )

    except Exception as e:
        flash(f'Error fetching prescription: {e}', 'error')
@bp.route('/delete_dispensed_drug/<int:dispensed_drug_id>', methods=['GET'])
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

    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting dispensed drug: {e}", "error")

    return redirect(request.referrer)  # ✅ Redirects to the previous page
@bp.route('/save_dispensed_drugs', methods=['POST'])
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
            return redirect(url_for('pharmacy.index'))

        # Parse updated drugs from form data
        print("DEBUG: Parsing updated drugs from form data")
        updated_drugs = []
        for key in request.form:
            if key.startswith("updatedDrugs["):
                drug_id = key[len("updatedDrugs["):-1]
                quantity = request.form.get(key)
                updated_drugs.append({"id": drug_id, "quantity": quantity})
                print(f"DEBUG: Found updated drug - ID: {drug_id}, Quantity: {quantity}")

        if not updated_drugs:
            print("DEBUG: No updated drugs found in form data")
            flash("No drugs to update!", "error")
            return redirect(url_for('pharmacy.index'))

        print(f"DEBUG: Processing {len(updated_drugs)} updated drugs")
        for drug_data in updated_drugs:
            drug_id = drug_data.get("id")
            new_quantity = int(drug_data.get("quantity", 0))
            print(f"DEBUG: Processing drug ID: {drug_id}, New Quantity: {new_quantity}")

            if new_quantity <= 0:
                print(f"DEBUG: Invalid quantity ({new_quantity}) for drug ID {drug_id}")
                flash("Quantity must be greater than 0!", "error")
                return redirect(url_for('pharmacy.index'))

            dispensed_drug = DispensedDrug.query.get(drug_id)
            if not dispensed_drug:
                print(f"DEBUG: Dispensed drug ID {drug_id} not found")
                flash(f"Dispensed drug {drug_id} not found!", "error")
                return redirect(url_for('pharmacy.index'))

            drug = Drug.query.get(dispensed_drug.drug_id)
            if not drug:
                print(f"DEBUG: Drug ID {dispensed_drug.drug_id} not found")
                flash("Drug not found!", "error")
                return redirect(url_for('pharmacy.index'))

            quantity_difference = new_quantity - dispensed_drug.quantity_dispensed
            print(f"DEBUG: Quantity difference for drug ID {drug_id}: {quantity_difference}")

            if quantity_difference > 0:
                # Get all batches for this drug, ordered by expiry date
                batches = Batch.query.filter_by(drug_id=dispensed_drug.drug_id).order_by(Batch.expiry_date.asc()).all()
                if not batches:
                    print(f"DEBUG: No batches found for drug ID {dispensed_drug.drug_id}")
                    flash("No batches available for this drug!", "error")
                    return redirect(url_for('pharmacy.index'))

                # Calculate total available stock across all batches
                total_available = sum(batch.quantity_in_stock for batch in batches)
                print(f"DEBUG: Total available stock for drug ID {dispensed_drug.drug_id}: {total_available}")

                if total_available < quantity_difference:
                    # Dispense all available stock and report shortfall
                    remaining_quantity = quantity_difference
                    total_dispensed = 0

                    for batch in batches:
                        if remaining_quantity <= 0:
                            break
                        qty_to_dispense = min(remaining_quantity, batch.quantity_in_stock)
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
                                    status=dispensed_drug.status
                                )
                                db.session.add(new_dispensed_drug)
                            batch.quantity_in_stock -= qty_to_dispense
                            drug.quantity_in_stock -= qty_to_dispense
                            total_dispensed += qty_to_dispense
                            remaining_quantity -= qty_to_dispense
                            print(f"DEBUG: Dispensed {qty_to_dispense} from batch ID {batch.id}, Remaining: {remaining_quantity}")

                    shortfall = quantity_difference - total_dispensed
                    print(f"DEBUG: Insufficient stock. Dispensed: {total_dispensed}, Shortfall: {shortfall}")
                    flash(f"Insufficient stock for {drug.generic_name}. Dispensed: {total_dispensed}, Shortfall: {shortfall}", "warning")

                else:
                    # Enough stock available, dispense as before
                    remaining_quantity = quantity_difference
                    batch_index = 0
                    original_batch = Batch.query.filter_by(id=dispensed_drug.batch_id).first()

                    if original_batch and original_batch.quantity_in_stock > 0:
                        qty_from_original = min(remaining_quantity, original_batch.quantity_in_stock)
                        original_batch.quantity_in_stock -= qty_from_original
                        drug.quantity_in_stock -= qty_from_original
                        dispensed_drug.quantity_dispensed += qty_from_original
                        remaining_quantity -= qty_from_original
                        print(f"DEBUG: Dispensed {qty_from_original} from original batch ID {original_batch.id}, Remaining: {remaining_quantity}")
                        db.session.add(original_batch)
                        db.session.add(drug)
                        db.session.add(dispensed_drug)

                    while remaining_quantity > 0 and batch_index < len(batches):
                        next_batch = batches[batch_index]
                        if next_batch.id == dispensed_drug.batch_id:
                            batch_index += 1
                            continue
                        qty_from_next = min(remaining_quantity, next_batch.quantity_in_stock)
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
                                status=dispensed_drug.status
                            )
                            remaining_quantity -= qty_from_next
                            print(f"DEBUG: Dispensed {qty_from_next} from next batch ID {next_batch.id}, Remaining: {remaining_quantity}")
                            db.session.add(next_batch)
                            db.session.add(drug)
                            db.session.add(new_dispensed_drug)
                        batch_index += 1

            elif quantity_difference < 0:
                # Reduce quantity in original batch
                batch = Batch.query.filter_by(id=dispensed_drug.batch_id).first()
                batch.quantity_in_stock -= quantity_difference  # Adds back since difference is negative
                drug.quantity_in_stock -= quantity_difference
                dispensed_drug.quantity_dispensed = new_quantity
                print(f"DEBUG: Reduced quantity - Batch stock: {batch.quantity_in_stock}, Drug stock: {drug.quantity_in_stock}, Dispensed qty: {dispensed_drug.quantity_dispensed}")
                db.session.add(batch)
                db.session.add(drug)
                db.session.add(dispensed_drug)

        # Debug: Check prescribed medicines
        print(f"DEBUG: Querying prescribed medicines for prescription_id '{prescription_id}'")
        prescribed_meds = db.session.query(PrescribedMedicine).filter_by(prescription_id=prescription_id).all()
        print(f"DEBUG: Found {len(prescribed_meds)} prescribed medicines")
        for med in prescribed_meds:
            print(f"DEBUG: PrescribedMedicine ID: {med.id}, Status: {med.status}, Prescription ID: '{med.prescription_id}'")

        if not prescribed_meds:
            print(f"DEBUG: No prescribed medicines found for prescription_id '{prescription_id}'")
            flash("Prescription not found!", "error")
            return redirect(url_for('pharmacy.index'))

        # Update status of all matching prescribed medicines
        print(f"DEBUG: Updating status to 1 for prescription_id '{prescription_id}'")
        updated_rows = db.session.query(PrescribedMedicine).filter_by(prescription_id=prescription_id).update({"status": 1})
        print(f"DEBUG: Updated {updated_rows} prescribed medicine rows to status=1")

        # Verify before commit
        pre_commit_meds = db.session.query(PrescribedMedicine).filter_by(prescription_id=prescription_id).all()
        for med in pre_commit_meds:
            print(f"DEBUG PRE-COMMIT: PrescribedMedicine ID: {med.id}, Status: {med.status}")

        db.session.commit()
        print("DEBUG: Database commit successful")

        # Verify after commit
        post_commit_meds = db.session.query(PrescribedMedicine).filter_by(prescription_id=prescription_id).all()
        for med in post_commit_meds:
            print(f"DEBUG POST-COMMIT: PrescribedMedicine ID: {med.id}, Status: {med.status}")

        flash("Dispensed drugs updated successfully!", "success")
        print("DEBUG: Redirecting to pharmacy.index with success message")
        return redirect(url_for('pharmacy.index'))

    except Exception as e:
        db.session.rollback()
        print(f"DEBUG: Exception occurred: {str(e)}")
        flash(f"Error saving dispensed drugs: {e}", "error")
        print("DEBUG: Redirecting to pharmacy.index with error message")
        return redirect(url_for('pharmacy.index'))

@bp.route('/save_prescription/<string:prescription_id>', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin')
def save_prescription(prescription_id):
    """Save dispensed drugs for a specific prescription."""
    try:
        # Extract form data
        drug_ids = request.form.getlist('drugs[]')  # List of selected drug IDs
        quantities = request.form.getlist('quantity[]')  # List of quantities
        batch_numbers = request.form.getlist('batch_number[]')  # List of batch numbers

        # Validate input
        if not drug_ids or not quantities or not batch_numbers:
            flash('Invalid input! Please select drugs and specify their quantities.', 'error')
            return redirect(url_for('pharmacy.view_prescriptions', patient_id=request.form.get('patient_id')))

        if len(drug_ids) != len(quantities) or len(drug_ids) != len(batch_numbers):
            flash('Mismatched input! Ensure each drug has a corresponding quantity and batch number.', 'error')
            return redirect(url_for('pharmacy.view_prescriptions', patient_id=request.form.get('patient_id')))

        # Fetch the first prescribed medicine to get patient ID
        prescribed_medicines = PrescribedMedicine.query.filter_by(prescription_id=prescription_id).all()
        if not prescribed_medicines:
            flash(f'Prescription with ID {prescription_id} does not exist!', 'error')
            return redirect(url_for('pharmacy.index'))

        patient_id = prescribed_medicines[0].patient_id

        # Process each drug in the form
        for i, drug_id in enumerate(drug_ids):
            try:
                quantity_dispensed = int(quantities[i])
                if quantity_dispensed <= 0:
                    raise ValueError("Quantity must be greater than zero.")
            except (ValueError, IndexError):
                flash(f'Invalid quantity for drug with ID {drug_id}!', 'error')
                return redirect(url_for('pharmacy.view_prescriptions', patient_id=patient_id))

            batch_number = batch_numbers[i]

            # Check if the drug exists
            drug = Drug.query.get(drug_id)
            if not drug:
                flash(f'Drug with ID {drug_id} does not exist!', 'error')
                return redirect(url_for('pharmacy.view_prescriptions', patient_id=patient_id))

            # Check if the batch exists and has sufficient stock
            batch = Batch.query.filter_by(batch_number=batch_number, drug_id=drug.id).first()
            if not batch or batch.quantity_in_stock < quantity_dispensed:
                flash(f'Insufficient stock for {drug.generic_name} ({drug.brand_name}). Available: {batch.quantity_in_stock if batch else "N/A"}', 'error')
                return redirect(url_for('pharmacy.view_prescriptions', patient_id=patient_id))

            # Create a new dispensed drug entry
            new_dispensed_drug = DispensedDrug(
                drug_id=drug.id,
                batch_id=batch.id,
                patient_id=patient_id,
                prescription_id=prescription_id,
                quantity_dispensed=quantity_dispensed,
                date_dispensed=datetime.today().date()
            )
            db.session.add(new_dispensed_drug)

            # Update batch stock
            batch.quantity_in_stock -= quantity_dispensed
            db.session.add(batch)

        # Commit changes to the database
        db.session.commit()

        flash('Drugs dispensed successfully!', 'success')
        return redirect(url_for('pharmacy.view_prescriptions', patient_id=patient_id))

    except Exception as e:
        flash(f'An error occurred while dispensing drugs: {e}', 'error')
        print(f"Debug: Error in pharmacy.save_prescription: {e}")
        db.session.rollback()  # Rollback changes in case of error
        return redirect(url_for('pharmacy.view_prescriptions', patient_id=request.form.get('patient_id')))    

 
@bp.route('/remove_dispensed/<int:dispense_id>', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin')
def remove_dispensed(dispense_id):
    """Remove a dispensed drug entry."""
    try:
        # Fetch the dispensed drug entry by ID
        dispensed_drug = DispensedDrug.query.get(dispense_id)
        if not dispensed_drug:
            flash(f'Dispensed drug with ID {dispense_id} does not exist!', 'error')
            return redirect(url_for('pharmacy.dispense_prescription', prescription_id=request.form.get('prescription_id')))

        # Restore stock in the batch
        batch = Batch.query.get(dispensed_drug.batch_id) if dispensed_drug.batch_id else None
        if batch:
            batch.quantity_in_stock += dispensed_drug.quantity_dispensed
            db.session.add(batch)

        # Delete the dispensed drug entry
        db.session.delete(dispensed_drug)
        db.session.commit()

        flash(f'{dispensed_drug.drug.generic_name} removed from dispensing list!', 'success')
        return redirect(url_for('pharmacy.dispense_prescription', prescription_id=dispensed_drug.prescription_id))

    except Exception as e:
        flash(f'Error removing dispensed drug: {e}', 'error')
        print(f"Debug: Error in pharmacy.remove_dispensed: {e}")  # Debugging
        db.session.rollback()
        return redirect(url_for('pharmacy.dispense_prescription', prescription_id=request.form.get('prescription_id'))) 
@bp.route('/low_stock', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def low_stock():
    """Displays drugs with stock levels below their reorder threshold."""
    try:
        # Subquery to identify drugs with total stock below reorder_level
        total_stock_subquery = db.session.query(
            Batch.drug_id,
            func.sum(Batch.quantity_in_stock).label('total_stock')
        ).group_by(Batch.drug_id).having(
            func.sum(Batch.quantity_in_stock) < db.session.query(Drug.reorder_level).filter(Drug.id == Batch.drug_id).scalar_subquery()
        ).subquery()

        # Main query to fetch drug details with total stock
        low_stock_drugs = db.session.query(
            Drug.generic_name,
            Drug.brand_name,
            Drug.dosage_form,
            Drug.strength,
            Drug.reorder_level,
            total_stock_subquery.c.total_stock.label('current_stock')
        ).join(
            total_stock_subquery, Drug.id == total_stock_subquery.c.drug_id
        ).all()

        # Debug: Print results to verify data
        if not low_stock_drugs:
            print("Debug: No low stock drugs found.")
        for drug in low_stock_drugs:
            print(f"Drug: {drug.generic_name}, Stock: {drug.current_stock}, Reorder: {drug.reorder_level}")

        return render_template(
            'pharmacy/low_stock.html',
            low_stock_drugs=low_stock_drugs
        )

    except Exception as e:
        flash(f'Error fetching low stock data: {e}', 'error')
        print(f"Debug: Error in pharmacy.low_stock: {e}")
        return redirect(url_for('pharmacy.index'))
import uuid

@bp.route('/drug-requests', methods=['GET', 'POST'])
@login_required
@roles_required('pharmacy', 'admin')
def drug_requests():
    """Handles drug requests from the store (only latest request shown)."""
    try:
        if request.method == 'POST':
            drug_id = request.form.get('drug_id')
            quantity = request.form.get('quantity')

            if not drug_id or not quantity or int(quantity) <= 0:
                flash('Please select a drug and enter a valid quantity.', 'error')
                return redirect(url_for('pharmacy.drug_requests'))

            # Check if an open request exists for the user
            existing_request = DrugRequest.query.filter_by(
                requested_by=current_user.id,
                status='Submitted'
            ).first()

            if not existing_request:
                # Create a new request with a unique UUID
                existing_request = DrugRequest(
                    request_uuid=str(uuid.uuid4()),  # Generate a unique UUID
                    request_date=datetime.today().date(),
                    status='Pending',
                    requested_by=current_user.id
                )
                db.session.add(existing_request)
                db.session.flush()  # Get request ID before commit

            # Add drug to request
            item = RequestItem(
                request_id=existing_request.id,
                drug_id=int(drug_id),
                quantity_requested=int(quantity),
                quantity_issued=0  # Default
            )
            db.session.add(item)
            db.session.commit()

            flash('Drug request added successfully.', 'success')
            return redirect(url_for('pharmacy.drug_requests'))

        # Fetch only the latest request for the current user
        latest_request = DrugRequest.query.filter_by(requested_by=current_user.id, status='Pending') \
            .order_by(DrugRequest.request_date.desc()).first()

        all_drugs = Drug.query.all()  # Get available drugs

        return render_template(
            'pharmacy/drug_requests.html',
            all_drugs=all_drugs,
            latest_request=latest_request
        )

    except Exception as e:
        db.session.rollback()
        flash(f'Error processing request: {e}', 'error')
        print(f"Debug: Error in pharmacy.drug_requests: {e}")
        return redirect(url_for('pharmacy.index'))
#save order
@bp.route('/save-order', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def save_order():
    """Finalizes the current drug request and redirects to the dashboard."""
    try:
        # Get the latest pending request for the user
        latest_request = DrugRequest.query.filter_by(
            requested_by=current_user.id,
            status='Pending'
        ).first()

        if latest_request:
            latest_request.status = 'Submitted'  # Mark the request as submitted
            db.session.commit()
            flash('Order saved successfully.', 'success')

        return redirect(url_for('pharmacy.index'))  # Redirect to pharmacy dashboard

    except Exception as e:
        db.session.rollback()
        flash(f'Error saving order: {e}', 'error')
        print(f"Debug: Error in pharmacy.save_order: {e}")
        return redirect(url_for('pharmacy.index'))
@bp.route('/pending-requests', methods=['GET', 'POST'])
@login_required
@roles_required('pharmacy', 'admin')
def pending_requests():
    """Displays pending drug requests with date range filtering."""
    try:
        # Query with User's username instead of ID
        query = db.session.query(
            DrugRequest.id,
            DrugRequest.request_date,
            DrugRequest.status,
            User.username.label("requested_by")  # Fetch `username`
        ).join(User, User.id == DrugRequest.requested_by) \
         .filter(DrugRequest.status == 'Submitted') \
         .order_by(DrugRequest.request_date.desc())

        # Date filtering
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        if start_date and end_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            query = query.filter(DrugRequest.request_date.between(start_date, end_date))

        pending_requests = query.limit(5).all()

        return render_template('pharmacy/pending_requests.html', pending_requests=pending_requests)

    except Exception as e:
        flash(f'Error fetching pending requests: {e}', 'error')
        return redirect(url_for('pharmacy.index'))
#pending requests details
@bp.route('/pending-request-details/<int:request_id>', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def pending_request_details(request_id):
    """Displays the details of a pending drug request."""
    # Fetch request details with username
    request_details = db.session.query(
        DrugRequest,
        User.username.label("requested_by")
    ).join(User, User.id == DrugRequest.requested_by) \
     .filter(DrugRequest.id == request_id) \
     .first_or_404()

    # Extract the DrugRequest instance
    drug_request = request_details.DrugRequest  # This is the actual model instance
    requested_by = request_details.requested_by  # Extract the username

    return render_template('pharmacy/pending_request_details.html', drug_request=drug_request, requested_by=requested_by)    


@bp.route('/served-requests', methods=['GET', 'POST'])
@login_required
@roles_required('pharmacy', 'admin')
def served_requests():
    """Displays served drug requests with date range filtering."""
    try:
        # Query with User's username instead of ID
        query = db.session.query(
            DrugRequest.id,
            DrugRequest.request_date,
            DrugRequest.status,
            User.username.label("requested_by")  # Fetch `username`
        ).join(User, User.id == DrugRequest.requested_by) \
         .filter(DrugRequest.status == 'Completed') \
         .order_by(DrugRequest.request_date.desc())

        # Date filtering
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        if start_date and end_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            query = query.filter(DrugRequest.request_date.between(start_date, end_date))

        served_requests = query.limit(5).all()  # Fetch last 5 served requests

        return render_template('pharmacy/served_requests.html', served_requests=served_requests)

    except Exception as e:
        flash(f'Error fetching served requests: {e}', 'error')
        return redirect(url_for('pharmacy.index'))


# Served Request Details Route
@bp.route('/served-request-details/<int:request_id>', methods=['GET'])
@login_required
@roles_required('pharmacy', 'admin')
def served_requests_details(request_id):
    """Displays the details of a served drug request."""
    # Fetch request details with username
    request_details = db.session.query(
        DrugRequest,
        User.username.label("requested_by")
    ).join(User, User.id == DrugRequest.requested_by) \
     .filter(DrugRequest.id == request_id) \
     .first_or_404()

    # Extract variables properly
    drug_request, requested_by = request_details  # Correct unpacking

    return render_template('pharmacy/served_request_details.html', drug_request=drug_request, requested_by=requested_by)

@bp.route('/get_all_batches', methods=['GET'])
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
            return jsonify({'error': 'No available drugs'}), 200

        # Convert to JSON format
        batch_list = [
            {
                'drug_id': batch.drug_id,
                'generic_name': batch.drug.generic_name,
                'brand_name': batch.drug.brand_name,
                'dosage_form': batch.drug.dosage_form,
                'strength': batch.drug.strength,
                'selling_price': batch.drug.selling_price,
                'batch_qty': batch.quantity_in_stock,
                'batch_id': batch.id
            }
            for batch in batches
        ]

        print(f"📌 DEBUG: API Response: {len(batch_list)} unique batches returned")
        return jsonify(batch_list), 200

    except Exception as e:
        print(f"❌ ERROR: {e}")  # Debugging
        return jsonify({'error': f'Error fetching batches: {str(e)}'}), 500

@bp.route('/dispense/process/<string:prescription_id>', methods=['POST'])
@login_required
@roles_required('pharmacy', 'admin')
def process_dispense(prescription_id):
    """
    Handles dispensing of drugs, updates stock, and renders the dispense prescription page with dispensed drug details.
    """
    # Check user permissions
    try:
        # Extract form data
        drug_id = request.form.get('drug_id')
        batch_id = request.form.get('batch_id')
        quantity_dispensed = request.form.get('quantity_dispensed')

        # Validate form inputs
        if not drug_id or not batch_id or not quantity_dispensed:
            flash('Invalid input! Please select a drug and specify its quantity.', 'error')
            return redirect(url_for('pharmacy.dispense_prescription', prescription_id=prescription_id))

        try:
            quantity_dispensed = int(quantity_dispensed)
            if quantity_dispensed <= 0:
                raise ValueError("Quantity must be greater than zero.")
        except ValueError:
            flash('Invalid quantity! Enter a positive number.', 'error')
            return redirect(url_for('pharmacy.dispense_prescription', prescription_id=prescription_id))

        # Fetch the selected drug
        drug = Drug.query.get(drug_id)
        if not drug:
            flash(f'Drug with ID {drug_id} does not exist!', 'error')
            return redirect(url_for('pharmacy.dispense_prescription', prescription_id=prescription_id))

        # Fetch the selected batch
        batch = Batch.query.filter_by(id=batch_id, drug_id=drug.id).first()
        if not batch:
            flash(f'Batch ID {batch_id} does not exist for {drug.generic_name}.', 'error')
            return redirect(url_for('pharmacy.dispense_prescription', prescription_id=prescription_id))

        if batch.quantity_in_stock < quantity_dispensed:
            flash(f'Insufficient stock for {drug.generic_name}. Available: {batch.quantity_in_stock}', 'error')
            return redirect(url_for('pharmacy.dispense_prescription', prescription_id=prescription_id))

        # Fetch the prescription
        prescribed_medicine = PrescribedMedicine.query.filter_by(prescription_id=prescription_id).first()
        if not prescribed_medicine:
            flash('Prescription not found.', 'error')
            return redirect(url_for('pharmacy.index'))

        # Get patient ID from the prescription
        patient_id = prescribed_medicine.patient_id

        # Create a new dispensed drug entry
        new_dispensed_drug = DispensedDrug(
            drug_id=drug.id,
            batch_id=batch.id,
            patient_id=patient_id,
            prescription_id=prescription_id,
            quantity_dispensed=quantity_dispensed,
            date_dispensed=datetime.today().date(),
            status="Pending"
        )
        db.session.add(new_dispensed_drug)

        # Update batch stock
        batch.quantity_in_stock -= quantity_dispensed
        db.session.add(batch)

        # Commit changes to the database
        db.session.commit()
        flash(f'{drug.generic_name} ({quantity_dispensed} units) dispensed successfully!', 'success')

        # Fetch all dispensed drugs for this prescription
        dispensed_drugs = (
            db.session.query(
                Drug.generic_name,
                Drug.brand_name,
                Drug.dosage_form,
                Drug.strength,
                Drug.selling_price,
                DispensedDrug.id,
                DispensedDrug.batch_id,
                DispensedDrug.quantity_dispensed,
                (Drug.selling_price * DispensedDrug.quantity_dispensed).label("total"),
            )
            .join(DispensedDrug, DispensedDrug.drug_id == Drug.id)
            .filter(DispensedDrug.prescription_id == prescription_id)
            .all()
        )

        dispensed_drugs_list = [
            {
                "generic_name": d.generic_name,
                "id": d.id,
                "brand_name": d.brand_name,
                "dosage_form": d.dosage_form,
                "strength": d.strength,
                "selling_price": d.selling_price,
                "batch_id": d.batch_id,
                "quantity_dispensed": d.quantity_dispensed,
                "total": d.total,
            }
            for d in dispensed_drugs
        ]

        prescribed_medicines = PrescribedMedicine.query.filter_by(prescription_id=prescription_id).all()
        drug_batches = {
            b.drug_id: b
            for b in Batch.query.filter(
                Batch.drug_id.in_([m.medicine_id for m in prescribed_medicines])
            ).all()
        }

        return render_template(
            'pharmacy/dispense_prescription.html',
            prescription_id=prescription_id,
            prescribed_medicines=prescribed_medicines,
            drug_batches=drug_batches,
            drugs=Drug.query.all(),
            dispensed_drugs=dispensed_drugs_list
        )

    except Exception as e:
        db.session.rollback()
        flash(f'Error processing dispense: {e}', 'error')
        print(f"Debug: Error in pharmacy.process_dispense: {e}")
        return redirect(url_for('pharmacy.index'))

@bp.route('/patient_history', methods=['GET', 'POST'])
@login_required
def patient_history():
    """API endpoint to retrieve patient history by patient_id with clinical data."""
    from departments.models.medicine import SOAPNote, RequestedLab, LabTest, RequestedImage, Imaging, AdmittedPatient
    from departments.models.nursing import Vitals

    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            patient_id = data.get('patient_id', '').strip()
        else:
            patient_id = request.form.get('patient_id', '').strip()

        if not patient_id:
            if request.is_json:
                return jsonify({'error': 'Patient ID is required'}), 400
            flash('Please provide a patient ID.', 'error')
            return render_template('pharmacy/pharmacy_dashboard.html', patient_history_form=True)

        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            if request.is_json:
                return jsonify({'error': f'No patient found with ID {patient_id}'}), 404
            flash(f'No patient found with ID {patient_id}.', 'error')
            return render_template('pharmacy/pharmacy_dashboard.html', patient_history_form=True)

        prescribed_meds = PrescribedMedicine.query.filter_by(patient_id=patient_id).options(joinedload(PrescribedMedicine.medicine)).all()
        dispensed_drugs = DispensedDrug.query.filter_by(patient_id=patient_id).options(joinedload(DispensedDrug.drug), joinedload(DispensedDrug.batch)).all()
        requested_labs = RequestedLab.query.filter_by(patient_id=patient_id).options(joinedload(RequestedLab.lab_test)).all()
        lab_results = LabResult.query.filter_by(patient_id=patient_id).options(joinedload(LabResult.lab_test)).all()

        test_presentations = {}
        for result in lab_results:
            results_dict = {}
            try:
                results_dict = json.loads(result.result) if result.result else {}
            except json.JSONDecodeError:
                flash(f'Invalid result format for result ID {result.result_id}.', 'warning')

            parameters = LabResultTemplate.query.filter_by(test_id=result.lab_test_id).all()
            test_presentation = []
            for param in parameters:
                result_value = results_dict.get(str(param.id))
                try:
                    result_value_float = float(result_value) if result_value is not None else None
                except (ValueError, TypeError):
                    result_value_float = None

                status = ("Invalid Result" if result_value_float is None else
                          "Low" if result_value_float < param.normal_range_low else
                          "High" if result_value_float > param.normal_range_high else
                          "Normal")

                test_presentation.append({
                    'parameter_name': param.parameter_name,
                    'normal_range_low': param.normal_range_low,
                    'normal_range_high': param.normal_range_high,
                    'unit': param.unit,
                    'result': result_value if result_value is not None else "N/A",
                    'status': status
                })

            test_presentations[result.result_id] = {
                'test_name': result.lab_test.test_name if result.lab_test else "Unknown Test",
                'test_date': result.test_date,
                'result_notes': result.result_notes,
                'parameters': test_presentation
            }

        requested_images = RequestedImage.query.filter_by(patient_id=patient_id).options(joinedload(RequestedImage.imaging)).all()
        imaging_results = ImagingResult.query.filter_by(patient_id=patient_id).options(joinedload(ImagingResult.imaging)).all()
        vitals = Vitals.query.filter_by(patient_id=patient_id).all()
        soap_notes = SOAPNote.query.filter_by(patient_id=patient_id).all()
        clinic_bookings = ClinicBooking.query.filter_by(patient_id=patient_id).options(joinedload(ClinicBooking.clinic)).all()
        admissions = AdmittedPatient.query.filter_by(patient_id=patient_id).options(joinedload(AdmittedPatient.ward)).all()
        billing = Billing.query.filter_by(patient_id=patient_id).options(joinedload(Billing.charge)).all()

        history_data = {
            'patient': {
                'id': patient.id,
                'patient_id': patient.patient_id,
                'name': patient.name,
                'sex': patient.sex,
                'contact': patient.contact,
                'date_registered': patient.date_registered.strftime('%Y-%m-%d %H:%M:%S') if patient.date_registered else None
            },
            'prescribed_meds': [{'id': m.id, 'medicine': m.medicine.generic_name if m.medicine else 'N/A', 'dosage': m.dosage, 'frequency': m.frequency} for m in prescribed_meds],
            'dispensed_drugs': [{'id': d.id, 'drug': d.drug.generic_name if d.drug else 'N/A', 'quantity': d.quantity_dispensed, 'date': d.date_dispensed.strftime('%Y-%m-%d')} for d in dispensed_drugs],
            'requested_labs': [{'id': l.id, 'test_name': l.lab_test.test_name if l.lab_test else 'N/A', 'status': l.status} for l in requested_labs],
            'lab_results': test_presentations,
            'requested_images': [{'id': i.id, 'type': i.imaging.imaging_type if i.imaging else 'N/A', 'status': i.status} for i in requested_images],
            'imaging_results': [{'id': ir.id, 'type': ir.imaging.imaging_type if ir.imaging else 'N/A', 'findings': ir.ai_findings} for ir in imaging_results],
            'vitals': [{'temp': v.temperature, 'bp': f"{v.blood_pressure_systolic}/{v.blood_pressure_diastolic}", 'pulse': v.pulse} for v in vitals],
            'soap_notes': [{'id': s.id, 'assessment': s.assessment, 'recommendation': s.recommendation} for s in soap_notes],
            'clinic_bookings': [{'id': cb.id, 'clinic': cb.clinic.name if cb.clinic else 'N/A', 'date': cb.clinic_date.strftime('%Y-%m-%d')} for cb in clinic_bookings],
            'admissions': [{'id': a.id, 'ward': a.ward.name if a.ward else 'N/A', 'admitted_on': a.admitted_on.strftime('%Y-%m-%d')} for a in admissions],
            'billing': [{'id': b.id, 'charge': b.charge.name if b.charge else 'N/A', 'cost': float(b.total_cost), 'status': b.status} for b in billing]
        }

        if request.is_json:
            return jsonify(history_data), 200

        return render_template(
            'pharmacy/pharmacy_dashboard.html',
            patient=patient,
            prescribed_meds=prescribed_meds,
            dispensed_drugs=dispensed_drugs,
            requested_labs=requested_labs,
            lab_results=test_presentations,
            requested_images=requested_images,
            imaging_results=imaging_results,
            vitals=vitals,
            soap_notes=soap_notes,
            clinic_bookings=clinic_bookings,
            admissions=admissions,
            billing=billing,
            patient_history_form=True
        )

    return render_template('pharmacy/pharmacy_dashboard.html', patient_history_form=True)


@bp.route('/analytics', methods=['GET', 'POST'])
@login_required
def analytics():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    if request.method == 'POST':
        start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d')

    # Sales Trends (from drugs_bill, only paid bills)
    sales = DrugsBill.query.filter(
        DrugsBill.billed_at.between(start_date, end_date),
        DrugsBill.status == 1
    ).all()
    sales_data = {}
    for sale in sales:
        date_key = sale.billed_at.strftime('%Y-%m-%d')
        sales_data[date_key] = sales_data.get(date_key, 0) + float(sale.total_cost)

    # Top Dispensed Drugs (with additional fields)
    dispensed = DispensedDrug.query.filter(
        DispensedDrug.date_dispensed.between(start_date, end_date)
    ).join(Drug, DispensedDrug.drug_id == Drug.id).all()
    drug_counts = Counter([d.drug.generic_name for d in dispensed]).most_common(5)
    top_drugs = []
    for name, count in drug_counts:
        drug = Drug.query.filter_by(generic_name=name).first()  # Get drug details
        top_drugs.append({
            'generic_name': name,
            'brand_name': drug.brand_name if drug else 'N/A',
            'strength': drug.strength if drug else 'N/A',
            'dosage_form': drug.dosage_form if drug else 'N/A',
            'count': count
        })

    # Inventory Usage Rate (with additional fields)
    drugs = Drug.query.all()
    usage_rates = []
    for drug in drugs:
        dispensed_qty = db.session.query(
            db.func.sum(DispensedDrug.quantity_dispensed)
        ).filter(
            DispensedDrug.drug_id == drug.id,
            DispensedDrug.date_dispensed.between(start_date, end_date)
        ).scalar() or 0
        initial_stock = drug.quantity_in_stock + dispensed_qty
        usage = (dispensed_qty / initial_stock * 100) if initial_stock > 0 else 0
        usage_rates.append({
            'generic_name': drug.generic_name,
            'brand_name': drug.brand_name or 'N/A',
            'strength': drug.strength,
            'dosage_form': drug.dosage_form,
            'remaining': drug.quantity_in_stock,
            'usage': usage
        })

    # Expiry Risks (with additional fields)
    expiry_risks = Batch.query.filter(
        Batch.expiry_date <= (datetime.now() + timedelta(days=90)),
        Batch.quantity_in_stock > 0
    ).join(Drug, Batch.drug_id == Drug.id).all()
    expiry_data = [
        {
            'generic_name': batch.drug.generic_name,
            'brand_name': batch.drug.brand_name or 'N/A',
            'strength': batch.drug.strength,
            'dosage_form': batch.drug.dosage_form,
            'batch': batch.batch_number,
            'expiry': batch.expiry_date.strftime('%Y-%m-%d'),
            'qty': batch.quantity_in_stock
        }
        for batch in expiry_risks
    ]

    # Peak Dispensing Hours
    peak_hours = Counter([d.date_dispensed.hour for d in dispensed])
    hours_data = [{'hour': h, 'count': peak_hours.get(h, 0)} for h in range(24)]

    # Store sales_data in session for export
    session['sales_data'] = sales_data
    session['start_date'] = start_date.strftime('%Y-%m-%d')
    session['end_date'] = end_date.strftime('%Y-%m-%d')

    return render_template(
        'pharmacy/analytics.html',
        sales_data=json.dumps(list(sales_data.items())),
        top_drugs=top_drugs,
        usage_rates=usage_rates,
        expiry_data=expiry_data,
        hours_data=json.dumps(hours_data),
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

@bp.route('/analytics/export')
@login_required
def export_analytics():
    sales_data = session.get('sales_data', {})
    start_date = session.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = session.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    if not sales_data:
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        sales = DrugsBill.query.filter(
            DrugsBill.billed_at.between(start_date_dt, end_date_dt),
            DrugsBill.status == 1
        ).all()
        sales_data = {}
        for sale in sales:
            date_key = sale.billed_at.strftime('%Y-%m-%d')
            sales_data[date_key] = sales_data.get(date_key, 0) + float(sale.total_cost)

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Sales'])
    for date, total in sales_data.items():
        writer.writerow([date, total])
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": f"attachment;filename=analytics_sales_{start_date}_to_{end_date}.csv"}
    )

# --- AI Drug Discovery Routes ---

@bp.route('/ai_discovery', methods=['GET', 'POST'])
@login_required
@roles_required('pharmacy', 'admin')
def ai_discovery():
    """Renders the AI Drug Discovery page and handles form submissions for MolMIM and DiffDock."""
    # Lazy import to avoid circular dependencies
    from departments.nlp.src.nvidia_client import NvidiaNIMClient
    
    results = None
    tool_used = None
    
    if request.method == 'POST':
        client = NvidiaNIMClient()
        action = request.form.get('action')
        
        if action == 'molmim':
            properties = request.form.get('target_properties', '').strip()
            if properties:
                molecules = client.generate_molecules(properties)
                results = {'molecules': molecules, 'properties': properties}
                tool_used = 'molmim'
            else:
                flash('Please enter target properties.', 'error')
                
        elif action == 'diffdock':
            ligand = request.form.get('ligand_smiles', '').strip()
            protein = request.form.get('protein_sequence', '').strip()
            if ligand and protein:
                docking_results = client.predict_docking(ligand, protein)
                results = {'docking': docking_results, 'ligand': ligand}
                tool_used = 'diffdock'
            else:
                flash('Please provide both a ligand SMILES string and a protein sequence.', 'error')
                
    return render_template('pharmacy/ai_discovery.html', results=results, tool_used=tool_used)