from flask import render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from extensions import db
from datetime import datetime
from decimal import Decimal, InvalidOperation  # Added InvalidOperation
from . import bp  # Import the blueprint
from departments.models.records import Patient,ClinicBooking,Clinic  # Import Patient model
from departments.models.billing import Billing, Charge, DrugsBill, PaidBill,LabBill, ClinicBill, TheatreBill, ImagingBill, WardBill
from departments.models.pharmacy import Drug,DispensedDrug # Import Drug model
from departments.models.medicine import TheatreList,TheatreProcedure,AdmittedPatient,RequestedImage,RequestedLab,LabTest,Imaging
from sqlalchemy.orm import joinedload
import logging  # Added for logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@bp.route('/index')
@login_required
def index():
    if current_user.role not in ['billing', 'admin']:     
        return redirect(url_for('login'))

    # Fetch all unpaid billings and drug bills with related data (eager loading)
    unpaid_billings = Billing.query.filter_by(status=0).options(
        joinedload(Billing.patient),
        joinedload(Billing.charge)
    ).all()

    unpaid_drug_bills = DrugsBill.query.filter_by(status=0).options(
        joinedload(DrugsBill.patient),
        joinedload(DrugsBill.drug)
    ).all()

    # Group unpaid bills by patient and calculate total cost
    patient_bills = {}
    for bill in unpaid_billings + unpaid_drug_bills:
        if bill.patient_id not in patient_bills:
            patient_bills[bill.patient_id] = {
                "patient": bill.patient,
                "total_cost": Decimal('0')  # Use Decimal for precise arithmetic
            }

        # Add bill's total cost to the patient's total
        if bill.patient:  # Ensure the patient relationship exists
            patient_bills[bill.patient_id]["total_cost"] += Decimal(str(bill.total_cost or '0'))

    # Convert the dictionary to a list for rendering
    patients_with_unpaid_bills = [
        {"patient": data["patient"], "total_cost": data["total_cost"]}
        for patient_id, data in patient_bills.items()
    ]

    return render_template('billing/index.html', patients=patients_with_unpaid_bills)

@bp.route('/new_drugs_billing', methods=['GET', 'POST'])
@login_required
def new_drugs_billing():
    if current_user.role not in ['billing', 'admin']:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Extract form data
        patient_id = request.form.get('patient_id')
        drug_id = request.form.get('drug_id')
        quantity = request.form.get('quantity')

        # Validate input
        if not patient_id or not drug_id or not quantity:
            flash('All fields are required!', 'error')
            return redirect(url_for('billing.new_drugs_billing'))

        try:
            quantity = int(quantity)
            if quantity <= 0:
                raise ValueError("Quantity must be greater than zero.")
        except ValueError:
            flash('Invalid quantity! Please enter a positive integer.', 'error')
            return redirect(url_for('billing.new_drugs_billing'))

        # Check if the patient exists
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f'Patient with ID {patient_id} does not exist!', 'error')
            return redirect(url_for('billing.new_drugs_billing'))

        # Check if the drug exists
        drug = Drug.query.get(drug_id)
        if not drug:
            flash(f'Drug with ID {drug_id} does not exist!', 'error')
            return redirect(url_for('billing.new_drugs_billing'))

        # Calculate total cost explicitly
        total_cost = Decimal(str(quantity)) * Decimal(str(drug.selling_price))

        # Create a new drugs billing entry
        new_drugs_billing = DrugsBill(
            patient_id=patient_id,
            drug_id=drug_id,
            quantity=quantity,
            total_cost=total_cost  # Explicitly set total_cost
        )

        db.session.add(new_drugs_billing)
        db.session.commit()

        logger.debug(f"Total Cost Calculated: {new_drugs_billing.total_cost}")  # Debugging
        flash(f'Drugs billing created successfully for {patient.name}! Total Cost: Kshs {new_drugs_billing.total_cost}', 'success')
        return redirect(url_for('billing.index'))

    # Fetch all patients and drugs for the dropdowns
    patients = Patient.query.all()
    drugs = Drug.query.all()

    return render_template('billing/new_drugs_billing.html', patients=patients, drugs=drugs)

@bp.route('/view/<int:billing_id>')
@login_required
def view_billing(billing_id):
    if current_user.role not in ['billing', 'admin']:
        return redirect(url_for('login'))

    # Fetch the billing entry by ID
    billing = Billing.query.options(
        joinedload(Billing.patient),
        joinedload(Billing.charge)
    ).get(billing_id) or DrugsBill.query.options(
        joinedload(DrugsBill.patient),
        joinedload(DrugsBill.drug)
    ).get(billing_id)

    if not billing:
        flash(f'Billing with ID {billing_id} does not exist!', 'error')
        return redirect(url_for('billing.index'))

    return render_template('billing/view_billing.html', billing=billing)

@bp.route('/update_status/<int:billing_id>', methods=['GET', 'POST'])
@login_required
def update_status(billing_id):
    if current_user.role not in ['billing', 'admin']:
        return redirect(url_for('login'))

    # Fetch the billing entry by ID
    billing = Billing.query.get(billing_id) or DrugsBill.query.get(billing_id)

    if not billing:
        flash(f'Billing with ID {billing_id} does not exist!', 'error')
        return redirect(url_for('billing.index'))

    if request.method == 'POST':
        # Update the status
        new_status = request.form.get('status')
        if new_status not in ['Pending', 'Paid']:
            flash('Invalid status! Please select "Pending" or "Paid".', 'error')
            return redirect(url_for('billing.update_status', billing_id=billing_id))

        billing.status = 1 if new_status == 'Paid' else 0  # 1 for Paid, 0 for Pending
        db.session.commit()

        flash(f'Status updated successfully for Billing ID {billing_id}!', 'success')
        return redirect(url_for('billing.index'))

    return render_template('billing/update_status.html', billing=billing)

@bp.route('/search_patients', methods=['GET'])
@login_required
def search_patients():
    if current_user.role not in ['billing', 'admin']:
        return jsonify({"status": "error", "message": "Unauthorized access!"}), 403

    # Get the search term from the request
    search_term = request.args.get('term', '').strip()

    # Query the database for matching patients
    if search_term:
        patients = Patient.query.filter(
            Patient.name.ilike(f"%{search_term}%") | Patient.patient_id.ilike(f"%{search_term}%")
        ).all()
    else:
        patients = []

    # Format the results as a list of dictionaries
    results = [
        {
            "id": patient.patient_id,
            "text": f"{patient.name} ({patient.patient_id})"
        }
        for patient in patients
    ]

    return jsonify({"results": results})

@bp.route('/new_billing', methods=['GET', 'POST'])
@login_required
def new_billing():
    if current_user.role not in ['billing', 'admin']:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Extract form data
        patient_id = request.form.get('patient_id')
        charge_id = request.form.get('charge_id')
        quantity = request.form.get('quantity')

        # Validate input
        if not patient_id or not charge_id or not quantity:
            flash('All fields are required!', 'error')
            return redirect(url_for('billing.new_billing'))

        try:
            quantity = int(quantity)
            if quantity <= 0:
                raise ValueError("Quantity must be greater than zero.")
        except ValueError:
            flash('Invalid quantity! Please enter a positive integer.', 'error')
            return redirect(url_for('billing.new_billing'))

        # Check if the patient exists
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f'Patient with ID {patient_id} does not exist!', 'error')
            return redirect(url_for('billing.new_billing'))

        # Check if the charge exists
        charge = Charge.query.get(charge_id)
        if not charge:
            flash(f'Charge with ID {charge_id} does not exist!', 'error')
            return redirect(url_for('billing.new_billing'))

        # Create a new billing entry
        new_billing = Billing(
            patient_id=patient_id,
            charge_id=charge_id,
            quantity=quantity
        )
        new_billing.calculate_total()  # Ensure this method exists in the Billing model
        db.session.add(new_billing)
        db.session.commit()

        flash(f'Billing created successfully for {patient.name}!', 'success')
        return redirect(url_for('billing.index'))

    # Fetch all patients and charges for the dropdowns
    patients = Patient.query.all()
    charges = Charge.query.all()

    return render_template('billing/new_billing.html', patients=patients, charges=charges)

@bp.route('/new_invoice', methods=['GET', 'POST'])
@login_required
def new_invoice():
    if current_user.role not in ['billing', 'admin']:
        return redirect(url_for('login'))

    # Placeholder logic (to be implemented later)
    flash("This feature is under development!", "info")
    return redirect(url_for('billing.index'))

@bp.route('/view_unpaid_bills/<patient_id>')
@login_required
def view_unpaid_bills(patient_id):
    if current_user.role not in ['billing', 'admin']:
        return redirect(url_for('login'))

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        flash(f'Patient with ID {patient_id} does not exist!', 'error')
        return redirect(url_for('billing.index'))

    # Fetch existing unpaid bills
    unpaid_billings = Billing.query.filter_by(patient_id=patient_id, status='Pending').all()
    unpaid_drug_bills = DrugsBill.query.filter_by(patient_id=patient_id, status=0).all()

    # Fetch unbilled items (receipt_number IS NULL)
    dispensed_drugs = DispensedDrug.query.filter_by(patient_id=patient_id, receipt_number=None).all()
    requested_labs = RequestedLab.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all()
    clinic_bookings = ClinicBooking.query.filter_by(patient_id=patient_id, seen=1, receipt_number=None).all()
    theatre_list = TheatreList.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all()
    requested_images = RequestedImage.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all()
    admitted_patients = AdmittedPatient.query.filter_by(patient_id=patient_id, discharged_on=None, receipt_number=None).all()

    # Pre-calculate ward admissions
    admitted_patients_data = []
    for admission in admitted_patients:
        days = (datetime.utcnow() - admission.admitted_on).days + 1
        total_cost = Decimal(str(admission.ward.daily_charge or 0)) * Decimal(days)
        admitted_patients_data.append({
            'id': admission.id,
            'ward_name': admission.ward.name,
            'days': days,
            'daily_charge': admission.ward.daily_charge,
            'total_cost': total_cost
        })

    # Calculate totals
    totals = {
        'Dispensed Drugs': sum(Decimal(str(d.drug.selling_price or 0)) * Decimal(str(d.quantity_dispensed or 0)) for d in dispensed_drugs) if dispensed_drugs else Decimal('0'),
        'Lab Tests': sum(Decimal(str(r.lab_test.cost or 0)) for r in requested_labs) if requested_labs else Decimal('0'),
        'Clinic Bookings': sum(Decimal(str(b.clinic.fee or 0)) for b in clinic_bookings) if clinic_bookings else Decimal('0'),
        'Theatre Procedures': sum(Decimal(str(t.procedure.cost or 0)) for t in theatre_list) if theatre_list else Decimal('0'),
        'Imaging': sum(Decimal(str(i.imaging.cost or 0)) for i in requested_images) if requested_images else Decimal('0'),
        'Ward Admissions': sum(Decimal(str(a['total_cost'])) for a in admitted_patients_data) if admitted_patients_data else Decimal('0'),
        'Existing Billings': sum(Decimal(str(b.total_cost or 0)) for b in unpaid_billings) if unpaid_billings else Decimal('0'),
        'Existing Drug Bills': sum(Decimal(str(b.total_cost or 0)) for b in unpaid_drug_bills) if unpaid_drug_bills else Decimal('0')
    }
    grand_total = sum(totals.values())

    return render_template(
        'billing/view_unpaid_bills.html',
        patient=patient,
        dispensed_drugs=dispensed_drugs,
        requested_labs=requested_labs,
        clinic_bookings=clinic_bookings,
        theatre_list=theatre_list,
        requested_images=requested_images,
        admitted_patients=admitted_patients_data,
        unpaid_billings=unpaid_billings,
        unpaid_drug_bills=unpaid_drug_bills,
        totals=totals,
        grand_total=grand_total
    )
@bp.route('/pay_all/<patient_id>', methods=['POST'])
@login_required
def pay_all(patient_id):
    if current_user.role not in ['billing', 'admin']:
        return redirect(url_for('login'))

    # Fetch the patient by ID
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        flash(f'Patient with ID {patient_id} does not exist!', 'error')
        return redirect(url_for('billing.index'))

    # Fetch all unpaid billings and drug bills for the patient
    unpaid_billings = Billing.query.filter_by(patient_id=patient_id, status=0).all()
    unpaid_drug_bills = DrugsBill.query.filter_by(patient_id=patient_id, status=0).all()

    if not unpaid_billings and not unpaid_drug_bills:
        flash(f'No pending bills to mark as paid for {patient.name}.', 'info')
        return redirect(url_for('billing.view_unpaid_bills', patient_id=patient_id))

    try:
        # Extract form data
        amount_paid = request.form.get('amount_paid')
        payment_method = request.form.get('payment_method')

        # Validate input
        if not amount_paid or not payment_method:
            flash('Amount Paid and Payment Method are required!', 'error')
            return redirect(url_for('billing.view_unpaid_bills', patient_id=patient_id))

        try:
            amount_paid = Decimal(amount_paid)
            if amount_paid <= 0:
                raise ValueError("Amount Paid must be greater than zero.")
        except (ValueError, InvalidOperation):
            flash('Invalid Amount Paid! Please enter a positive number.', 'error')
            return redirect(url_for('billing.view_unpaid_bills', patient_id=patient_id))

        # Calculate grand total of all unpaid bills
        grand_total = sum(Decimal(str(bill.total_cost or '0')) for bill in unpaid_billings + unpaid_drug_bills)

        # Validate payment against grand total
        if amount_paid > grand_total:
            flash(f'Amount Paid cannot exceed the Grand Total (Kshs {grand_total})!', 'error')
            return redirect(url_for('billing.view_unpaid_bills', patient_id=patient_id))

        # Calculate remaining balance
        balance = max(grand_total - amount_paid, Decimal('0'))

        # Generate a unique receipt number
        receipt_number = PaidBill.generate_receipt_number()

        # Create a new paid_bill entry
        new_paid_bill = PaidBill(
            receipt_number=receipt_number,
            patient_id=patient_id,
            grand_total=grand_total,
            amount_paid=amount_paid,
            balance=balance,
            paid_at=datetime.utcnow(),
            payment_method=payment_method
        )
        db.session.add(new_paid_bill)

        # Mark all unpaid bills as paid and assign the receipt number
        for bill in unpaid_billings + unpaid_drug_bills:
            bill.status = 1  # Mark as paid
            bill.receipt_number = receipt_number  # Assign receipt number
            db.session.add(bill)

        db.session.commit()

        flash(f'Payment recorded successfully for {patient.name}! Receipt Number: {receipt_number}, Amount Paid: Kshs {amount_paid}, Remaining Balance: Kshs {balance}', 'success')
        return redirect(url_for('billing.view_unpaid_bills', patient_id=patient_id))

    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while marking bills as paid: {e}', 'error')
        return redirect(url_for('billing.view_unpaid_bills', patient_id=patient_id))

@bp.route('/paid_bills/<patient_id>')
@login_required
def paid_bills(patient_id):
    if current_user.role not in ['billing', 'admin']:
        return redirect(url_for('login'))

    # Fetch the patient by ID
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        flash(f'Patient with ID {patient_id} does not exist!', 'error')
        return redirect(url_for('billing.index'))

    # Fetch all paid bills for the patient
    paid_bills = PaidBill.query.filter_by(patient_id=patient_id).all()

    return render_template('billing/paid_bills.html', patient=patient, paid_bills=paid_bills)

@bp.route('/pay_bills/<patient_id>', methods=['GET', 'POST'])
@login_required
def pay_bills(patient_id):
    """Handle billing for a patient's unpaid items, including partial payments and selected items."""
    logger.debug(f"Entering pay_bills function for patient_id: {patient_id}")

    if current_user.role not in ['billing', 'admin']:
        logger.warning(f"Unauthorized access attempt by user: {current_user.id}")
        return redirect(url_for('login'))

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        logger.error(f"Patient with ID {patient_id} does not exist!")
        flash(f'Patient with ID {patient_id} does not exist!', 'error')
        return redirect(url_for('billing.index'))

    if request.method == 'POST':
        logger.debug("Processing POST request")
        action = request.form.get('action')
        amount_paid = Decimal(request.form.get('amount_paid', '0'))
        payment_method = request.form.get('payment_method')
        payment_reference = request.form.get('payment_reference', '')

        logger.debug(f"Action: {action}, Amount Paid: {amount_paid}, Payment Method: {payment_method}, Payment Reference: {payment_reference}")

        now = datetime.utcnow()
        receipt_base = f"REC-{now.strftime('%Y%m%d')}"
        last_bill = db.session.query(LabBill).filter(LabBill.receipt_number.like(f"{receipt_base}%")).order_by(LabBill.id.desc()).first()
        sequence = int(last_bill.receipt_number.split('-')[-1]) + 1 if last_bill else 1
        receipt_number = f"{receipt_base}-{sequence:03d}"

        logger.debug(f"Generated receipt number: {receipt_number}")

        all_items = {
            'dispensed_drugs': DispensedDrug.query.filter_by(patient_id=patient_id, receipt_number=None).all(),
            'requested_labs': RequestedLab.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all(),
            'clinic_bookings': ClinicBooking.query.filter_by(patient_id=patient_id, seen=1, receipt_number=None).all(),
            'theatre_list': TheatreList.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all(),
            'requested_images': RequestedImage.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all(),
            'admitted_patients': AdmittedPatient.query.filter_by(patient_id=patient_id, discharged_on=None, receipt_number=None).all()
        }

        logger.debug(f"Retrieved all items for patient: {patient_id}")

        selected_items = {}
        if action == 'pay_selected':
            logger.debug("Processing selected items for payment")
            for category in all_items:
                selected_ids = request.form.getlist(category)
                selected_items[category] = [item for item in all_items[category] if str(item.id) in selected_ids]
        else:  # pay_all
            logger.debug("Processing all items for payment")
            selected_items = all_items

        logger.debug(f"Selected items: {selected_items}")

        totals = {
            'Dispensed Drugs': sum(Decimal(str(d.drug.selling_price or 0)) * Decimal(str(d.quantity_dispensed or 0)) for d in selected_items['dispensed_drugs']) if selected_items['dispensed_drugs'] else Decimal('0'),
            'Lab Tests': sum(Decimal(str(r.lab_test.cost or 0)) for r in selected_items['requested_labs']) if selected_items['requested_labs'] else Decimal('0'),
            'Clinic Bookings': sum(Decimal(str(b.clinic.fee or 0)) for b in selected_items['clinic_bookings']) if selected_items['clinic_bookings'] else Decimal('0'),
            'Theatre Procedures': sum(Decimal(str(t.procedure.cost or 0)) for t in selected_items['theatre_list']) if selected_items['theatre_list'] else Decimal('0'),
            'Imaging': sum(Decimal(str(i.imaging.cost or 0)) for i in selected_items['requested_images']) if selected_items['requested_images'] else Decimal('0'),
            'Ward Admissions': sum(
                Decimal(str(a.ward.daily_charge or 0)) * Decimal((datetime.utcnow() - a.admitted_on).days + 1)
                for a in selected_items['admitted_patients']
            ) if selected_items['admitted_patients'] else Decimal('0')
        }
        grand_total = sum(totals.values())

        logger.debug(f"Calculated totals: {totals}, Grand Total: {grand_total}")

        if grand_total == 0:
            logger.warning("No items selected for payment")
            flash('No items selected for payment!', 'warning')
            return redirect(url_for('billing.pay_bills', patient_id=patient_id))
        if amount_paid < 0:
            logger.warning("Amount paid is negative")
            flash('Amount paid cannot be negative!', 'error')
            return redirect(url_for('billing.pay_bills', patient_id=patient_id))
        if amount_paid < grand_total:
            logger.warning(f"Partial payment detected: Amount paid (Kshs {amount_paid:,.2f}) is less than total (Kshs {grand_total:,.2f})")
            flash(f'Amount paid (Kshs {amount_paid:,.2f}) is less than total (Kshs {grand_total:,.2f}). Partial payment recorded.', 'warning')
        elif amount_paid > grand_total:
            logger.info(f"Amount paid exceeds total: Amount paid (Kshs {amount_paid:,.2f}) exceeds total (Kshs {grand_total:,.2f})")
            flash(f'Amount paid (Kshs {amount_paid:,.2f}) exceeds total (Kshs {grand_total:,.2f}). Change: Kshs {(amount_paid - grand_total):,.2f}', 'info')

        # Prepare paid items for confirmation
        paid_items = {}
        for category, items in selected_items.items():
            if items:
                if category == 'dispensed_drugs':
                    paid_items['Dispensed Drugs'] = [f"{item.drug.generic_name} ({item.quantity_dispensed})" for item in items]
                elif category == 'requested_labs':
                    paid_items['Lab Tests'] = [item.lab_test.test_name for item in items]
                elif category == 'clinic_bookings':
                    paid_items['Clinic Bookings'] = [item.clinic.name for item in items]
                elif category == 'theatre_list':
                    paid_items['Theatre Procedures'] = [item.procedure.name for item in items]
                elif category == 'requested_images':
                    paid_items['Imaging'] = [item.imaging.imaging_type for item in items]
                elif category == 'admitted_patients':
                    paid_items['Ward Admissions'] = [f"{item.ward.name} ({(datetime.utcnow() - item.admitted_on).days + 1} days)" for item in items]

        logger.debug(f"Prepared paid items: {paid_items}")

        try:
            for category, items in selected_items.items():
                if items:
                    total = totals[category.replace('_', ' ').title()]
                    for item in items:
                        item.receipt_number = receipt_number
                    if category == 'dispensed_drugs':
                        drugs_bill = DrugsBill(
                            patient_id=patient_id,
                            drug_id=None,
                            quantity=len(items),
                            total_cost=total,
                            status=1,
                            receipt_number=receipt_number,
                            billed_at=datetime.utcnow(),
                            payment_method=payment_method,
                            payment_reference=payment_reference
                        )
                        db.session.add(drugs_bill)
                    elif category == 'requested_labs':
                        lab_bill = LabBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference
                        )
                        db.session.add(lab_bill)
                    elif category == 'clinic_bookings':
                        clinic_bill = ClinicBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference
                        )
                        db.session.add(clinic_bill)
                    elif category == 'theatre_list':
                        theatre_bill = TheatreBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference
                        )
                        db.session.add(theatre_bill)
                    elif category == 'requested_images':
                        imaging_bill = ImagingBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference
                        )
                        db.session.add(imaging_bill)
                    elif category == 'admitted_patients':
                        ward_bill = WardBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference
                        )
                        db.session.add(ward_bill)

            db.session.commit()
            logger.info(f"Payment processed successfully! Receipt Number: {receipt_number}")
            flash(f'Payment processed successfully! Receipt Number: {receipt_number}', 'success')
            return render_template(
                'billing/view_unpaid_bills.html',
                patient=patient,
                payment_success=True,
                receipt_number=receipt_number,
                amount_paid=amount_paid,
                payment_method=payment_method,
                payment_reference=payment_reference,
                paid_totals=totals,
                paid_items=paid_items
            )

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing payment: {str(e)}")
            flash(f'Error processing payment: {str(e)}', 'error')
            return redirect(url_for('billing.pay_bills', patient_id=patient_id))

    # GET request: Show payment form
    logger.debug("Processing GET request")
    dispensed_drugs = DispensedDrug.query.filter_by(patient_id=patient_id, receipt_number=None).all()
    requested_labs = RequestedLab.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all()
    clinic_bookings = ClinicBooking.query.filter_by(patient_id=patient_id, seen=1, receipt_number=None).all()
    theatre_list = TheatreList.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all()
    requested_images = RequestedImage.query.filter_by(patient_id=patient_id, status=1, receipt_number=None).all()
    admitted_patients = AdmittedPatient.query.filter_by(patient_id=patient_id, discharged_on=None, receipt_number=None).all()
    unpaid_billings = Billing.query.filter_by(patient_id=patient_id, status='Pending').all()
    unpaid_drug_bills = DrugsBill.query.filter_by(patient_id=patient_id, status=0).all()

    admitted_patients_data = []
    for admission in admitted_patients:
        days = (datetime.utcnow() - admission.admitted_on).days + 1
        total_cost = Decimal(str(admission.ward.daily_charge or 0)) * Decimal(days)
        admitted_patients_data.append({
            'id': admission.id,
            'ward_name': admission.ward.name,
            'days': days,
            'daily_charge': admission.ward.daily_charge,
            'total_cost': total_cost
        })

    totals = {
        'Dispensed Drugs': sum(Decimal(str(d.drug.selling_price or 0)) * Decimal(str(d.quantity_dispensed or 0)) for d in dispensed_drugs) if dispensed_drugs else Decimal('0'),
        'Lab Tests': sum(Decimal(str(r.lab_test.cost or 0)) for r in requested_labs) if requested_labs else Decimal('0'),
        'Clinic Bookings': sum(Decimal(str(b.clinic.fee or 0)) for b in clinic_bookings) if clinic_bookings else Decimal('0'),
        'Theatre Procedures': sum(Decimal(str(t.procedure.cost or 0)) for t in theatre_list) if theatre_list else Decimal('0'),
        'Imaging': sum(Decimal(str(i.imaging.cost or 0)) for i in requested_images) if requested_images else Decimal('0'),
        'Ward Admissions': sum(Decimal(str(a['total_cost'])) for a in admitted_patients_data) if admitted_patients_data else Decimal('0'),
        'Existing Billings': sum(Decimal(str(b.total_cost or 0)) for b in unpaid_billings) if unpaid_billings else Decimal('0'),
        'Existing Drug Bills': sum(Decimal(str(b.total_cost or 0)) for b in unpaid_drug_bills) if unpaid_drug_bills else Decimal('0')
    }
    grand_total = sum(totals.values())

    logger.debug(f"Calculated totals for GET request: {totals}, Grand Total: {grand_total}")

    return render_template(
        'billing/view_unpaid_bills.html',
        patient=patient,
        dispensed_drugs=dispensed_drugs,
        requested_labs=requested_labs,
        clinic_bookings=clinic_bookings,
        theatre_list=theatre_list,
        requested_images=requested_images,
        admitted_patients=admitted_patients_data,
        totals=totals,
        grand_total=grand_total,
        unpaid_billings=unpaid_billings,
        unpaid_drug_bills=unpaid_drug_bills,
        payment_success=False
    )