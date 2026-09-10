import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy.orm import joinedload

from departments.api.audit import log_audit_event
from departments.models.billing import (
    Billing,
    Charge,
    ChargeCategory,
    ClinicBill,
    DrugsBill,
    ImagingBill,
    LabBill,
    PaidBill,
    TheatreBill,
    WardBill,
)
from departments.models.medicine import (
    AdmittedPatient,
    RequestedImage,
    RequestedLab,
    TheatreList,
)
from departments.models.pharmacy import DispensedDrug, Drug
from departments.models.records import ClinicBooking, Patient
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@bp.route("/")
@bp.route("/index")
@login_required
@roles_required("billing", "admin")
def index():
    # Fetch all unpaid billings and drug bills with related data (eager loading)
    unpaid_billings = (
        Billing.query.filter_by(status=0)
        .options(joinedload(Billing.patient), joinedload(Billing.charge))
        .all()
    )

    unpaid_drug_bills = (
        DrugsBill.query.filter_by(status=0)
        .options(joinedload(DrugsBill.patient), joinedload(DrugsBill.drug))
        .all()
    )

    # Group unpaid bills by patient and calculate total cost
    patient_bills = {}
    for bill in unpaid_billings + unpaid_drug_bills:
        if bill.patient_id not in patient_bills:
            patient_bills[bill.patient_id] = {
                "patient": bill.patient,
                "total_cost": Decimal("0"),  # Use Decimal for precise arithmetic
            }

        # Add bill's total cost to the patient's total
        if bill.patient:  # Ensure the patient relationship exists
            patient_bills[bill.patient_id]["total_cost"] += Decimal(
                str(bill.total_cost or "0")
            )

    # Convert the dictionary to a list for rendering
    patients_with_unpaid_bills = [
        {"patient": data["patient"], "total_cost": data["total_cost"]}
        for patient_id, data in patient_bills.items()
    ]

    return render_template("billing/index.html", patients=patients_with_unpaid_bills)


@bp.route("/billings", methods=["GET"])
@login_required
@roles_required("billing", "admin")
def list_billings():
    """Lists all billing and invoice entries across patients."""

    billings = Billing.query.options(
        joinedload(Billing.patient), joinedload(Billing.charge)
    ).all()

    drug_bills = DrugsBill.query.options(
        joinedload(DrugsBill.patient), joinedload(DrugsBill.drug)
    ).all()

    all_billings = sorted(
        billings + drug_bills, key=lambda b: b.billed_at, reverse=True
    )
    return render_template("billing/invoices_list.html", billings=all_billings)


@bp.route("/new_drugs_billing", methods=["GET", "POST"])
@login_required
@roles_required("billing", "admin")
def new_drugs_billing():
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        drug_id = request.form.get("drug_id")
        quantity = request.form.get("quantity")

        if not patient_id or not drug_id or not quantity:
            flash("All fields are required!", "error")
            return redirect(url_for("billing.new_drugs_billing"))

        try:
            quantity = int(quantity)
            if quantity <= 0:
                raise ValueError("Quantity must be greater than zero.")
        except ValueError:
            flash("Invalid quantity! Please enter a positive integer.", "error")
            return redirect(url_for("billing.new_drugs_billing"))

        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f"Patient with ID {patient_id} does not exist!", "error")
            return redirect(url_for("billing.new_drugs_billing"))

        drug = Drug.query.get(drug_id)
        if not drug:
            flash(f"Drug with ID {drug_id} does not exist!", "error")
            return redirect(url_for("billing.new_drugs_billing"))

        total_cost = Decimal(str(quantity)) * Decimal(str(drug.selling_price))

        new_drugs_billing = DrugsBill(
            patient_id=patient_id,
            drug_id=drug_id,
            quantity=quantity,
            total_cost=total_cost,
        )

        db.session.add(new_drugs_billing)
        db.session.commit()

        logger.debug(f"Total Cost Calculated: {new_drugs_billing.total_cost}")
        flash(
            f"Drugs billing created successfully for {patient.name}! Total Cost: Kshs {new_drugs_billing.total_cost}",
            "success",
        )
        return redirect(url_for("billing.index"))

    patients = Patient.query.all()
    drugs = Drug.query.all()

    return render_template(
        "billing/new_drugs_billing.html", patients=patients, drugs=drugs
    )


@bp.route("/view/<int:billing_id>")
@login_required
@roles_required("billing", "admin")
def view_billing(billing_id):
    billing = Billing.query.options(
        joinedload(Billing.patient), joinedload(Billing.charge)
    ).get(billing_id) or DrugsBill.query.options(
        joinedload(DrugsBill.patient), joinedload(DrugsBill.drug)
    ).get(billing_id)

    if not billing:
        flash(f"Billing with ID {billing_id} does not exist!", "error")
        return redirect(url_for("billing.index"))

    return render_template("billing/view_billing.html", billing=billing)


@bp.route("/update_status/<int:billing_id>", methods=["GET", "POST"])
@login_required
@roles_required("billing", "admin")
def update_status(billing_id):
    billing = Billing.query.get(billing_id) or DrugsBill.query.get(billing_id)

    if not billing:
        flash(f"Billing with ID {billing_id} does not exist!", "error")
        return redirect(url_for("billing.index"))

    if request.method == "POST":
        new_status = request.form.get("status")
        if new_status not in ["Pending", "Paid"]:
            flash('Invalid status! Please select "Pending" or "Paid".', "error")
            return redirect(url_for("billing.update_status", billing_id=billing_id))

        billing.status = 1 if new_status == "Paid" else 0
        db.session.commit()

        flash(f"Status updated successfully for Billing ID {billing_id}!", "success")
        return redirect(url_for("billing.index"))

    return render_template("billing/update_status.html", billing=billing)


@bp.route("/search_patients", methods=["GET"])
@login_required
@roles_required("billing", "admin")
def search_patients():
    search_term = request.args.get("term", "").strip()

    if search_term:
        patients = Patient.query.filter(
            Patient.name.ilike(f"%{search_term}%")
            | Patient.patient_id.ilike(f"%{search_term}%")
        ).all()
    else:
        patients = []

    results = [
        {"id": patient.patient_id, "text": f"{patient.name} ({patient.patient_id})"}
        for patient in patients
    ]

    return jsonify({"results": results})


@bp.route("/new_billing", methods=["GET", "POST"])
@login_required
@roles_required("billing", "admin")
def new_billing():
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        charge_id = request.form.get("charge_id")
        quantity = request.form.get("quantity")

        if not patient_id or not charge_id or not quantity:
            flash("All fields are required!", "error")
            return redirect(url_for("billing.new_billing"))

        try:
            quantity = int(quantity)
            if quantity <= 0:
                raise ValueError("Quantity must be greater than zero.")
        except ValueError:
            flash("Invalid quantity! Please enter a positive integer.", "error")
            return redirect(url_for("billing.new_billing"))

        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f"Patient with ID {patient_id} does not exist!", "error")
            return redirect(url_for("billing.new_billing"))

        charge = Charge.query.get(charge_id)
        if not charge:
            flash(f"Charge with ID {charge_id} does not exist!", "error")
            return redirect(url_for("billing.new_billing"))

        new_billing = Billing(
            patient_id=patient_id, charge_id=charge_id, quantity=quantity
        )
        new_billing.calculate_total()
        db.session.add(new_billing)
        db.session.commit()

        flash(f"Billing created successfully for {patient.name}!", "success")
        return redirect(url_for("billing.index"))

    patients = Patient.query.all()
    charges = Charge.query.all()

    return render_template(
        "billing/new_billing.html", patients=patients, charges=charges
    )


@bp.route("/new_invoice", methods=["GET", "POST"])
@login_required
@roles_required("billing", "admin")
def new_invoice():
    """Create a new invoice / charge for a patient."""
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        charge_id = request.form.get("charge_id")
        quantity = request.form.get("quantity", 1, type=int)

        if not patient_id or not charge_id:
            flash("Patient and Charge service are required!", "error")
            return redirect(url_for("billing.new_invoice"))

        patient = Patient.query.filter_by(patient_id=patient_id).first()
        charge = Charge.query.get(charge_id)

        if not patient or not charge:
            flash("Invalid patient or charge selected.", "error")
            return redirect(url_for("billing.new_invoice"))

        billing_entry = Billing(
            patient_id=patient_id,
            charge_id=charge_id,
            quantity=quantity,
            total_cost=charge.cost * quantity,
            status=0,
        )
        db.session.add(billing_entry)
        db.session.commit()

        # Trigger invoice due notification email
        from departments.notifications.triggers import trigger_invoice_due

        trigger_invoice_due(billing_entry)

        flash(
            f"Invoice created successfully for {patient.name} (Amount: Kshs {billing_entry.total_cost})",
            "success",
        )
        return redirect(url_for("billing.list_billings"))

    patients = Patient.query.all()
    charges = Charge.query.all()
    return render_template(
        "billing/new_billing.html", patients=patients, charges=charges
    )


@bp.route("/view_unpaid_bills/<patient_id>")
@login_required
@roles_required("billing", "admin")
def view_unpaid_bills(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        flash(f"Patient with ID {patient_id} does not exist!", "error")
        return redirect(url_for("billing.index"))

    unpaid_billings = Billing.query.filter_by(patient_id=patient_id, status=0).all()
    unpaid_drug_bills = DrugsBill.query.filter_by(patient_id=patient_id, status=0).all()

    dispensed_drugs = DispensedDrug.query.filter_by(
        patient_id=patient_id, receipt_number=None
    ).all()
    requested_labs = RequestedLab.query.filter_by(
        patient_id=patient_id, status=1, receipt_number=None
    ).all()
    clinic_bookings = ClinicBooking.query.filter_by(
        patient_id=patient_id, seen=1, receipt_number=None
    ).all()
    theatre_list = TheatreList.query.filter_by(
        patient_id=patient_id, status=1, receipt_number=None
    ).all()
    requested_images = RequestedImage.query.filter_by(
        patient_id=patient_id, status=1, receipt_number=None
    ).all()
    admitted_patients = AdmittedPatient.query.filter_by(
        patient_id=patient_id, discharged_on=None, receipt_number=None
    ).all()

    admitted_patients_data = []
    for admission in admitted_patients:
        days = (datetime.now(timezone.utc) - admission.admitted_on).days + 1
        total_cost = Decimal(str(admission.ward.daily_charge or 0)) * Decimal(days)
        admitted_patients_data.append(
            {
                "id": admission.id,
                "ward_name": admission.ward.name,
                "days": days,
                "daily_charge": admission.ward.daily_charge,
                "total_cost": total_cost,
            }
        )

    totals = {
        "Dispensed Drugs": sum(
            Decimal(str(d.drug.selling_price or 0))
            * Decimal(str(d.quantity_dispensed or 0))
            for d in dispensed_drugs
        )
        if dispensed_drugs
        else Decimal("0"),
        "Lab Tests": sum(Decimal(str(r.lab_test.cost or 0)) for r in requested_labs)
        if requested_labs
        else Decimal("0"),
        "Clinic Bookings": sum(Decimal(str(b.clinic.fee or 0)) for b in clinic_bookings)
        if clinic_bookings
        else Decimal("0"),
        "Theatre Procedures": sum(
            Decimal(str(t.procedure.cost or 0)) for t in theatre_list
        )
        if theatre_list
        else Decimal("0"),
        "Imaging": sum(Decimal(str(i.imaging.cost or 0)) for i in requested_images)
        if requested_images
        else Decimal("0"),
        "Ward Admissions": sum(
            Decimal(str(a["total_cost"])) for a in admitted_patients_data
        )
        if admitted_patients_data
        else Decimal("0"),
        "Existing Billings": sum(
            Decimal(str(b.total_cost or 0)) for b in unpaid_billings
        )
        if unpaid_billings
        else Decimal("0"),
        "Existing Drug Bills": sum(
            Decimal(str(b.total_cost or 0)) for b in unpaid_drug_bills
        )
        if unpaid_drug_bills
        else Decimal("0"),
    }
    grand_total = sum(totals.values())

    return render_template(
        "billing/view_unpaid_bills.html",
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
        grand_total=grand_total,
    )


@bp.route("/pay_all/<patient_id>", methods=["POST"])
@login_required
@roles_required("billing", "admin")
def pay_all(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        flash(f"Patient with ID {patient_id} does not exist!", "error")
        return redirect(url_for("billing.index"))

    unpaid_billings = Billing.query.filter_by(patient_id=patient_id, status=0).all()
    unpaid_drug_bills = DrugsBill.query.filter_by(patient_id=patient_id, status=0).all()

    if not unpaid_billings and not unpaid_drug_bills:
        flash(f"No pending bills to mark as paid for {patient.name}.", "info")
        return redirect(url_for("billing.view_unpaid_bills", patient_id=patient_id))

    try:
        amount_paid = request.form.get("amount_paid")
        payment_method = request.form.get("payment_method")

        if not amount_paid or not payment_method:
            flash("Amount Paid and Payment Method are required!", "error")
            return redirect(url_for("billing.view_unpaid_bills", patient_id=patient_id))

        try:
            amount_paid = Decimal(amount_paid)
            if amount_paid <= 0:
                raise ValueError("Amount Paid must be greater than zero.")
        except (ValueError, InvalidOperation):
            flash("Invalid Amount Paid! Please enter a positive number.", "error")
            return redirect(url_for("billing.view_unpaid_bills", patient_id=patient_id))

        grand_total = sum(
            Decimal(str(bill.total_cost or "0"))
            for bill in unpaid_billings + unpaid_drug_bills
        )

        if amount_paid > grand_total:
            flash(
                f"Amount Paid cannot exceed the Grand Total (Kshs {grand_total})!",
                "error",
            )
            return redirect(url_for("billing.view_unpaid_bills", patient_id=patient_id))

        balance = max(grand_total - amount_paid, Decimal("0"))
        receipt_number = PaidBill.generate_receipt_number()

        new_paid_bill = PaidBill(
            receipt_number=receipt_number,
            patient_id=patient_id,
            grand_total=grand_total,
            amount_paid=amount_paid,
            balance=balance,
            paid_at=datetime.now(timezone.utc),
            payment_method=payment_method,
        )
        db.session.add(new_paid_bill)

        for bill in unpaid_billings + unpaid_drug_bills:
            bill.status = 1
            bill.receipt_number = receipt_number
            db.session.add(bill)

        db.session.commit()

        flash(
            f"Payment recorded successfully for {patient.name}! Receipt Number: {receipt_number}, Amount Paid: Kshs {amount_paid}, Remaining Balance: Kshs {balance}",
            "success",
        )
        return redirect(url_for("billing.view_unpaid_bills", patient_id=patient_id))

    except Exception:
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("billing.view_unpaid_bills", patient_id=patient_id))


@bp.route("/paid_bills/<patient_id>")
@login_required
@roles_required("billing", "admin")
def paid_bills(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        flash(f"Patient with ID {patient_id} does not exist!", "error")
        return redirect(url_for("billing.index"))

    paid_bills = (
        PaidBill.query.filter_by(patient_id=patient_id)
        .order_by(PaidBill.paid_at.desc())
        .all()
    )
    return render_template(
        "billing/paid_bills.html", patient=patient, paid_bills=paid_bills
    )


@bp.route("/pay_bills/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required("billing", "admin")
def pay_bills(patient_id):
    """Handle billing for a patient's unpaid items, including partial payments and selected items."""
    logger.debug(f"Entering pay_bills function for patient_id: {patient_id}")

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        logger.error(f"Patient with ID {patient_id} does not exist!")
        flash(f"Patient with ID {patient_id} does not exist!", "error")
        return redirect(url_for("billing.index"))

    if request.method == "POST":
        logger.debug("Processing POST request")
        action = request.form.get("action")
        amount_paid = Decimal(request.form.get("amount_paid", "0"))
        payment_method = request.form.get("payment_method")
        payment_reference = request.form.get("payment_reference", "")

        receipt_number = PaidBill.generate_receipt_number()
        logger.debug(f"Generated receipt number: {receipt_number}")

        all_items = {
            "dispensed_drugs": DispensedDrug.query.filter_by(
                patient_id=patient_id, receipt_number=None
            ).all(),
            "requested_labs": RequestedLab.query.filter_by(
                patient_id=patient_id, status=1, receipt_number=None
            ).all(),
            "clinic_bookings": ClinicBooking.query.filter_by(
                patient_id=patient_id, seen=1, receipt_number=None
            ).all(),
            "theatre_list": TheatreList.query.filter_by(
                patient_id=patient_id, status=1, receipt_number=None
            ).all(),
            "requested_images": RequestedImage.query.filter_by(
                patient_id=patient_id, status=1, receipt_number=None
            ).all(),
            "admitted_patients": AdmittedPatient.query.filter_by(
                patient_id=patient_id, discharged_on=None, receipt_number=None
            ).all(),
        }

        selected_items = {}
        if action == "pay_selected":
            for category in all_items:
                selected_ids = request.form.getlist(category)
                selected_items[category] = [
                    item for item in all_items[category] if str(item.id) in selected_ids
                ]
        else:
            selected_items = all_items

        totals = {
            "Dispensed Drugs": sum(
                Decimal(str(d.drug.selling_price or 0))
                * Decimal(str(d.quantity_dispensed or 0))
                for d in selected_items["dispensed_drugs"]
            )
            if selected_items["dispensed_drugs"]
            else Decimal("0"),
            "Lab Tests": sum(
                Decimal(str(r.lab_test.cost or 0))
                for r in selected_items["requested_labs"]
            )
            if selected_items["requested_labs"]
            else Decimal("0"),
            "Clinic Bookings": sum(
                Decimal(str(b.clinic.fee or 0))
                for b in selected_items["clinic_bookings"]
            )
            if selected_items["clinic_bookings"]
            else Decimal("0"),
            "Theatre Procedures": sum(
                Decimal(str(t.procedure.cost or 0))
                for t in selected_items["theatre_list"]
            )
            if selected_items["theatre_list"]
            else Decimal("0"),
            "Imaging": sum(
                Decimal(str(i.imaging.cost or 0))
                for i in selected_items["requested_images"]
            )
            if selected_items["requested_images"]
            else Decimal("0"),
            "Ward Admissions": sum(
                Decimal(str(a.ward.daily_charge or 0))
                * Decimal((datetime.now(timezone.utc) - a.admitted_on).days + 1)
                for a in selected_items["admitted_patients"]
            )
            if selected_items["admitted_patients"]
            else Decimal("0"),
        }
        grand_total = sum(totals.values())

        if grand_total == 0:
            flash("No items selected for payment!", "warning")
            return redirect(url_for("billing.pay_bills", patient_id=patient_id))
        if amount_paid < 0:
            flash("Amount paid cannot be negative!", "error")
            return redirect(url_for("billing.pay_bills", patient_id=patient_id))

        balance = max(grand_total - amount_paid, Decimal("0"))

        paid_items = {}
        for category, items in selected_items.items():
            if items:
                if category == "dispensed_drugs":
                    paid_items["Dispensed Drugs"] = [
                        f"{item.drug.generic_name} ({item.quantity_dispensed})"
                        for item in items
                    ]
                elif category == "requested_labs":
                    paid_items["Lab Tests"] = [
                        item.lab_test.test_name for item in items
                    ]
                elif category == "clinic_bookings":
                    paid_items["Clinic Bookings"] = [item.clinic.name for item in items]
                elif category == "theatre_list":
                    paid_items["Theatre Procedures"] = [
                        item.procedure.name for item in items
                    ]
                elif category == "requested_images":
                    paid_items["Imaging"] = [
                        item.imaging.imaging_type for item in items
                    ]
                elif category == "admitted_patients":
                    paid_items["Ward Admissions"] = [
                        f"{item.ward.name} ({(datetime.now(timezone.utc) - item.admitted_on).days + 1} days)"
                        for item in items
                    ]

        try:
            # Register master PaidBill entry for patient payment history
            paid_bill_record = PaidBill(
                receipt_number=receipt_number,
                patient_id=patient_id,
                grand_total=grand_total,
                amount_paid=amount_paid,
                balance=balance,
                paid_at=datetime.now(timezone.utc),
                payment_method=payment_method or "Cash",
            )
            db.session.add(paid_bill_record)

            # Record category bills and link item receipts
            for category, items in selected_items.items():
                if items:
                    total = totals[category.replace("_", " ").title()]
                    for item in items:
                        item.receipt_number = receipt_number
                    if category == "dispensed_drugs":
                        drugs_bill = DrugsBill(
                            patient_id=patient_id,
                            drug_id=None,
                            quantity=len(items),
                            total_cost=total,
                            status=1,
                            receipt_number=receipt_number,
                            billed_at=datetime.now(timezone.utc),
                            payment_method=payment_method,
                            payment_reference=payment_reference,
                        )
                        db.session.add(drugs_bill)
                    elif category == "requested_labs":
                        lab_bill = LabBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference,
                        )
                        db.session.add(lab_bill)
                    elif category == "clinic_bookings":
                        clinic_bill = ClinicBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference,
                        )
                        db.session.add(clinic_bill)
                    elif category == "theatre_list":
                        theatre_bill = TheatreBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference,
                        )
                        db.session.add(theatre_bill)
                    elif category == "requested_images":
                        imaging_bill = ImagingBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference,
                        )
                        db.session.add(imaging_bill)
                    elif category == "admitted_patients":
                        ward_bill = WardBill(
                            patient_id=patient_id,
                            total_paid=total,
                            receipt_number=receipt_number,
                            payment_method=payment_method,
                            payment_reference=payment_reference,
                        )
                        db.session.add(ward_bill)

            # Update status of existing pending Billing and DrugsBill items
            Billing.query.filter_by(patient_id=patient_id, status=0).update(
                {"status": 1, "receipt_number": receipt_number}
            )
            DrugsBill.query.filter_by(patient_id=patient_id, status=0).update(
                {"status": 1, "receipt_number": receipt_number}
            )

            db.session.commit()

            # Phase 2: settlement may complete the visit.
            from departments.shared.visit_closure import maybe_close_encounter

            maybe_close_encounter(patient_id)
            logger.info(
                f"Payment processed successfully! Receipt Number: {receipt_number}"
            )
            log_audit_event(
                "BILL_PAYMENT",
                resource_type="PaidBill",
                resource_id=receipt_number,
                details={
                    "patient_id": patient_id,
                    "amount_paid": str(amount_paid),
                    "payment_method": payment_method,
                },
            )

            # Trigger payment receipt notification email
            from departments.notifications.triggers import trigger_payment_received

            trigger_payment_received(paid_bill_record)

            flash(
                f"Payment processed successfully! Receipt Number: {receipt_number}",
                "success",
            )
            return render_template(
                "billing/view_unpaid_bills.html",
                patient=patient,
                payment_success=True,
                receipt_number=receipt_number,
                amount_paid=amount_paid,
                payment_method=payment_method,
                payment_reference=payment_reference,
                paid_totals=totals,
                paid_items=paid_items,
            )

        except Exception as e:
            db.session.rollback()
            logger.error(
                f"Error processing payment for patient {patient_id}: {e}", exc_info=True
            )
            flash(
                "Payment processing failed. Please try again or contact support.",
                "error",
            )
            return redirect(url_for("billing.pay_bills", patient_id=patient_id))

    # GET request: Show payment form
    dispensed_drugs = DispensedDrug.query.filter_by(
        patient_id=patient_id, receipt_number=None
    ).all()
    requested_labs = RequestedLab.query.filter_by(
        patient_id=patient_id, status=1, receipt_number=None
    ).all()
    clinic_bookings = ClinicBooking.query.filter_by(
        patient_id=patient_id, seen=1, receipt_number=None
    ).all()
    theatre_list = TheatreList.query.filter_by(
        patient_id=patient_id, status=1, receipt_number=None
    ).all()
    requested_images = RequestedImage.query.filter_by(
        patient_id=patient_id, status=1, receipt_number=None
    ).all()
    admitted_patients = AdmittedPatient.query.filter_by(
        patient_id=patient_id, discharged_on=None, receipt_number=None
    ).all()
    unpaid_billings = Billing.query.filter_by(patient_id=patient_id, status=0).all()
    unpaid_drug_bills = DrugsBill.query.filter_by(patient_id=patient_id, status=0).all()

    admitted_patients_data = []
    for admission in admitted_patients:
        days = (datetime.now(timezone.utc) - admission.admitted_on).days + 1
        total_cost = Decimal(str(admission.ward.daily_charge or 0)) * Decimal(days)
        admitted_patients_data.append(
            {
                "id": admission.id,
                "ward_name": admission.ward.name,
                "days": days,
                "daily_charge": admission.ward.daily_charge,
                "total_cost": total_cost,
            }
        )

    totals = {
        "Dispensed Drugs": sum(
            Decimal(str(d.drug.selling_price or 0))
            * Decimal(str(d.quantity_dispensed or 0))
            for d in dispensed_drugs
        )
        if dispensed_drugs
        else Decimal("0"),
        "Lab Tests": sum(Decimal(str(r.lab_test.cost or 0)) for r in requested_labs)
        if requested_labs
        else Decimal("0"),
        "Clinic Bookings": sum(Decimal(str(b.clinic.fee or 0)) for b in clinic_bookings)
        if clinic_bookings
        else Decimal("0"),
        "Theatre Procedures": sum(
            Decimal(str(t.procedure.cost or 0)) for t in theatre_list
        )
        if theatre_list
        else Decimal("0"),
        "Imaging": sum(Decimal(str(i.imaging.cost or 0)) for i in requested_images)
        if requested_images
        else Decimal("0"),
        "Ward Admissions": sum(
            Decimal(str(a["total_cost"])) for a in admitted_patients_data
        )
        if admitted_patients_data
        else Decimal("0"),
        "Existing Billings": sum(
            Decimal(str(b.total_cost or 0)) for b in unpaid_billings
        )
        if unpaid_billings
        else Decimal("0"),
        "Existing Drug Bills": sum(
            Decimal(str(b.total_cost or 0)) for b in unpaid_drug_bills
        )
        if unpaid_drug_bills
        else Decimal("0"),
    }
    grand_total = sum(totals.values())

    return render_template(
        "billing/view_unpaid_bills.html",
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
        payment_success=False,
    )


# ─────────────────────────────────────────────
# SERVICE CHARGES (FEE SCHEDULE CATALOG)
# ─────────────────────────────────────────────
@bp.route("/charges")
@login_required
@roles_required("billing", "admin")
def charges():
    charges_list = Charge.query.options(joinedload(Charge.category)).all()
    categories = ChargeCategory.query.all()
    return render_template(
        "billing/charges_list.html", charges=charges_list, categories=categories
    )


@bp.route("/charges/add", methods=["GET", "POST"])
@login_required
@roles_required("billing", "admin")
def add_charge():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        category_id = request.form.get("category_id")
        new_category_name = request.form.get("new_category_name", "").strip()
        cost = request.form.get("cost", "").strip()
        description = request.form.get("description", "").strip()

        if not name or not cost:
            flash("Charge name and cost are required.", "error")
            return redirect(url_for("billing.add_charge"))

        try:
            cost = Decimal(cost)
            if cost < 0:
                raise ValueError()
        except Exception:
            flash("Cost must be a positive number.", "error")
            return redirect(url_for("billing.add_charge"))

        if new_category_name:
            cat = ChargeCategory.query.filter_by(name=new_category_name).first()
            if not cat:
                cat = ChargeCategory(name=new_category_name)
                db.session.add(cat)
                db.session.commit()
            category_id = cat.id

        if not category_id:
            flash("Please select or enter a valid category.", "error")
            return redirect(url_for("billing.add_charge"))

        new_charge = Charge(
            name=name, category_id=int(category_id), cost=cost, description=description
        )
        db.session.add(new_charge)
        db.session.commit()
        flash(f'Charge "{name}" added successfully!', "success")
        return redirect(url_for("billing.charges"))

    categories = ChargeCategory.query.all()
    return render_template("billing/add_charge.html", categories=categories)


@bp.route("/charges/<int:charge_id>/edit", methods=["GET", "POST"])
@login_required
@roles_required("billing", "admin")
def edit_charge(charge_id):
    charge = Charge.query.get_or_404(charge_id)
    if request.method == "POST":
        charge.name = request.form.get("name", charge.name).strip()
        category_id = request.form.get("category_id")
        if category_id:
            charge.category_id = int(category_id)
        cost_str = request.form.get("cost", "").strip()
        try:
            charge.cost = Decimal(cost_str)
        except Exception:
            flash("Invalid cost value.", "error")
            return redirect(url_for("billing.edit_charge", charge_id=charge_id))
        charge.description = request.form.get("description", charge.description).strip()
        db.session.commit()
        flash("Service charge updated successfully!", "success")
        return redirect(url_for("billing.charges"))

    categories = ChargeCategory.query.all()
    return render_template(
        "billing/add_charge.html", charge=charge, categories=categories
    )


# ─────────────────────────────────────────────
# OFFICIAL RECEIPT & RECEIPT LOG
# ─────────────────────────────────────────────
@bp.route("/receipt/<receipt_number>")
@login_required
@roles_required("billing", "admin")
def official_receipt(receipt_number):
    paid_record = PaidBill.query.filter_by(receipt_number=receipt_number).first_or_404()
    patient = Patient.query.filter_by(patient_id=paid_record.patient_id).first_or_404()

    # Gather itemized records associated with this receipt_number
    dispensed = DispensedDrug.query.filter_by(receipt_number=receipt_number).all()
    labs = RequestedLab.query.filter_by(receipt_number=receipt_number).all()
    clinics = ClinicBooking.query.filter_by(receipt_number=receipt_number).all()
    theatres = TheatreList.query.filter_by(receipt_number=receipt_number).all()
    images = RequestedImage.query.filter_by(receipt_number=receipt_number).all()
    admissions = AdmittedPatient.query.filter_by(receipt_number=receipt_number).all()
    billings = Billing.query.filter_by(receipt_number=receipt_number).all()
    drugs_bills = DrugsBill.query.filter_by(receipt_number=receipt_number).all()

    return render_template(
        "billing/receipt.html",
        paid_record=paid_record,
        patient=patient,
        dispensed=dispensed,
        labs=labs,
        clinics=clinics,
        theatres=theatres,
        images=images,
        admissions=admissions,
        billings=billings,
        drugs_bills=drugs_bills,
    )


@bp.route("/receipts")
@login_required
@roles_required("billing", "admin")
def receipts_list():
    q = request.args.get("q", "").strip()
    query = PaidBill.query.options(joinedload(PaidBill.patient))
    if q:
        query = query.filter(
            (PaidBill.receipt_number.ilike(f"%{q}%"))
            | (PaidBill.patient_id.ilike(f"%{q}%"))
        )
    receipts = query.order_by(PaidBill.paid_at.desc()).limit(100).all()
    return render_template(
        "billing/receipts_list.html", receipts=receipts, search_query=q
    )


# ─────────────────────────────────────────────
# FINANCIAL REPORTS
# ─────────────────────────────────────────────
@bp.route("/reports/daily_revenue")
@login_required
@roles_required("billing", "admin")
def daily_revenue_report():
    from datetime import date, timedelta

    start_str = request.args.get("start")
    end_str = request.args.get("end")
    try:
        start_date = (
            datetime.strptime(start_str, "%Y-%m-%d").date()
            if start_str
            else date.today() - timedelta(days=29)
        )
        end_date = (
            datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else date.today()
        )
    except ValueError:
        start_date = date.today() - timedelta(days=29)
        end_date = date.today()

    paid_records = (
        PaidBill.query.filter(
            db.func.date(PaidBill.paid_at) >= start_date,
            db.func.date(PaidBill.paid_at) <= end_date,
        )
        .options(joinedload(PaidBill.patient))
        .order_by(PaidBill.paid_at.desc())
        .all()
    )

    total_revenue = (
        sum(r.amount_paid for r in paid_records) if paid_records else Decimal("0")
    )
    total_balance = (
        sum(r.balance for r in paid_records) if paid_records else Decimal("0")
    )

    # Group revenue by payment method
    by_method = {}
    for r in paid_records:
        method = r.payment_method or "Cash"
        by_method[method] = float(
            by_method.get(method, Decimal("0")) + Decimal(str(r.amount_paid or 0))
        )

    return render_template(
        "billing/reports_daily_revenue.html",
        paid_records=paid_records,
        total_revenue=total_revenue,
        total_balance=total_balance,
        by_method=by_method,
        start_date=start_date,
        end_date=end_date,
    )


@bp.route("/reports/outstanding")
@login_required
@roles_required("billing", "admin")
def outstanding_report():
    unpaid_billings = (
        Billing.query.filter_by(status=0)
        .options(joinedload(Billing.patient), joinedload(Billing.charge))
        .all()
    )
    unpaid_drug_bills = (
        DrugsBill.query.filter_by(status=0)
        .options(joinedload(DrugsBill.patient), joinedload(DrugsBill.drug))
        .all()
    )

    patient_debt = {}
    for b in unpaid_billings + unpaid_drug_bills:
        pid = b.patient_id
        if pid not in patient_debt:
            patient_debt[pid] = {
                "patient": b.patient,
                "total": Decimal("0"),
                "items_count": 0,
            }
        patient_debt[pid]["total"] += Decimal(str(b.total_cost or 0))
        patient_debt[pid]["items_count"] += 1

    outstanding_list = sorted(
        patient_debt.values(), key=lambda x: x["total"], reverse=True
    )
    grand_debt = sum(item["total"] for item in outstanding_list)

    return render_template(
        "billing/reports_outstanding.html",
        outstanding_list=outstanding_list,
        grand_debt=grand_debt,
    )


@bp.route("/analytics/revenue-by-encounter-type", methods=["GET"])
@login_required
def revenue_by_encounter_type():
    """Revenue per Encounter Type using encounter_id tags on InvoiceLineItem."""
    from departments.models.encounter import Encounter
    from extensions import db
    from sqlalchemy import func

    results = db.session.query(
        Encounter.encounter_type,
        func.count(InvoiceLineItem.id).label("line_items"),
        func.sum(InvoiceLineItem.total).label("total_revenue"),
    ).join(
        Encounter, InvoiceLineItem.encounter_id == Encounter.id
    ).group_by(
        Encounter.encounter_type
    ).order_by(func.sum(InvoiceLineItem.total).desc()).all()

    return jsonify({
        "report": "Revenue per Encounter Type",
        "data": [
            {
                "encounter_type": r.encounter_type,
                "line_items": r.line_items,
                "total_revenue": float(r.total_revenue or 0),
            }
            for r in results
        ]
    })
