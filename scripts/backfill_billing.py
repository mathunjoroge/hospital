"""
Historical Billing Backfill Script.

Walks through all existing legacy charges and payments in the database
and calls sync_charge() and sync_payment() to populate the unified Invoice system.

Idempotent: Safe to run multiple times without creating duplicate line items or payments.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_billing")


def run_backfill():
    from app import app
    from departments.billing.sync import sync_charge, sync_payment
    from departments.models.billing import (
        Billing,
        Charge,
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
        LabTest,
        RequestedLab,
        TheatreList,
    )
    from departments.models.pharmacy import DispensedDrug
    from departments.models.records import ClinicBooking
    from extensions import db

    with app.app_context():
        logger.info("Starting historical billing backfill...")
        counts = {"charges": 0, "payments": 0, "errors": 0}

        # 1. RequestedLab
        labs = RequestedLab.query.all()
        for lab in labs:
            try:
                lab_test = db.session.get(LabTest, lab.lab_test_id) if lab.lab_test_id else None
                test_name = lab_test.test_name if lab_test else "Lab Test"
                cost = float(lab_test.cost or 0) if lab_test else 0.0
                res = sync_charge(
                    patient_id=lab.patient_id,
                    source_table="requested_lab",
                    source_id=lab.id,
                    description=f"Lab Test: {test_name}",
                    category="lab",
                    amount=cost,
                )
                if res:
                    counts["charges"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error backfilling RequestedLab {lab.id}: {e}")
                counts["errors"] += 1

        # 2. DispensedDrug
        drugs = DispensedDrug.query.all()
        for d in drugs:
            try:
                res = sync_charge(
                    patient_id=d.patient_id,
                    source_table="dispensed_drug",
                    source_id=d.id,
                    description=f"Drug: {getattr(d, 'drug_name', 'Medication')}",
                    category="drug",
                    amount=float(getattr(d, "total_price", 0) or 0),
                    quantity=int(getattr(d, "quantity", 1) or 1),
                )
                if res:
                    counts["charges"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error backfilling DispensedDrug {d.id}: {e}")
                counts["errors"] += 1

        # 3. ClinicBooking
        bookings = ClinicBooking.query.all()
        for b in bookings:
            try:
                res = sync_charge(
                    patient_id=b.patient_id,
                    source_table="clinic_booking",
                    source_id=b.id,
                    description="Clinic Consultation",
                    category="consult",
                    amount=float(getattr(b, "amount", 0) or 0),
                )
                if res:
                    counts["charges"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error backfilling ClinicBooking {b.id}: {e}")
                counts["errors"] += 1

        # 4. TheatreList
        theatres = TheatreList.query.all()
        for t in theatres:
            try:
                res = sync_charge(
                    patient_id=t.patient_id,
                    source_table="theatre_list",
                    source_id=t.id,
                    description=f"Theatre: {getattr(t, 'procedure_name', 'Surgery')}",
                    category="theatre",
                    amount=float(getattr(t, "amount", 0) or 0),
                )
                if res:
                    counts["charges"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error backfilling TheatreList {t.id}: {e}")
                counts["errors"] += 1

        # 5. AdmittedPatient
        admissions = AdmittedPatient.query.all()
        for a in admissions:
            try:
                res = sync_charge(
                    patient_id=a.patient_id,
                    source_table="admitted_patient",
                    source_id=a.id,
                    description="Ward Admission",
                    category="ward",
                    amount=float(getattr(a, "amount", 0) or 0),
                )
                if res:
                    counts["charges"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error backfilling AdmittedPatient {a.id}: {e}")
                counts["errors"] += 1

        # 6. Legacy Billing
        billings = Billing.query.all()
        for bill in billings:
            try:
                charge = db.session.get(Charge, bill.charge_id) if bill.charge_id else None
                charge_name = charge.name if charge else "Hospital Charge"
                res = sync_charge(
                    patient_id=bill.patient_id,
                    source_table="billing",
                    source_id=bill.id,
                    description=f"Charge: {charge_name}",
                    category="other",
                    amount=float(bill.total_cost or 0),
                    quantity=int(bill.quantity or 1),
                )
                if res:
                    counts["charges"] += 1
                if getattr(bill, "status", 0) == 1 or bill.receipt_number:
                    p_res = sync_payment(
                        patient_id=bill.patient_id,
                        amount=float(bill.total_cost or 0),
                        payment_method="cash",
                        receipt_number=getattr(bill, "receipt_number", None),
                    )
                    if p_res:
                        counts["payments"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error backfilling Billing {bill.id}: {e}")
                counts["errors"] += 1

        # 7. Legacy DrugsBill
        drug_bills = DrugsBill.query.all()
        for dbill in drug_bills:
            try:
                res = sync_charge(
                    patient_id=dbill.patient_id,
                    source_table="drugs_bill",
                    source_id=dbill.id,
                    description="Drug Bill",
                    category="drug",
                    amount=float(dbill.total_cost or 0),
                    quantity=int(dbill.quantity or 1),
                )
                if res:
                    counts["charges"] += 1
                if getattr(dbill, "status", 0) == 1 or dbill.receipt_number:
                    p_res = sync_payment(
                        patient_id=dbill.patient_id,
                        amount=float(dbill.total_cost or 0),
                        payment_method=getattr(dbill, "payment_method", "cash") or "cash",
                        reference_number=getattr(dbill, "payment_reference", None),
                        receipt_number=getattr(dbill, "receipt_number", None),
                    )
                    if p_res:
                        counts["payments"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error backfilling DrugsBill {dbill.id}: {e}")
                counts["errors"] += 1

        # 8. Legacy PaidBill
        paid_bills = PaidBill.query.all()
        for pb in paid_bills:
            try:
                p_res = sync_payment(
                    patient_id=pb.patient_id,
                    amount=float(pb.amount_paid or 0),
                    payment_method=getattr(pb, "payment_method", "cash") or "cash",
                    receipt_number=pb.receipt_number,
                )
                if p_res:
                    counts["payments"] += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error backfilling PaidBill {pb.id}: {e}")
                counts["errors"] += 1

        # 9. Department-Specific Bills
        for model in [LabBill, ClinicBill, TheatreBill, ImagingBill, WardBill]:
            bills = model.query.all()
            for b in bills:
                try:
                    amount_paid = float(getattr(b, "total_paid", 0) or 0)
                    if amount_paid > 0:
                        p_res = sync_payment(
                            patient_id=b.patient_id,
                            amount=amount_paid,
                            payment_method=getattr(b, "payment_method", "cash") or "cash",
                            reference_number=getattr(b, "payment_reference", None),
                            receipt_number=getattr(b, "receipt_number", None),
                        )
                        if p_res:
                            counts["payments"] += 1
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error backfilling {model.__name__} {b.id}: {e}")
                    counts["errors"] += 1

        db.session.commit()
        logger.info(f"✅ Historical billing backfill completed successfully: {counts}")


if __name__ == "__main__":
    run_backfill()
