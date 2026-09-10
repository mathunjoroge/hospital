"""
SQLAlchemy event listeners for automatic billing sync.

Uses session-level 'after_flush' event to safely sync charges to the
unified Invoice system after legacy records are flushed to the database.
"""
from departments.models.medicine import RequestedLab, RequestedImage, PrescribedMedicine
from departments.models.billing import InvoiceLineItem
from sqlalchemy import event, select
import logging


from extensions import db

logger = logging.getLogger(__name__)


def register_billing_sync_listeners():
    """Register all event listeners for billing charges and payments sync."""

    try:
        # Import models
        from departments.billing.sync import sync_charge, sync_payment
        from departments.models.billing import (
            Billing,
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

        @event.listens_for(db.session, "after_flush_postexec")
        def sync_billing_events(session, flush_context):
            """
            Process newly inserted and updated billing objects to sync charges and payments
            to the unified Invoice system after the flush completes.
            """
            processed = set()
            # KNOWN LIMITATION (AUDIT FINDING - PRIORITY 5):
            # Only session.new and session.dirty are inspected here. session.deleted is intentionally not handled,
            # meaning deleted or reversed legacy bill records leave an orphaned InvoiceLineItem on the unified invoice.
            # See DECISIONS_PENDING.md Section 10 for product governance options.
            items_to_check = list(session.new) + list(session.dirty)

            for instance in items_to_check:
                if id(instance) in processed:
                    continue
                processed.add(id(instance))

                try:
                    # 1. Lab Requests
                    if isinstance(instance, RequestedLab):
                        if hasattr(instance, "id") and instance.id is not None:
                            lab_test = session.get(LabTest, instance.lab_test_id)
                            if lab_test:
                                sync_charge(
                                    patient_id=instance.patient_id,
                                    source_table="requested_lab",
                                    source_id=instance.id,
                                    description=f"Lab Test: {lab_test.test_name}",
                                    category="lab",
                                    amount=float(lab_test.cost or 0),
                                )

                    # 2. Dispensed Drugs
                    elif isinstance(instance, DispensedDrug):
                        if hasattr(instance, "id") and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table="dispensed_drug",
                                source_id=instance.id,
                                description=f"Drug: {getattr(instance, 'drug_name', 'Medication')}",
                                category="drug",
                                amount=float(getattr(instance, "total_price", 0) or 0),
                                quantity=int(getattr(instance, "quantity", 1) or 1),
                            )

                    # 3. Clinic Bookings
                    elif isinstance(instance, ClinicBooking):
                        if hasattr(instance, "id") and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table="clinic_booking",
                                source_id=instance.id,
                                description="Clinic Consultation",
                                category="consult",
                                amount=float(getattr(instance, "amount", 0) or 0),
                            )

                    # 4. Theatre Bookings
                    elif isinstance(instance, TheatreList):
                        if hasattr(instance, "id") and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table="theatre_list",
                                source_id=instance.id,
                                description=f"Theatre: {getattr(instance, 'procedure_name', 'Surgery')}",
                                category="theatre",
                                amount=float(getattr(instance, "amount", 0) or 0),
                            )

                    # 5. Admissions
                    elif isinstance(instance, AdmittedPatient):
                        if hasattr(instance, "id") and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table="admitted_patient",
                                source_id=instance.id,
                                description="Ward Admission",
                                category="ward",
                                amount=float(getattr(instance, "amount", 0) or 0),
                            )

                    # 6. Legacy PaidBill (Receipt generation in cashier flow)
                    elif isinstance(instance, PaidBill):
                        if hasattr(instance, "id") and instance.id is not None:
                            sync_payment(
                                patient_id=instance.patient_id,
                                amount=float(instance.amount_paid or 0),
                                payment_method=getattr(
                                    instance, "payment_method", "cash"
                                )
                                or "cash",
                                receipt_number=instance.receipt_number,
                            )

                    # 7. Legacy DrugsBill (Charges + Payments)
                    elif isinstance(instance, DrugsBill):
                        if hasattr(instance, "id") and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table="drugs_bill",
                                source_id=instance.id,
                                description=f"Drug Bill: {getattr(instance, 'drug_name', 'Medication')}",
                                category="drug",
                                amount=float(instance.total_cost or 0),
                                quantity=int(getattr(instance, "quantity", 1) or 1),
                            )
                            if (
                                getattr(instance, "status", 0) == 1
                                or instance.receipt_number
                            ):
                                sync_payment(
                                    patient_id=instance.patient_id,
                                    amount=float(instance.total_cost or 0),
                                    payment_method=getattr(
                                        instance, "payment_method", "cash"
                                    )
                                    or "cash",
                                    reference_number=getattr(
                                        instance, "payment_reference", None
                                    ),
                                    receipt_number=getattr(
                                        instance, "receipt_number", None
                                    ),
                                )

                    # 8. Legacy Billing (Charges + Payments)
                    elif isinstance(instance, Billing):
                        if hasattr(instance, "id") and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table="billing",
                                source_id=instance.id,
                                description="Hospital Charge",
                                category="other",
                                amount=float(instance.total_cost or 0),
                                quantity=int(getattr(instance, "quantity", 1) or 1),
                            )
                            if (
                                getattr(instance, "status", 0) == 1
                                or instance.receipt_number
                            ):
                                sync_payment(
                                    patient_id=instance.patient_id,
                                    amount=float(instance.total_cost or 0),
                                    payment_method="cash",
                                    receipt_number=getattr(
                                        instance, "receipt_number", None
                                    ),
                                )

                    # 9. Department-Specific Bills (LabBill, ClinicBill, TheatreBill, ImagingBill, WardBill)
                    elif isinstance(
                        instance,
                        (LabBill, ClinicBill, TheatreBill, ImagingBill, WardBill),
                    ):
                        if (
                            hasattr(instance, "id")
                            and instance.id is not None
                            and getattr(instance, "total_paid", 0)
                        ):
                            amount_paid = float(getattr(instance, "total_paid", 0) or 0)
                            if amount_paid > 0:
                                sync_payment(
                                    patient_id=instance.patient_id,
                                    amount=amount_paid,
                                    payment_method=getattr(
                                        instance, "payment_method", "cash"
                                    )
                                    or "cash",
                                    reference_number=getattr(
                                        instance, "payment_reference", None
                                    ),
                                    receipt_number=getattr(
                                        instance, "receipt_number", None
                                    ),
                                )

                except Exception as e:
                    logger.error(
                        f"Error syncing billing event for {type(instance).__name__} ID {getattr(instance, 'id', 'unknown')}: {e}"
                    )

        logger.info(
            "✅ Billing sync event listeners registered successfully (charges + payments)"
        )

    except Exception as e:
        logger.error(f"❌ Error registering billing sync listeners: {e}")
        import traceback

        traceback.print_exc()


def unregister_billing_sync_listeners():
    """Unregister all event listeners (for testing/cleanup)."""
    logger.info("Billing sync listener unregistration not implemented")


# --- T3.8: Encounter Scoping for Billing ---
@event.listens_for(InvoiceLineItem, 'before_insert')
@event.listens_for(InvoiceLineItem, 'before_update')
def populate_encounter_id_on_line_item(mapper, connection, target):
    """
    Intercepts the creation/update of an InvoiceLineItem and backfills 
    the encounter_id from the source clinical service if it's missing.
    """
    if getattr(target, 'encounter_id', None):
        return

    source_id = None
    source_type = None

    # Note: Adjust attribute names (e.g., requested_lab_id) if your schema differs
    if hasattr(target, 'lab_request_id') and target.lab_request_id:
        source_id = target.lab_request_id
        source_type = 'lab'
    elif hasattr(target, 'image_request_id') and target.image_request_id:
        source_id = target.image_request_id
        source_type = 'image'
    elif hasattr(target, 'prescription_id') and target.prescription_id:
        source_id = target.prescription_id
        source_type = 'med'

    if not source_id:
        return

    try:
        encounter_id = None
        if source_type == 'lab':
            encounter_id = connection.execute(
                select(RequestedLab.encounter_id).where(RequestedLab.id == source_id)
            ).scalar_one_or_none()
        elif source_type == 'image':
            encounter_id = connection.execute(
                select(RequestedImage.encounter_id).where(RequestedImage.id == source_id)
            ).scalar_one_or_none()
        elif source_type == 'med':
            encounter_id = connection.execute(
                select(PrescribedMedicine.encounter_id).where(PrescribedMedicine.id == source_id)
            ).scalar_one_or_none()

        if encounter_id:
            target.encounter_id = encounter_id
    except Exception:
        # Fail gracefully: do not block billing creation if a source record is missing
        pass
# ---------------------------------------------
