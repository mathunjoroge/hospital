"""
SQLAlchemy event listeners for automatic billing sync.

Uses session-level 'after_flush' event to safely sync charges to the
unified Invoice system after legacy records are flushed to the database.
"""
import logging

from sqlalchemy import event

from extensions import db

logger = logging.getLogger(__name__)


def register_billing_sync_listeners():
    """Register all event listeners for billing sync."""

    try:
        # Import models
        from departments.billing.sync import sync_charge
        from departments.models.medicine import (
            AdmittedPatient,
            LabTest,
            RequestedLab,
            TheatreList,
        )
        from departments.models.pharmacy import DispensedDrug
        from departments.models.records import ClinicBooking

        @event.listens_for(db.session, 'after_flush')
        def sync_billing_charges(session, flush_context):
            """
            Process newly inserted objects and sync them to the unified Invoice system.
            This runs after the flush completes, so all PKs are assigned and it's safe
            to query related data and add new InvoiceLineItems to the session.
            """
            # Iterate over newly added objects in this flush
            # We use list(session.new) to avoid modifying the set while iterating
            for instance in list(session.new):
                try:
                    # 1. Lab Requests
                    if isinstance(instance, RequestedLab):
                        if hasattr(instance, 'id') and instance.id is not None:
                            lab_test = session.get(LabTest, instance.lab_test_id)
                            if lab_test:
                                sync_charge(
                                    patient_id=instance.patient_id,
                                    source_table='requested_lab',
                                    source_id=instance.id,
                                    description=f"Lab Test: {lab_test.test_name}",
                                    category='lab',
                                    amount=float(lab_test.cost or 0)
                                )

                    # 2. Dispensed Drugs
                    elif isinstance(instance, DispensedDrug):
                        if hasattr(instance, 'id') and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table='dispensed_drug',
                                source_id=instance.id,
                                description=f"Drug: {getattr(instance, 'drug_name', 'Medication')}",
                                category='drug',
                                amount=float(getattr(instance, 'total_price', 0) or 0),
                                quantity=int(getattr(instance, 'quantity', 1) or 1)
                            )

                    # 3. Clinic Bookings
                    elif isinstance(instance, ClinicBooking):
                        if hasattr(instance, 'id') and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table='clinic_booking',
                                source_id=instance.id,
                                description="Clinic Consultation",
                                category='consult',
                                amount=float(getattr(instance, 'amount', 0) or 0)
                            )

                    # 4. Theatre Bookings
                    elif isinstance(instance, TheatreList):
                        if hasattr(instance, 'id') and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table='theatre_list',
                                source_id=instance.id,
                                description=f"Theatre: {getattr(instance, 'procedure_name', 'Surgery')}",
                                category='theatre',
                                amount=float(getattr(instance, 'amount', 0) or 0)
                            )

                    # 5. Admissions
                    elif isinstance(instance, AdmittedPatient):
                        if hasattr(instance, 'id') and instance.id is not None:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table='admitted_patient',
                                source_id=instance.id,
                                description="Ward Admission",
                                category='ward',
                                amount=float(getattr(instance, 'amount', 0) or 0)
                            )

                except Exception as e:
                    logger.error(f"Error syncing billing charge for {type(instance).__name__} ID {getattr(instance, 'id', 'unknown')}: {e}")

        logger.info("✅ Billing sync event listeners registered successfully (using after_flush)")

    except Exception as e:
        logger.error(f"❌ Error registering billing sync listeners: {e}")
        import traceback
        traceback.print_exc()


def unregister_billing_sync_listeners():
    """Unregister all event listeners (for testing/cleanup)."""
    logger.info("Billing sync listener unregistration not implemented")
