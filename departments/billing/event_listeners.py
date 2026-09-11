"""
T3.8 Billing Event Listeners — Production-Safe Implementation
─────────────────────────────────────────────────────────────
Two-phase capture with independent session and thread-safe storage.

Phase 1 (after_flush): Capture billing objects as plain values while
    session.new still contains them.
Phase 2 (after_flush_postexec): Process captured data using an INDEPENDENT
    session to avoid "Session is already flushing" errors.

Covers: RequestedLab, RequestedImage, PrescribedMedicine, DispensedDrug,
        TheatreList, ClinicBooking, PaidBill, DrugsBill, Billing

Note: AdmittedPatient/ward charges are intentionally NOT auto-billed here.
    Ward.daily_charge is a per-day rate, not a one-time admission fee, and
    /nursing/mar/auto_bill (departments/nursing/mar.py) is the single,
    explicit, auditable engine for billing it once per admitted day. Syncing
    a charge here too would double-bill day one every time.

Thread-safe: Uses threading.local() instead of module-level globals to
    prevent race conditions under concurrent requests.
"""

import logging
import threading

from sqlalchemy import event
from sqlalchemy.orm import Session

from extensions import db

logger = logging.getLogger(__name__)

# Thread-local storage to prevent race conditions under concurrent requests
_thread_local = threading.local()


def _get_pending_charges():
    """Get thread-local pending charges list."""
    if not hasattr(_thread_local, 'pending_charges'):
        _thread_local.pending_charges = []
    return _thread_local.pending_charges


def _clear_pending_charges():
    """Clear thread-local pending charges."""
    _thread_local.pending_charges = []


@event.listens_for(db.session, "after_flush")
def capture_pending_charges(session, flush_context):
    """
    Phase 1: Capture billing data as plain values before objects move
    to the identity map.

    We extract all needed data as plain Python values (strings, ints, floats)
    instead of holding references to SQLAlchemy objects, which may become
    invalid when we try to use them with an independent session.
    """
    from departments.models.billing import Billing, DrugsBill, PaidBill
    from departments.models.medicine import (
        PrescribedMedicine,
        RequestedImage,
        RequestedLab,
        TheatreList,
    )
    from departments.models.pharmacy import DispensedDrug
    from departments.models.records import ClinicBooking

    pending = _get_pending_charges()

    for instance in list(session.new):
        try:
            if isinstance(instance, RequestedLab):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'RequestedLab',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'lab_test_id': instance.lab_test_id,
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    })

            elif isinstance(instance, RequestedImage):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'RequestedImage',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'imaging_id': instance.imaging_id,
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    })

            elif isinstance(instance, PrescribedMedicine):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'PrescribedMedicine',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'medicine_id': instance.medicine_id,
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    })

            elif isinstance(instance, DispensedDrug):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'DispensedDrug',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'drug_name': getattr(instance, 'drug_name', 'Medication'),
                        'unit_price': float(getattr(instance, 'unit_price', 0) or 0),
                        'quantity': int(getattr(instance, 'quantity', 1) or 1),
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    })

            elif isinstance(instance, TheatreList):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'TheatreList',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'procedure_id': instance.procedure_id,
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    })

            elif isinstance(instance, ClinicBooking):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'ClinicBooking',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'consultation_fee': float(getattr(instance, 'consultation_fee', 0) or 0),
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    })

            # Payment sync
            elif isinstance(instance, PaidBill):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'PaidBill',
                        'patient_id': instance.patient_id,
                        'amount': float(getattr(instance, 'amount_paid', 0) or 0),
                        'payment_method': getattr(instance, 'payment_method', 'cash'),
                        'receipt_number': getattr(instance, 'receipt_number', None),
                    })

            elif isinstance(instance, DrugsBill):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'DrugsBill',
                        'patient_id': instance.patient_id,
                        'amount': float(getattr(instance, 'total', 0) or 0),
                        'payment_method': getattr(instance, 'payment_method', 'cash'),
                        'receipt_number': getattr(instance, 'receipt_number', None),
                    })

            elif isinstance(instance, Billing):
                if hasattr(instance, "id") and instance.id is not None:
                    pending.append({
                        'type': 'Billing',
                        'patient_id': instance.patient_id,
                        'amount': float(getattr(instance, 'amount', 0) or 0),
                        'payment_method': getattr(instance, 'payment_method', 'cash'),
                        'receipt_number': getattr(instance, 'receipt_number', None),
                    })

        except Exception as e:
            logger.error(f"Error capturing charge data: {e}", exc_info=True)
            continue

    if pending:
        logger.debug(f"Captured {len(pending)} pending billing charges")


@event.listens_for(db.session, "after_flush_postexec")
def sync_billing_events(session, flush_context):
    """
    Phase 2: Sync billing using an INDEPENDENT session.

    We use the plain values captured in Phase 1 to look up related objects
    (LabTest, Imaging, Medicine, etc.) using the independent session, then
    call sync_charge/sync_payment with that session.
    """
    from flask import current_app

    from departments.models.medicine import (
        Imaging,
        LabTest,
        Medicine,
        TheatreProcedure,
    )

    from .sync import sync_charge, sync_payment

    pending = _get_pending_charges()

    if not pending:
        return

    logger.info(f"Processing {len(pending)} pending billing charges")

    # SQLite in-memory databases are isolated per connection, so an
    # independent Session(bind=db.engine) would write to a different
    # database than the one tests read from. In test mode we instead bind
    # a *separate* Session object to the outer session's own connection —
    # this keeps it on the same in-memory database/transaction (so it can
    # see the not-yet-committed rows and participates in the same
    # commit/rollback), while still being a distinct Session instance.
    # Reusing db.session itself here would trip SQLAlchemy's "Session is
    # already flushing" guard, since after_flush_postexec fires while the
    # outer session's flush() call is still on the stack.
    should_commit = not current_app.config.get("TESTING", False)
    sync_session = (
        Session(bind=db.engine) if should_commit else Session(bind=session.connection())
    )

    try:
        for charge_data in pending:
            try:
                charge_type = charge_data['type']

                # ── Charge sync ──────────────────────────────────────
                if charge_type == 'RequestedLab':
                    lab_test = sync_session.get(LabTest, charge_data['lab_test_id'])
                    if lab_test:
                        sync_charge(
                            patient_id=charge_data['patient_id'],
                            source_table="requested_lab",
                            source_id=charge_data['source_id'],
                            description=f"Lab Test: {lab_test.test_name}",
                            category="lab",
                            amount=float(lab_test.cost or 0),
                            source_encounter_id=charge_data.get('encounter_id'),
                            _session=sync_session,
                        )
                        logger.info(f"Synced lab charge for RequestedLab #{charge_data['source_id']}")

                elif charge_type == 'RequestedImage':
                    imaging = sync_session.get(Imaging, charge_data['imaging_id'])
                    if imaging:
                        sync_charge(
                            patient_id=charge_data['patient_id'],
                            source_table="requested_image",
                            source_id=charge_data['source_id'],
                            description=f"Imaging: {imaging.imaging_type}",
                            category="imaging",
                            amount=float(imaging.cost or 0),
                            source_encounter_id=charge_data.get('encounter_id'),
                            _session=sync_session,
                        )
                        logger.info(f"Synced imaging charge for RequestedImage #{charge_data['source_id']}")

                elif charge_type == 'PrescribedMedicine':
                    med = sync_session.get(Medicine, charge_data['medicine_id'])
                    med_name = med.generic_name if med else "Medication"
                    sync_charge(
                        patient_id=charge_data['patient_id'],
                        source_table="prescribed_medicine",
                        source_id=charge_data['source_id'],
                        description=f"Prescription: {med_name}",
                        category="drug",
                        amount=0.0,  # Cost resolved at dispensing
                        source_encounter_id=charge_data.get('encounter_id'),
                        _session=sync_session,
                    )
                    logger.info(f"Synced prescription charge for PrescribedMedicine #{charge_data['source_id']}")

                elif charge_type == 'DispensedDrug':
                    sync_charge(
                        patient_id=charge_data['patient_id'],
                        source_table="dispensed_drug",
                        source_id=charge_data['source_id'],
                        description=f"Dispensed: {charge_data['drug_name']}",
                        category="drug",
                        amount=charge_data['unit_price'],
                        quantity=charge_data['quantity'],
                        source_encounter_id=charge_data.get('encounter_id'),
                        _session=sync_session,
                    )
                    logger.info(f"Synced drug charge for DispensedDrug #{charge_data['source_id']}")

                elif charge_type == 'TheatreList':
                    procedure = sync_session.get(TheatreProcedure, charge_data['procedure_id'])
                    if procedure:
                        sync_charge(
                            patient_id=charge_data['patient_id'],
                            source_table="theatre_list",
                            source_id=charge_data['source_id'],
                            description=f"Theatre: {procedure.name}",
                            category="theatre",
                            amount=float(procedure.cost or 0),
                            source_encounter_id=charge_data.get('encounter_id'),
                            _session=sync_session,
                        )
                        logger.info(f"Synced theatre charge for TheatreList #{charge_data['source_id']}")

                elif charge_type == 'ClinicBooking':
                    sync_charge(
                        patient_id=charge_data['patient_id'],
                        source_table="clinic_booking",
                        source_id=charge_data['source_id'],
                        description="Consultation Fee",
                        category="consult",
                        amount=charge_data['consultation_fee'],
                        source_encounter_id=charge_data.get('encounter_id'),
                        _session=sync_session,
                    )
                    logger.info(f"Synced consult charge for ClinicBooking #{charge_data['source_id']}")

                # ── Payment sync ─────────────────────────────────────
                elif charge_type in ('PaidBill', 'DrugsBill', 'Billing'):
                    if charge_data['amount'] > 0:
                        sync_payment(
                            patient_id=charge_data['patient_id'],
                            amount=charge_data['amount'],
                            payment_method=charge_data['payment_method'],
                            receipt_number=charge_data.get('receipt_number'),
                                _session=sync_session,
                            )
                        logger.info(f"Synced payment for {charge_type} #{charge_data.get('source_id', 'N/A')}")

            except Exception as e:
                logger.error(f"Error syncing billing for {charge_data.get('type', 'unknown')}: {e}", exc_info=True)
                sync_session.rollback()
                continue

        # sync_session is always its own Session object now (either bound to
        # the engine in production, or to the outer session's connection in
        # test mode). Committing it is safe and necessary in both cases:
        # in production it commits an independent transaction; in test mode
        # (connection-bound) it only releases that Session's own SAVEPOINT
        # into the still-open outer transaction, making the synced rows
        # visible to the outer db.session without prematurely committing it.
        sync_session.commit()
        logger.info(
            "Billing sync committed successfully"
            if should_commit else
            "Billing sync completed (test mode, savepoint released)"
        )

    except Exception as e:
        logger.error(f"Billing sync session error: {e}", exc_info=True)
        sync_session.rollback()

    finally:
        # Safe in both cases: when bound to an explicit Connection (test
        # mode) close() only expunges objects/detaches — it does not close
        # a connection the Session doesn't own.
        sync_session.close()
        _clear_pending_charges()


def register_billing_sync_listeners():
    """Register billing sync event listeners during app initialization."""
    logger.info("✅ Billing sync event listeners registered (two-phase + independent session + thread-safe)")
