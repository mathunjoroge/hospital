"""
T3.8 Billing Event Listeners — Two-Phase Capture with Independent Session
─────────────────────────────────────────────────────────────────────────
Phase 1 (after_flush): Capture billing data as plain values (not objects)
Phase 2 (after_flush_postexec): Process using an INDEPENDENT session

We capture plain values instead of object references to avoid ObjectDeletedError
when the original session state changes between phases.
"""

import logging
from sqlalchemy import event
from sqlalchemy.orm import Session
from extensions import db
from departments.models.billing import InvoiceLineItem
from departments.models.medicine import (
    RequestedLab,
    RequestedImage,
    PrescribedMedicine,
    LabTest,
)
from .sync import sync_charge

logger = logging.getLogger(__name__)

# Global list to capture billing data as plain values
_pending_charges = []


@event.listens_for(db.session, "after_flush")
def capture_pending_charges(session, flush_context):
    """
    Phase 1: Capture billing data as plain values before objects move to identity map.
    
    We extract all needed data as plain Python values (strings, ints, floats)
    instead of holding references to SQLAlchemy objects, which may become
    invalid when we try to use them with an independent session.
    """
    global _pending_charges
    _pending_charges = []
    
    for instance in list(session.new):
        try:
            if isinstance(instance, RequestedLab):
                if hasattr(instance, "id") and instance.id is not None:
                    # Extract all data as plain values
                    charge_data = {
                        'type': 'RequestedLab',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'lab_test_id': instance.lab_test_id,
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    }
                    _pending_charges.append(charge_data)
            
            elif isinstance(instance, RequestedImage):
                if hasattr(instance, "id") and instance.id is not None:
                    charge_data = {
                        'type': 'RequestedImage',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'imaging_id': instance.imaging_id,
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    }
                    _pending_charges.append(charge_data)
            
            elif isinstance(instance, PrescribedMedicine):
                if hasattr(instance, "id") and instance.id is not None:
                    charge_data = {
                        'type': 'PrescribedMedicine',
                        'patient_id': instance.patient_id,
                        'source_id': instance.id,
                        'medicine_id': instance.medicine_id,
                        'encounter_id': getattr(instance, 'encounter_id', None),
                    }
                    _pending_charges.append(charge_data)
        
        except Exception as e:
            logger.error(f"Error capturing charge data: {e}", exc_info=True)
            continue
    
    if _pending_charges:
        logger.debug(f"Captured {len(_pending_charges)} pending billing charges")


@event.listens_for(db.session, "after_flush_postexec")
def sync_billing_events(session, flush_context):
    """
    Phase 2: Sync billing using an INDEPENDENT session.
    
    We use the plain values captured in Phase 1 to look up related objects
    (LabTest, Imaging, Medicine) using the independent session, then call
    sync_charge with that session.
    """
    global _pending_charges
    
    if not _pending_charges:
        return
    
    logger.info(f"Processing {len(_pending_charges)} pending billing charges")
    
    # Open an independent session for billing writes
    sync_session = Session(bind=db.engine)
    
    try:
        for charge_data in _pending_charges:
            try:
                charge_type = charge_data['type']
                patient_id = charge_data['patient_id']
                source_id = charge_data['source_id']
                encounter_id = charge_data.get('encounter_id')
                
                if charge_type == 'RequestedLab':
                    lab_test_id = charge_data['lab_test_id']
                    lab_test = sync_session.get(LabTest, lab_test_id)
                    if lab_test:
                        sync_charge(
                            patient_id=patient_id,
                            source_table="requested_lab",
                            source_id=source_id,
                            description=f"Lab Test: {lab_test.test_name}",
                            category="lab",
                            amount=float(lab_test.cost or 0),
                            source_encounter_id=encounter_id,
                            _session=sync_session,
                        )
                        logger.info(f"Synced lab charge for RequestedLab #{source_id}")
                
                elif charge_type == 'RequestedImage':
                    from departments.models.medicine import Imaging
                    imaging_id = charge_data['imaging_id']
                    imaging = sync_session.get(Imaging, imaging_id)
                    if imaging:
                        sync_charge(
                            patient_id=patient_id,
                            source_table="requested_image",
                            source_id=source_id,
                            description=f"Imaging: {imaging.imaging_type}",
                            category="imaging",
                            amount=float(imaging.cost or 0),
                            source_encounter_id=encounter_id,
                            _session=sync_session,
                        )
                        logger.info(f"Synced imaging charge for RequestedImage #{source_id}")
                
                elif charge_type == 'PrescribedMedicine':
                    from departments.models.medicine import Medicine
                    medicine_id = charge_data['medicine_id']
                    med = sync_session.get(Medicine, medicine_id)
                    med_name = med.generic_name if med else "Medication"
                    sync_charge(
                        patient_id=patient_id,
                        source_table="prescribed_medicine",
                        source_id=source_id,
                        description=f"Prescription: {med_name}",
                        category="drug",
                        amount=0.0,
                        source_encounter_id=encounter_id,
                        _session=sync_session,
                    )
                    logger.info(f"Synced prescription charge for PrescribedMedicine #{source_id}")
            
            except Exception as e:
                logger.error(f"Error syncing billing for {charge_data.get('type', 'unknown')}: {e}", exc_info=True)
                sync_session.rollback()
                continue
        
        # Commit all billing writes in the independent session
        sync_session.commit()
        logger.info("Billing sync committed successfully")
    
    except Exception as e:
        logger.error(f"Billing sync session error: {e}", exc_info=True)
        sync_session.rollback()
    
    finally:
        sync_session.close()
        _pending_charges = []


def register_billing_sync_listeners():
    """Register billing sync event listeners during app initialization."""
    logger.info("✅ Billing sync event listeners registered (two-phase + independent session)")
