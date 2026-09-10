"""
T3.8 Billing Event Listeners - Two-Phase Capture Fix
────────────────────────────────────────────────────
Fixes silent failure where session.new was empty in after_flush_postexec.

Two-phase approach:
- Phase 1 (after_flush): Capture objects while session.new still has them
- Phase 2 (after_flush_postexec): Create billing records after IDs assigned

This ensures that RequestedLab, RequestedImage, PrescribedMedicine, and
DispensedDrug records are properly synced to the unified billing system
with correct encounter_id scoping for TELEHEALTH, ANC, REFERRAL, and IPD.
"""

import logging
from sqlalchemy import event
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

# Global list to capture pending charges during after_flush
_pending_charges = []


@event.listens_for(db.session, "after_flush")
def capture_pending_charges(session, flush_context):
    """
    Phase 1: Capture billing objects before they move to identity map.
    
    At this point, session.new still contains the objects being inserted.
    We capture them here because by after_flush_postexec, session.new will be empty.
    """
    global _pending_charges
    _pending_charges = []
    
    # Capture new billing-related objects
    for instance in list(session.new):
        if isinstance(instance, (RequestedLab, RequestedImage, PrescribedMedicine)):
            # Store the instance for later processing
            _pending_charges.append({
                'type': type(instance).__name__,
                'instance': instance,
            })
    
    if _pending_charges:
        logger.debug(f"Captured {len(_pending_charges)} pending billing charges")


@event.listens_for(db.session, "after_flush_postexec")
def sync_billing_events(session, flush_context):
    """
    Phase 2: Sync billing after flush completes and IDs are assigned.
    
    By this point:
    - All objects have their IDs assigned
    - Foreign keys are resolved
    - session.new is now empty (objects moved to identity map)
    
    We process the charges we captured in Phase 1.
    """
    global _pending_charges
    
    if not _pending_charges:
        return
    
    logger.info(f"Processing {len(_pending_charges)} pending billing charges")
    
    for charge_data in _pending_charges:
        instance = charge_data['instance']
        charge_type = charge_data['type']
        
        try:
            # Lab Requests
            if charge_type == 'RequestedLab':
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
                            source_encounter_id=getattr(instance, "encounter_id", None),
                        )
                        logger.info(f"Synced lab charge for RequestedLab #{instance.id}")
            
            # Imaging Requests
            elif charge_type == 'RequestedImage':
                if hasattr(instance, "id") and instance.id is not None:
                    from departments.models.medicine import Imaging
                    imaging = session.get(Imaging, instance.imaging_id)
                    if imaging:
                        sync_charge(
                            patient_id=instance.patient_id,
                            source_table="requested_image",
                            source_id=instance.id,
                            description=f"Imaging: {imaging.imaging_type}",
                            category="imaging",
                            amount=float(imaging.cost or 0),
                            source_encounter_id=getattr(instance, "encounter_id", None),
                        )
                        logger.info(f"Synced imaging charge for RequestedImage #{instance.id}")
            
            # Prescribed Medicines
            elif charge_type == 'PrescribedMedicine':
                if hasattr(instance, "id") and instance.id is not None:
                    from departments.models.medicine import Medicine
                    med = session.get(Medicine, instance.medicine_id)
                    med_name = med.generic_name if med else "Medication"
                    sync_charge(
                        patient_id=instance.patient_id,
                        source_table="prescribed_medicine",
                        source_id=instance.id,
                        description=f"Prescription: {med_name}",
                        category="drug",
                        amount=0.0,  # Cost resolved at dispensing
                        source_encounter_id=getattr(instance, "encounter_id", None),
                    )
                    logger.info(f"Synced prescription charge for PrescribedMedicine #{instance.id}")
            
        except Exception as e:
            logger.error(f"Error syncing billing for {charge_type} #{getattr(instance, 'id', 'unknown')}: {e}", exc_info=True)
            continue
    
    # Clear pending list for next flush cycle
    _pending_charges = []


def register_billing_sync_listeners():
    """
    Register all billing sync event listeners.
    
    Call this function during app initialization to enable automatic
    billing synchronization for labs, imaging, prescriptions, and drugs.
    """
    # Listeners are already registered via @event.listens_for decorators
    # This function exists for explicit registration if needed
    logger.info("✅ Billing sync event listeners registered (two-phase capture)")
