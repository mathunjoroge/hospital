#!/usr/bin/env python3
"""
T3.8 Billing Event Listener — Final Fix
Use independent session in after_flush_postexec to avoid "Session is already flushing"
"""

from pathlib import Path

print("🔧 Applying T3.8 final fix (independent session pattern)...")

listener_file = Path("departments/billing/event_listeners.py")

new_content = '''"""
T3.8 Billing Event Listeners — Two-Phase Capture with Independent Session
─────────────────────────────────────────────────────────────────────────
Phase 1 (after_flush): Capture billing objects while session.new still has them
Phase 2 (after_flush_postexec): Process using an INDEPENDENT session to avoid
    "Session is already flushing" errors. The main session is still in its
    flush cycle, so we cannot touch db.session.

This ensures RequestedLab, RequestedImage, PrescribedMedicine charges are
properly synced to the unified billing system with correct encounter_id scoping.
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

# Global list to capture pending charges during after_flush
_pending_charges = []


@event.listens_for(db.session, "after_flush")
def capture_pending_charges(session, flush_context):
    """Phase 1: Capture billing objects before they move to identity map."""
    global _pending_charges
    _pending_charges = []
    
    for instance in list(session.new):
        if isinstance(instance, (RequestedLab, RequestedImage, PrescribedMedicine)):
            _pending_charges.append({
                'type': type(instance).__name__,
                'instance': instance,
            })
    
    if _pending_charges:
        logger.debug(f"Captured {len(_pending_charges)} pending billing charges")


@event.listens_for(db.session, "after_flush_postexec")
def sync_billing_events(session, flush_context):
    """
    Phase 2: Sync billing using an INDEPENDENT session.
    
    We CANNOT use db.session here because the main session is still in its
    flush cycle. Using db.session.add() or db.session.flush() would raise
    "Session is already flushing". Instead, we open a new session bound
    to the same engine and pass it as _session to sync_charge.
    """
    global _pending_charges
    
    if not _pending_charges:
        return
    
    logger.info(f"Processing {len(_pending_charges)} pending billing charges")
    
    # Open an independent session for billing writes
    sync_session = Session(bind=db.engine)
    
    try:
        for charge_data in _pending_charges:
            instance = charge_data['instance']
            charge_type = charge_data['type']
            
            try:
                if charge_type == 'RequestedLab':
                    if hasattr(instance, "id") and instance.id is not None:
                        lab_test = sync_session.get(LabTest, instance.lab_test_id)
                        if lab_test:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table="requested_lab",
                                source_id=instance.id,
                                description=f"Lab Test: {lab_test.test_name}",
                                category="lab",
                                amount=float(lab_test.cost or 0),
                                source_encounter_id=getattr(instance, "encounter_id", None),
                                _session=sync_session,
                            )
                            logger.info(f"Synced lab charge for RequestedLab #{instance.id}")
                
                elif charge_type == 'RequestedImage':
                    if hasattr(instance, "id") and instance.id is not None:
                        from departments.models.medicine import Imaging
                        imaging = sync_session.get(Imaging, instance.imaging_id)
                        if imaging:
                            sync_charge(
                                patient_id=instance.patient_id,
                                source_table="requested_image",
                                source_id=instance.id,
                                description=f"Imaging: {imaging.imaging_type}",
                                category="imaging",
                                amount=float(imaging.cost or 0),
                                source_encounter_id=getattr(instance, "encounter_id", None),
                                _session=sync_session,
                            )
                            logger.info(f"Synced imaging charge for RequestedImage #{instance.id}")
                
                elif charge_type == 'PrescribedMedicine':
                    if hasattr(instance, "id") and instance.id is not None:
                        from departments.models.medicine import Medicine
                        med = sync_session.get(Medicine, instance.medicine_id)
                        med_name = med.generic_name if med else "Medication"
                        sync_charge(
                            patient_id=instance.patient_id,
                            source_table="prescribed_medicine",
                            source_id=instance.id,
                            description=f"Prescription: {med_name}",
                            category="drug",
                            amount=0.0,
                            source_encounter_id=getattr(instance, "encounter_id", None),
                            _session=sync_session,
                        )
                        logger.info(f"Synced prescription charge for PrescribedMedicine #{instance.id}")
            
            except Exception as e:
                logger.error(f"Error syncing billing for {charge_type} #{getattr(instance, 'id', 'unknown')}: {e}")
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
'''

# Backup
backup = listener_file.with_suffix('.py.pre_independent_session')
backup.write_text(listener_file.read_text())
print(f"   📦 Backup: {backup.name}")

# Write
listener_file.write_text(new_content)
print("   ✅ event_listeners.py rewritten with independent session pattern")

# Verify sync.py has _session parameter
sync_file = Path("departments/billing/sync.py")
sync_content = sync_file.read_text()
if "_session=None" in sync_content:
    print("   ✅ sync.py has _session parameter")
else:
    print("   ⚠️ sync.py missing _session parameter — patching now...")
    # Add _session to sync_charge signature
    sync_content = sync_content.replace(
        "    quantity: int = 1,\n) -> InvoiceLineItem:",
        "    quantity: int = 1,\n    _session=None,\n) -> InvoiceLineItem:"
    )
    # Add _session to get_or_create_open_invoice signature
    sync_content = sync_content.replace(
        "def get_or_create_open_invoice(patient_id: str) -> Invoice:",
        "def get_or_create_open_invoice(patient_id: str, _session=None) -> Invoice:"
    )
    # Use sess instead of db.session in get_or_create_open_invoice
    sync_content = sync_content.replace(
        "    # Find the most recent active encounter",
        "    sess = _session if _session is not None else db.session\n\n    # Find the most recent active encounter"
    )
    sync_content = sync_content.replace(
        "    active_encounter = Encounter.query.filter_by(",
        "    active_encounter = sess.query(Encounter).filter_by("
    )
    sync_content = sync_content.replace(
        "    invoice = Invoice.query.filter_by(",
        "    invoice = sess.query(Invoice).filter_by("
    )
    sync_content = sync_content.replace(
        "        db.session.add(invoice)\n        db.session.flush()",
        "        sess.add(invoice)\n        sess.flush()"
    )
    # Use sess in sync_charge
    sync_content = sync_content.replace(
        "    # Get or create the patient's open invoice\n    invoice = get_or_create_open_invoice(patient_id)",
        "    sess = _session if _session is not None else db.session\n    invoice = get_or_create_open_invoice(patient_id, _session=sess)"
    )
    sync_content = sync_content.replace(
        "    db.session.add(line_item)",
        "    sess.add(line_item)"
    )
    sync_file.write_text(sync_content)
    print("   ✅ sync.py patched with _session support")

print("\n✅ Fix complete. Now verify with the diagnostic:")
print("   python3 << 'PYEOF' ... (same diagnostic as before)")
print("\nExpected output:")
print("   after_flush: 1 items: ['RequestedLab']")
print("   after_flush_postexec: 0 items: []")
print("   InvoiceLineItem created: True")
print("   encounter_id: <correct_id>")