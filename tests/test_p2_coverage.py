"""
Priority 2 (P2) Edge Case & High-Value Module Coverage Suite.

Tests billing sync edge cases.
"""

from departments.billing.sync import sync_invoice_status
from departments.models.billing import Invoice, InvoiceStatus
from extensions import db

# ==============================================================================
# Billing Sync Edge Cases
# ==============================================================================

def test_sync_invoice_status_lifecycle(client):
    """Test sync_invoice_status handles status transitions correctly."""
    inv = Invoice(patient_id=1, total_amount=100.00, paid_amount=0.00, status=InvoiceStatus.DRAFT)
    db.session.add(inv)
    db.session.commit()

    # Total 100, Paid 0 -> UNPAID (or DRAFT if zero paid)
    sync_invoice_status(inv)
    assert inv.status in (InvoiceStatus.DRAFT, InvoiceStatus.UNPAID)

    # Partial payment -> PARTIAL
    inv.paid_amount = 40.00
    sync_invoice_status(inv)
    assert inv.status == InvoiceStatus.PARTIAL

    # Full payment -> PAID
    inv.paid_amount = 100.00
    sync_invoice_status(inv)
    assert inv.status == InvoiceStatus.PAID
