"""
Billing sync utilities - bridges legacy per-department billing with unified Invoice system.

This module provides the sync layer that ensures charges created through the legacy
billing workflow (DrugsBill, LabBill, etc.) are also reflected in the unified Invoice
system used by the patient portal, M-Pesa, and insurance claims.
"""

import logging
from datetime import datetime

from flask import current_app

from departments.models.billing import Invoice, InvoiceLineItem, InvoiceStatus, Payment
from departments.models.encounter import Encounter
from extensions import db

logger = logging.getLogger(__name__)


def get_or_create_open_invoice(patient_id: str) -> Invoice:
    """
    Get the patient's current open invoice, or create one if none exists.

    An "open" invoice is one with status DRAFT or ISSUED that can still accept
    new line items. Once an invoice is marked PAID or PARTIAL, it's closed and
    a new one will be created for subsequent charges.

    Args:
        patient_id: The patient's business key (e.g., "P0001")

    Returns:
        Invoice: The patient's current open invoice
    """
    # Find the most recent active encounter for this patient
    active_encounter = Encounter.query.filter_by(
        patient_id=patient_id, status="ACTIVE"
    ).order_by(Encounter.started_at.desc()).first()
    enc_id = active_encounter.id if active_encounter else None

    # Check for existing open invoice scoped to this encounter (or unscoped if no encounter)
    invoice = Invoice.query.filter_by(
        patient_id=patient_id,
        status=InvoiceStatus.DRAFT,
        encounter_id=enc_id
    ).first()

    if not invoice:
        # Create new invoice scoped to the encounter
        invoice = Invoice(
            patient_id=patient_id,
            encounter_id=enc_id,
            status=InvoiceStatus.DRAFT,
            grand_total=0.0,
            amount_paid=0.0,
            created_at=datetime.utcnow(),
        )
        db.session.add(invoice)
        db.session.flush()  # Get the ID without committing
        logger.info(f"Created new invoice {invoice.id} for patient {patient_id} (Encounter: {enc_id})")

    return invoice


# KNOWN LIMITATION (AUDIT FINDING - PRIORITY 5):
# sync_charge() is idempotent for insertions (if existing: return existing), which prevents duplicate line items.
# However, post-facto updates (e.g. modifying price/quantity on an existing legacy bill) or deletions of legacy
# charges will NOT automatically sync or update the InvoiceLineItem. Deleted legacy bills leave orphaned line items.
# See DECISIONS_PENDING.md Section 10 for proposed product options (retroactive invoice adjustments vs credit lines).
def sync_charge(
    patient_id: str,
    source_table: str,
    source_id: int,
    description: str,
    category: str,
    amount: float,
    quantity: int = 1,
) -> InvoiceLineItem:
    """
    Sync a charge from legacy billing to the unified Invoice system.

    This is idempotent - calling it multiple times with the same source_table/source_id
    will return the existing line item without creating duplicates.

    Args:
        patient_id: Patient's business key
        source_table: Name of the source table (e.g., 'requested_lab', 'dispensed_drug')
        source_id: ID in the source table
        description: Human-readable description of the charge
        category: Charge category (drug, lab, imaging, theatre, ward, consult)
        amount: Unit price
        quantity: Number of units (default 1)

    Returns:
        InvoiceLineItem: The created or existing line item
    """
    # Check if sync is enabled
    if not current_app.config.get("BILLING_SYNC_ENABLED", True):
        logger.debug(
            f"Billing sync disabled, skipping charge sync for {source_table}:{source_id}"
        )
        return None

    # Check for existing line item (idempotency)
    existing = InvoiceLineItem.query.filter_by(
        source_table=source_table, source_id=source_id
    ).first()

    if existing:
        logger.debug(f"Line item already exists for {source_table}:{source_id}")
        return existing

    # Get or create the patient's open invoice
    invoice = get_or_create_open_invoice(patient_id)

    # Create the line item (inherit encounter_id from the invoice)
    line_item = InvoiceLineItem(
        invoice_id=invoice.id,
        encounter_id=invoice.encounter_id,
        description=description,
        category=category,
        quantity=quantity,
        unit_price=amount,
        total=amount * quantity,
        source_table=source_table,
        source_id=source_id,
    )

    db.session.add(line_item)

    # Update invoice total
    current_gt = float(invoice.grand_total or 0)
    invoice.grand_total = current_gt + (float(amount) * float(quantity))
    invoice.balance = invoice.grand_total - float(invoice.amount_paid or 0)

    logger.info(
        f"Synced charge: {description} (${amount}x{quantity}) to invoice {invoice.id}"
    )

    return line_item


def sync_payment(
    patient_id: str,
    amount: float,
    payment_method: str,
    reference_number: str = None,
    receipt_number: str = None,
) -> Payment:
    """
    Record a payment against the patient's open invoice.

    This should be called whenever a payment is recorded in the legacy system
    (via pay_bills, pay_all, etc.) to keep the unified Invoice system in sync.

    Args:
        patient_id: Patient's business key
        amount: Amount paid
        payment_method: Payment method (cash, mpesa, insurance, etc.)
        reference_number: External reference (e.g., M-Pesa transaction ID)
        receipt_number: Internal receipt number

    Returns:
        Payment: The created payment record
    """
    if not current_app.config.get("BILLING_SYNC_ENABLED", True):
        logger.debug("Billing sync disabled, skipping payment sync")
        return None

    if amount is None or float(amount) <= 0:
        return None

    # Idempotency checks
    if receipt_number:
        existing = Payment.query.filter_by(receipt_number=receipt_number).first()
        if existing:
            logger.debug(f"Payment already synced for receipt {receipt_number}")
            return existing
    if reference_number:
        existing = Payment.query.filter_by(reference=reference_number).first()
        if existing:
            logger.debug(f"Payment already synced for reference {reference_number}")
            return existing

    # Get or create the patient's open invoice
    invoice = Invoice.query.filter_by(patient_id=patient_id, status=InvoiceStatus.DRAFT).first()
    if not invoice:
        invoice = get_or_create_open_invoice(patient_id)

    # Create the payment
    payment = Payment(
        invoice_id=invoice.id,
        amount=amount,
        payment_method=payment_method,
        reference_number=reference_number,
        receipt_number=receipt_number,
        payment_date=datetime.utcnow(),
    )

    db.session.add(payment)

    # Update invoice
    current_paid = float(invoice.amount_paid or 0)
    invoice.amount_paid = current_paid + float(amount)
    current_gt = float(invoice.grand_total or 0)
    invoice.balance = current_gt - invoice.amount_paid

    # Recalculate status
    if invoice.balance <= 0:
        invoice.status = InvoiceStatus.PAID
        invoice.paid_at = datetime.utcnow()
    elif invoice.amount_paid > 0:
        invoice.status = InvoiceStatus.PARTIAL

    logger.info(
        f"Synced payment: ${amount} via {payment_method} to invoice {invoice.id}"
    )

    return payment


def check_billing_sync_enabled() -> bool:
    """Check if billing sync is currently enabled."""
    return current_app.config.get("BILLING_SYNC_ENABLED", True)


def sync_invoice_status(invoice: Invoice) -> InvoiceStatus:
    """
    Recalculate and update the status of an Invoice based on total, paid amount, and balance.
    """
    grand_total = float(getattr(invoice, "grand_total", None) or getattr(invoice, "total_amount", 0) or 0)
    amount_paid = float(getattr(invoice, "amount_paid", None) or getattr(invoice, "paid_amount", 0) or 0)
    invoice.balance = grand_total - amount_paid

    if amount_paid >= grand_total and grand_total > 0:
        invoice.status = InvoiceStatus.PAID
    elif amount_paid > 0:
        invoice.status = InvoiceStatus.PARTIAL
    else:
        invoice.status = InvoiceStatus.DRAFT
    db.session.commit()
    return invoice.status

