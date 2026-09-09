# Payment-state helper for the Phase-1 pharmacy dispensing gate.
from departments.models.billing import Billing, DrugsBill, Invoice, InvoiceStatus


def unpaid_charge_count(patient_id: str) -> int:
    count = Billing.query.filter_by(patient_id=patient_id, status=0).count()
    count += DrugsBill.query.filter_by(patient_id=patient_id, status=0).count()
    count += Invoice.query.filter(
        Invoice.patient_id == patient_id,
        Invoice.status.in_([InvoiceStatus.DRAFT, InvoiceStatus.ISSUED]),
        Invoice.balance > 0,
    ).count()
    return count


def has_unpaid_charges(patient_id: str) -> bool:
    return unpaid_charge_count(patient_id) > 0
