"""
tests/test_unified_billing.py
─────────────────────────────
Unit tests for Task 2.2: Unified Invoice / InvoiceLineItem / Payment models.
Legacy billing tables are not tested here — they are already covered by
tests/test_billing.py.
"""
from datetime import date
from decimal import Decimal

from departments.models.billing import (
    Invoice,
    InvoiceLineItem,
    InvoiceStatus,
    Payment,
    PaymentMethod,
)
from departments.models.records import Patient
from extensions import db

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_patient(pid):
    p = Patient(
        patient_id=pid,
        name=f"Test Patient {pid}",
        place_of_residence="Nairobi",
        sex="Male",
        date_of_birth=date(1990, 1, 1),
        marital_status="Single",
        contact=f"07000{pid[-4:]}",
        next_of_kin="Kin",
        relationship_with_next_of_kin="Parent",
        next_of_kin_contact="0711111111",
        national_id=None,
        emergency_contact="0722222222",
    )
    db.session.add(p)
    db.session.commit()
    return p


def _make_invoice(patient_id, discount=0):
    inv = Invoice(
        invoice_number=Invoice.generate_invoice_number(),
        patient_id=patient_id,
        status=InvoiceStatus.DRAFT,
        discount=Decimal(str(discount)),
        subtotal=Decimal("0"),
        grand_total=Decimal("0"),
        amount_paid=Decimal("0"),
        balance=Decimal("0"),
    )
    db.session.add(inv)
    db.session.commit()
    return inv


def _add_line(invoice, description, category, unit_price, quantity=1, discount=0):
    total = Decimal(str(unit_price)) * Decimal(str(quantity)) - Decimal(str(discount))
    li = InvoiceLineItem(
        invoice_id=invoice.id,
        description=description,
        category=category,
        unit_price=Decimal(str(unit_price)),
        quantity=Decimal(str(quantity)),
        discount=Decimal(str(discount)),
        total=total,
    )
    db.session.add(li)
    db.session.commit()
    return li


# ─────────────────────────────────────────────
# 1. Invoice creation & number generation
# ─────────────────────────────────────────────

class TestInvoiceCreation:
    def test_create_invoice_defaults(self, app):
        with app.app_context():
            _make_patient("INV001")
            inv = _make_invoice("INV001")
            assert inv.id is not None
            assert inv.status == InvoiceStatus.DRAFT
            assert inv.invoice_number.startswith("INV-")

    def test_invoice_number_sequential(self, app):
        """Two invoices created in same context get different sequential numbers."""
        with app.app_context():
            _make_patient("INV002")
            n1 = Invoice.generate_invoice_number()
            inv1 = Invoice(
                invoice_number=n1,
                patient_id="INV002",
                status=InvoiceStatus.DRAFT,
                subtotal=0, discount=0, grand_total=0, amount_paid=0, balance=0,
            )
            db.session.add(inv1)
            db.session.commit()
            n2 = Invoice.generate_invoice_number()
            assert n1 != n2

    def test_invoice_links_to_patient(self, app):
        with app.app_context():
            _make_patient("INV003")
            inv = _make_invoice("INV003")
            loaded = db.session.get(Invoice, inv.id)
            assert loaded.patient_id == "INV003"


# ─────────────────────────────────────────────
# 2. InvoiceLineItem
# ─────────────────────────────────────────────

class TestInvoiceLineItem:
    def test_add_single_line_item(self, app):
        with app.app_context():
            _make_patient("LI001")
            inv = _make_invoice("LI001")
            li = _add_line(inv, "Consultation", "consult", 500)
            assert li.id is not None
            assert li.total == Decimal("500")

    def test_add_multiple_line_items(self, app):
        with app.app_context():
            _make_patient("LI002")
            inv = _make_invoice("LI002")
            _add_line(inv, "Paracetamol 500mg x10", "drug", 15, quantity=10)
            _add_line(inv, "FBC Test", "lab", 300)
            items = InvoiceLineItem.query.filter_by(invoice_id=inv.id).all()
            assert len(items) == 2

    def test_line_item_total_with_discount(self, app):
        with app.app_context():
            _make_patient("LI003")
            inv = _make_invoice("LI003")
            li = _add_line(inv, "Theatre Charge", "theatre", 10000, discount=500)
            assert li.total == Decimal("9500")

    def test_calculate_total_method(self, app):
        with app.app_context():
            _make_patient("LI004")
            inv = _make_invoice("LI004")
            li = InvoiceLineItem(
                invoice_id=inv.id,
                description="Ward Charge",
                category="ward",
                unit_price=Decimal("2000"),
                quantity=Decimal("3"),
                discount=Decimal("100"),
                total=Decimal("0"),
            )
            li.calculate_total()
            assert li.total == Decimal("5900")


# ─────────────────────────────────────────────
# 3. Payment
# ─────────────────────────────────────────────

class TestPayment:
    def test_create_cash_payment(self, app):
        with app.app_context():
            _make_patient("PAY001")
            inv = _make_invoice("PAY001")
            pmt = Payment(
                invoice_id=inv.id,
                patient_id="PAY001",
                amount=Decimal("500"),
                method=PaymentMethod.CASH,
                receipt_number="REC-001",
            )
            db.session.add(pmt)
            db.session.commit()
            saved = Payment.query.filter_by(receipt_number="REC-001").first()
            assert saved is not None
            assert saved.method == PaymentMethod.CASH

    def test_create_mpesa_payment_with_reference(self, app):
        with app.app_context():
            _make_patient("PAY002")
            inv = _make_invoice("PAY002")
            pmt = Payment(
                invoice_id=inv.id,
                patient_id="PAY002",
                amount=Decimal("1500"),
                method=PaymentMethod.MPESA,
                reference="QJN8A1234X",
                receipt_number="REC-MPE-001",
                mpesa_checkout_id="ws_CO_12345",
            )
            db.session.add(pmt)
            db.session.commit()
            saved = Payment.query.filter_by(receipt_number="REC-MPE-001").first()
            assert saved.reference == "QJN8A1234X"
            assert saved.mpesa_checkout_id == "ws_CO_12345"

    def test_multiple_partial_payments(self, app):
        with app.app_context():
            _make_patient("PAY003")
            inv = _make_invoice("PAY003")
            for i, amt in enumerate([500, 300, 200], start=1):
                db.session.add(Payment(
                    invoice_id=inv.id,
                    patient_id="PAY003",
                    amount=Decimal(str(amt)),
                    method=PaymentMethod.CASH,
                    receipt_number=f"REC-PART-{i:03d}",
                ))
            db.session.commit()
            payments = Payment.query.filter_by(invoice_id=inv.id).all()
            assert len(payments) == 3
            assert sum(p.amount for p in payments) == Decimal("1000")


# ─────────────────────────────────────────────
# 4. Invoice.recalculate()
# ─────────────────────────────────────────────

class TestInvoiceRecalculate:
    def test_recalculate_sets_totals(self, app):
        with app.app_context():
            _make_patient("CALC01")
            inv = _make_invoice("CALC01", discount=0)
            _add_line(inv, "Consultation", "consult", 500)
            _add_line(inv, "Lab FBC", "lab", 300)
            inv.recalculate()
            db.session.commit()
            assert inv.subtotal == Decimal("800")
            assert inv.grand_total == Decimal("800")
            assert inv.amount_paid == Decimal("0")
            assert inv.balance == Decimal("800")

    def test_recalculate_status_paid(self, app):
        with app.app_context():
            _make_patient("CALC02")
            inv = _make_invoice("CALC02")
            _add_line(inv, "Consultation", "consult", 500)
            db.session.add(Payment(
                invoice_id=inv.id,
                patient_id="CALC02",
                amount=Decimal("500"),
                method=PaymentMethod.CASH,
                receipt_number="REC-CALC-001",
            ))
            db.session.commit()
            inv.recalculate()
            assert inv.status == InvoiceStatus.PAID
            assert inv.balance <= 0

    def test_recalculate_status_partial(self, app):
        with app.app_context():
            _make_patient("CALC03")
            inv = _make_invoice("CALC03")
            _add_line(inv, "Ward charge", "ward", 2000)
            db.session.add(Payment(
                invoice_id=inv.id,
                patient_id="CALC03",
                amount=Decimal("1000"),
                method=PaymentMethod.MPESA,
                receipt_number="REC-CALC-002",
            ))
            db.session.commit()
            inv.recalculate()
            assert inv.status == InvoiceStatus.PARTIAL
            assert inv.balance == Decimal("1000")

    def test_recalculate_with_invoice_discount(self, app):
        with app.app_context():
            _make_patient("CALC04")
            inv = _make_invoice("CALC04", discount=100)
            _add_line(inv, "Theatre", "theatre", 1000)
            inv.recalculate()
            assert inv.grand_total == Decimal("900")
            assert inv.balance == Decimal("900")


# ─────────────────────────────────────────────
# 5. Legacy source tracking
# ─────────────────────────────────────────────

class TestLegacySourceTracking:
    def test_legacy_source_fields_stored(self, app):
        """Backfilled records should carry legacy_source and legacy_id."""
        with app.app_context():
            _make_patient("LEG001")
            inv = Invoice(
                invoice_number=Invoice.generate_invoice_number(),
                patient_id="LEG001",
                status=InvoiceStatus.PAID,
                subtotal=Decimal("500"),
                discount=Decimal("0"),
                grand_total=Decimal("500"),
                amount_paid=Decimal("500"),
                balance=Decimal("0"),
                legacy_source="drugs_bill",
                legacy_id=42,
            )
            db.session.add(inv)
            db.session.commit()
            saved = Invoice.query.filter_by(legacy_source="drugs_bill", legacy_id=42).first()
            assert saved is not None
            assert saved.grand_total == Decimal("500")
