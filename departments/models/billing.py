import enum
from datetime import datetime, timezone

from extensions import db  # Use absolute import for db


class ChargeCategory(db.Model):
    """Categories of charges (Consultation, Lab, Surgery, etc.)."""

    __tablename__ = "charge_categories"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)


class Charge(db.Model):
    """Stores hospital charges."""

    __tablename__ = "charges"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category_id = db.Column(
        db.Integer, db.ForeignKey("charge_categories.id"), nullable=False
    )
    cost = db.Column(db.Numeric(10, 2), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime, default=db.func.current_timestamp(), nullable=False
    )

    category = db.relationship(
        "ChargeCategory", backref=db.backref("charges", lazy=True)
    )


class Billing(db.Model):
    """Links patients to charges, tracks payment status."""

    __tablename__ = "billing"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    charge_id = db.Column(db.Integer, db.ForeignKey("charges.id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    total_cost = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(
        db.Integer, nullable=False, default=0
    )  # 0 for Pending, 1 for Paid
    billed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    receipt_number = db.Column(
        db.String(50), nullable=True
    )  # Receipt number for paid bills

    patient = db.relationship("Patient", backref=db.backref("bills", lazy=True))
    charge = db.relationship("Charge", backref=db.backref("billings", lazy=True))

    def calculate_total(self):
        """Calculate total cost based on charge."""
        self.total_cost = self.quantity * self.charge.cost


class DrugsBill(db.Model):
    """Links patients to drugs and tracks payment status."""

    __tablename__ = "drugs_bill"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    drug_id = db.Column(
        db.Integer, db.ForeignKey("drugs.id"), nullable=True
    )  # Nullable for summary drug bills
    quantity = db.Column(db.Integer, nullable=False, default=1)
    total_cost = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(
        db.Integer, nullable=False, default=0
    )  # 0 for Pending, 1 for Paid
    billed_at = db.Column(
        db.DateTime, default=db.func.current_timestamp(), nullable=False
    )
    receipt_number = db.Column(db.String(50), nullable=True)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    # Relationships
    patient = db.relationship("Patient", backref=db.backref("drug_bills", lazy=True))
    drug = db.relationship("Drug", backref=db.backref("bills", lazy=True))

    def calculate_total(self):
        """Calculate total cost based on drug selling price."""
        if self.drug:
            self.total_cost = self.quantity * self.drug.selling_price
        else:
            self.total_cost = 0


class PaidBill(db.Model):
    """Stores details of paid bills."""

    __tablename__ = "paid_bills"

    id = db.Column(db.Integer, primary_key=True)
    receipt_number = db.Column(
        db.String(50), unique=True, nullable=False
    )  # Unique receipt number
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )  # Link to Patient
    grand_total = db.Column(
        db.Numeric(10, 2), nullable=False
    )  # Total cost of all unpaid bills
    amount_paid = db.Column(
        db.Numeric(10, 2), nullable=False
    )  # Amount paid in this transaction
    balance = db.Column(
        db.Numeric(10, 2), nullable=False
    )  # Remaining balance after payment
    paid_at = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False
    )  # Timestamp of payment
    payment_method = db.Column(
        db.String(50), nullable=True
    )  # Payment method (e.g., Cash, M-Pesa)

    # Relationships
    patient = db.relationship("Patient", backref=db.backref("paid_bills", lazy=True))

    @staticmethod
    def generate_receipt_number():
        """Generate a unique receipt number."""
        now = datetime.now(timezone.utc)
        prefix = f"REC-{now.strftime('%Y%m%d')}"
        last_paid_bill = (
            PaidBill.query.filter(PaidBill.receipt_number.like(f"{prefix}%"))
            .order_by(PaidBill.id.desc())
            .first()
        )
        if last_paid_bill:
            try:
                last_number = int(last_paid_bill.receipt_number.split("-")[-1])
            except ValueError:
                last_number = 0
        else:
            last_number = 0
        new_number = last_number + 1
        return f"{prefix}-{new_number:04d}"


class WardBill(db.Model):
    __tablename__ = "ward_bills"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship("Patient", backref=db.backref("ward_bills", lazy=True))

    def __repr__(self):
        return f"<WardBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"


# Lab Bills (for requested_labs)
class LabBill(db.Model):
    __tablename__ = "lab_bills"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship("Patient", backref=db.backref("lab_bills", lazy=True))

    def __repr__(self):
        return f"<LabBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"


# Clinic Bills (for clinic_bookings)
class ClinicBill(db.Model):
    __tablename__ = "clinic_bills"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship("Patient", backref=db.backref("clinic_bills", lazy=True))

    def __repr__(self):
        return f"<ClinicBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"


# Theatre Bills (for theatre_list)
class TheatreBill(db.Model):
    __tablename__ = "theatre_bills"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship("Patient", backref=db.backref("theatre_bills", lazy=True))

    def __repr__(self):
        return f"<TheatreBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"


# Imaging Bills (for requested_images)
class ImagingBill(db.Model):
    __tablename__ = "imaging_bills"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    total_paid = db.Column(db.Numeric(10, 2), nullable=False)
    receipt_number = db.Column(db.String(50), unique=True, nullable=True)
    billed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    payment_method = db.Column(db.String(50), nullable=True)
    payment_reference = db.Column(db.String(255), nullable=True)

    patient = db.relationship("Patient", backref=db.backref("imaging_bills", lazy=True))

    def __repr__(self):
        return f"<ImagingBill(id={self.id}, patient_id={self.patient_id}, total_paid={self.total_paid}, receipt_number={self.receipt_number})>"


# ═══════════════════════════════════════════════════════
# UNIFIED BILLING — Task 2.2
# Invoice / InvoiceLineItem / Payment
# Legacy tables above are kept for zero-breakage backcompat.
# All new billing code should write to these models.
# ═══════════════════════════════════════════════════════


class InvoiceStatus(str, enum.Enum):
    DRAFT = "draft"
    ISSUED = "issued"
    UNPAID = "issued"  # noqa: PIE796
    PARTIAL = "partial"
    PAID = "paid"
    VOID = "void"


class PaymentMethod(str, enum.Enum):
    CASH = "cash"
    MPESA = "mpesa"
    INSURANCE = "insurance"
    BANK = "bank"
    WAIVER = "waiver"
    OTHER = "other"


class Invoice(db.Model):
    """
    Single unified running invoice per patient.

    Operates as an open running balance for a patient until fully settled (status PAID/PARTIAL).
    Subsequent charges after settlement initiate a new open invoice. Invoices are automatically
    scoped to the patient's active Encounter when available, enabling true per-visit billing.
    Replaces the 6 legacy *Bill tables for new billing operations.
    """

    __tablename__ = "invoices"
    facility_id = db.Column(db.Integer, db.ForeignKey("facilities.id"), nullable=True, index=True)

    id = db.Column(db.Integer, primary_key=True)
    invoice_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )

    status = db.Column(
        db.Enum(InvoiceStatus), nullable=False, default=InvoiceStatus.DRAFT
    )
    issued_at = db.Column(db.DateTime, nullable=True)
    due_date = db.Column(db.Date, nullable=True)

    # Totals (denormalised for query performance)
    subtotal = db.Column(db.Numeric(12, 2), nullable=False, default=0)
    discount = db.Column(db.Numeric(12, 2), nullable=False, default=0)
    grand_total = db.Column(db.Numeric(12, 2), nullable=False, default=0)
    amount_paid = db.Column(db.Numeric(12, 2), nullable=False, default=0)
    balance = db.Column(db.Numeric(12, 2), nullable=False, default=0)

    # Insurance
    insurance_scheme_id = db.Column(db.Integer, nullable=True)  # FK added by 2.3
    insurance_claim_ref = db.Column(db.String(100), nullable=True)
    encounter_id = db.Column(db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True)

    # Audit
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    notes = db.Column(db.Text, nullable=True)

    # Legacy source tracking (to link backfilled records)
    legacy_source = db.Column(
        db.String(30), nullable=True
    )  # e.g. 'drugs_bill', 'ward_bill'
    legacy_id = db.Column(db.Integer, nullable=True)

    patient = db.relationship("Patient", backref=db.backref("invoices", lazy="dynamic"))
    line_items = db.relationship(
        "InvoiceLineItem",
        back_populates="invoice",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    payments = db.relationship(
        "Payment",
        back_populates="invoice",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def __init__(self, **kwargs):
        if "total_amount" in kwargs:
            kwargs["grand_total"] = kwargs.pop("total_amount")
        if "paid_amount" in kwargs:
            kwargs["amount_paid"] = kwargs.pop("paid_amount")
        if "balance_due" in kwargs:
            kwargs["balance"] = kwargs.pop("balance_due")
        if "invoice_number" not in kwargs:
            kwargs["invoice_number"] = Invoice.generate_invoice_number()
        if "status" in kwargs and isinstance(kwargs["status"], str):
            st_val = kwargs["status"].upper()
            if st_val == "UNPAID":
                kwargs["status"] = InvoiceStatus.ISSUED
            elif hasattr(InvoiceStatus, st_val):
                kwargs["status"] = getattr(InvoiceStatus, st_val)
            elif kwargs["status"].lower() in [e.value for e in InvoiceStatus]:
                kwargs["status"] = InvoiceStatus(kwargs["status"].lower())
        super().__init__(**kwargs)

    @property
    def total_amount(self):
        return self.grand_total

    @total_amount.setter
    def total_amount(self, value):
        self.grand_total = value

    @property
    def paid_amount(self):
        return self.amount_paid

    @paid_amount.setter
    def paid_amount(self, value):
        self.amount_paid = value

    @property
    def balance_due(self):
        return self.balance

    @balance_due.setter
    def balance_due(self, value):
        self.balance = value

    @staticmethod
    def generate_invoice_number():
        """Generate unique sequential invoice number INV-YYYYMMDD-NNNN-UUID."""
        import uuid

        now = datetime.now(timezone.utc)
        prefix = f"INV-{now.strftime('%Y%m%d')}"
        last = (
            Invoice.query.filter(Invoice.invoice_number.like(f"{prefix}-%"))
            .order_by(Invoice.id.desc())
            .first()
        )
        seq = 1
        if last:
            try:
                parts = last.invoice_number.split("-")
                if len(parts) >= 3:
                    seq = int(parts[2]) + 1
            except (ValueError, IndexError):
                pass
        unique_id = str(uuid.uuid4())[:6].upper()
        return f"{prefix}-{seq:04d}-{unique_id}"

    def recalculate(self):
        """Recompute subtotal, grand_total, amount_paid, balance from child records."""
        sub = sum(float(li.total) for li in self.line_items)
        disc = float(self.discount or 0)
        paid = sum(float(p.amount) for p in self.payments)

        if sub > 0:
            self.subtotal = sub
            self.grand_total = sub - disc
        self.amount_paid = paid
        self.balance = float(self.grand_total or 0) - paid
        if self.balance <= 0:
            self.status = InvoiceStatus.PAID
            # Fully settled: allow the visit to close if clinical work is done.
            if self.encounter_id:
                from departments.shared.visit_closure import maybe_close_encounter

                maybe_close_encounter(self.patient_id)
        elif self.amount_paid > 0:
            self.status = InvoiceStatus.PARTIAL

    def __repr__(self):
        return f"<Invoice {self.invoice_number} [{self.status}]>"


class InvoiceLineItem(db.Model):
    """A single charge line on an invoice (drug, lab, ward, etc.)."""

    __tablename__ = "invoice_line_items"

    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(
        db.Integer, db.ForeignKey("invoices.id"), nullable=False, index=True
    )

    description = db.Column(db.String(255), nullable=False)
    category = db.Column(
        db.String(50), nullable=False
    )  # drug / lab / ward / theatre / imaging / consult / other
    quantity = db.Column(db.Numeric(10, 3), nullable=False, default=1)
    unit_price = db.Column(db.Numeric(12, 2), nullable=False)
    discount = db.Column(db.Numeric(12, 2), nullable=False, default=0)
    total = db.Column(db.Numeric(12, 2), nullable=False)

    # Optional FK back to source domain tables
    charge_id = db.Column(db.Integer, db.ForeignKey("charges.id"), nullable=True)
    encounter_id = db.Column(db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True)

    invoice = db.relationship("Invoice", back_populates="line_items")

    # Source tracking for billing sync (Phase 1)
    source_table = db.Column(db.String(50), nullable=True)  # e.g., 'requested_lab', 'dispensed_drug'
    source_id = db.Column(db.Integer, nullable=True)  # ID in the source table

    __table_args__ = (
        db.UniqueConstraint('source_table', 'source_id', name='uq_invoice_line_item_source'),
    )


    def __init__(self, **kwargs):
        if "total_price" in kwargs:
            kwargs["total"] = kwargs.pop("total_price")
        if "amount" in kwargs:
            val = kwargs.pop("amount")
            kwargs["unit_price"] = val
            kwargs["total"] = val
        if "created_at" in kwargs:
            kwargs.pop("created_at")
        if "category" not in kwargs:
            kwargs["category"] = "other"
        super().__init__(**kwargs)

    @property
    def amount(self):
        return self.total

    @amount.setter
    def amount(self, value):
        self.unit_price = value
        self.total = value

    def calculate_total(self):
        self.total = (self.unit_price * self.quantity) - self.discount

    def __repr__(self):
        return f"<LineItem {self.description} x{self.quantity} = {self.total}>"


class Payment(db.Model):
    """A single payment transaction applied to an invoice (supports partial payments)."""

    __tablename__ = "payments"

    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(
        db.Integer, db.ForeignKey("invoices.id"), nullable=False, index=True
    )
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )

    amount = db.Column(db.Numeric(12, 2), nullable=False)
    method = db.Column(
        db.Enum(PaymentMethod), nullable=False, default=PaymentMethod.CASH
    )
    reference = db.Column(
        db.String(100), nullable=True
    )  # M-Pesa receipt, bank ref, etc.
    receipt_number = db.Column(db.String(30), unique=True, nullable=True)

    paid_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    recorded_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    notes = db.Column(db.Text, nullable=True)

    # M-Pesa specific (populated by Task 2.4 Daraja integration)
    mpesa_checkout_id = db.Column(db.String(100), nullable=True)
    mpesa_result_code = db.Column(db.Integer, nullable=True)

    _is_reconciled = db.Column(
        "is_reconciled", db.Boolean, default=False, nullable=True
    )

    payment_method = db.synonym("method")
    payment_reference = db.synonym("reference")
    is_reconciled = db.synonym("_is_reconciled")

    invoice = db.relationship("Invoice", back_populates="payments")
    patient = db.relationship(
        "Patient", backref=db.backref("unified_payments", lazy="dynamic")
    )

    def __init__(self, **kwargs):
        if "payment_method" in kwargs:
            kwargs["method"] = kwargs.pop("payment_method")
        if "payment_reference" in kwargs:
            kwargs["reference"] = kwargs.pop("payment_reference")
        if "is_reconciled" in kwargs:
            kwargs["_is_reconciled"] = kwargs.pop("is_reconciled")
        if "method" in kwargs and isinstance(kwargs["method"], str):
            m_val = kwargs["method"].upper()
            if hasattr(PaymentMethod, m_val):
                kwargs["method"] = getattr(PaymentMethod, m_val)
            elif kwargs["method"].lower() in [e.value for e in PaymentMethod]:
                kwargs["method"] = PaymentMethod(kwargs["method"].lower())
        if (
            "invoice_id" in kwargs
            and "patient_id" not in kwargs
            and kwargs["invoice_id"]
        ):
            inv = Invoice.query.get(kwargs["invoice_id"])
            if inv:
                kwargs["patient_id"] = inv.patient_id
        super().__init__(**kwargs)

    def __repr__(self):
        return f"<Payment {self.receipt_number} {self.amount} via {self.method}>"
