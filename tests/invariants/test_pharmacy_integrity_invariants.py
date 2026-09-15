"""
tests/invariants/test_pharmacy_integrity_invariants.py
───────────────────────────────────────────────────────
Invariants for pharmacy stock and dispensing integrity.

These tests encode properties that must NEVER be violated:
  - Stock quantity can never go below zero
  - Dispensing more than available stock must fail loudly, not silently
  - FEFO allocation never double-counts the same batch units
  - A drug with zero stock cannot be dispensed
"""

from decimal import Decimal

import pytest

# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def pharmacy_seed(app):
    """Seed one drug category, one drug and two batches for tests."""
    from departments.models.pharmacy import Batch, Drug, DrugCategory
    from extensions import db

    with app.app_context():
        cat = DrugCategory(name="Analgesics")
        db.session.add(cat)
        db.session.flush()

        drug = Drug(
            generic_name="Paracetamol",
            brand_name="Panadol",
            category_id=cat.id,
            dosage_form="Tablet",
            strength="500mg",
            buying_price=Decimal("1.00"),
            selling_price=Decimal("2.00"),
            quantity_in_stock=20,
        )
        db.session.add(drug)
        db.session.flush()

        from datetime import date, timedelta

        batch_a = Batch(
            drug_id=drug.id,
            batch_number="BATCH-A",
            expiry_date=date.today() + timedelta(days=90),
            quantity_in_stock=12,
        )
        batch_b = Batch(
            drug_id=drug.id,
            batch_number="BATCH-B",
            expiry_date=date.today() + timedelta(days=180),
            quantity_in_stock=8,
        )
        db.session.add_all([batch_a, batch_b])
        db.session.commit()

        yield {"drug_id": drug.id, "batch_a_id": batch_a.id, "batch_b_id": batch_b.id}


# ─── invariant tests ────────────────────────────────────────────────────────

class TestStockNeverNegative:
    """
    INVARIANT: drug.quantity_in_stock and batch.quantity_in_stock must
    never be stored as a negative number.
    """

    def test_dispensing_exact_available_stock_succeeds(self, app, pharmacy_seed):
        from departments.models.pharmacy import Drug
        from departments.pharmacy.fefo import dispense_medication_fefo
        from extensions import db

        with app.app_context():
            drug_id = pharmacy_seed["drug_id"]
            records = dispense_medication_fefo("P-TEST", drug_id, 20)
            assert len(records) > 0

            drug = db.session.get(Drug, drug_id)
            assert drug.quantity_in_stock == 0, (
                "After dispensing all stock, quantity_in_stock must be exactly 0, not negative."
            )

    def test_overdispensing_raises_error(self, app, pharmacy_seed):
        """Attempting to dispense more than available must raise ValueError."""
        from departments.pharmacy.fefo import dispense_medication_fefo

        with app.app_context():
            with pytest.raises(ValueError, match="Insufficient stock"):
                dispense_medication_fefo("P-TEST", pharmacy_seed["drug_id"], 999)

    def test_batch_stock_non_negative_after_fefo(self, app, pharmacy_seed):
        """No batch.quantity_in_stock should ever drop below 0 after FEFO dispensing."""
        from departments.models.pharmacy import Batch
        from departments.pharmacy.fefo import dispense_medication_fefo

        with app.app_context():
            dispense_medication_fefo("P-TEST", pharmacy_seed["drug_id"], 15)

            batches = Batch.query.filter_by(drug_id=pharmacy_seed["drug_id"]).all()
            for batch in batches:
                assert batch.quantity_in_stock >= 0, (
                    f"INVARIANT VIOLATION: batch {batch.batch_number} has "
                    f"quantity_in_stock={batch.quantity_in_stock} (negative stock)."
                )

    def test_zero_stock_drug_cannot_be_dispensed(self, app, pharmacy_seed):
        """Dispensing from a drug with 0 stock must fail, never silently succeed."""
        from departments.models.pharmacy import Batch
        from departments.pharmacy.fefo import dispense_medication_fefo
        from extensions import db

        with app.app_context():
            # Zero out all batches
            Batch.query.filter_by(drug_id=pharmacy_seed["drug_id"]).update(
                {"quantity_in_stock": 0}
            )
            db.session.commit()

            with pytest.raises(ValueError):
                dispense_medication_fefo("P-TEST", pharmacy_seed["drug_id"], 1)


class TestFEFOAllocationIntegrity:
    """
    INVARIANT: FEFO allocation must never allocate more units than exist
    across all batches, and must prioritize earliest-expiring batches.
    """

    def test_fefo_allocates_earliest_expiry_first(self, app, pharmacy_seed):
        """FEFO must allocate from BATCH-A (shorter expiry) before BATCH-B."""
        from departments.pharmacy.fefo import allocate_drug_fefo

        with app.app_context():
            allocations = allocate_drug_fefo(pharmacy_seed["drug_id"], 5)
            assert len(allocations) >= 1
            # First allocation must come from batch_a (earliest expiry)
            assert allocations[0]["batch_id"] == pharmacy_seed["batch_a_id"], (
                "FEFO violation: allocation did not start with the earliest-expiring batch."
            )

    def test_fefo_total_allocated_equals_requested(self, app, pharmacy_seed):
        """Total allocated quantity must exactly equal the requested quantity."""
        from departments.pharmacy.fefo import allocate_drug_fefo

        with app.app_context():
            requested = 15
            allocations = allocate_drug_fefo(pharmacy_seed["drug_id"], requested)
            total_allocated = sum(a["allocated_quantity"] for a in allocations)
            assert total_allocated == requested, (
                f"INVARIANT VIOLATION: requested {requested} units but "
                f"FEFO allocated {total_allocated}."
            )

    def test_fefo_no_batch_allocated_more_than_its_stock(self, app, pharmacy_seed):
        """No single batch allocation can exceed that batch's available stock."""
        from departments.models.pharmacy import Batch
        from departments.pharmacy.fefo import allocate_drug_fefo

        with app.app_context():
            batches = {b.id: b.quantity_in_stock for b in Batch.query.filter_by(
                drug_id=pharmacy_seed["drug_id"]
            ).all()}

            allocations = allocate_drug_fefo(pharmacy_seed["drug_id"], 18)
            for alloc in allocations:
                available = batches[alloc["batch_id"]]
                assert alloc["allocated_quantity"] <= available, (
                    f"INVARIANT VIOLATION: batch {alloc['batch_id']} has {available} units "
                    f"but FEFO tried to allocate {alloc['allocated_quantity']}."
                )


class TestFinancialFieldPrecision:
    """
    INVARIANT: financial fields on Drug must be stored as Numeric (not Float)
    and must preserve decimal precision exactly.
    """

    def test_drug_prices_stored_as_numeric(self, app):
        """Drug.buying_price and selling_price must use Numeric, not Float."""
        from sqlalchemy import Numeric as NumericType

        from departments.models.pharmacy import Drug

        with app.app_context():
            buying_col = Drug.__table__.c.buying_price
            selling_col = Drug.__table__.c.selling_price
            assert isinstance(buying_col.type, NumericType), (
                "Drug.buying_price must be Numeric, not Float. "
                "Float arithmetic produces rounding errors in financial calculations."
            )
            assert isinstance(selling_col.type, NumericType), (
                "Drug.selling_price must be Numeric, not Float."
            )

    def test_decimal_price_roundtrip_exact(self, app, pharmacy_seed):
        """A price like 1.15 must be stored and retrieved exactly, not as 1.1499999..."""
        from decimal import Decimal

        from departments.models.pharmacy import Drug
        from extensions import db

        with app.app_context():
            drug = db.session.get(Drug, pharmacy_seed["drug_id"])
            drug.selling_price = Decimal("1.15")
            db.session.commit()

            db.session.expire(drug)
            drug = db.session.get(Drug, pharmacy_seed["drug_id"])
            assert drug.selling_price == Decimal("1.15"), (
                f"INVARIANT VIOLATION: stored Decimal('1.15') but retrieved "
                f"{drug.selling_price!r} — precision lost (Float type suspected)."
            )
