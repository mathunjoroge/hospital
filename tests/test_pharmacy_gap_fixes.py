"""
tests/test_pharmacy_gap_fixes.py
────────────────────────────────
Regression tests for the pharmacy gap-analysis fixes:

  - P0-1: previously-unprotected routes now 403 for non-pharmacy roles.
  - P0-2: drug-request lifecycle — items go to the PENDING cart, and a
    Submitted cart is never appended to again.
  - P0-3: record_purchase writes RECEIVED ledger rows and keeps
    Drug.quantity_in_stock in sync with batch stock.
  - P0-5: voiding a dispensed (and already-billed) drug creates a legacy
    DrugsBill credit row.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def pharmacy_user(app):
    """A pharmacy-role user for request-context auth."""
    from departments.models.user import User
    from extensions import db

    with app.app_context():
        user = User.query.filter_by(username="pharm_gapfix").first()
        if not user:
            from werkzeug.security import generate_password_hash

            user = User(
                username="pharm_gapfix",
                password=generate_password_hash("pharm123", method="pbkdf2:sha256"),
                role="pharmacy",
            )
            db.session.add(user)
            db.session.commit()
        yield user


@pytest.fixture
def logged_in_pharmacist(client, pharmacy_user):
    client.post(
        "/login",
        data={"username": "pharm_gapfix", "password": "pharm123"},
        follow_redirects=True,
    )
    return client


def _make_drug(app, name, qty=0, reorder=10, with_batch=True):
    """Create a drug (and optional batch). Returns drug_id (int)."""
    from departments.models.pharmacy import Batch, Drug, DrugCategory
    from extensions import db

    with app.app_context():
        cat = DrugCategory.query.filter_by(name="GapFix").first()
        if not cat:
            cat = DrugCategory(name="GapFix")
            db.session.add(cat)
            db.session.flush()

        drug = Drug(
            generic_name=name,
            brand_name=f"{name}Brand",
            category_id=cat.id,
            dosage_form="Tablet",
            strength="10mg",
            buying_price=Decimal("1.00"),
            selling_price=Decimal("2.00"),
            quantity_in_stock=qty,
            reorder_level=reorder,
        )
        db.session.add(drug)
        db.session.flush()

        batch_id = None
        if with_batch and qty > 0:
            batch = Batch(
                drug_id=drug.id,
                batch_number=f"B-{name}",
                quantity_in_stock=qty,
            )
            db.session.add(batch)
            db.session.flush()
            batch_id = batch.id

        db.session.commit()
        return drug.id, batch_id


def _make_dispensed(app, drug_id, batch_id, qty=4, status="1", receipt=None):
    from departments.models.pharmacy import DispensedDrug
    from extensions import db

    with app.app_context():
        dispensed = DispensedDrug(
            drug_id=drug_id,
            batch_id=batch_id,
            patient_id="P-GAPFIX-01",
            prescription_id="RX-GAPFIX-1",
            quantity_dispensed=qty,
            date_dispensed=datetime.now(timezone.utc),
            status=status,
            receipt_number=receipt,
        )
        db.session.add(dispensed)
        db.session.commit()
        return dispensed.id


# ─── P0-2: drug-request lifecycle ───────────────────────────────────────────


class TestDrugRequestLifecycle:
    def test_submitted_cart_not_reused(self, app, logged_in_pharmacist):
        """
        After save_order marks the cart Submitted, adding a new item must
        create a FRESH Pending request — not silently append to the one
        already sent to the store.
        """
        client = logged_in_pharmacist
        drug_id, _ = _make_drug(app, "LifecycleDrug", qty=0, with_batch=False)

        # 1. Add an item -> creates a Pending cart
        resp = client.post(
            "/pharmacy/drug-requests",
            data={"drug_id": drug_id, "quantity": "5"},
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            from departments.models.pharmacy import DrugRequest
            from departments.models.user import User

            user = User.query.filter_by(username="pharm_gapfix").first()
            carts = DrugRequest.query.filter_by(requested_by=user.id).all()
            assert len(carts) == 1
            assert carts[0].status == "Pending"
            cart_id = carts[0].id

        # 2. Submit the order
        resp = client.post("/pharmacy/save-order", follow_redirects=True)
        assert resp.status_code == 200

        with app.app_context():
            from departments.models.pharmacy import DrugRequest
            from extensions import db

            submitted = db.session.get(DrugRequest, cart_id)
            assert submitted.status == "Submitted"

        # 3. Add another item after submission -> must create a NEW Pending cart
        resp = client.post(
            "/pharmacy/drug-requests",
            data={"drug_id": drug_id, "quantity": "3"},
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            from departments.models.pharmacy import DrugRequest, RequestItem
            from departments.models.user import User

            user = User.query.filter_by(username="pharm_gapfix").first()
            carts = DrugRequest.query.filter_by(requested_by=user.id).all()
            assert (
                len(carts) == 2
            ), "A fresh Pending cart must be created after submission."

            new_cart = next(c for c in carts if c.id != cart_id)
            assert new_cart.status == "Pending"
            item = RequestItem.query.filter_by(request_id=new_cart.id).first()
            assert item is not None
            assert item.quantity_requested == 3

            # The submitted cart must be untouched
            old_items = RequestItem.query.filter_by(request_id=cart_id).all()
            assert len(old_items) == 1
            assert old_items[0].quantity_requested == 5

    def test_save_order_is_post_only(self, client, logged_in_pharmacist):
        """save_order mutates state — GET must now be rejected (405)."""
        resp = client.get("/pharmacy/save-order")
        assert resp.status_code == 405


# ─── P0-3: record_purchase ledger + drug-level sync ────────────────────────


class TestRecordPurchaseLedger:
    def test_purchase_writes_received_ledger_and_syncs_drug_stock(
        self, app, logged_in_pharmacist
    ):
        from departments.models.pharmacy import Batch, Drug
        from departments.models.stock_movement import StockMovement
        from extensions import db

        client = logged_in_pharmacist
        drug_id, _ = _make_drug(app, "LedgerDrug", qty=20)

        resp = client.post(
            "/pharmacy/record_purchase",
            data={
                "drug_ids[]": str(drug_id),
                "batch_numbers[]": "B-NEW-1",
                "quantities[]": "30",
                "unit_costs[]": "1.50",
                "expiry_dates[]": "2028-01-31",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            drug = db.session.get(Drug, drug_id)
            assert drug.quantity_in_stock == 50, "Drug-level cache must sync (20 + 30)."

            batch = Batch.query.filter_by(
                drug_id=drug_id, batch_number="B-NEW-1"
            ).first()
            assert batch is not None
            assert batch.quantity_in_stock == 30
            assert batch.expiry_date is not None, "New batch must capture expiry date."

            mv = StockMovement.query.filter_by(
                item_type="DRUG",
                item_id=drug_id,
                movement_type="RECEIVED",
                reference_type="DIRECT_PURCHASE",
            ).first()
            assert (
                mv is not None
            ), "record_purchase must append a RECEIVED ledger row."
            assert mv.quantity_delta == 30
            assert mv.balance_after == 50

    def test_new_batch_without_expiry_rejected(self, app, logged_in_pharmacist):
        """New batches require an expiry date — no more NULL-expiry black holes."""
        from departments.models.pharmacy import Batch

        client = logged_in_pharmacist
        drug_id, _ = _make_drug(app, "NoExpiryDrug", qty=10)

        resp = client.post(
            "/pharmacy/record_purchase",
            data={
                "drug_ids[]": str(drug_id),
                "batch_numbers[]": "B-NOEXP",
                "quantities[]": "5",
                "unit_costs[]": "1.00",
                # no expiry_dates[]
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            assert (
                Batch.query.filter_by(batch_number="B-NOEXP").first() is None
            ), "Batch creation must be rejected without an expiry date."


# ─── P0-5: void reverses legacy billing ────────────────────────────────────


class TestVoidReversesLegacyBilling:
    def test_voided_billed_dispense_creates_credit_bill(self, app):
        from departments.models.billing import DrugsBill
        from departments.pharmacy.void_service import void_dispensed_drug
        from extensions import db

        with app.app_context():
            drug_id, batch_id = _make_drug(app, "VoidBillDrug", qty=100)
            dispense_id = _make_dispensed(
                app, drug_id, batch_id, qty=4, receipt="RCP-GAPFIX-001"
            )

            # Legacy PAID bill covering this dispense
            db.session.add(
                DrugsBill(
                    patient_id="P-GAPFIX-01",
                    drug_id=drug_id,
                    quantity=4,
                    total_cost=Decimal("8.00"),
                    status=1,
                    receipt_number="RCP-GAPFIX-001",
                )
            )
            db.session.commit()

            void_dispensed_drug(dispense_id, "Wrong strength dispensed", user_id=1)
            db.session.commit()

            credit = DrugsBill.query.filter_by(
                receipt_number="RCP-GAPFIX-001",
                payment_method="VOID_REVERSAL",
            ).first()
            assert credit is not None, "Void must create a legacy credit DrugsBill."
            assert credit.quantity == -4
            assert float(credit.total_cost) == -8.00

            from departments.models.pharmacy import DispensedDrug

            dispensed = db.session.get(DispensedDrug, dispense_id)
            assert dispensed.status == "VOIDED"

    def test_double_void_raises(self, app):
        from departments.pharmacy.void_service import VoidError, void_dispensed_drug
        from extensions import db

        with app.app_context():
            drug_id, batch_id = _make_drug(app, "DoubleVoidDrug", qty=10)
            dispense_id = _make_dispensed(app, drug_id, batch_id, qty=1)

            void_dispensed_drug(dispense_id, "first void", user_id=1)
            db.session.commit()

            with pytest.raises(VoidError, match="already been voided"):
                void_dispensed_drug(dispense_id, "second void", user_id=1)


# ─── P0-1: RBAC on previously-unprotected routes ───────────────────────────


@pytest.fixture
def hr_user(app):
    """A non-pharmacy (hr) user for RBAC denial tests."""
    from departments.models.user import User
    from extensions import db

    with app.app_context():
        user = User.query.filter_by(username="hr_gapfix").first()
        if not user:
            from werkzeug.security import generate_password_hash

            user = User(
                username="hr_gapfix",
                password=generate_password_hash("hr123", method="pbkdf2:sha256"),
                role="hr",
            )
            db.session.add(user)
            db.session.commit()
        yield user


class TestRBACOnSensitiveRoutes:
    def test_save_dispensed_drugs_rejects_non_pharmacy_role(self, app, hr_user):
        """A logged-in user with a non-pharmacy role must get 403."""
        client = app.test_client()
        client.post(
            "/login",
            data={"username": "hr_gapfix", "password": "hr123"},
            follow_redirects=True,
        )
        resp = client.post("/pharmacy/save_dispensed_drugs", data={})
        assert resp.status_code == 403, (
            f"save_dispensed_drugs must 403 for non-pharmacy roles, "
            f"got {resp.status_code}"
        )

    def test_patient_history_rejects_non_pharmacy_role(self, app, hr_user):
        client = app.test_client()
        client.post(
            "/login",
            data={"username": "hr_gapfix", "password": "hr123"},
            follow_redirects=True,
        )
        resp = client.get("/pharmacy/patient_history")
        assert resp.status_code == 403

    def test_get_all_batches_rejects_non_pharmacy_role(self, app, hr_user):
        client = app.test_client()
        client.post(
            "/login",
            data={"username": "hr_gapfix", "password": "hr123"},
            follow_redirects=True,
        )
        resp = client.get("/pharmacy/get_all_batches")
        assert resp.status_code == 403

    def test_analytics_rejects_non_pharmacy_role(self, app, hr_user):
        client = app.test_client()
        client.post(
            "/login",
            data={"username": "hr_gapfix", "password": "hr123"},
            follow_redirects=True,
        )
        resp = client.get("/pharmacy/analytics")
        assert resp.status_code == 403
