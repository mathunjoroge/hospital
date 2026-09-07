"""
tests/test_supply_chain_complete.py
────────────────────────────────────
Tests for:
1. Manual Purchase Order Creation (/pharmacy/po/create)
2. PO Over-Receipt Variance Detection & RECEIVED_WITH_DISCREPANCY Status
3. Direct Receipt VoteHead Budget Encumbrance Validation
4. Multi-Department Commodity Requisitions (/stores/requisitions/create)
5. Physical Stock Take Audit & StockMovement Ledger Adjustment (/stores/stock-take)
6. Automated Batch Expiry Notification Scanner
"""

from datetime import date, timedelta

import pytest
from werkzeug.security import generate_password_hash

from departments.models.budget import VoteHead
from departments.models.pharmacy import Batch, Drug, DrugCategory
from departments.models.stock_movement import StockMovement
from departments.models.stores import NonPharmCategory, NonPharmItem, OtherOrder
from departments.models.supplier import PurchaseOrder, Supplier
from departments.models.user import User
from departments.notifications.triggers import trigger_batch_expiry_check
from extensions import db


@pytest.fixture
def sc_user(app):
    with app.app_context():
        u = User(
            username="sc_complete_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="stores",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def setup_complete_sc_data(app):
    with app.app_context():
        supplier = Supplier(name="Global Med Ltd", is_active=True)
        db.session.add(supplier)
        db.session.flush()

        cat = DrugCategory.query.first() or DrugCategory(name="Antibiotics")
        db.session.add(cat)
        db.session.flush()

        drug = Drug(
            generic_name="Amoxicillin 500mg",
            category_id=cat.id,
            dosage_form="Capsule",
            strength="500mg",
            buying_price=20.0,
            selling_price=30.0,
            quantity_in_stock=100,
            reorder_level=20,
        )
        db.session.add(drug)
        db.session.flush()

        np_cat = NonPharmCategory.query.first() or NonPharmCategory(name="Cleaning")
        db.session.add(np_cat)
        db.session.flush()

        non_pharm = NonPharmItem(
            name="Detergent 5L",
            category_id=np_cat.id,
            unit="Liters",
            unit_cost=15.0,
            stock_level=50,
        )
        db.session.add(non_pharm)
        db.session.flush()

        vh = VoteHead(
            code="VOTE-STORES-2026",
            name="Stores Consumables Budget",
            department="stores",
            financial_year="2026",
            allocated_amount=5000.0,
        )
        db.session.add(vh)
        db.session.commit()

        yield supplier.id, drug.id, non_pharm.id, vh.id


def test_manual_po_creation_api(client, sc_user, setup_complete_sc_data):
    """Test manual creation of draft PO with supplier and line items."""
    supplier_id, drug_id, non_pharm_id, vh_id = setup_complete_sc_data
    client.post(
        "/login", data={"username": "sc_complete_user", "password": "Password123!"}
    )

    res = client.post(
        "/pharmacy/po/create",
        json={
            "supplier_id": supplier_id,
            "notes": "Urgent manual stock replenishment",
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": drug_id,
                    "quantity_ordered": 50,
                    "unit_cost": 20.0,
                    "vote_head_id": vh_id,
                },
                {
                    "item_type": "NON_PHARM",
                    "non_pharm_item_id": non_pharm_id,
                    "quantity_ordered": 10,
                    "unit_cost": 15.0,
                    "vote_head_id": vh_id,
                },
            ],
        },
    )
    assert res.status_code == 201
    data = res.get_json()["purchase_order"]
    assert data["status"] == "DRAFT"
    assert len(data["items"]) == 2
    assert "PO-MAN-" in data["po_number"]


def test_po_shipment_over_receipt_and_discrepancy_flag(
    client, app, sc_user, setup_complete_sc_data
):
    """Test receiving shipment with quantity variance flags RECEIVED_WITH_DISCREPANCY status."""
    supplier_id, drug_id, _, _ = setup_complete_sc_data
    client.post(
        "/login", data={"username": "sc_complete_user", "password": "Password123!"}
    )

    res_create = client.post(
        "/pharmacy/po/create",
        json={
            "supplier_id": supplier_id,
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": drug_id,
                    "quantity_ordered": 100,
                    "unit_cost": 20.0,
                }
            ],
        },
    )
    po_id = res_create.get_json()["purchase_order"]["id"]

    # Order PO
    with app.app_context():
        po = db.session.get(PurchaseOrder, po_id)
        po.status = "ORDERED"
        db.session.commit()

    # Receive 120 (over-receipt variance of +20)
    res_rcv = client.post(
        f"/pharmacy/po/{po_id}/receive",
        json={
            "items": [
                {
                    "drug_id": drug_id,
                    "quantity_received": 120,
                    "batch_number": "B-OVER-120",
                    "expiry_date": "2027-12-31",
                }
            ]
        },
    )
    assert res_rcv.status_code == 200
    body = res_rcv.get_json()
    assert body["purchase_order"]["status"] == "RECEIVED_WITH_DISCREPANCY"
    assert "discrepancies" in body
    assert body["discrepancies"][0]["kind"] == "OVER_RECEIPT"
    assert body["discrepancies"][0]["variance"] == 20


def test_direct_receipt_optional_vote_head_budget_check(
    client, app, sc_user, setup_complete_sc_data
):
    """Test direct receipt encumbers VoteHead budget when specified and rejects if cap exceeded."""
    supplier_id, drug_id, _, vh_id = setup_complete_sc_data
    client.post(
        "/login", data={"username": "sc_complete_user", "password": "Password123!"}
    )

    # Direct receipt exceeding available budget (300 units @ 20.0 = KES 6,000 > KES 5,000 budget)
    res_fail = client.post(
        "/pharmacy/receipt/direct",
        json={
            "supplier_id": supplier_id,
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": drug_id,
                    "quantity": 300,
                    "unit_cost": 20.0,
                    "batch_number": "B-DIR-FAIL",
                    "expiry_date": "2027-06-01",
                    "vote_head_id": vh_id,
                }
            ],
        },
    )
    assert res_fail.status_code == 400
    assert "cap exceeded" in res_fail.get_json()["error"].lower()

    # Valid direct receipt within budget (50 units @ 20.0 = KES 1,000 <= KES 5,000)
    res_ok = client.post(
        "/pharmacy/receipt/direct",
        json={
            "supplier_id": supplier_id,
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": drug_id,
                    "quantity": 50,
                    "unit_cost": 20.0,
                    "batch_number": "B-DIR-OK",
                    "expiry_date": "2027-06-01",
                    "vote_head_id": vh_id,
                }
            ],
        },
    )
    assert res_ok.status_code in (200, 201)
    with app.app_context():
        vh = db.session.get(VoteHead, vh_id)
        assert float(vh.encumbered_amount) == 1000.0


def test_commodity_requisition_creation_api(
    client, app, sc_user, setup_complete_sc_data
):
    """Test multi-department requisition creation for non-pharm commodities."""
    _, _, non_pharm_id, _ = setup_complete_sc_data
    client.post(
        "/login", data={"username": "sc_complete_user", "password": "Password123!"}
    )

    res = client.post(
        "/stores/requisitions/create",
        json={
            "item_id": non_pharm_id,
            "quantity_requested": 5,
            "department": "kitchen",
        },
    )
    assert res.status_code == 201
    assert res.get_json()["requisition"]["department"] == "kitchen"

    with app.app_context():
        req = OtherOrder.query.filter_by(item_id=non_pharm_id).first()
        assert req is not None
        assert req.quantity_requested == 5
        assert "kitchen" in req.notes


def test_physical_stock_take_audit_and_ledger_adjustment(
    client, app, sc_user, setup_complete_sc_data
):
    """Test physical stock count audit updates physical stock level and logs STOCK_TAKE_ADJUSTMENT ledger entries."""
    _, drug_id, non_pharm_id, _ = setup_complete_sc_data
    client.post(
        "/login", data={"username": "sc_complete_user", "password": "Password123!"}
    )

    # Book stock: drug = 100, non_pharm = 50
    # Audit count: drug = 92 (-8 shrinkage), non_pharm = 55 (+5 surplus count)
    res = client.post(
        "/stores/stock-take/create",
        json={
            "notes": "Q3 Routine Physical Audit",
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": drug_id,
                    "physical_quantity": 92,
                    "reason": "Damaged vials discarded during count",
                },
                {
                    "item_type": "NON_PHARM",
                    "non_pharm_item_id": non_pharm_id,
                    "physical_quantity": 55,
                    "reason": "Unrecorded delivery box found on shelf",
                },
            ],
        },
    )
    assert res.status_code == 201
    take_data = res.get_json()["stock_take"]
    assert take_data["total_items_counted"] == 2
    assert take_data["total_variance_count"] == 2

    with app.app_context():
        drug = db.session.get(Drug, drug_id)
        non_pharm = db.session.get(NonPharmItem, non_pharm_id)

        assert drug.quantity_in_stock == 92
        assert non_pharm.stock_level == 55

        # Verify STOCK_TAKE_ADJUSTMENT movements in StockMovement ledger
        mov_drug = StockMovement.query.filter_by(
            item_type="DRUG", item_id=drug_id, movement_type="STOCK_TAKE_ADJUSTMENT"
        ).first()
        assert mov_drug is not None
        assert mov_drug.quantity_delta == -8

        mov_np = StockMovement.query.filter_by(
            item_type="NON_PHARM",
            item_id=non_pharm_id,
            movement_type="STOCK_TAKE_ADJUSTMENT",
        ).first()
        assert mov_np is not None
        assert mov_np.quantity_delta == 5


def test_batch_expiry_notification_trigger(app, setup_complete_sc_data):
    """Test automated batch expiry scanner queries expiring stock and dispatches alerts."""
    _, drug_id, _, _ = setup_complete_sc_data
    with app.app_context():
        today = date.today()

        # Batch 1: Expiring in 10 days
        b1 = Batch(
            drug_id=drug_id,
            batch_number="B-EXPIRING-10",
            quantity_in_stock=25,
            expiry_date=today + timedelta(days=10),
        )
        # Batch 2: Expired 5 days ago
        b2 = Batch(
            drug_id=drug_id,
            batch_number="B-EXPIRED-PAST",
            quantity_in_stock=10,
            expiry_date=today - timedelta(days=5),
        )
        db.session.add_all([b1, b2])
        db.session.commit()

        sent_count = trigger_batch_expiry_check(app, window_days=30)
        assert sent_count >= 2
