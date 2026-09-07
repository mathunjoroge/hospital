"""
tests/test_budget_and_transfers.py
───────────────────────────────────
Tests for:
1. Phase D — Budget & Vote-Head Procurement Control
2. Phase E — Inter-Facility Stock Transfer Engine (Dispatch & Receive)
"""

import pytest
from werkzeug.security import generate_password_hash

from departments.models.budget import VoteHead
from departments.models.facility import Facility
from departments.models.pharmacy import Drug, DrugCategory
from departments.models.stock_movement import StockMovement
from departments.models.supplier import PurchaseOrder, PurchaseOrderItem, Supplier
from departments.models.user import User
from extensions import db


@pytest.fixture
def stores_user(app):
    with app.app_context():
        u = User(
            username="test_transfers_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="stores",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def approver_user(app):
    with app.app_context():
        u = User(
            username="test_approver_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="pharmacy",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def setup_budget_and_facilities(app):
    with app.app_context():
        # Facilities
        target_fac = Facility(
            name="Sub-County Dispensary North",
            facility_code="KMHFL-DISP-002",
            facility_type="Dispensary",
            is_active=True,
            is_self=False,
        )
        db.session.add(target_fac)

        # VoteHead with KES 10,000 allocation
        vh = VoteHead(
            code="VOTE-PHARM-2026",
            name="Pharmacy Budget",
            department="pharmacy",
            financial_year="2026",
            allocated_amount=10000.0,
        )
        db.session.add(vh)

        cat = DrugCategory.query.first()
        if not cat:
            cat = DrugCategory(name="Analgesics")
            db.session.add(cat)
            db.session.commit()

        drug = Drug(
            generic_name="Ibuprofen 400mg",
            category_id=cat.id,
            dosage_form="Tablet",
            strength="400mg",
            buying_price=10.0,
            selling_price=15.0,
            quantity_in_stock=500,
            reorder_level=50,
        )
        supplier = Supplier(name="Budget Supplier Co", is_active=True)
        db.session.add_all([drug, supplier])
        db.session.commit()

        yield target_fac.id, vh.id, drug.id, supplier.id


def test_vote_head_budget_encumbrance_and_cap_blocking(
    client, app, stores_user, approver_user, setup_budget_and_facilities
):
    """Test Phase D: PO approval encumbers vote-head budget; rejects if cost exceeds available balance."""
    _, vh_id, drug_id, supplier_id = setup_budget_and_facilities

    client.post(
        "/login", data={"username": "test_transfers_user", "password": "Password123!"}
    )

    # 1. Create draft PO exceeding budget (1,500 units @ KES 10 = KES 15,000 > KES 10,000 allocated)
    with app.app_context():
        creator = db.session.merge(stores_user)
        po_exceed = PurchaseOrder(
            po_number="PO-BUDGET-EXCEED",
            supplier_id=supplier_id,
            status="DRAFT",
            created_by_id=creator.id,
        )
        db.session.add(po_exceed)
        db.session.flush()

        poi = PurchaseOrderItem(
            po_id=po_exceed.id,
            drug_id=drug_id,
            vote_head_id=vh_id,
            quantity_ordered=1500,
            unit_cost=10.0,
        )
        db.session.add(poi)
        db.session.commit()
        po_exceed_id = po_exceed.id

    # Log in as approver to attempt approval -> 400 Bad Request (budget exceeded)
    client.post(
        "/login", data={"username": "test_approver_user", "password": "Password123!"}
    )
    res_fail = client.post(f"/pharmacy/po/{po_exceed_id}/order")
    assert res_fail.status_code == 400
    assert "budget vote-head cap exceeded" in res_fail.get_json()["error"].lower()

    # 2. Create valid PO within budget (500 units @ KES 10 = KES 5,000 <= KES 10,000)
    with app.app_context():
        creator = db.session.merge(stores_user)
        po_ok = PurchaseOrder(
            po_number="PO-BUDGET-OK",
            supplier_id=supplier_id,
            status="DRAFT",
            created_by_id=creator.id,
        )
        db.session.add(po_ok)
        db.session.flush()

        poi = PurchaseOrderItem(
            po_id=po_ok.id,
            drug_id=drug_id,
            vote_head_id=vh_id,
            quantity_ordered=500,
            unit_cost=10.0,
        )
        db.session.add(poi)
        db.session.commit()
        po_ok_id = po_ok.id

    res_ok = client.post(f"/pharmacy/po/{po_ok_id}/order")
    assert res_ok.status_code == 200

    with app.app_context():
        vh = db.session.get(VoteHead, vh_id)
        assert float(vh.encumbered_amount) == 5000.0
        assert vh.available_amount == 5000.0


def test_inter_facility_transfer_lifecycle(
    client, app, stores_user, setup_budget_and_facilities
):
    """Test Phase E: Inter-facility transfer creation, outbound dispatch, and inbound receiving."""
    target_fac_id, _, drug_id, _ = setup_budget_and_facilities

    client.post(
        "/login", data={"username": "test_transfers_user", "password": "Password123!"}
    )

    # 1. Create draft transfer order
    res_create = client.post(
        "/stores/transfers/create",
        json={
            "target_facility_id": target_fac_id,
            "notes": "Emergency transfer of analgesics",
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": drug_id,
                    "quantity_requested": 100,
                }
            ],
        },
    )
    assert res_create.status_code == 201
    transfer_data = res_create.get_json()["transfer"]
    transfer_id = transfer_data["id"]
    assert transfer_data["status"] == "DRAFT"

    # 2. Dispatch outbound stock -> status DISPATCHED, stock deducted, TRANSFER_OUT ledger recorded
    res_dispatch = client.post(
        f"/stores/transfers/{transfer_id}/dispatch",
        json={
            "items": [
                {
                    "drug_id": drug_id,
                    "quantity_dispatched": 100,
                    "expiry_date": "2028-05-20",
                    "batch_number": "BATCH-TR-100",
                }
            ]
        },
    )
    assert res_dispatch.status_code == 200
    assert res_dispatch.get_json()["transfer"]["status"] == "DISPATCHED"

    with app.app_context():
        drug = db.session.get(Drug, drug_id)
        assert drug.quantity_in_stock == 400  # 500 - 100

        movement = StockMovement.query.filter_by(
            item_type="DRUG", item_id=drug_id, movement_type="TRANSFER_OUT"
        ).first()
        assert movement is not None
        assert movement.quantity_delta == -100

    # 3. Receive inbound stock (from external facility to home facility) -> status RECEIVED, stock added, TRANSFER_IN ledger recorded
    with app.app_context():
        from departments.models.facility import get_home_facility
        from departments.models.transfer import TransferOrder, TransferOrderItem

        inbound_tr = TransferOrder(
            transfer_number="TR-INBOUND-001",
            source_facility_id=target_fac_id,
            target_facility_id=get_home_facility().id,
            status="DISPATCHED",
        )
        db.session.add(inbound_tr)
        db.session.flush()
        toi = TransferOrderItem(
            transfer_id=inbound_tr.id,
            item_type="DRUG",
            drug_id=drug_id,
            quantity_requested=100,
            quantity_dispatched=100,
        )
        db.session.add(toi)
        db.session.commit()
        inbound_transfer_id = inbound_tr.id

    res_receive = client.post(
        f"/stores/transfers/{inbound_transfer_id}/receive",
        json={
            "items": [
                {
                    "drug_id": drug_id,
                    "quantity_received": 100,
                    "expiry_date": "2028-05-20",
                    "batch_number": "BATCH-TR-100",
                }
            ]
        },
    )
    assert res_receive.status_code == 200
    assert res_receive.get_json()["transfer"]["status"] == "RECEIVED"

    with app.app_context():
        drug = db.session.get(Drug, drug_id)
        assert drug.quantity_in_stock == 500  # 400 + 100

        movement_in = StockMovement.query.filter_by(
            item_type="DRUG", item_id=drug_id, movement_type="TRANSFER_IN"
        ).first()
        assert movement_in is not None
        assert movement_in.quantity_delta == 100
