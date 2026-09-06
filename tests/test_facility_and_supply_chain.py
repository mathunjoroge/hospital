"""
tests/test_facility_and_supply_chain.py
─────────────────────────────────────────
Tests for:
1. Facility model & get_home_facility() helper (Phase B.1)
2. Segregation of duties for Purchase Orders (Phase C.2)
3. Append-only StockMovement ledger & reconciliation (Phase C.4 / Phase F)
4. Bin Card API and Reconciliation Report routes (Phase F)
"""


import pytest
from werkzeug.security import generate_password_hash

from departments.models.facility import Facility, get_home_facility
from departments.models.pharmacy import Drug, DrugCategory
from departments.models.stock_movement import (
    StockMovement,
    reconcile_stock_balance,
    record_movement,
)
from departments.models.supplier import PurchaseOrder, PurchaseOrderItem, Supplier
from departments.models.transfer import TransferOrder, TransferOrderItem
from departments.models.user import User
from extensions import db


@pytest.fixture
def user_creator(app):
    with app.app_context():
        u = User(
            username="sc_creator_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="pharmacy",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def user_approver(app):
    with app.app_context():
        u = User(
            username="sc_approver_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="pharmacy",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def user_store_admin(app):
    """A user permitted to both dispatch and receive inter-facility transfers."""
    with app.app_context():
        u = User(
            username="sc_store_admin_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="admin",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def test_setup(app):
    with app.app_context():
        cat = DrugCategory.query.first()
        if not cat:
            cat = DrugCategory(name="Analgesics")
            db.session.add(cat)
            db.session.commit()

        drug = Drug(
            generic_name="Paracetamol 500mg",
            category_id=cat.id,
            dosage_form="Tablet",
            strength="500mg",
            buying_price=5.0,
            selling_price=10.0,
            quantity_in_stock=100,
            reorder_level=20,
        )
        supplier = Supplier(name="Pharma Supply Co", is_active=True)
        db.session.add_all([drug, supplier])
        db.session.flush()

        record_movement(
            item_type="DRUG",
            item_id=drug.id,
            movement_type="ADJUSTMENT",
            quantity_delta=100,
            balance_after=100,
            notes="Initial stock baseline",
        )
        db.session.commit()

        yield drug.id, supplier.id


def test_facility_home_seeding(app):
    """Test Facility model and get_home_facility helper."""
    with app.app_context():
        home = get_home_facility()
        assert home is not None
        assert home.is_self is True
        assert home.facility_code == "KMHFL-MAIN-001"



def test_segregation_of_duties_approval_restriction(client, app, user_creator, test_setup):
    """Creator attempting to approve their own PO must receive 403 Forbidden."""
    drug_id, supplier_id = test_setup

    client.post("/login", data={"username": "sc_creator_user", "password": "Password123!"})

    with app.app_context():
        u = db.session.merge(user_creator)
        po = PurchaseOrder(
            po_number="PO-SOD-001",
            supplier_id=supplier_id,
            status="DRAFT",
            created_by_id=u.id,
        )
        db.session.add(po)
        db.session.commit()
        po_id = po.id

    # Creator tries to approve PO -> 403 Forbidden
    res = client.post(f"/pharmacy/po/{po_id}/order")
    assert res.status_code == 403
    assert "segregation of duties" in res.get_json()["error"].lower()


def test_segregation_of_duties_success_and_sod_warning(client, app, user_creator, user_approver, test_setup):
    """Different user approving PO succeeds; receiving by approver triggers sod_warning."""
    drug_id, supplier_id = test_setup

    # Log in as creator to create PO
    with app.app_context():
        creator = db.session.merge(user_creator)
        db.session.merge(user_approver)
        po = PurchaseOrder(
            po_number="PO-SOD-002",
            supplier_id=supplier_id,
            status="DRAFT",
            created_by_id=creator.id,
        )
        db.session.add(po)
        db.session.flush()
        poi = PurchaseOrderItem(po_id=po.id, drug_id=drug_id, quantity_ordered=50, unit_cost=5.0)
        db.session.add(poi)
        db.session.commit()
        po_id = po.id

    # Log in as approver to approve PO
    client.post("/login", data={"username": "sc_approver_user", "password": "Password123!"})
    res_order = client.post(f"/pharmacy/po/{po_id}/order")
    assert res_order.status_code == 200
    assert res_order.get_json()["purchase_order"]["status"] == "ORDERED"

    # Approver receives the shipment -> receiving succeeds but sod_warning becomes True
    res_receive = client.post(
        f"/pharmacy/po/{po_id}/receive",
        json={
            "items": [
                {
                    "drug_id": drug_id,
                    "quantity_received": 50,
                    "expiry_date": "2027-12-31",
                    "batch_number": "BATCH-SOD-002",
                }
            ]
        },
    )
    assert res_receive.status_code == 200
    po_data = res_receive.get_json()["purchase_order"]
    assert po_data["status"] == "RECEIVED"
    assert po_data["sod_warning"] is True


def test_stock_movement_ledger_and_reconciliation(client, app, user_approver, test_setup):
    """Test append-only movement ledger entries and reconciliation helper."""
    drug_id, supplier_id = test_setup

    client.post("/login", data={"username": "sc_approver_user", "password": "Password123!"})

    with app.app_context():
        po = PurchaseOrder(po_number="PO-LEDGER-001", supplier_id=supplier_id, status="ORDERED")
        db.session.add(po)
        db.session.flush()
        poi = PurchaseOrderItem(po_id=po.id, drug_id=drug_id, quantity_ordered=100, unit_cost=5.0)
        db.session.add(poi)
        db.session.commit()
        po_id = po.id

    res = client.post(
        f"/pharmacy/po/{po_id}/receive",
        json={
            "items": [
                {
                    "drug_id": drug_id,
                    "quantity_received": 100,
                    "expiry_date": "2028-01-01",
                    "batch_number": "BATCH-LEDGER-1",
                }
            ]
        },
    )
    assert res.status_code == 200

    with app.app_context():
        # Check StockMovement row was created
        movement = StockMovement.query.filter_by(
            item_type="DRUG", item_id=drug_id, reference_id="PO-LEDGER-001"
        ).first()
        assert movement is not None
        assert movement.quantity_delta == 100
        assert movement.movement_type == "RECEIVED"

        # Check reconciliation helper matches
        recon = reconcile_stock_balance("DRUG", drug_id)
        assert recon["match"] is True
        assert recon["ledger_balance"] == 200
        assert recon["cached_balance"] == 200  # initial 100 + 100 received



def test_bin_card_and_reconciliation_report_apis(client, app, user_approver, test_setup):
    """Test GET /stores/bin-card/DRUG/<id> and GET /stores/reconciliation-report endpoints."""
    drug_id, _ = test_setup

    client.post("/login", data={"username": "sc_approver_user", "password": "Password123!"})

    # Test Bin Card API
    res_bincard = client.get(
        f"/stores/bin-card/DRUG/{drug_id}",
        headers={"Accept": "application/json"},
    )
    assert res_bincard.status_code == 200
    json_bc = res_bincard.get_json()
    assert json_bc["item_type"] == "DRUG"
    assert json_bc["item_id"] == drug_id
    assert "movements" in json_bc
    assert "reconciliation" in json_bc

    # Test Reconciliation Report API
    res_recon = client.get(
        "/stores/reconciliation-report",
        headers={"Accept": "application/json"},
    )
    assert res_recon.status_code == 200
    json_rr = res_recon.get_json()
    assert "drugs" in json_rr
    assert "non_pharm" in json_rr
    assert "has_discrepancies" in json_rr


def test_suppliers_directory_ui_view(client, user_approver, test_setup):
    """Test GET /stores/suppliers renders suppliers directory UI page."""
    client.post("/login", data={"username": "sc_approver_user", "password": "Password123!"})
    res = client.get("/stores/suppliers")
    assert res.status_code == 200
    assert b"Suppliers &amp; Vendors Directory" in res.data or b"Suppliers & Vendors Directory" in res.data
    assert b"Pharma Supply Co" in res.data


def test_inter_facility_transfers_ui_views(client, user_approver):
    """Test GET /stores/transfers and /stores/transfers/new UI routes."""
    client.post("/login", data={"username": "sc_approver_user", "password": "Password123!"})

    res_list = client.get("/stores/transfers")
    assert res_list.status_code == 200
    assert b"Inter-Facility Stock Transfers" in res_list.data

    res_new = client.get("/stores/transfers/new")
    assert res_new.status_code == 200
    assert b"Initiate Inter-Facility Transfer" in res_new.data



def test_dispatch_blocked_for_non_source_facility(client, app, user_store_admin, test_setup):
    """A transfer whose source facility isn't this installation cannot be dispatched here."""
    drug_id, _supplier_id = test_setup
    with app.app_context():
        other_facility = Facility(name="Other County Hospital", is_self=False)
        db.session.add(other_facility)
        db.session.flush()

        transfer = TransferOrder(
            transfer_number="TR-TESTSRC01",
            source_facility_id=other_facility.id,
            target_facility_id=get_home_facility().id,
            status="DRAFT",
        )
        db.session.add(transfer)
        db.session.flush()
        db.session.add(TransferOrderItem(
            transfer_id=transfer.id,
            item_type="DRUG",
            drug_id=drug_id,
            quantity_requested=10,
        ))
        db.session.commit()
        transfer_id = transfer.id

    client.post("/login", data={"username": "sc_store_admin_user", "password": "Password123!"})
    res = client.post(f"/stores/transfers/{transfer_id}/dispatch", json={})
    assert res.status_code == 403
    assert "Facility boundary violation" in res.get_json()["error"]


def test_receive_blocked_for_non_target_facility(client, app, user_store_admin, test_setup):
    """A transfer this installation dispatched to another facility cannot be 'received' here too."""
    drug_id, _supplier_id = test_setup
    client.post("/login", data={"username": "sc_store_admin_user", "password": "Password123!"})

    with app.app_context():
        other_facility = Facility(name="Other County Hospital", is_self=False)
        db.session.add(other_facility)
        db.session.commit()
        other_facility_id = other_facility.id

    res_create = client.post("/stores/transfers/create", json={
        "target_facility_id": other_facility_id,
        "items": [{"item_type": "DRUG", "drug_id": drug_id, "quantity_requested": 10}],
    })
    assert res_create.status_code == 201
    transfer_id = res_create.get_json()["transfer"]["id"]

    res_dispatch = client.post(f"/stores/transfers/{transfer_id}/dispatch", json={})
    assert res_dispatch.status_code == 200

    res_receive = client.post(f"/stores/transfers/{transfer_id}/receive", json={})
    assert res_receive.status_code == 403
    assert "Facility boundary violation" in res_receive.get_json()["error"]


def test_receive_succeeds_and_flags_variance_for_target_facility(client, app, user_store_admin, test_setup):
    """This installation, as the genuine target facility, can receive and short/over receipt is flagged."""
    drug_id, _supplier_id = test_setup
    with app.app_context():
        other_facility = Facility(name="Sending County Hospital", is_self=False)
        db.session.add(other_facility)
        db.session.flush()

        transfer = TransferOrder(
            transfer_number="TR-TESTVAR01",
            source_facility_id=other_facility.id,
            target_facility_id=get_home_facility().id,
            status="DISPATCHED",
            dispatched_at=db.func.now(),
        )
        db.session.add(transfer)
        db.session.flush()
        toi = TransferOrderItem(
            transfer_id=transfer.id,
            item_type="DRUG",
            drug_id=drug_id,
            quantity_requested=10,
            quantity_dispatched=10,
        )
        db.session.add(toi)
        db.session.commit()
        transfer_id = transfer.id
        item_id = toi.id

    client.post("/login", data={"username": "sc_store_admin_user", "password": "Password123!"})

    # Receive 12 against 10 dispatched — previously silently accepted with no flag at all.
    res = client.post(f"/stores/transfers/{transfer_id}/receive", json={
        "items": [{"item_id": item_id, "quantity_received": 12, "expiry_date": "2027-01-01"}],
    })
    assert res.status_code == 200
    body = res.get_json()
    assert "discrepancies" in body
    assert body["discrepancies"][0]["kind"] == "OVER_RECEIPT"
    assert body["discrepancies"][0]["variance"] == 2
    assert body["transfer"]["status"] == "RECEIVED_WITH_DISCREPANCY"

    # A genuine short-receipt should stay open (DISPATCHED) rather than being marked received.
    with app.app_context():
        transfer2 = TransferOrder(
            transfer_number="TR-TESTVAR02",
            source_facility_id=db.session.get(TransferOrder, transfer_id).source_facility_id,
            target_facility_id=get_home_facility().id,
            status="DISPATCHED",
        )
        db.session.add(transfer2)
        db.session.flush()
        toi2 = TransferOrderItem(
            transfer_id=transfer2.id,
            item_type="DRUG",
            drug_id=drug_id,
            quantity_requested=10,
            quantity_dispatched=10,
        )
        db.session.add(toi2)
        db.session.commit()
        transfer2_id = transfer2.id
        item2_id = toi2.id

    res_short = client.post(f"/stores/transfers/{transfer2_id}/receive", json={
        "items": [{"item_id": item2_id, "quantity_received": 7}],
    })
    assert res_short.status_code == 200
    body_short = res_short.get_json()
    assert body_short["discrepancies"][0]["kind"] == "SHORT_RECEIPT"
    assert body_short["transfer"]["status"] == "DISPATCHED"
