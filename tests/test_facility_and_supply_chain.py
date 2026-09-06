"""
tests/test_facility_and_supply_chain.py
─────────────────────────────────────────
Tests for:
1. Facility model & get_home_facility() helper (Phase B.1)
2. Segregation of duties for Purchase Orders (Phase C.2)
3. Append-only StockMovement ledger & reconciliation (Phase C.4 / Phase F)
4. Bin Card API and Reconciliation Report routes (Phase F)
"""

from datetime import date
import pytest
from werkzeug.security import generate_password_hash

from departments.models.facility import Facility, get_home_facility
from departments.models.pharmacy import Batch, Drug, DrugCategory, DrugRequest, RequestItem
from departments.models.stock_movement import StockMovement, record_movement, reconcile_stock_balance
from departments.models.supplier import PurchaseOrder, PurchaseOrderItem, Supplier
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
        approver = db.session.merge(user_approver)
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
