"""
tests/test_pharmacy_po.py
──────────────────────────
Phase F — Automated Pharmacy Inventory & Supplier Purchase Orders Test Suite
"""

import pytest
from werkzeug.security import generate_password_hash

from departments.models.pharmacy import Batch, Drug
from departments.models.supplier import Supplier
from departments.models.user import User
from extensions import db


@pytest.fixture
def pharmacy_user(app):
    with app.app_context():
        u = User(
            username="pharmacy_po_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="pharmacy",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def approver_user(app):
    with app.app_context():
        u = User(
            username="pharmacy_approver_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="pharmacy",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def sample_supplier(app):
    with app.app_context():
        s = Supplier(
            name="KEMSA Pharma Supplies",
            contact_email="orders@kemsa.co.ke",
            phone="0700000000",
            lead_time_days=3,
        )
        db.session.add(s)
        db.session.commit()
        yield s



@pytest.fixture
def low_stock_drug(app):
    with app.app_context():
        from departments.models.pharmacy import DrugCategory
        cat = DrugCategory.query.first()
        if not cat:
            cat = DrugCategory(name="Antibiotics")
            db.session.add(cat)
            db.session.commit()

        d = Drug(
            generic_name="Amoxicillin 500mg Caps",
            category_id=cat.id,
            dosage_form="Capsule",
            strength="500mg",
            buying_price=10.00,
            selling_price=15.00,
            quantity_in_stock=10,  # below reorder level 50
            reorder_level=50,
        )
        db.session.add(d)
        db.session.commit()
        yield d


def test_supplier_creation_and_listing(client, pharmacy_user):
    """POST /pharmacy/suppliers creates supplier, GET lists them."""
    client.post("/login", data={"username": pharmacy_user.username, "password": "Password123!"})

    # Create supplier
    resp = client.post(
        "/pharmacy/suppliers",
        json={"name": "MEDS Kenya Ltd", "contact_email": "info@meds.or.ke", "lead_time_days": 2},
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["supplier"]["name"] == "MEDS Kenya Ltd"

    # List suppliers
    list_resp = client.get("/pharmacy/suppliers")
    assert list_resp.status_code == 200
    suppliers = list_resp.get_json()["suppliers"]
    assert any(s["name"] == "MEDS Kenya Ltd" for s in suppliers)


def test_low_stock_inventory_scan(client, pharmacy_user, low_stock_drug):
    """GET /pharmacy/low-stock detects drugs below reorder level."""
    client.post("/login", data={"username": pharmacy_user.username, "password": "Password123!"})
    resp = client.get("/pharmacy/low-stock")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["low_stock_count"] >= 1
    assert any(d["name"] == low_stock_drug.generic_name for d in data["drugs"])


def test_auto_generate_po_and_lifecycle(client, app, pharmacy_user, approver_user, sample_supplier, low_stock_drug):
    """POST /pharmacy/po/auto-generate creates draft PO, orders, and receives shipment."""
    client.post("/login", data={"username": pharmacy_user.username, "password": "Password123!"})

    # 1. Auto-generate draft PO
    po_resp = client.post(
        "/pharmacy/po/auto-generate",
        json={"supplier_id": sample_supplier.id},
    )
    assert po_resp.status_code == 201
    po_data = po_resp.get_json()["purchase_order"]
    po_id = po_data["id"]
    assert po_data["status"] == "DRAFT"
    assert len(po_data["items"]) >= 1

    # 2. Submit order as a different user to satisfy Segregation of Duties
    client.post("/login", data={"username": approver_user.username, "password": "Password123!"})
    order_resp = client.post(f"/pharmacy/po/{po_id}/order")
    assert order_resp.status_code == 200
    assert order_resp.get_json()["purchase_order"]["status"] == "ORDERED"


    # 3. Receive shipment
    item_id = po_data["items"][0]["drug_id"]
    rec_resp = client.post(
        f"/pharmacy/po/{po_id}/receive",
        json={
            "items": [
                {
                    "drug_id": item_id,
                    "quantity_received": po_data["items"][0]["quantity_ordered"],
                    "expiry_date": "2027-12-31",
                    "batch_number": f"B-PO-{po_id}-{item_id}"
                }
            ]
        }
    )
    assert rec_resp.status_code == 200
    assert rec_resp.get_json()["purchase_order"]["status"] == "RECEIVED"

    # 4. Verify DB stock increment and FEFO batch creation
    with app.app_context():
        refreshed_drug = db.session.get(Drug, low_stock_drug.id)
        assert refreshed_drug.quantity_in_stock > 10  # incremented by ordered qty

        batch = Batch.query.filter_by(drug_id=low_stock_drug.id).order_by(Batch.id.desc()).first()
        assert batch is not None
        assert batch.batch_number.startswith("B-PO-")


def test_receive_non_pharm_po_shipment(client, app, pharmacy_user, sample_supplier):
    """Receiving PO with non-pharm items increments NonPharmItem.stock_level and records StockMovement."""
    client.post("/login", data={"username": pharmacy_user.username, "password": "Password123!"})

    with app.app_context():
        from departments.models.stores import NonPharmCategory, NonPharmItem
        from departments.models.supplier import PurchaseOrder, PurchaseOrderItem

        cat = NonPharmCategory(name="General Supplies")
        db.session.add(cat)
        db.session.commit()

        np_item = NonPharmItem(name="Surgical Gloves", category_id=cat.id, unit="boxes", unit_cost=250.0, stock_level=10)
        db.session.add(np_item)
        db.session.commit()

        po = PurchaseOrder(po_number="PO-NP-001", supplier_id=sample_supplier.id, status="ORDERED", total_cost=2500.0)
        db.session.add(po)
        db.session.flush()

        po_item = PurchaseOrderItem(po_id=po.id, item_type="NON_PHARM", non_pharm_item_id=np_item.id, quantity_ordered=10, unit_cost=250.0)
        db.session.add(po_item)
        db.session.commit()
        po_id = po.id
        np_id = np_item.id

    rec_resp = client.post(
        f"/pharmacy/po/{po_id}/receive",
        json={
            "items": [
                {
                    "non_pharm_item_id": np_id,
                    "quantity_received": 10,
                    "expiry_date": "2027-12-31"
                }
            ]
        }
    )
    assert rec_resp.status_code == 200
    assert rec_resp.get_json()["purchase_order"]["status"] == "RECEIVED"

    with app.app_context():
        from departments.models.stock_movement import StockMovement
        from departments.models.stores import NonPharmItem
        refreshed = db.session.get(NonPharmItem, np_id)
        assert refreshed.stock_level == 20  # 10 + 10

        sm = StockMovement.query.filter_by(item_type="NON_PHARM", item_id=np_id).first()
        assert sm is not None
        assert sm.quantity_delta == 10
        assert sm.balance_after == 20


def test_record_direct_receipt(client, app, pharmacy_user, sample_supplier, low_stock_drug):
    """POST /pharmacy/receipt/direct records goods received without prior PO."""
    client.post("/login", data={"username": pharmacy_user.username, "password": "Password123!"})

    with app.app_context():
        from departments.models.stores import NonPharmCategory, NonPharmItem
        cat = NonPharmCategory(name="Lab Disposables")
        db.session.add(cat)
        db.session.commit()

        np_item = NonPharmItem(name="Test Tubes", category_id=cat.id, unit="packs", unit_cost=50.0, stock_level=5)
        db.session.add(np_item)
        db.session.commit()
        np_id = np_item.id

    resp = client.post(
        "/pharmacy/receipt/direct",
        json={
            "supplier_id": sample_supplier.id,
            "notes": "Direct Emergency Delivery",
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": low_stock_drug.id,
                    "quantity": 30,
                    "unit_cost": 10.0,
                    "batch_number": "B-DIR-DRUG-1",
                    "expiry_date": "2028-06-30"
                },
                {
                    "item_type": "NON_PHARM",
                    "non_pharm_item_id": np_id,
                    "quantity": 15,
                    "unit_cost": 50.0,
                    "expiry_date": "2028-06-30"
                }
            ]
        }
    )
    assert resp.status_code == 201
    po_data = resp.get_json()["purchase_order"]
    assert po_data["status"] == "RECEIVED"
    assert len(po_data["items"]) == 2

    with app.app_context():
        from departments.models.stores import NonPharmItem
        refreshed_drug = db.session.get(Drug, low_stock_drug.id)
        assert refreshed_drug.quantity_in_stock == 40  # 10 + 30

        refreshed_np = db.session.get(NonPharmItem, np_id)
        assert refreshed_np.stock_level == 20  # 5 + 15


def test_stores_po_ui_routes(client, pharmacy_user):
    """GET UI pages for purchase orders, direct receipt, and receipt history."""
    client.post("/login", data={"username": pharmacy_user.username, "password": "Password123!"})

    r1 = client.get("/stores/purchase-orders")
    assert r1.status_code == 200
    assert b"Purchase Orders" in r1.data

    r2 = client.get("/stores/receipt/direct")
    assert r2.status_code == 200
    assert b"Record Direct Supplier Receipt" in r2.data

    r3 = client.get("/stores/receipt-history")
    assert r3.status_code == 200
    assert b"Goods Receipt History" in r3.data

