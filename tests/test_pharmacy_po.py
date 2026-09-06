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


def test_auto_generate_po_and_lifecycle(client, app, pharmacy_user, sample_supplier, low_stock_drug):
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

    # 2. Submit order
    order_resp = client.post(f"/pharmacy/po/{po_id}/order")
    assert order_resp.status_code == 200
    assert order_resp.get_json()["purchase_order"]["status"] == "ORDERED"

    # 3. Receive shipment
    rec_resp = client.post(f"/pharmacy/po/{po_id}/receive")
    assert rec_resp.status_code == 200
    assert rec_resp.get_json()["purchase_order"]["status"] == "RECEIVED"

    # 4. Verify DB stock increment and FEFO batch creation
    with app.app_context():
        refreshed_drug = db.session.get(Drug, low_stock_drug.id)
        assert refreshed_drug.quantity_in_stock > 10  # incremented by ordered qty

        batch = Batch.query.filter_by(drug_id=low_stock_drug.id).order_by(Batch.id.desc()).first()
        assert batch is not None
        assert batch.batch_number.startswith("B-PO-")
