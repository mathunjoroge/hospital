"""
tests/test_supply_chain_defects.py
───────────────────────────────────
Phase A defect fixes:
1. Stop fabricating expiry dates on PO receiving and drug request issuing.
2. Verify explicit expiry_date and batch_number validation.
"""

from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from departments.models.pharmacy import (
    Batch,
    Drug,
    DrugCategory,
    DrugRequest,
    RequestItem,
)
from departments.models.supplier import PurchaseOrder, PurchaseOrderItem, Supplier
from departments.models.user import User
from extensions import db


@pytest.fixture
def stores_user(app):
    with app.app_context():
        u = User(
            username="sc_stores_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="stores",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def pharm_user(app):
    with app.app_context():
        u = User(
            username="sc_pharm_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="pharmacy",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def drug_and_supplier(app):
    """Seed a minimal drug + category + supplier."""
    with app.app_context():
        cat = DrugCategory.query.first()
        if not cat:
            cat = DrugCategory(name="Antibiotics")
            db.session.add(cat)
            db.session.commit()

        drug = Drug(
            generic_name="Amoxicillin 500mg",
            category_id=cat.id,
            dosage_form="Capsule",
            strength="500mg",
            buying_price=15.0,
            selling_price=25.0,
            quantity_in_stock=10,
            reorder_level=50,
        )
        supplier = Supplier(name="Defect Test Supplier", is_active=True)
        db.session.add_all([drug, supplier])
        db.session.commit()
        yield drug, supplier


def test_po_receive_requires_expiry_date(client, app, pharm_user, drug_and_supplier):
    """POST /pharmacy/po/<id>/receive must fail with HTTP 400 if expiry_date is omitted."""
    drug, supplier = drug_and_supplier

    client.post(
        "/login", data={"username": "sc_pharm_user", "password": "Password123!"}
    )

    with app.app_context():
        drug = db.session.merge(drug)
        supplier = db.session.merge(supplier)

        po = PurchaseOrder(
            po_number="PO-DEFECT-001", supplier_id=supplier.id, status="ORDERED"
        )
        db.session.add(po)
        db.session.flush()

        poi = PurchaseOrderItem(
            po_id=po.id, drug_id=drug.id, quantity_ordered=50, unit_cost=15.0
        )
        db.session.add(poi)
        db.session.commit()

        po_id = po.id
        drug_id = drug.id

    # 1. No body → 400
    res_no_body = client.post(f"/pharmacy/po/{po_id}/receive")
    assert res_no_body.status_code == 400
    assert "explicit item details" in res_no_body.get_json()["error"].lower()

    # 2. Item present but no expiry_date → 400
    res_missing = client.post(
        f"/pharmacy/po/{po_id}/receive",
        json={"items": [{"drug_id": drug_id, "quantity_received": 50}]},
    )
    assert res_missing.status_code == 400
    assert "expiry_date" in res_missing.get_json()["error"].lower()

    # 3. Bad date format → 400
    res_bad_fmt = client.post(
        f"/pharmacy/po/{po_id}/receive",
        json={
            "items": [
                {
                    "drug_id": drug_id,
                    "quantity_received": 50,
                    "expiry_date": "NOT-A-DATE",
                }
            ]
        },
    )
    assert res_bad_fmt.status_code == 400
    assert "invalid expiry_date format" in res_bad_fmt.get_json()["error"].lower()

    # 4. Valid payload → 200 + batch has real expiry date
    res_ok = client.post(
        f"/pharmacy/po/{po_id}/receive",
        json={
            "items": [
                {
                    "drug_id": drug_id,
                    "quantity_received": 50,
                    "expiry_date": "2028-06-30",
                    "batch_number": "REAL-BATCH-2028",
                }
            ]
        },
    )
    assert res_ok.status_code == 200, res_ok.get_json()

    with app.app_context():
        batch = Batch.query.filter_by(batch_number="REAL-BATCH-2028").first()
        assert batch is not None
        # The batch must carry the REAL expiry date, not any system-computed value
        assert batch.expiry_date == date(2028, 6, 30)


def test_issue_request_requires_expiry_date(
    client, app, stores_user, drug_and_supplier
):
    """POST /stores/issue_request/<id> must fail if expiry_date field is absent."""
    drug, _ = drug_and_supplier

    client.post(
        "/login", data={"username": "sc_stores_user", "password": "Password123!"}
    )

    with app.app_context():
        drug = db.session.merge(drug)

        req_obj = DrugRequest(
            request_date=date.today(), status="Pending", requested_by=stores_user.id
        )
        db.session.add(req_obj)
        db.session.flush()

        req_item = RequestItem(
            request_id=req_obj.id, drug_id=drug.id, quantity_requested=20
        )
        db.session.add(req_item)
        db.session.commit()

        req_id = req_obj.id
        item_id = req_item.id

    # 1. POST without expiry date → 400
    resp = client.post(
        f"/stores/issue_request/{req_id}",
        data={f"quantity_issued_{item_id}": 20},
    )
    assert resp.status_code == 400

    # 2. POST with explicit expiry date and batch_number → 302 redirect (success)
    resp_ok = client.post(
        f"/stores/issue_request/{req_id}",
        data={
            f"quantity_issued_{item_id}": 20,
            f"expiry_date_{item_id}": "2027-11-15",
            f"batch_number_{item_id}": "BATCH-STORES-99",
        },
    )
    assert resp_ok.status_code == 302

    with app.app_context():
        batch = Batch.query.filter_by(batch_number="BATCH-STORES-99").first()
        assert batch is not None
        # Batch must carry the real expiry date entered by staff
        assert batch.expiry_date == date(2027, 11, 15)
