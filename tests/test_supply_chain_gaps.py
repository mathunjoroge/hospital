"""
tests/test_supply_chain_gaps.py
─────────────────────────────────
Automated tests for Supply Chain Gaps & Enhancements:
1. Return to Vendor (RTV) creation, dispatch, and RETURN_TO_VENDOR stock movement ledgers.
2. Stock Disposal & Write-off Board creation, approval, and DISCARDED stock movement ledgers.
3. Supplier On-Time In-Full (OTIF) metrics calculation.
4. Smart Reorder AI Average Daily Consumption (ADC) and Dynamic ROP calculation.
5. Server-rendered UI routes for RTV, Disposal Board, and Smart Reorder.
"""

import pytest
from werkzeug.security import generate_password_hash

from departments.models.pharmacy import Batch, Drug, DrugCategory
from departments.models.stock_movement import StockMovement
from departments.models.stores import NonPharmCategory, NonPharmItem
from departments.models.supplier import Supplier
from departments.models.user import User
from extensions import db


@pytest.fixture
def gap_user(app):
    with app.app_context():
        u = User(
            username="sc_gap_user",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="pharmacy",
        )
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def gap_supplier(app):
    with app.app_context():
        s = Supplier(
            name="Global Pharma Distributors",
            contact_email="orders@globalpharma.co.ke",
            phone="0711111111",
            lead_time_days=4,
        )
        db.session.add(s)
        db.session.commit()
        yield s


@pytest.fixture
def gap_drug_and_item(app):
    with app.app_context():
        cat1 = DrugCategory(name="Vaccines")
        db.session.add(cat1)
        db.session.commit()

        d = Drug(
            generic_name="BCG Vaccine 20 Doses",
            category_id=cat1.id,
            dosage_form="Vial",
            strength="20 Doses",
            buying_price=100.0,
            selling_price=150.0,
            quantity_in_stock=50,
            reorder_level=10,
            storage_condition="Cold Chain (2-8°C)",
        )
        db.session.add(d)
        db.session.commit()

        cat2 = NonPharmCategory(name="Syringes")
        db.session.add(cat2)
        db.session.commit()

        np = NonPharmItem(
            name="Auto-Disable Syringe 0.5ml",
            category_id=cat2.id,
            unit="pieces",
            unit_cost=15.0,
            stock_level=100,
            storage_condition="Ambient",
        )
        db.session.add(np)
        db.session.commit()

        yield d.id, np.id


def test_rtv_creation_and_dispatch(
    client, app, gap_user, gap_supplier, gap_drug_and_item
):
    """Test creating draft RTV and dispatching it to supplier."""
    drug_id, np_id = gap_drug_and_item
    client.post(
        "/login", data={"username": gap_user.username, "password": "Password123!"}
    )

    # Create draft RTV
    create_resp = client.post(
        "/pharmacy/rtv/create",
        json={
            "supplier_id": gap_supplier.id,
            "reason": "Damaged Shipment",
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": drug_id,
                    "batch_number": "B-RTV-001",
                    "quantity": 10,
                    "unit_cost": 100.0,
                    "reason": "Vials broken in transit",
                },
                {
                    "item_type": "NON_PHARM",
                    "non_pharm_item_id": np_id,
                    "quantity": 20,
                    "unit_cost": 15.0,
                    "reason": "Defective packaging",
                },
            ],
        },
    )
    assert create_resp.status_code == 201
    rtv_data = create_resp.get_json()["supplier_return"]
    rtv_id = rtv_data["id"]
    assert rtv_data["status"] == "DRAFT"
    assert rtv_data["total_credit_amount"] == 1300.0

    # Dispatch RTV
    dispatch_resp = client.post(f"/pharmacy/rtv/{rtv_id}/dispatch")
    assert dispatch_resp.status_code == 200
    assert dispatch_resp.get_json()["supplier_return"]["status"] == "DISPATCHED"

    # Verify inventory deduction and StockMovement ledgers
    with app.app_context():
        refreshed_drug = db.session.get(Drug, drug_id)
        assert refreshed_drug.quantity_in_stock == 40  # 50 - 10

        refreshed_np = db.session.get(NonPharmItem, np_id)
        assert refreshed_np.stock_level == 80  # 100 - 20

        m1 = StockMovement.query.filter_by(
            item_type="DRUG", item_id=drug_id, movement_type="RETURN_TO_VENDOR"
        ).first()
        assert m1 is not None
        assert m1.quantity_delta == -10

        m2 = StockMovement.query.filter_by(
            item_type="NON_PHARM", item_id=np_id, movement_type="RETURN_TO_VENDOR"
        ).first()
        assert m2 is not None
        assert m2.quantity_delta == -20


def test_stock_disposal_creation_and_approval(client, app, gap_user, gap_drug_and_item):
    """Test draft stock disposal board creation and execution of write-off."""
    drug_id, np_id = gap_drug_and_item
    client.post(
        "/login", data={"username": gap_user.username, "password": "Password123!"}
    )

    with app.app_context():
        b = Batch(drug_id=drug_id, batch_number="B-EXP-001", quantity_in_stock=10)
        db.session.add(b)
        db.session.commit()
        batch_id = b.id

    # Create draft disposal
    create_resp = client.post(
        "/stores/disposal/create",
        json={
            "reason": "Expired Vaccine Destruction",
            "items": [
                {
                    "item_type": "DRUG",
                    "drug_id": drug_id,
                    "batch_id": batch_id,
                    "quantity": 5,
                    "unit_cost": 100.0,
                    "reason": "Expired",
                }
            ],
        },
    )
    assert create_resp.status_code == 201
    disp_data = create_resp.get_json()["disposal"]
    disp_id = disp_data["id"]
    assert disp_data["status"] == "DRAFT"
    assert disp_data["total_loss_value"] == 500.0

    # Approve disposal
    appr_resp = client.post(f"/stores/disposal/{disp_id}/approve")
    assert appr_resp.status_code == 200
    assert appr_resp.get_json()["disposal"]["status"] == "APPROVED"

    # Verify stock deduction and DISCARDED ledger movement
    with app.app_context():
        refreshed_drug = db.session.get(Drug, drug_id)
        assert refreshed_drug.quantity_in_stock == 45  # 50 - 5

        refreshed_batch = db.session.get(Batch, batch_id)
        assert refreshed_batch.quantity_in_stock == 5  # 10 - 5

        m = StockMovement.query.filter_by(
            item_type="DRUG", item_id=drug_id, movement_type="DISCARDED"
        ).first()
        assert m is not None
        assert m.quantity_delta == -5


def test_supplier_otif_metrics_api(client, gap_user, gap_supplier):
    """Test GET /pharmacy/suppliers/<id>/metrics returns OTIF metrics."""
    client.post(
        "/login", data={"username": gap_user.username, "password": "Password123!"}
    )

    resp = client.get(f"/pharmacy/suppliers/{gap_supplier.id}/metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["supplier_id"] == gap_supplier.id
    assert "promised_lead_time_days" in data
    assert "actual_avg_lead_time_days" in data
    assert "on_time_delivery_rate" in data
    assert "fill_rate_percentage" in data


def test_smart_reorder_calculation(client, gap_user, gap_drug_and_item):
    """Test GET /pharmacy/smart-reorder returns ADC & dynamic ROP proposals."""
    client.post(
        "/login", data={"username": gap_user.username, "password": "Password123!"}
    )

    resp = client.get("/pharmacy/smart-reorder")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "total_items_analyzed" in data
    assert "proposals" in data
    assert len(data["proposals"]) >= 2
    assert any(
        p["storage_condition"] == "Cold Chain (2-8°C)" for p in data["proposals"]
    )


def test_gaps_ui_routes(client, gap_user):
    """Test UI rendering of RTV log, RTV new, Disposal Board, and Smart Reorder pages."""
    client.post(
        "/login", data={"username": gap_user.username, "password": "Password123!"}
    )

    r1 = client.get("/stores/rtv")
    assert r1.status_code == 200
    assert b"Return to Vendor" in r1.data

    r2 = client.get("/stores/rtv/new")
    assert r2.status_code == 200
    assert b"Initiate Return to Vendor" in r2.data

    r3 = client.get("/stores/disposals")
    assert r3.status_code == 200
    assert (
        b"Quarantine &amp; Stock Disposal Board" in r3.data
        or b"Quarantine & Stock Disposal Board" in r3.data
    )

    r4 = client.get("/stores/smart-reorder")
    assert r4.status_code == 200
    assert (
        b"Smart Reorder &amp; Consumption Analytics" in r4.data
        or b"Smart Reorder & Consumption Analytics" in r4.data
    )
