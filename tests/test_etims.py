"""
tests/test_etims.py
───────────────────
Unit and integration tests for KRA eTIMS Tax Compliance & Fiscalization engine.
"""

from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from departments.billing.etims_service import (
    fiscalize_invoice,
    get_or_create_etims_config,
    ping_etims_vscu_connection,
    update_etims_config,
)
from departments.models import Patient
from departments.models.billing import Invoice, InvoiceStatus
from departments.models.user import User
from extensions import db


@pytest.fixture
def admin_username(app):
    with app.app_context():
        user = User.query.filter_by(username="etims_admin").first()
        if not user:
            user = User(
                username="etims_admin",
                role="admin",
                password=generate_password_hash("Password123!"),
            )
            db.session.add(user)
            db.session.commit()
        return "etims_admin"


def test_etims_config_service(app):
    """Test getting and updating KRA eTIMS configuration."""
    with app.app_context():
        config = get_or_create_etims_config()
        assert config.kra_pin == "P051234567A"

        updated = update_etims_config(
            kra_pin="P059999999Z",
            branch_code="01",
            device_serial="OSCU-TEST-9999",
            cmc_key="NEW-CMC-KEY",
            vscu_server_url="https://etims-api.kra.go.ke/test",
            is_sandbox=True,
            enabled=True,
        )
        assert updated.kra_pin == "P059999999Z"
        assert updated.device_serial == "OSCU-TEST-9999"


def test_etims_connection_ping(app):
    """Test VSCU connection ping service."""
    with app.app_context():
        success, msg = ping_etims_vscu_connection()
        assert success is True
        assert "KRA eTIMS" in msg


def test_etims_admin_console_route(client, admin_username):
    """Test GET /admin/etims renders the administration console."""
    client.post("/login", data={"username": admin_username, "password": "Password123!"})
    res = client.get("/admin/etims")
    assert res.status_code == 200
    assert b"KRA eTIMS" in res.data
    assert b"Facility Registration Credentials" in res.data


def test_etims_config_save_api(client, admin_username):
    """Test POST /admin/etims/config saves credentials."""
    client.post("/login", data={"username": admin_username, "password": "Password123!"})
    payload = {
        "kra_pin": "P058888888B",
        "branch_code": "00",
        "device_serial": "VSCU-DEVICE-777",
        "cmc_key": "CMC-SECRET",
        "vscu_server_url": "https://etims-api.kra.go.ke/etims-api/v1",
        "is_sandbox": True,
        "enabled": True,
    }
    res = client.post(
        "/admin/etims/config", json=payload, headers={"Accept": "application/json"}
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["config"]["kra_pin"] == "P058888888B"


def test_invoice_fiscalization(app):
    """Test fiscalizing a settled invoice generates KRA Control Unit QR Code."""
    with app.app_context():
        patient = Patient(
            patient_id="P_ETIMS_01",
            first_name="John",
            last_name="Doe",
            gender="Male",
            date_of_birth=date(1990, 1, 1),
        )
        db.session.add(patient)
        db.session.commit()

        invoice = Invoice(
            patient_id=patient.patient_id,
            grand_total=1500.0,
            amount_paid=1500.0,
            balance=0.0,
            status=InvoiceStatus.PAID,
        )
        db.session.add(invoice)
        db.session.commit()

        receipt = fiscalize_invoice(invoice.id)
        assert receipt is not None
        assert receipt.invoice_id == invoice.id
        assert "KRA-" in receipt.cu_invoice_number
        assert "verifyInvoice.htm" in receipt.qr_code_url
        assert len(receipt.fiscal_signature) == 64
