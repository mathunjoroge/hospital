"""
tests/test_mpesa.py
───────────────────
Unit tests for Task 2.4: Safaricom Daraja M-Pesa Integration
"""

from datetime import date

import pytest

from departments.billing.mpesa import (
    format_phone_number,
    generate_stk_password,
    initiate_stk_push,
    process_mpesa_callback,
    reconcile_pending_mpesa_payments,
)
from departments.models.billing import Invoice, InvoiceLineItem, InvoiceStatus, Payment
from departments.models.records import Patient
from extensions import db


@pytest.fixture
def sample_invoice(app):
    patient = Patient(
        patient_id="PTMPESA01",
        name="M-Pesa Test Patient",
        place_of_residence="Nairobi",
        sex="Female",
        date_of_birth=date(1995, 5, 5),
        marital_status="Single",
        contact="0700000000",
        next_of_kin="Next Kin",
        relationship_with_next_of_kin="Parent",
        next_of_kin_contact="0711111111",
        emergency_contact="0722222222"
    )
    db.session.add(patient)
    db.session.commit()

    inv = Invoice(patient_id=patient.id, total_amount=2500.0, balance_due=2500.0, status="UNPAID")
    db.session.add(inv)
    db.session.commit()

    item = InvoiceLineItem(invoice_id=inv.id, description="Consultation", amount=2500.0)
    db.session.add(item)
    db.session.commit()
    inv.recalculate()
    return inv


class TestPhoneFormatting:
    def test_format_07_prefix(self):
        assert format_phone_number("0712345678") == "254712345678"

    def test_format_01_prefix(self):
        assert format_phone_number("0112345678") == "254112345678"

    def test_format_254_prefix(self):
        assert format_phone_number("254712345678") == "254712345678"

    def test_format_with_plus(self):
        assert format_phone_number("+254712345678") == "254712345678"


class TestStkPassword:
    def test_password_generation(self):
        pwd = generate_stk_password("174379", "passkey123", "20260905120000")
        assert isinstance(pwd, str)
        assert len(pwd) > 0


class TestStkPushInitiation:
    def test_initiate_stk_push_creates_pending_payment(self, app, sample_invoice):
        res = initiate_stk_push(
            phone_number="0712345678",
            amount=2500.0,
            account_reference=f"INV-{sample_invoice.id}",
            invoice_id=sample_invoice.id
        )
        assert res["success"] is True
        assert "CheckoutRequestID" in res

        # Verify pending payment created in database
        payment = Payment.query.filter_by(invoice_id=sample_invoice.id).first()
        assert payment is not None
        assert payment.payment_method == "mpesa"
        assert payment.is_reconciled is False
        assert payment.amount == 2500.0


class TestMpesaCallbackProcessing:
    def test_successful_callback_reconciles_payment(self, app, sample_invoice):
        # 1. Initiate STK Push
        res = initiate_stk_push("0712345678", 2500.0, "INV-1", sample_invoice.id)
        checkout_id = res["CheckoutRequestID"]

        # 2. Simulate Safaricom Success Callback
        callback_payload = {
            "Body": {
                "stkCallback": {
                    "MerchantRequestID": "29182-1000000-1",
                    "CheckoutRequestID": checkout_id,
                    "ResultCode": 0,
                    "ResultDesc": "The service request is processed successfully.",
                    "CallbackMetadata": {
                        "Item": [
                            {"Name": "Amount", "Value": 2500.0},
                            {"Name": "MpesaReceiptNumber", "Value": "QHK1234567"},
                            {"Name": "TransactionDate", "Value": 20260905120000},
                            {"Name": "PhoneNumber", "Value": 254712345678}
                        ]
                    }
                }
            }
        }

        result = process_mpesa_callback(callback_payload)
        assert result["status"] == "success"
        assert result["receipt"] == "QHK1234567"

        # 3. Verify Payment & Invoice State
        payment = Payment.query.filter_by(payment_reference="QHK1234567").first()
        assert payment is not None
        assert payment.is_reconciled is True

        sample_invoice.recalculate()
        assert sample_invoice.status in ("PAID", "paid", InvoiceStatus.PAID) or str(sample_invoice.status).upper() == "PAID"
        assert sample_invoice.balance_due == 0.0

    def test_failed_callback_updates_notes(self, app, sample_invoice):
        res = initiate_stk_push("0712345678", 2500.0, "INV-1", sample_invoice.id)
        checkout_id = res["CheckoutRequestID"]

        callback_payload = {
            "Body": {
                "stkCallback": {
                    "MerchantRequestID": "29182-1000000-1",
                    "CheckoutRequestID": checkout_id,
                    "ResultCode": 1032,
                    "ResultDesc": "Request cancelled by user."
                }
            }
        }

        result = process_mpesa_callback(callback_payload)
        assert result["status"] == "failed"

        payment = Payment.query.filter_by(payment_reference=checkout_id).first()
        assert payment is not None
        assert payment.is_reconciled is False
        assert "cancelled by user" in payment.notes


class TestReconciliation:
    def test_reconcile_pending_payments(self, app, sample_invoice):
        initiate_stk_push("0712345678", 2500.0, "INV-1", sample_invoice.id)
        count = reconcile_pending_mpesa_payments()
        assert count == 1


class TestMpesaEndpoints:
    def test_stkpush_api_endpoint(self, client, sample_invoice):
        resp = client.post('/api/mpesa/stkpush', json={
            "phone_number": "0712345678",
            "amount": 1500.0,
            "invoice_id": sample_invoice.id
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_callback_api_endpoint(self, client, sample_invoice):
        res = initiate_stk_push("0712345678", 2500.0, "INV-1", sample_invoice.id)
        checkout_id = res["CheckoutRequestID"]

        callback_payload = {
            "Body": {
                "stkCallback": {
                    "MerchantRequestID": "29182-1000000-1",
                    "CheckoutRequestID": checkout_id,
                    "ResultCode": 0,
                    "ResultDesc": "Success",
                    "CallbackMetadata": {
                        "Item": [
                            {"Name": "Amount", "Value": 2500.0},
                            {"Name": "MpesaReceiptNumber", "Value": "QHK7654321"},
                            {"Name": "PhoneNumber", "Value": 254712345678}
                        ]
                    }
                }
            }
        }

        resp = client.post('/api/mpesa/callback', json=callback_payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ResultCode"] == 0
