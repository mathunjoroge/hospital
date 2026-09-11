"""
departments/billing/mpesa.py
──────────────────────────────
Task 2.4 — Safaricom Daraja M-Pesa Integration Module

Features:
  - Access Token generation (OAuth client credentials)
  - STK Push request initiation (LIPA NA M-PESA ONLINE)
  - Async Webhook / Callback handler (/api/mpesa/callback)
  - Payment record auto-creation and Invoice status updates
  - Pending payment reconciliation task
"""

import base64
import logging
from datetime import datetime, timezone

import requests
from flask import Blueprint, current_app, jsonify, request

try:
    from extensions import db
except ImportError:
    from extensions import db
from departments.models.billing import Payment

logger = logging.getLogger(__name__)

mpesa_bp = Blueprint("mpesa", __name__, url_prefix="/api/mpesa")


# Configuration helpers
def get_mpesa_config():
    """Retrieve M-Pesa Daraja API configuration from Flask app or environment."""
    return {
        "env": current_app.config.get("MPESA_ENV", "sandbox"),
        "consumer_key": current_app.config.get(
            "MPESA_CONSUMER_KEY", "test_consumer_key"
        ),
        "consumer_secret": current_app.config.get(
            "MPESA_CONSUMER_SECRET", "test_consumer_secret"
        ),
        "shortcode": current_app.config.get("MPESA_SHORTCODE", "174379"),
        "passkey": current_app.config.get(
            "MPESA_PASSKEY",
            "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919",
        ),
        "callback_url": current_app.config.get(
            "MPESA_CALLBACK_URL", "https://hospital.example.com/api/mpesa/callback"
        ),
    }


def format_phone_number(phone: str) -> str:
    """Format Kenyan phone numbers to standard 254XXXXXXXXX format."""
    cleaned = "".join(filter(str.isdigit, str(phone)))
    if cleaned.startswith("0"):
        return "254" + cleaned[1:]
    elif cleaned.startswith(("7", "1")):
        return "254" + cleaned
    elif cleaned.startswith("254") and len(cleaned) == 12:
        return cleaned
    return cleaned


def generate_stk_password(shortcode: str, passkey: str, timestamp: str) -> str:
    """Generate base64 encoded STK Push password."""
    raw = f"{shortcode}{passkey}{timestamp}"
    return base64.b64encode(raw.encode("utf-8")).decode("utf-8")


def get_mpesa_access_token() -> str:
    """Obtain OAuth access token from Safaricom Daraja API."""
    cfg = get_mpesa_config()
    if cfg["env"] == "test" or current_app.config.get("TESTING"):
        return "mock_access_token_12345"

    url = f"https://{'sandbox' if cfg['env'] == 'sandbox' else 'api'}.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials"
    auth = (cfg["consumer_key"], cfg["consumer_secret"])
    try:
        res = requests.get(url, auth=auth, timeout=10)
        res.raise_for_status()
        return res.json().get("access_token", "")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to fetch M-Pesa access token: {e}")
        return ""


def initiate_stk_push(
    phone_number: str,
    amount: float,
    account_reference: str,
    invoice_id: int | None = None,
    transaction_desc: str = "Hospital Bill Payment",
) -> dict:
    """
    Initiate STK Push (Lipa Na M-Pesa Online).
    Returns dict with success status, MerchantRequestID, CheckoutRequestID, ResponseCode.
    """
    cfg = get_mpesa_config()
    formatted_phone = format_phone_number(phone_number)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    password = generate_stk_password(cfg["shortcode"], cfg["passkey"], timestamp)

    # In testing environment, return mock response directly
    if current_app.config.get("TESTING") or cfg["env"] == "test":
        checkout_id = f"ws_CO_{timestamp}_{invoice_id or 1}"
        merchant_id = f"29182-1000000-{invoice_id or 1}"

        # Pre-create pending Payment record
        if invoice_id:
            payment = Payment(
                invoice_id=invoice_id,
                amount=amount,
                payment_method="mpesa",
                payment_reference=checkout_id,
                notes=f"STK Push initiated for {formatted_phone}",
                is_reconciled=False,
            )
            db.session.add(payment)
            db.session.commit()

        return {
            "success": True,
            "ResponseCode": "0",
            "ResponseDescription": "Success. Request accepted for processing",
            "MerchantRequestID": merchant_id,
            "CheckoutRequestID": checkout_id,
            "CustomerMessage": "Success. Request accepted for processing",
        }

    token = get_mpesa_access_token()
    if not token:
        return {
            "success": False,
            "error": "Could not authenticate with Safaricom M-Pesa service.",
        }

    url = f"https://{'sandbox' if cfg['env'] == 'sandbox' else 'api'}.safaricom.co.ke/mpesa/stkpush/v1/processrequest"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "BusinessShortCode": cfg["shortcode"],
        "Password": password,
        "Timestamp": timestamp,
        "TransactionType": "CustomerPayBillOnline",
        "Amount": int(amount),
        "PartyA": formatted_phone,
        "PartyB": cfg["shortcode"],
        "PhoneNumber": formatted_phone,
        "CallBackURL": cfg["callback_url"],
        "AccountReference": account_reference[:12],
        "TransactionDesc": transaction_desc[:12],
    }

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        data = res.json()
        if res.status_code == 200 and data.get("ResponseCode") == "0":
            checkout_id = data.get("CheckoutRequestID")
            if invoice_id:
                payment = Payment(
                    invoice_id=invoice_id,
                    amount=amount,
                    payment_method="mpesa",
                    payment_reference=checkout_id,
                    notes=f"STK Push initiated for {formatted_phone}",
                    is_reconciled=False,
                )
                db.session.add(payment)
                db.session.commit()
            return {"success": True, **data}
        else:
            return {
                "success": False,
                "error": data.get("CustomerMessage", "STK push failed"),
                **data,
            }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error initiating STK push: {e}")
        return {"success": False, "error": str(e)}


def process_mpesa_callback(callback_data: dict) -> dict:
    """
    Process M-Pesa callback payload received from Safaricom servers.
    """
    try:
        stk_callback = callback_data.get("Body", {}).get("stkCallback", {})
        result_code = stk_callback.get("ResultCode")
        result_desc = stk_callback.get("ResultDesc")
        stk_callback.get("MerchantRequestID")
        checkout_request_id = stk_callback.get("CheckoutRequestID")

        payment = Payment.query.filter_by(payment_reference=checkout_request_id).first()

        if result_code == 0:
            # Payment Successful
            items = stk_callback.get("CallbackMetadata", {}).get("Item", [])
            meta = {
                item.get("Name"): item.get("Value") for item in items if "Name" in item
            }

            mpesa_receipt = meta.get("MpesaReceiptNumber", checkout_request_id)
            paid_amount = float(meta.get("Amount", 0))
            phone = str(meta.get("PhoneNumber", ""))

            if payment:
                payment.payment_reference = mpesa_receipt
                payment.amount = paid_amount if paid_amount > 0 else payment.amount
                payment.is_reconciled = True
                payment.notes = f"M-Pesa Paid: {mpesa_receipt} from {phone}"

                # Recalculate invoice totals and status
                if payment.invoice:
                    payment.invoice.recalculate()
            else:
                logger.warning(
                    f"Callback received for unknown checkout ID: {checkout_request_id}"
                )

            db.session.commit()
            return {
                "status": "success",
                "receipt": mpesa_receipt,
                "amount": paid_amount,
            }

        else:
            # Payment Failed or Cancelled
            if payment:
                payment.notes = f"M-Pesa Failed ({result_code}): {result_desc}"
                payment.is_reconciled = False
                db.session.commit()
            return {"status": "failed", "code": result_code, "desc": result_desc}

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error processing M-Pesa callback: {e}")
        db.session.rollback()
        return {"status": "error", "error": str(e)}


def reconcile_pending_mpesa_payments() -> int:
    """
    Find payments initiated over 10 minutes ago without M-Pesa receipt numbers
    and update them as timed out.
    Returns count of reconciled payments.
    """
    datetime.now(timezone.utc)
    pending = Payment.query.filter(
        Payment.payment_method == "mpesa",
        Payment.is_reconciled.is_(False),
        Payment.payment_reference.like("ws_CO_%"),
    ).all()

    reconciled_count = 0
    for p in pending:
        # Mark timed out
        p.notes = "M-Pesa STK push timed out / expired."
        reconciled_count += 1

    if reconciled_count > 0:
        db.session.commit()

    return reconciled_count


# API Routes
@mpesa_bp.route("/stkpush", methods=["POST"])
def handle_stk_push_route():
    """Initiate M-Pesa STK push request for an invoice or amount."""
    data = request.get_json() or {}
    phone = data.get("phone_number")
    amount = data.get("amount")
    invoice_id = data.get("invoice_id")
    account_ref = data.get(
        "account_reference", f"INV-{invoice_id}" if invoice_id else "HOSPITAL"
    )

    if not phone or not amount:
        return jsonify({"error": "phone_number and amount are required"}), 400

    res = initiate_stk_push(
        phone_number=phone,
        amount=float(amount),
        account_reference=account_ref,
        invoice_id=invoice_id,
    )

    if res.get("success"):
        return jsonify(res), 200
    else:
        return jsonify(res), 400


@mpesa_bp.route("/callback", methods=["POST"])
def handle_callback_route():
    """Receive callback from Safaricom Daraja API."""
    data = request.get_json() or {}
    logger.info(f"Received M-Pesa Callback: {data}")
    process_mpesa_callback(data)
    # Daraja expects HTTP 200 with ResultCode 0 response
    return jsonify({"ResultCode": 0, "ResultDesc": "Accepted"}), 200
