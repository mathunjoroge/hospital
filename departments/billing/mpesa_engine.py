"""
M-Pesa (Safaricom Daraja API) Integration Engine.
Handles STK Push initiation, transaction tracking, and asynchronous callbacks.
"""

import base64
import logging
import os
from datetime import datetime, timezone

import requests

from extensions import db

logger = logging.getLogger(__name__)


class MpesaTransaction(db.Model):
    """Tracks M-Pesa STK Push transactions and their lifecycle."""

    __tablename__ = "mpesa_transactions"

    id = db.Column(db.Integer, primary_key=True)
    reference = db.Column(
        db.String(50), nullable=False, index=True
    )  # Invoice # or Patient ID
    phone_number = db.Column(db.String(15), nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)

    # Daraja specific IDs
    merchant_request_id = db.Column(db.String(50), nullable=True)
    checkout_request_id = db.Column(
        db.String(50), nullable=True, unique=True, index=True
    )
    mpesa_receipt_number = db.Column(db.String(50), nullable=True, unique=True)

    # PENDING, COMPLETED, FAILED, CANCELLED
    status = db.Column(db.String(20), nullable=False, default="PENDING")
    result_desc = db.Column(db.Text, nullable=True)

    initiated_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)


class MpesaService:
    """Service class for interacting with the Safaricom Daraja API."""

    def __init__(self):
        self.consumer_key = os.getenv("MPESA_CONSUMER_KEY")
        self.consumer_secret = os.getenv("MPESA_CONSUMER_SECRET")
        self.shortcode = os.getenv("MPESA_BUSINESS_SHORTCODE")
        self.passkey = os.getenv("MPESA_PASSKEY")
        self.callback_url = os.getenv("MPESA_CALLBACK_URL")

        env = os.getenv("MPESA_ENV", "sandbox")
        self.base_url = (
            "https://api.safaricom.co.ke"
            if env == "production"
            else "https://sandbox.safaricom.co.ke"
        )

    def _get_access_token(self) -> str:
        """Generates an OAuth2 Bearer Token from Daraja."""
        url = f"{self.base_url}/oauth/v1/generate?grant_type=client_credentials"
        response = requests.get(
            url, auth=(self.consumer_key, self.consumer_secret), timeout=10
        )
        response.raise_for_status()
        return response.json().get("access_token")

    def _generate_password(self, timestamp: str) -> str:
        """Generates the Base64 encoded password required for STK Push."""
        raw_str = f"{self.shortcode}{self.passkey}{timestamp}"
        return base64.b64encode(raw_str.encode("utf-8")).decode("utf-8")

    def initiate_stk_push(
        self, phone_number: str, amount: float, reference: str
    ) -> MpesaTransaction:
        """
        Triggers an STK Push to the patient's phone.
        Returns the MpesaTransaction record.
        """
        if not all(
            [self.consumer_key, self.shortcode, self.passkey, self.callback_url]
        ):
            raise ValueError(
                "M-Pesa credentials are not fully configured in the environment."
            )

        # Format phone number to 2547XXXXXXXX
        phone_number = phone_number.replace(" ", "").replace("-", "")
        if phone_number.startswith("0"):
            phone_number = "254" + phone_number[1:]
        elif phone_number.startswith("+"):
            phone_number = phone_number[1:]
        elif not phone_number.startswith("254"):
            phone_number = "254" + phone_number

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        password = self._generate_password(timestamp)
        access_token = self._get_access_token()

        headers = {"Authorization": f"Bearer {access_token}"}
        payload = {
            "BusinessShortCode": self.shortcode,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": int(amount),  # Daraja expects integer amount
            "PartyA": phone_number,
            "PartyB": self.shortcode,
            "PhoneNumber": phone_number,
            "CallBackURL": self.callback_url,
            "AccountReference": reference[:12],  # Daraja limit is 12 chars
            "TransactionDesc": f"Payment for {reference}",
        }

        url = f"{self.base_url}/mpesa/stkpush/v1/processrequest"
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("ResponseCode") != "0":
            raise ValueError(f"Daraja API Error: {data.get('ResponseDescription')}")

        transaction = MpesaTransaction(
            phone_number=phone_number,
            amount=amount,
            reference=reference,
            merchant_request_id=data.get("MerchantRequestID"),
            checkout_request_id=data.get("CheckoutRequestID"),
            status="PENDING",
        )
        db.session.add(transaction)
        db.session.commit()

        logger.info(
            "STK Push initiated for %s, amount %s. CheckoutID: %s",
            phone_number,
            amount,
            transaction.checkout_request_id,
        )
        return transaction

    def process_callback(self, payload: dict) -> bool:
        """
        Processes the asynchronous callback from Safaricom.
        Updates the transaction status based on the payment result.
        """
        body = payload.get("Body", {}).get("stkCallback", {})
        checkout_request_id = body.get("CheckoutRequestID")

        transaction = MpesaTransaction.query.filter_by(
            checkout_request_id=checkout_request_id
        ).first()
        if not transaction:
            logger.warning(
                "Received M-Pesa callback for unknown CheckoutRequestID: %s",
                checkout_request_id,
            )
            return False

        result_code = body.get("ResultCode")
        result_desc = body.get("ResultDesc")

        if result_code == 0:
            # Success
            items = body.get("CallbackMetadata", {}).get("Item", [])
            receipt_number = None
            for item in items:
                if item.get("Name") == "Mpesa Receipt Number":
                    receipt_number = item.get("Value")
                    break

            transaction.status = "COMPLETED"
            transaction.mpesa_receipt_number = receipt_number
            transaction.completed_at = datetime.now(timezone.utc)
            logger.info(
                "M-Pesa Payment COMPLETED for %s. Receipt: %s",
                transaction.phone_number,
                receipt_number,
            )
        else:
            transaction.status = "FAILED"
            logger.warning(
                "M-Pesa Payment FAILED for %s. Reason: %s",
                transaction.phone_number,
                result_desc,
            )

        transaction.result_desc = result_desc
        db.session.commit()
        return True
