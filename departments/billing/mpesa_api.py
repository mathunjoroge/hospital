"""
M-Pesa API Routes.
Provides endpoints for initiating STK Push, polling status, and receiving callbacks.
"""

from flask import Blueprint, current_app, jsonify, request
from flask_login import login_required

from extensions import csrf

from .mpesa_engine import MpesaService, MpesaTransaction

mpesa_api_bp = Blueprint("mpesa_api", __name__, url_prefix="/billing/mpesa")


@mpesa_api_bp.route("/initiate", methods=["POST"])
@login_required
def initiate_payment():
    """Triggers an STK Push to the patient's phone."""
    data = request.get_json(silent=True) or {}
    phone = data.get("phone_number")
    amount = data.get("amount")
    reference = data.get("reference")

    if not all([phone, amount, reference]):
        return (
            jsonify({"error": "phone_number, amount, and reference are required"}),
            400,
        )

    try:
        service = MpesaService()
        transaction = service.initiate_stk_push(phone, float(amount), str(reference))
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "STK Push sent to phone. Please enter PIN.",
                    "checkout_request_id": transaction.checkout_request_id,
                }
            ),
            201,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        current_app.logger.exception("M-Pesa initiation failed")
        return jsonify({"error": "Failed to initiate payment."}), 500


@mpesa_api_bp.route("/status/<string:checkout_request_id>", methods=["GET"])
@login_required
def check_status(checkout_request_id: str):
    """Polls the status of an STK Push transaction."""
    transaction = MpesaTransaction.query.filter_by(
        checkout_request_id=checkout_request_id
    ).first()
    if not transaction:
        return jsonify({"error": "Transaction not found"}), 404

    return (
        jsonify(
            {
                "status": transaction.status,
                "receipt_number": transaction.mpesa_receipt_number,
                "amount": float(transaction.amount),
                "reference": transaction.reference,
            }
        ),
        200,
    )


@mpesa_api_bp.route("/callback", methods=["POST"])
@csrf.exempt
def mpesa_callback():
    """
    Receives asynchronous payment confirmations from Safaricom.
    MUST be exempt from CSRF and login requirements.
    """
    payload = request.get_json(silent=True) or {}
    try:
        service = MpesaService()
        success = service.process_callback(payload)
        if success:
            return jsonify({"ResultCode": "0", "ResultDesc": "Success"}), 200
        return jsonify({"ResultCode": "1", "ResultDesc": "Failed"}), 200
    except Exception:
        current_app.logger.exception("M-Pesa callback processing failed")
        return jsonify({"ResultCode": "1", "ResultDesc": "Internal Error"}), 500
