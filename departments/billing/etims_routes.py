"""
departments/billing/etims_routes.py
───────────────────────────────────
Flask HTTP routes for KRA eTIMS Tax Compliance & Fiscalization Admin Console.

Endpoints:
  GET  /admin/etims               – KRA eTIMS Admin UI Console
  POST /admin/etims/config        – Save KRA PIN & VSCU credentials
  POST /admin/etims/test          – Test KRA VSCU API connectivity
  POST /admin/etims/fiscalize/<id>– Fiscalize invoice & return KRA QR code
"""

import logging

from flask import Blueprint, jsonify, render_template, request
from flask_login import login_required

from departments.models.billing import EtimsFiscalReceipt
from departments.rbac import roles_required

from .etims_service import (
    fiscalize_invoice,
    get_or_create_etims_config,
    ping_etims_vscu_connection,
    update_etims_config,
)

etims_bp = Blueprint("etims", __name__, url_prefix="/admin/etims")
logger = logging.getLogger(__name__)


@etims_bp.route("", methods=["GET"])
@login_required
@roles_required("admin", "billing")
def etims_console():
    """
    GET /admin/etims
    Render the KRA eTIMS Tax Compliance & Registration Console.
    """
    config = get_or_create_etims_config()
    recent_receipts = (
        EtimsFiscalReceipt.query.order_by(EtimsFiscalReceipt.fiscalized_at.desc())
        .limit(20)
        .all()
    )
    total_fiscalized = EtimsFiscalReceipt.query.count()

    return render_template(
        "admin/etims.html",
        config=config,
        recent_receipts=recent_receipts,
        total_fiscalized=total_fiscalized,
    )


@etims_bp.route("/config", methods=["POST"])
@login_required
@roles_required("admin")
def save_etims_config():
    """
    POST /admin/etims/config
    Save or update KRA PIN, branch code, device serial, and VSCU credentials.
    """
    data = request.get_json(silent=True) or request.form

    kra_pin = data.get("kra_pin")
    branch_code = data.get("branch_code", "00")
    device_serial = data.get("device_serial")
    cmc_key = data.get("cmc_key")
    vscu_server_url = data.get("vscu_server_url")
    is_sandbox = str(data.get("is_sandbox")).lower() in ("true", "1", "on")
    enabled = str(data.get("enabled")).lower() in ("true", "1", "on")
    exemptions_note = data.get("exemptions_note")

    if not kra_pin or not device_serial:
        return jsonify(
            {"error": "KRA PIN and Control Unit Serial Number are required."}
        ), 400

    config = update_etims_config(
        kra_pin=kra_pin,
        branch_code=branch_code,
        device_serial=device_serial,
        cmc_key=cmc_key,
        vscu_server_url=vscu_server_url,
        is_sandbox=is_sandbox,
        enabled=enabled,
        exemptions_note=exemptions_note,
    )

    if request.headers.get("Accept") == "application/json" or request.is_json:
        return jsonify(
            {
                "status": "success",
                "message": f"KRA eTIMS configuration saved successfully for PIN {config.kra_pin}",
                "config": {
                    "kra_pin": config.kra_pin,
                    "branch_code": config.branch_code,
                    "device_serial": config.device_serial,
                    "is_sandbox": config.is_sandbox,
                    "enabled": config.enabled,
                },
            }
        ), 200

    return jsonify({"status": "success", "message": "Settings saved."})


@etims_bp.route("/test", methods=["POST"])
@login_required
@roles_required("admin", "billing")
def test_connection():
    """
    POST /admin/etims/test
    Ping / test connection to the configured KRA eTIMS VSCU endpoint.
    """
    success, message = ping_etims_vscu_connection()
    status_code = 200 if success else 400
    return jsonify({"success": success, "message": message}), status_code


@etims_bp.route("/fiscalize/<int:invoice_id>", methods=["POST"])
@login_required
@roles_required("admin", "billing", "records")
def fiscalize_invoice_api(invoice_id: int):
    """
    POST /admin/etims/fiscalize/<invoice_id>
    Fiscalize an invoice and return KRA Control Unit QR Code payload.
    """
    try:
        receipt = fiscalize_invoice(invoice_id)
        return jsonify(
            {
                "status": "success",
                "message": "Invoice fiscalized with KRA eTIMS VSCU.",
                "cu_invoice_number": receipt.cu_invoice_number,
                "cu_serial_number": receipt.cu_serial_number,
                "qr_code_url": receipt.qr_code_url,
                "fiscal_signature": receipt.fiscal_signature,
                "fiscalized_at": receipt.fiscalized_at.isoformat(),
            }
        ), 200
    except Exception as exc:
        logger.exception("Error fiscalizing invoice #%d", invoice_id)
        return jsonify({"error": str(exc)}), 422
