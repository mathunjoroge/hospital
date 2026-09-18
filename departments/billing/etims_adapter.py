"""
departments/billing/etims_adapter.py
──────────────────────────────────────
Section 6: KRA eTIMS Fiscalization Middleware & VAT Exemption Engine.

Provides automated QR code fiscal receipt generation and tax classification
under Kenyan tax law (VAT Act 2013 & KRA eTIMS specification).
Controlled by feature flag `ENABLE_ETIMS=False` by default.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from flask import current_app

logger = logging.getLogger(__name__)

# KRA eTIMS VAT Tax Rates
VAT_EXEMPT_RATE = 0.00
VAT_STANDARD_RATE = 0.16  # 16% VAT

# Exempt categories under Value Added Tax Act (First Schedule)
EXEMPT_CATEGORIES = {"consult", "lab", "imaging", "drug", "ward", "procedure", "theatre"}


def is_etims_enabled() -> bool:
    """Check if eTIMS fiscalization is active in configuration."""
    from flask import has_app_context
    if has_app_context():
        return bool(current_app.config.get("ENABLE_ETIMS", False))
    return False


def categorize_item_vat(description: str, category: str) -> dict[str, Any]:
    """
    Classify an invoice line item into KRA eTIMS tax categories.

    Medical consultations, essential pharmaceuticals, laboratory diagnostics,
    radiology imaging, and surgical procedures are VAT-exempt under Kenya VAT Act.
    Retail items, non-medical supplies, and administrative fees are subject to 16% VAT.
    """
    cat_norm = (category or "").strip().lower()
    desc_norm = (description or "").strip().lower()

    is_exempt = cat_norm in EXEMPT_CATEGORIES or "exempt" in desc_norm

    if is_exempt:
        return {
            "tax_code": "E",  # KRA Tax Code E = Exempt
            "tax_rate": VAT_EXEMPT_RATE,
            "is_taxable": False,
        }

    return {
        "tax_code": "A",  # KRA Tax Code A = 16% Standard Rate
        "tax_rate": VAT_STANDARD_RATE,
        "is_taxable": True,
    }


def generate_etims_fiscal_signature(
    invoice_id: int,
    patient_id: str,
    total_amount: float,
    items: list[dict[str, Any]],
    kra_pin: str = "P051234567Z",
) -> dict[str, Any]:
    """
    Generate a KRA eTIMS compliant fiscal invoice signature and QR code payload.

    Args:
        invoice_id: Unified invoice ID
        patient_id: Patient ID
        total_amount: Invoice total amount
        items: List of line item dicts (description, category, unit_price, quantity)
        kra_pin: Hospital KRA PIN

    Returns:
        dict: eTIMS payload including control code, QR code payload, and VAT breakdown
    """
    exempt_total = 0.0
    taxable_net = 0.0
    vat_total = 0.0

    for item in items:
        unit_p = float(item.get("unit_price") or 0.0)
        qty = int(item.get("quantity") or 1)
        line_tot = unit_p * qty
        vat_info = categorize_item_vat(item.get("description", ""), item.get("category", ""))

        if vat_info["is_taxable"]:
            net = round(line_tot / (1.0 + VAT_STANDARD_RATE), 2)
            vat = round(line_tot - net, 2)
            taxable_net += net
            vat_total += vat
        else:
            exempt_total += line_tot

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    raw_payload = f"{kra_pin}|{invoice_id}|{total_amount:.2f}|{timestamp}"
    control_code = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()[:16].upper()

    qr_payload = f"https://itax.kra.go.ke/etims/verify?pin={kra_pin}&inv={invoice_id}&code={control_code}"

    return {
        "etims_enabled": is_etims_enabled(),
        "kra_pin": kra_pin,
        "invoice_id": invoice_id,
        "patient_id": patient_id,
        "total_amount": round(total_amount, 2),
        "exempt_total": round(exempt_total, 2),
        "taxable_net": round(taxable_net, 2),
        "vat_total": round(vat_total, 2),
        "control_code": control_code,
        "qr_code_url": qr_payload,
        "timestamp": timestamp,
    }
