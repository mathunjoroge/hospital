"""
tests/test_etims_adapter.py
────────────────────────────
Unit tests for KRA eTIMS fiscalization & VAT classification adapter (Section 6).
"""

from departments.billing.etims_adapter import (
    categorize_item_vat,
    generate_etims_fiscal_signature,
    is_etims_enabled,
)


def test_etims_feature_flag_disabled_by_default(app):
    with app.app_context():
        assert is_etims_enabled() is False


def test_vat_exemption_medical_consultation():
    vat = categorize_item_vat("General Consultation", "consult")
    assert vat["tax_code"] == "E"
    assert vat["tax_rate"] == 0.0
    assert vat["is_taxable"] is False


def test_vat_exemption_essential_drug():
    vat = categorize_item_vat("Paracetamol 500mg", "drug")
    assert vat["tax_code"] == "E"
    assert vat["is_taxable"] is False


def test_vat_taxable_retail_supply():
    vat = categorize_item_vat("Cosmetic Lotion", "retail")
    assert vat["tax_code"] == "A"
    assert vat["tax_rate"] == 0.16
    assert vat["is_taxable"] is True


def test_generate_etims_signature(app):
    with app.app_context():
        items = [
            {"description": "Consultation Fee", "category": "consult", "unit_price": 1000.0, "quantity": 1},
            {"description": "Retail Cosmetic Item", "category": "retail", "unit_price": 116.0, "quantity": 1},
        ]
        sig = generate_etims_fiscal_signature(
            invoice_id=101,
            patient_id="P-001",
            total_amount=1116.0,
            items=items,
            kra_pin="P051234567Z",
        )
        assert sig["invoice_id"] == 101
        assert sig["exempt_total"] == 1000.0
        assert sig["vat_total"] == 16.0
        assert len(sig["control_code"]) == 16
        assert "itax.kra.go.ke" in sig["qr_code_url"]
