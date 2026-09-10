"""
tests/test_billing_event_listeners.py
──────────────────────────────────────
T3.8 — Billing Event Listeners: InvoiceLineItem.encounter_id is sourced
from the originating service record (RequestedLab, RequestedImage,
PrescribedMedicine) rather than defaulting to the invoice-level encounter.
This guarantees that TELEHEALTH, ANC, REFERRAL, and IPD charges land on
the correct visit even when multiple encounter types co-exist for a patient.
"""

from datetime import date, datetime
from decimal import Decimal
import pytest

from departments.billing.sync import sync_charge
from departments.models.billing import Invoice, InvoiceLineItem, InvoiceStatus
from departments.models.encounter import Encounter
from departments.models.medicine import (
    Imaging,
    LabTest,
    PrescribedMedicine,
    RequestedImage,
    RequestedLab,
)
from departments.models.records import Patient
from extensions import db


# ── helpers ────────────────────────────────────────────────────────
def _patient(pid: str) -> Patient:
    p = Patient(
        patient_id=pid,
        name=f"Test {pid}",
        sex="F",
        date_of_birth=date(1990, 1, 1),
    )
    db.session.add(p)
    db.session.commit()
    return p


def _encounter(patient_id: str, enc_type: str = "OPD", stage: str = "IN_CONSULTATION") -> Encounter:
    enc = Encounter(
        patient_id=patient_id,
        encounter_type=enc_type,
        stage=stage,
        status="ACTIVE",
    )
    db.session.add(enc)
    db.session.commit()
    return enc


def _lab_test(name: str = "CBC", cost: float = 500.0) -> LabTest:
    lt = LabTest(test_name=name, cost=cost)
    db.session.add(lt)
    db.session.commit()
    return lt


def _imaging(name: str = "Chest X-Ray", cost: float = 1500.0) -> Imaging:
    img = Imaging(imaging_type=name, cost=cost)
    db.session.add(img)
    db.session.commit()
    return img


# ── T3.8: sync_charge with source_encounter_id ────────────────────
class TestSyncChargeEncounterTagging:
    def test_source_encounter_id_overrides_invoice_encounter(self, app):
        """sync_charge must use source_encounter_id when provided, not invoice.encounter_id."""
        with app.app_context():
            _patient("P-BILL-01")
            # Two encounters: the invoice will be scoped to enc_a but the lab is from enc_b
            enc_a = _encounter("P-BILL-01", enc_type="OPD")
            enc_b = _encounter("P-BILL-01", enc_type="TELEHEALTH")
            line = sync_charge(
                patient_id="P-BILL-01",
                source_table="requested_lab",
                source_id=9001,
                description="Lab Test: CBC",
                category="lab",
                amount=500.0,
                source_encounter_id=enc_b.id,
            )
            assert line is not None
            assert line.encounter_id == enc_b.id, (
                f"Expected enc_b ({enc_b.id}), got {line.encounter_id}"
            )

    def test_no_source_encounter_id_falls_back_to_invoice(self, app):
        """When source_encounter_id is None, line item inherits invoice.encounter_id."""
        with app.app_context():
            _patient("P-BILL-02")
            enc = _encounter("P-BILL-02", enc_type="OPD")
            line = sync_charge(
                patient_id="P-BILL-02",
                source_table="requested_lab",
                source_id=9002,
                description="Lab Test: Malaria RDT",
                category="lab",
                amount=300.0,
                source_encounter_id=None,
            )
            # The invoice was scoped to enc (the active encounter)
            assert line is not None
            assert line.encounter_id == enc.id

    def test_idempotency_preserved_with_source_encounter(self, app):
        """Calling sync_charge twice for the same source returns the same line item."""
        with app.app_context():
            _patient("P-BILL-03")
            enc = _encounter("P-BILL-03", enc_type="ANC")
            line1 = sync_charge(
                patient_id="P-BILL-03",
                source_table="requested_lab",
                source_id=9003,
                description="Lab: Haemoglobin",
                category="lab",
                amount=400.0,
                source_encounter_id=enc.id,
            )
            line2 = sync_charge(
                patient_id="P-BILL-03",
                source_table="requested_lab",
                source_id=9003,
                description="Lab: Haemoglobin",
                category="lab",
                amount=400.0,
                source_encounter_id=enc.id,
            )
            assert line1.id == line2.id, "Idempotency broken — duplicate line item created"


# ── T3.8: Event listener integration tests ────────────────────────
class TestEventListenerEncounterTagging:
    def test_requested_lab_gets_encounter_id_from_source(self, app):
        """After flush, InvoiceLineItem for a RequestedLab carries the lab's encounter_id."""
        with app.app_context():
            _patient("P-LAB-01")
            enc = _encounter("P-LAB-01", enc_type="OPD")
            lt = _lab_test("CBC", 500.0)
            lab_req = RequestedLab(
                patient_id="P-LAB-01",
                lab_test_id=lt.id,
                encounter_id=enc.id,
            )
            db.session.add(lab_req)
            db.session.commit()

            line = InvoiceLineItem.query.filter_by(
                source_table="requested_lab",
                source_id=lab_req.id,
            ).first()
            assert line is not None, "No InvoiceLineItem created for RequestedLab"
            assert line.encounter_id == enc.id, (
                f"Expected {enc.id}, got {line.encounter_id}"
            )
            assert line.category == "lab"
            assert float(line.unit_price) == 500.0

    def test_requested_image_gets_encounter_id_from_source(self, app):
        """After flush, InvoiceLineItem for a RequestedImage carries the image's encounter_id."""
        with app.app_context():
            _patient("P-IMG-01")
            enc = _encounter("P-IMG-01", enc_type="TELEHEALTH")
            img = _imaging("Chest X-Ray", 1500.0)
            img_req = RequestedImage(
                patient_id="P-IMG-01",
                imaging_id=img.id,
                encounter_id=enc.id,
            )
            db.session.add(img_req)
            db.session.commit()

            line = InvoiceLineItem.query.filter_by(
                source_table="requested_image",
                source_id=img_req.id,
            ).first()
            assert line is not None, "No InvoiceLineItem created for RequestedImage"
            assert line.encounter_id == enc.id, (
                f"Expected {enc.id}, got {line.encounter_id}"
            )
            assert line.category == "imaging"

    def test_lab_without_encounter_id_still_creates_line_item(self, app):
        """RequestedLab with no encounter_id (legacy) still creates a line item — no crash."""
        with app.app_context():
            _patient("P-LAB-02")
            lt = _lab_test("Urine FEME", 200.0)
            lab_req = RequestedLab(
                patient_id="P-LAB-02",
                lab_test_id=lt.id,
                encounter_id=None,
            )
            db.session.add(lab_req)
            db.session.commit()

            line = InvoiceLineItem.query.filter_by(
                source_table="requested_lab",
                source_id=lab_req.id,
            ).first()
            assert line is not None
            # encounter_id may be None or the active-encounter fallback — both are valid
            assert line.category == "lab"

    def test_anc_encounter_lab_tagged_correctly(self, app):
        """Lab ordered during an ANC encounter must carry the ANC encounter_id."""
        with app.app_context():
            _patient("P-ANC-BILL-01")
            anc_enc = _encounter("P-ANC-BILL-01", enc_type="ANC")
            lt = _lab_test("Syphilis RPR", 350.0)
            lab_req = RequestedLab(
                patient_id="P-ANC-BILL-01",
                lab_test_id=lt.id,
                encounter_id=anc_enc.id,
            )
            db.session.add(lab_req)
            db.session.commit()

            line = InvoiceLineItem.query.filter_by(
                source_table="requested_lab",
                source_id=lab_req.id,
            ).first()
            assert line is not None
            assert line.encounter_id == anc_enc.id
