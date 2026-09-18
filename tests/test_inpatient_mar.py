"""
tests/test_inpatient_mar.py
────────────────────────────
Unit tests for Task 3.6: Inpatient ADT & Medication Administration Record (MAR)

Updated to use authenticated clients after P0 fix: all three MAR endpoints
now require @login_required. Tests use a nursing-role user for charting
and an admin user for auto_bill.
"""

from datetime import datetime
from decimal import Decimal

import pytest
from werkzeug.security import generate_password_hash

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.billing import Invoice, InvoiceLineItem
from departments.models.medicine import AdmittedPatient, Ward
from departments.models.nursing import MedicationAdmin
from departments.models.records import Patient
from departments.models.user import User

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_user(app, username, role):
    """Idempotently create a user and return their id."""
    with app.app_context():
        u = User.query.filter_by(username=username).first()
        if not u:
            u = User(
                username=username,
                password=generate_password_hash("testpass", method="pbkdf2:sha256"),
                role=role,
            )
            db.session.add(u)
            db.session.commit()
        return u.id


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def nurse_id(app):
    return _make_user(app, "mar_nurse_fx", "nursing")


@pytest.fixture
def admin_id(app):
    return _make_user(app, "mar_admin_fx", "admin")


@pytest.fixture
def nurse_client(app, nurse_id):
    c = app.test_client()
    c.post("/login", data={"username": "mar_nurse_fx", "password": "testpass"})
    return c


@pytest.fixture
def admin_client(app, admin_id):
    c = app.test_client()
    c.post("/login", data={"username": "mar_admin_fx", "password": "testpass"})
    return c


@pytest.fixture
def mar_data(app):
    """Seed patient, ward, and admission."""
    with app.app_context():
        patient = Patient(
            patient_id="PT-MAR-01",
            name="MAR Test Patient",
            place_of_residence="Nairobi",
            sex="Male",
            date_of_birth=datetime(1985, 5, 5).date(),  # noqa: DTZ001
            marital_status="Married",
            contact="0700112233",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Spouse",
            next_of_kin_contact="0700112233",
            emergency_contact="0700112233",
        )
        db.session.add(patient)

        ward = Ward(
            name="General Male Ward",
            sex="Male",
            number_of_beds=20,
            occupied_beds=5,
            daily_charge=Decimal("1500.00"),
        )
        db.session.add(ward)
        db.session.commit()

        admission = AdmittedPatient(
            patient_id=patient.patient_id,
            ward_id=ward.id,
            admission_criteria="Severe Malaria",
            admitted_by=1,
        )
        db.session.add(admission)
        db.session.commit()

        return {
            "patient_id": patient.patient_id,
            "ward_id": ward.id,
            "admission_id": admission.id,
        }


# ── tests ──────────────────────────────────────────────────────────────────────


class TestInpatientMAR:
    # ── auth regression guards (must always pass) ─────────────────────────────

    def test_ward_occupancy_requires_auth(self, app):
        """Unauthenticated GET must be redirected to login (P0 regression guard)."""
        anon = app.test_client()
        resp = anon.get("/nursing/mar/occupancy")
        assert resp.status_code in (
            302,
            401,
            403,
        ), "REGRESSION: ward occupancy must require auth"

    def test_chart_requires_auth(self, app):
        """Unauthenticated POST to chart must be rejected (P0 regression guard)."""
        anon = app.test_client()
        resp = anon.post(
            "/nursing/mar/chart",
            json={"patient_id": "X", "medication": "X", "dosage": "1mg"},
        )
        assert resp.status_code in (
            302,
            401,
            403,
        ), "REGRESSION: chart_medication must require auth"

    def test_auto_bill_requires_auth(self, app):
        """Unauthenticated POST to auto_bill must be rejected (P0 regression guard)."""
        anon = app.test_client()
        resp = anon.post("/nursing/mar/auto_bill")
        assert resp.status_code in (
            302,
            401,
            403,
        ), "REGRESSION: auto_bill must require auth"

    # ── functional tests (with auth) ─────────────────────────────────────────

    def test_ward_occupancy(self, nurse_client, mar_data):
        resp = nurse_client.get("/nursing/mar/occupancy")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "occupancy" in data

        ward = next(
            (w for w in data["occupancy"] if w["ward_id"] == mar_data["ward_id"]),
            None,
        )
        assert ward is not None
        assert ward["name"] == "General Male Ward"
        assert ward["available_beds"] == 15

    def test_chart_medication(self, nurse_client, nurse_id, app, mar_data):
        """Nursing user can chart medication; recorded_by must come from session."""
        resp = nurse_client.post(
            "/nursing/mar/chart",
            json={
                "patient_id": mar_data["patient_id"],
                "medication": "Paracetamol IV",
                "dosage": "1000mg",
                # nurse_id deliberately omitted from body
            },
        )
        assert resp.status_code == 201
        body = resp.get_json()
        assert body["success"] is True

        with app.app_context():
            record = db.session.get(MedicationAdmin, body["record_id"])
            assert record is not None
            assert record.medication == "Paracetamol IV"
            assert record.dosage == "1000mg"
            # Must be the session user's id, not any forged value
            assert record.recorded_by == nurse_id

    def test_nurse_id_from_session_not_body(
        self, nurse_client, nurse_id, app, mar_data
    ):
        """Sending a forged nurse_id in the body must NOT override the session."""
        forged = nurse_id + 9999
        resp = nurse_client.post(
            "/nursing/mar/chart",
            json={
                "patient_id": mar_data["patient_id"],
                "medication": "Morphine",
                "dosage": "5mg IV",
                "nurse_id": forged,
            },
        )
        assert resp.status_code == 201
        with app.app_context():
            record = db.session.get(MedicationAdmin, resp.get_json()["record_id"])
            assert (
                record.recorded_by == nurse_id
            ), f"recorded_by should be session nurse {nurse_id}, not forged {forged}"

    def test_auto_billing(self, admin_client, app, mar_data):
        """Admin can trigger daily billing; invoices are created with Decimal amounts."""
        resp = admin_client.post("/nursing/mar/auto_bill")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert body["patients_billed"] >= 1

        with app.app_context():
            invoice = Invoice.query.filter_by(patient_id=mar_data["patient_id"]).first()
            assert invoice is not None
            assert Decimal(str(invoice.grand_total)) >= Decimal("1500.00")

            line_item = InvoiceLineItem.query.filter_by(invoice_id=invoice.id).first()
            assert line_item is not None
            assert "Daily Ward Charge" in line_item.description
