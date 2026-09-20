"""
tests/test_prescription_external_and_formulary.py
──────────────────────────────────────────────────
Unit and integration tests for:
  - Internal vs External Prescription Sign-off (Priority 1)
  - Formulary Drug linking and stock status indicators (Priority 1 & 4)
  - Stock Movement Ledger Primacy & Background Reconciliation (Priority 2)
  - Closed-loop Diagnostic Queue Transitions (Priority 3)
"""

from datetime import date
from decimal import Decimal

import pytest
from werkzeug.security import generate_password_hash

from departments.models.encounter import Encounter
from departments.models.medicine import PrescribedMedicine
from departments.models.pharmacy import Batch, Drug, DrugCategory
from departments.models.records import Patient
from departments.models.stock_movement import (
    reconcile_stock_balance,
    record_movement,
    run_all_stock_reconciliations,
)
from departments.models.user import User
from departments.shared.visit_closure import (
    advance_after_completion,
    determine_next_stage,
)
from extensions import db


@pytest.fixture
def test_setup(app):
    """Seed test patient, clinician, drug inventory, and active encounter."""
    with app.app_context():
        db.create_all()

        # Seed Doctor user
        doc = User.query.filter_by(username="dr_test_user").first()
        if not doc:
            doc = User(
                username="dr_test_user",
                password=generate_password_hash("DocPassword123!", method="pbkdf2:sha256"),
                role="medicine",
            )
            db.session.add(doc)

        # Seed Patient
        patient = Patient.query.filter_by(patient_id="P-TEST-99").first()
        if not patient:
            patient = Patient(
                patient_id="P-TEST-99",
                name="Test Formulary Patient",
                sex="F",
                date_of_birth=date(1995, 5, 15),
                contact="0712345678",
                national_id="99887766",
                place_of_residence="Nairobi",
                marital_status="Single",
                next_of_kin="Kin",
                relationship_with_next_of_kin="Parent",
                next_of_kin_contact="0722334455",
                emergency_contact="0722334455",
            )
            db.session.add(patient)

        # Seed DrugCategory
        cat = DrugCategory.query.filter_by(name="Antibiotics").first()
        if not cat:
            cat = DrugCategory(name="Antibiotics")
            db.session.add(cat)
            db.session.flush()

        # Seed In-Stock Drug (Amoxicillin)
        drug_amox = Drug.query.filter_by(generic_name="Amoxicillin").first()
        if not drug_amox:
            drug_amox = Drug(
                generic_name="Amoxicillin",
                brand_name="Amoxil",
                category_id=cat.id,
                dosage_form="Capsule",
                strength="500mg",
                buying_price=Decimal("10.00"),
                selling_price=Decimal("15.00"),
                quantity_in_stock=100,
            )
            db.session.add(drug_amox)
            db.session.flush()

            batch = Batch(
                drug_id=drug_amox.id,
                batch_number="BATCH-AMOX-01",
                quantity_in_stock=100,
                expiry_date=date(2028, 12, 31),
            )
            db.session.add(batch)

            record_movement(
                item_type="DRUG",
                item_id=drug_amox.id,
                movement_type="RECEIVED",
                quantity_delta=100,
                balance_after=100,
                batch_id=batch.id,
                notes="Initial stock seed for test",
            )

        # Seed Out-of-Stock Drug (Specialized Chemotherapy Drug)
        drug_chemo = Drug.query.filter_by(generic_name="Cisplatin").first()
        if not drug_chemo:
            drug_chemo = Drug(
                generic_name="Cisplatin",
                brand_name="Platinol",
                category_id=cat.id,
                dosage_form="Injection",
                strength="50mg",
                buying_price=Decimal("1800.00"),
                selling_price=Decimal("2500.00"),
                quantity_in_stock=0,
            )
            db.session.add(drug_chemo)

        db.session.commit()
        return {
            "patient_id": "P-TEST-99",
            "in_stock_drug": "Amoxicillin",
            "out_of_stock_drug": "Cisplatin",
        }


def test_internal_prescription_signoff(client, test_setup):
    """Test prescribing an in-stock drug creates an internal prescription with status PENDING (0)."""
    with client.session_transaction() as sess:
        doc = User.query.filter_by(username="dr_test_user").first()
        sess["_user_id"] = str(doc.id)
        sess["_fresh"] = True

    res = client.post(
        "/medicine/prescribe/signoff",
        json={
            "patient_id": test_setup["patient_id"],
            "prescriptions": [
                {
                    "name": test_setup["in_stock_drug"],
                    "dosage": "500mg",
                    "frequency": "TID",
                    "num_days": 7,
                    "is_external": False,
                }
            ],
        },
    )
    assert res.status_code == 201
    data = res.get_json()
    assert data["success"] is True
    assert len(data["items"]) == 1
    assert data["items"][0]["is_external"] is False
    assert data["items"][0]["stock_status"] == "IN_STOCK"
    assert data["total_internal_charge"] == 15.0

    # Verify DB record
    rx = PrescribedMedicine.query.filter_by(prescription_id=data["prescription_id"]).first()
    assert rx is not None
    assert rx.is_external is False
    assert rx.status == PrescribedMedicine.STATUS_PENDING


def test_external_prescription_for_out_of_stock_drug(client, test_setup):
    """Test prescribing an out-of-stock drug automatically flags as external (status 3) with zero internal charge."""
    with client.session_transaction() as sess:
        doc = User.query.filter_by(username="dr_test_user").first()
        sess["_user_id"] = str(doc.id)
        sess["_fresh"] = True

    res = client.post(
        "/medicine/prescribe/signoff",
        json={
            "patient_id": test_setup["patient_id"],
            "prescriptions": [
                {
                    "name": test_setup["out_of_stock_drug"],
                    "dosage": "50mg",
                    "frequency": "Once daily",
                    "num_days": 5,
                }
            ],
        },
    )
    assert res.status_code == 201
    data = res.get_json()
    assert data["success"] is True
    assert data["items"][0]["is_external"] is True
    assert data["items"][0]["stock_status"] == "OUT_OF_STOCK_EXTERNAL"
    assert data["total_internal_charge"] == 0.0
    assert "printable_url" in data

    # Verify DB record
    rx = PrescribedMedicine.query.filter_by(prescription_id=data["prescription_id"]).first()
    assert rx is not None
    assert rx.is_external is True
    assert rx.status == PrescribedMedicine.STATUS_EXTERNAL


def test_stock_ledger_reconciliation_helper(app, test_setup):
    """Test reconcile_stock_balance helper correctly matches cached balance vs ledger sum."""
    with app.app_context():
        drug = Drug.query.filter_by(generic_name=test_setup["in_stock_drug"]).first()
        res = reconcile_stock_balance("DRUG", drug.id)
        assert res["match"] is True
        assert res["variance"] == 0
        assert res["ledger_balance"] == res["cached_balance"]

        # Run background reconciliation helper over all items
        audit_res = run_all_stock_reconciliations()
        assert audit_res["variance_count"] == 0
        assert audit_res["total_items_checked"] >= 1


def test_closed_loop_diagnostic_queue_transition(app, test_setup):
    """Test that completing diagnostic tests advances encounter stage from AWAITING_LAB to WAITING_DOCTOR_RESULTS."""
    with app.app_context():
        patient_id = test_setup["patient_id"]
        enc = Encounter(patient_id=patient_id, stage="AWAITING_LAB", status="ACTIVE")
        db.session.add(enc)
        db.session.commit()

        # Check next stage when no pending labs remain
        next_stage = determine_next_stage(patient_id, enc)
        assert next_stage == "WAITING_DOCTOR_RESULTS"

        # Advance encounter
        updated_stage = advance_after_completion(patient_id)
        assert updated_stage == "WAITING_DOCTOR_RESULTS"
        assert enc.stage == "WAITING_DOCTOR_RESULTS"
