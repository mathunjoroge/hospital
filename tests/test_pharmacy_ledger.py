"""
tests/test_pharmacy_ledger.py
─────────────────────────────
Live-API tests for the pharmacy ledger + stock-sync patch (Findings A, B, C).

Finding A — every dispensing/void/write-off path now writes a StockMovement row.
Finding B — Drug.quantity_in_stock stays in sync with batch-level deductions.
Finding C — save_prescription warns (non-blocking) when dispensed drug name
            does not appear in the prescription's medicine list.
"""

from departments.models.pharmacy import Batch, DispensedDrug, Drug, DrugCategory
from departments.models.stock_movement import StockMovement, record_movement
from extensions import db

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_or_create_category(app):
    """Return a DrugCategory id, creating one if needed."""
    with app.app_context():
        cat = DrugCategory.query.filter_by(name="TestCategory").first()
        if not cat:
            cat = DrugCategory(name="TestCategory")
            db.session.add(cat)
            db.session.commit()
        return cat.id


def _make_drug(app, generic_name="Amoxicillin", qty=100):
    """Create and persist a Drug + one Batch. Returns (drug_id, batch_id)."""
    cat_id = _get_or_create_category(app)
    with app.app_context():
        drug = Drug(
            generic_name=generic_name,
            brand_name=f"{generic_name} Brand",
            category_id=cat_id,
            dosage_form="Tablet",
            strength="500mg",
            buying_price=5.0,
            selling_price=10.0,
            quantity_in_stock=qty,
            reorder_level=20,
            storage_condition="Ambient",
        )
        db.session.add(drug)
        db.session.flush()

        batch = Batch(
            drug_id=drug.id,
            batch_number=f"B-{drug.id}-001",
            quantity_in_stock=qty,
            expiry_date=None,
        )
        db.session.add(batch)
        db.session.commit()
        return drug.id, batch.id


# ─────────────────────────────────────────────────────────────────────────────
# record_movement helper — sanity unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordMovementHelper:
    def test_creates_row_with_correct_fields(self, app):
        drug_id, _ = _make_drug(app, "Paracetamol", qty=50)
        with app.app_context():
            mv = record_movement(
                item_type="DRUG",
                item_id=drug_id,
                movement_type="DISPENSED",
                quantity_delta=-10,
                balance_after=40,
                reference_type="PRESCRIPTION",
                reference_id="RX-TEST-001",
                notes="unit test",
            )
            db.session.commit()
            saved = db.session.get(StockMovement, mv.id)

        assert saved.item_type == "DRUG"
        assert saved.item_id == drug_id
        assert saved.movement_type == "DISPENSED"
        assert saved.quantity_delta == -10
        assert saved.balance_after == 40
        assert saved.reference_id == "RX-TEST-001"

    def test_void_return_is_positive_delta(self, app):
        drug_id, _ = _make_drug(app, "Ibuprofen", qty=30)
        with app.app_context():
            mv = record_movement(
                item_type="DRUG",
                item_id=drug_id,
                movement_type="VOID_RETURN",
                quantity_delta=5,
                balance_after=35,
            )
            db.session.commit()
            # Read inside context to avoid DetachedInstanceError
            assert mv.quantity_delta == 5
            assert mv.movement_type == "VOID_RETURN"


# ─────────────────────────────────────────────────────────────────────────────
# Finding A — DISPENSED/WRITEOFF ledger rows written by every dispense path
# ─────────────────────────────────────────────────────────────────────────────


class TestFindingA_LedgerEntries:
    def test_fefo_dispense_writes_dispensed_movement(self, app):
        """dispense_medication_fefo() must produce one DISPENSED row (single batch)."""
        from departments.pharmacy.fefo import dispense_medication_fefo

        drug_id, _ = _make_drug(app, "Metformin", qty=60)
        with app.app_context():
            before = StockMovement.query.filter_by(
                item_type="DRUG", item_id=drug_id, movement_type="DISPENSED"
            ).count()
            dispense_medication_fefo(
                patient_id="P-TEST-01",
                drug_id=drug_id,
                quantity=20,
                prescription_id="RX-FEFO-001",
                user_id=None,
            )
            after = StockMovement.query.filter_by(
                item_type="DRUG", item_id=drug_id, movement_type="DISPENSED"
            ).count()

        assert (
            after == before + 1
        ), "Finding A: dispense_medication_fefo() must write one DISPENSED ledger row."

    def test_fefo_dispense_ledger_delta_matches_quantity(self, app):
        """DISPENSED row's quantity_delta must equal -(dispensed quantity)."""
        from departments.pharmacy.fefo import dispense_medication_fefo

        drug_id, _ = _make_drug(app, "Atorvastatin", qty=80)
        with app.app_context():
            dispense_medication_fefo("P-TEST-02", drug_id, 25, "RX-FEFO-002")
            mv = StockMovement.query.filter_by(
                item_type="DRUG", item_id=drug_id, movement_type="DISPENSED"
            ).first()

        assert mv is not None
        assert mv.quantity_delta == -25

    def test_fefo_dispense_multi_batch_writes_row_per_batch(self, app):
        """Two-batch FEFO dispense must produce two DISPENSED ledger rows."""
        from datetime import date, timedelta

        from departments.pharmacy.fefo import dispense_medication_fefo

        cat_id = _get_or_create_category(app)
        with app.app_context():
            drug = Drug(
                generic_name="Ciprofloxacin",
                brand_name="Cipro",
                category_id=cat_id,
                dosage_form="Tablet",
                strength="250mg",
                buying_price=3.0,
                selling_price=5.0,
                quantity_in_stock=30,
                reorder_level=5,
                storage_condition="Ambient",
            )
            db.session.add(drug)
            db.session.flush()

            b1 = Batch(
                drug_id=drug.id,
                batch_number="B-CIP-001",
                quantity_in_stock=10,
                expiry_date=date.today() + timedelta(days=30),
            )
            b2 = Batch(
                drug_id=drug.id,
                batch_number="B-CIP-002",
                quantity_in_stock=20,
                expiry_date=date.today() + timedelta(days=90),
            )
            db.session.add_all([b1, b2])
            db.session.commit()
            drug_id = drug.id

            dispense_medication_fefo("P-TEST-03", drug_id, 25, "RX-FEFO-003")
            rows = StockMovement.query.filter_by(
                item_type="DRUG", item_id=drug_id, movement_type="DISPENSED"
            ).all()

        assert len(rows) == 2, "Two-batch FEFO must produce two DISPENSED rows."
        assert sum(abs(r.quantity_delta) for r in rows) == 25

    def test_remove_batch_writes_writeoff_movement(self, app, client, admin_user):
        """remove_batch() must write a WRITEOFF StockMovement row."""
        drug_id, batch_id = _make_drug(app, "Omeprazole", qty=40)

        resp = client.post(f"/pharmacy/remove_batch/{batch_id}", follow_redirects=True)
        assert resp.status_code == 200

        with app.app_context():
            mv = StockMovement.query.filter_by(
                item_type="DRUG", item_id=drug_id, movement_type="WRITEOFF"
            ).first()

        assert mv is not None, "Finding A: remove_batch() must write a WRITEOFF row."
        assert mv.quantity_delta == -40

    def test_remove_all_expiries_writes_writeoff_per_batch(
        self, app, client, admin_user
    ):
        """remove_all_expiries() must write one WRITEOFF row per expired batch."""
        from datetime import date, timedelta

        cat_id = _get_or_create_category(app)
        with app.app_context():
            drug = Drug(
                generic_name="ExpiredDrug",
                brand_name="ExpBrand",
                category_id=cat_id,
                dosage_form="Syrup",
                strength="100mg/5ml",
                buying_price=2.0,
                selling_price=8.0,
                quantity_in_stock=50,
                reorder_level=5,
                storage_condition="Ambient",
            )
            db.session.add(drug)
            db.session.flush()

            past = date.today() - timedelta(days=1)
            b1 = Batch(
                drug_id=drug.id,
                batch_number="EXP-001",
                quantity_in_stock=15,
                expiry_date=past,
            )
            b2 = Batch(
                drug_id=drug.id,
                batch_number="EXP-002",
                quantity_in_stock=35,
                expiry_date=past,
            )
            db.session.add_all([b1, b2])
            db.session.commit()
            drug_id = drug.id

        resp = client.post("/pharmacy/remove_all_expiries", follow_redirects=True)
        assert resp.status_code == 200

        with app.app_context():
            rows = StockMovement.query.filter_by(
                item_type="DRUG", item_id=drug_id, movement_type="WRITEOFF"
            ).all()

        assert len(rows) == 2, "remove_all_expiries must write one WRITEOFF per batch."
        assert sum(abs(r.quantity_delta) for r in rows) == 50


# ─────────────────────────────────────────────────────────────────────────────
# Finding B — Drug.quantity_in_stock stays in sync
# ─────────────────────────────────────────────────────────────────────────────


class TestFindingB_DrugLevelStockSync:
    def test_fefo_dispense_decrements_drug_quantity(self, app):
        """Drug.quantity_in_stock must drop by the dispensed amount."""
        from departments.pharmacy.fefo import dispense_medication_fefo

        drug_id, _ = _make_drug(app, "Salbutamol", qty=100)
        with app.app_context():
            dispense_medication_fefo("P-TEST-10", drug_id, 30, "RX-B-001")
            drug = db.session.get(Drug, drug_id)

        assert (
            drug.quantity_in_stock == 70
        ), "Finding B: Drug.quantity_in_stock must drop from 100 to 70."

    def test_remove_batch_decrements_drug_quantity(self, app, client, admin_user):
        """remove_batch() must reduce Drug.quantity_in_stock by batch qty."""
        drug_id, batch_id = _make_drug(app, "Fluconazole", qty=55)
        client.post(f"/pharmacy/remove_batch/{batch_id}", follow_redirects=True)

        with app.app_context():
            drug = db.session.get(Drug, drug_id)

        assert (
            drug.quantity_in_stock == 0
        ), "Finding B: removing only batch must reduce Drug.quantity_in_stock to 0."

    def test_void_via_remove_dispensed_restores_drug_quantity(
        self, app, client, admin_user
    ):
        """
        remove_dispensed() must restore Drug.quantity_in_stock.
        The original bug only restored the batch row, not the drug-level cache.
        """
        from departments.pharmacy.fefo import dispense_medication_fefo

        drug_id, _ = _make_drug(app, "Doxycycline", qty=80)

        with app.app_context():
            records = dispense_medication_fefo("P-TEST-11", drug_id, 20, "RX-B-002")
            dispense_id = records[0].id

        with app.app_context():
            assert db.session.get(Drug, drug_id).quantity_in_stock == 60

        resp = client.post(
            f"/pharmacy/remove_dispensed/{dispense_id}",
            data={
                "void_reason": "Test void for stock sync",
                "prescription_id": "RX-B-002",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            drug = db.session.get(Drug, drug_id)

        assert (
            drug.quantity_in_stock == 80
        ), "Finding B: voiding must restore Drug.quantity_in_stock to pre-dispense value."

    def test_void_writes_void_return_movement(self, app, client, admin_user):
        """remove_dispensed() must write a VOID_RETURN StockMovement row."""
        from departments.pharmacy.fefo import dispense_medication_fefo

        drug_id, _ = _make_drug(app, "Metronidazole", qty=50)
        with app.app_context():
            records = dispense_medication_fefo("P-TEST-12", drug_id, 10, "RX-B-003")
            dispense_id = records[0].id

        client.post(
            f"/pharmacy/remove_dispensed/{dispense_id}",
            data={"void_reason": "ledger test", "prescription_id": "RX-B-003"},
            follow_redirects=True,
        )

        with app.app_context():
            mv = StockMovement.query.filter_by(
                item_type="DRUG", item_id=drug_id, movement_type="VOID_RETURN"
            ).first()

        assert mv is not None, "VOID_RETURN ledger row must exist after voiding."
        assert mv.quantity_delta == 10


# ─────────────────────────────────────────────────────────────────────────────
# Finding C — name-mismatch guard in save_prescription
# ─────────────────────────────────────────────────────────────────────────────


class TestFindingC_NameMismatchGuard:
    def _setup_prescription(self, app, rx_id="RX-C-001"):
        """Create patient + medicine + PrescribedMedicine for a given rx_id."""
        from datetime import date

        from departments.models.medicine import Medicine, PrescribedMedicine
        from departments.models.records import Patient

        with app.app_context():
            p = Patient.query.filter_by(patient_id="P-C-001").first()
            if not p:
                p = Patient(
                    patient_id="P-C-001",
                    name="Test Patient C",
                    sex="M",
                    date_of_birth=date(1990, 1, 1),
                    contact="0700000000",
                )
                db.session.add(p)
                db.session.flush()

            med = Medicine(
                generic_name="Amoxicillin",
                brand_name="Amoxil",
                dosage="500mg TID",
            )
            db.session.add(med)
            db.session.flush()

            pm = PrescribedMedicine(
                patient_id="P-C-001",
                medicine_id=med.id,
                dosage="500mg",
                strength="500mg",
                frequency="TID",
                prescription_id=rx_id,
                num_days=5,
                status=0,
            )
            db.session.add(pm)
            db.session.commit()

    def test_name_match_dispense_succeeds(self, app, client, admin_user):
        """Happy path: name matches → dispense completes, no warning in logs."""
        self._setup_prescription(app, rx_id="RX-C-MATCH")
        drug_id, batch_id = _make_drug(app, "Amoxicillin", qty=100)

        with app.app_context():
            bn = db.session.get(Batch, batch_id).batch_number

        resp = client.post(
            "/pharmacy/save_prescription/RX-C-MATCH",
            data={
                "patient_id": "P-C-001",
                "drugs[]": [str(drug_id)],
                "quantity[]": ["10"],
                "batch_number[]": [bn],
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            dispensed = DispensedDrug.query.filter_by(
                prescription_id="RX-C-MATCH", drug_id=drug_id
            ).first()
        assert dispensed is not None, "Match case: dispense must complete."

    def test_name_mismatch_is_non_blocking(self, app, client, admin_user):
        """
        Mismatch: different drug dispensed → dispense still completes
        (pharmacist may be substituting a generic/brand equivalent).
        """
        self._setup_prescription(app, rx_id="RX-C-MISMATCH")
        drug_id, batch_id = _make_drug(app, "Ibuprofen", qty=100)

        with app.app_context():
            bn = db.session.get(Batch, batch_id).batch_number

        resp = client.post(
            "/pharmacy/save_prescription/RX-C-MISMATCH",
            data={
                "patient_id": "P-C-001",
                "drugs[]": [str(drug_id)],
                "quantity[]": ["5"],
                "batch_number[]": [bn],
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            dispensed = DispensedDrug.query.filter_by(
                prescription_id="RX-C-MISMATCH", drug_id=drug_id
            ).first()
        assert (
            dispensed is not None
        ), "Finding C: name-mismatch is non-blocking — dispense must still complete."

    def test_name_mismatch_still_writes_ledger(self, app, client, admin_user):
        """Even a mismatched dispense must write a DISPENSED ledger row (Finding A)."""
        self._setup_prescription(app, rx_id="RX-C-LEDGER")
        drug_id, batch_id = _make_drug(app, "Diclofenac", qty=60)

        with app.app_context():
            bn = db.session.get(Batch, batch_id).batch_number

        client.post(
            "/pharmacy/save_prescription/RX-C-LEDGER",
            data={
                "patient_id": "P-C-001",
                "drugs[]": [str(drug_id)],
                "quantity[]": ["8"],
                "batch_number[]": [bn],
            },
            follow_redirects=True,
        )

        with app.app_context():
            mv = StockMovement.query.filter_by(
                item_type="DRUG", item_id=drug_id, movement_type="DISPENSED"
            ).first()

        assert (
            mv is not None
        ), "Finding A+C: mismatch dispense must still write DISPENSED row."
        assert mv.quantity_delta == -8


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation — ledger sum must match Drug.quantity_in_stock
# ─────────────────────────────────────────────────────────────────────────────


class TestReconciliation:
    def test_ledger_matches_drug_stock_after_dispense(self, app):
        """
        After RECEIVED + DISPENSED movements, reconcile_stock_balance()
        must return match=True.
        """
        from departments.models.stock_movement import reconcile_stock_balance
        from departments.pharmacy.fefo import dispense_medication_fefo

        drug_id, _ = _make_drug(app, "Codeine", qty=200)

        with app.app_context():
            # Seed a RECEIVED row so the ledger total starts at 200
            record_movement(
                item_type="DRUG",
                item_id=drug_id,
                movement_type="RECEIVED",
                quantity_delta=200,
                balance_after=200,
                reference_type="MANUAL",
                notes="Initial stock baseline",
            )
            db.session.commit()

            dispense_medication_fefo("P-RECON-01", drug_id, 50, "RX-RECON-01")
            result = reconcile_stock_balance("DRUG", drug_id)

        assert result["match"], (
            f"Ledger ({result['ledger_balance']}) must match "
            f"Drug.quantity_in_stock ({result['cached_balance']}). "
            f"Variance={result['variance']}"
        )

    def test_writeoff_keeps_reconciliation_clean(self, app, client, admin_user):
        """After a bulk expiry write-off, reconcile_stock_balance returns match=True."""
        from datetime import date, timedelta

        from departments.models.stock_movement import reconcile_stock_balance

        cat_id = _get_or_create_category(app)
        with app.app_context():
            drug = Drug(
                generic_name="ReconWriteoff",
                brand_name="RWB",
                category_id=cat_id,
                dosage_form="Tablet",
                strength="100mg",
                buying_price=1.0,
                selling_price=2.0,
                quantity_in_stock=30,
                reorder_level=5,
                storage_condition="Ambient",
            )
            db.session.add(drug)
            db.session.flush()

            record_movement(
                item_type="DRUG",
                item_id=drug.id,
                movement_type="RECEIVED",
                quantity_delta=30,
                balance_after=30,
                reference_type="MANUAL",
                notes="Baseline for writeoff test",
            )

            batch = Batch(
                drug_id=drug.id,
                batch_number="RWB-001",
                quantity_in_stock=30,
                expiry_date=date.today() - timedelta(days=1),
            )
            db.session.add(batch)
            db.session.commit()
            drug_id = drug.id

        client.get("/pharmacy/remove_all_expiries", follow_redirects=True)

        with app.app_context():
            result = reconcile_stock_balance("DRUG", drug_id)

        assert result["match"], (
            f"Reconciliation mismatch after expiry write-off: "
            f"ledger={result['ledger_balance']} cached={result['cached_balance']}"
        )
