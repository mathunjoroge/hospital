"""
tests/test_mch_cold_chain.py
─────────────────────────────────────────────────────────────────────────────
Gap #6 — MCH / ANC & Immunization / Vaccine Lifecycle & Cold-Chain Tracking

Test coverage areas:
  T6.1  Cold-chain stock receipt & validation
  T6.2  FEFO dispensing & insufficient stock
  T6.3  Temperature monitoring & breach detection (FREEZE / TOO_HOT / OK)
  T6.4  Automated VVM escalation and batch quarantine on breach
  T6.5  Near-expiry alert reporting
  T6.6  Immunization cold-chain integration (lot traceability)
  T6.7  Immunization without cold-chain stock (graceful degradation)
  T6.8  Enhanced ANC vitals recording
  T6.9  API smoke tests for all cold-chain endpoints
"""

from datetime import date, timedelta

import pytest
from werkzeug.security import generate_password_hash

from departments.mch.cold_chain import (
    ColdChainEngine,
)
from departments.mch.engine import MchEngine
from departments.mch.models import (
    AncVisit,
    ImmunizationRecord,
    VaccineBatch,
    VaccineTemperatureLog,
)
from departments.models.records import Patient, PatientWaitingList
from departments.models.user import User
from departments.shared.queue_constants import QueueStatus
from extensions import db

# ── Fixtures ──────────────────────────────────────────────────────────────────

cc_engine = ColdChainEngine()
mch_engine = MchEngine()

FUTURE_DATE = date.today() + timedelta(days=365)
NEAR_EXPIRY_DATE = date.today() + timedelta(days=20)


@pytest.fixture
def nursing_user(app):
    with app.app_context():
        u = User(
            username="cc_nurse_001",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="nursing",
        )
        db.session.add(u)
        db.session.commit()
        yield u


def _patient(patient_id: str):
    from datetime import datetime

    p = Patient(
        patient_id=patient_id,
        name=f"Test {patient_id}",
        sex="F",
        date_of_birth=datetime(2023, 1, 1),  # noqa: DTZ001
    )
    db.session.add(p)
    db.session.add(
        PatientWaitingList(patient_id=patient_id, seen=QueueStatus.WAITING_TRIAGE)
    )
    db.session.commit()
    return p


def _batch(
    app,
    vaccine_name="BCG",
    batch_number="BCG-001",
    quantity=50,
    doses_per_vial=1,
    location="FRIDGE_A",
    expiry=None,
):
    """Helper: register a vaccine batch in cold-chain stock."""
    exp = expiry or FUTURE_DATE
    with app.app_context():
        return cc_engine.receive_vaccine_batch(
            vaccine_name=vaccine_name,
            batch_number=batch_number,
            manufacturer="KEMSA",
            quantity=quantity,
            doses_per_vial=doses_per_vial,
            expiry_date=exp,
            storage_location=location,
            supplied_by="UNICEF",
        )


# ── T6.1  Stock receipt ───────────────────────────────────────────────────────


def test_receive_batch_persists(app):
    """A received batch should be persisted with correct quantities."""
    with app.app_context():
        batch = cc_engine.receive_vaccine_batch(
            vaccine_name="OPV",
            batch_number="OPV-T61",
            manufacturer="WHO Labs",
            quantity=100,
            doses_per_vial=20,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_B",
        )
        saved = db.session.get(VaccineBatch, batch.id)
        assert saved is not None
        assert saved.vaccine_name == "OPV"
        assert saved.quantity_vials == 100
        assert saved.quantity_remaining_vials == 100
        assert saved.doses_per_vial == 20
        assert saved.vvm_stage == 1
        assert saved.is_cold_chain_breach is False


def test_receive_batch_rejects_duplicate(app):
    """Receiving the same batch number twice for the same vaccine should raise."""
    with app.app_context():
        cc_engine.receive_vaccine_batch(
            vaccine_name="BCG",
            batch_number="BCG-DUP",
            manufacturer="KEMSA",
            quantity=10,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_A",
        )
        with pytest.raises(ValueError, match="already registered"):
            cc_engine.receive_vaccine_batch(
                vaccine_name="BCG",
                batch_number="BCG-DUP",
                manufacturer="KEMSA",
                quantity=5,
                doses_per_vial=1,
                expiry_date=FUTURE_DATE,
                storage_location="FRIDGE_A",
            )


def test_receive_batch_rejects_expired(app):
    """Receiving an already-expired batch should raise."""
    with app.app_context():
        expired = date.today() - timedelta(days=1)
        with pytest.raises(ValueError, match="already expired"):
            cc_engine.receive_vaccine_batch(
                vaccine_name="BCG",
                batch_number="BCG-EXPIRED",
                manufacturer="KEMSA",
                quantity=10,
                doses_per_vial=1,
                expiry_date=expired,
                storage_location="FRIDGE_A",
            )


def test_receive_batch_rejects_zero_quantity(app):
    """quantity must be positive."""
    with app.app_context(), pytest.raises(ValueError, match="quantity must be > 0"):
        cc_engine.receive_vaccine_batch(
            vaccine_name="BCG",
            batch_number="BCG-ZERO",
            manufacturer="KEMSA",
            quantity=0,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_A",
        )


# ── T6.2  FEFO dispensing ────────────────────────────────────────────────────


def test_fefo_dispense_draws_earliest_expiry_first(app):
    """Dispense should use the batch with the earliest expiry date (FEFO)."""
    with app.app_context():
        # Batch that expires sooner
        cc_engine.receive_vaccine_batch(
            vaccine_name="PCV",
            batch_number="PCV-SOON",
            manufacturer="Pfizer",
            quantity=5,
            doses_per_vial=1,
            expiry_date=date.today() + timedelta(days=60),
            storage_location="FRIDGE_C",
        )
        # Batch that expires later
        cc_engine.receive_vaccine_batch(
            vaccine_name="PCV",
            batch_number="PCV-LATE",
            manufacturer="Pfizer",
            quantity=5,
            doses_per_vial=1,
            expiry_date=date.today() + timedelta(days=200),
            storage_location="FRIDGE_C",
        )
        drawn = cc_engine.dispense_vaccine("PCV", vials_needed=1)
        assert (
            drawn[0].batch_number == "PCV-SOON"
        ), "FEFO must draw the soonest-expiring batch"


def test_dispense_reduces_stock(app):
    """Dispensing vials must reduce quantity_remaining_vials correctly."""
    with app.app_context():
        cc_engine.receive_vaccine_batch(
            vaccine_name="Rotavirus",
            batch_number="ROT-001",
            manufacturer="GSK",
            quantity=10,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_A",
        )
        cc_engine.dispense_vaccine("Rotavirus", vials_needed=3)
        batch = VaccineBatch.query.filter_by(batch_number="ROT-001").first()
        assert batch.quantity_remaining_vials == 7


def test_dispense_insufficient_stock_raises(app):
    """Dispensing more than available stock should raise ValueError."""
    with app.app_context():
        cc_engine.receive_vaccine_batch(
            vaccine_name="IPV",
            batch_number="IPV-001",
            manufacturer="WHO Labs",
            quantity=2,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_A",
        )
        with pytest.raises(ValueError, match="Insufficient"):
            cc_engine.dispense_vaccine("IPV", vials_needed=10)


# ── T6.3  Temperature monitoring ─────────────────────────────────────────────


def test_normal_temperature_no_breach(app):
    """Temperature within [2, 8] °C should not flag a breach."""
    with app.app_context():
        log = cc_engine.log_temperature("FRIDGE_A", 5.0)
        assert log.is_breach is False
        assert log.breach_type is None


def test_temperature_too_hot_flags_breach(app):
    """Temperature above 8 °C should flag TOO_HOT breach."""
    with app.app_context():
        log = cc_engine.log_temperature("FRIDGE_A", 12.0)
        assert log.is_breach is True
        assert log.breach_type == "TOO_HOT"


def test_temperature_freeze_flags_breach(app):
    """Temperature below 0 °C should flag FREEZE breach."""
    with app.app_context():
        log = cc_engine.log_temperature("FRIDGE_A", -3.0)
        assert log.is_breach is True
        assert log.breach_type == "FREEZE"


def test_temperature_too_cold_flags_breach(app):
    """Temperature in (−inf, 2) but above 0 °C should flag TOO_COLD breach."""
    with app.app_context():
        log = cc_engine.log_temperature("FRIDGE_A", 1.0)
        assert log.is_breach is True
        assert log.breach_type == "TOO_COLD"


def test_temperature_log_persisted(app):
    """Temperature logs must be persisted to the database."""
    with app.app_context():
        log = cc_engine.log_temperature(
            "FRIDGE_B", 6.5, sensor_id="SENSOR-01", notes="routine"
        )
        saved = db.session.get(VaccineTemperatureLog, log.id)
        assert saved is not None
        assert saved.temperature_celsius == 6.5
        assert saved.sensor_id == "SENSOR-01"


# ── T6.4  VVM escalation & quarantine ────────────────────────────────────────


def test_freeze_breach_quarantines_freeze_sensitive_batches(app):
    """A FREEZE event should mark freeze-sensitive batches as breached with VVM→3."""
    with app.app_context():
        # Register a freeze-sensitive batch at this location
        cc_engine.receive_vaccine_batch(
            vaccine_name="Pentavalent",
            batch_number="PENT-FREEZE-01",
            manufacturer="KEMSA",
            quantity=20,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_FREEZE_TEST",
        )
        # Trigger freeze breach at same location
        cc_engine.log_temperature("FRIDGE_FREEZE_TEST", -2.0)

        batch = VaccineBatch.query.filter_by(batch_number="PENT-FREEZE-01").first()
        assert batch.is_cold_chain_breach is True
        assert batch.vvm_stage == 3


def test_heat_breach_escalates_vvm_stage(app):
    """A TOO_HOT event should increment VVM stage for batches at that location."""
    with app.app_context():
        cc_engine.receive_vaccine_batch(
            vaccine_name="Measles-Rubella",
            batch_number="MR-HEAT-01",
            manufacturer="Serum Institute",
            quantity=10,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_HEAT_TEST",
        )
        original_stage = (
            VaccineBatch.query.filter_by(batch_number="MR-HEAT-01").first().vvm_stage
        )

        cc_engine.log_temperature("FRIDGE_HEAT_TEST", 25.0)

        batch = VaccineBatch.query.filter_by(batch_number="MR-HEAT-01").first()
        assert batch.vvm_stage > original_stage


def test_breached_batch_excluded_from_dispensing(app):
    """Batches with is_cold_chain_breach=True must be excluded from FEFO dispensing."""
    with app.app_context():
        # Pentavalent is freeze-sensitive, so a freeze breach will quarantine it
        cc_engine.receive_vaccine_batch(
            vaccine_name="Pentavalent",
            batch_number="PENT-BREACH-EXCL",
            manufacturer="Sanofi",
            quantity=10,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_PENT_EXCL",
        )
        # Force a freeze breach at that location
        cc_engine.log_temperature("FRIDGE_PENT_EXCL", -5.0)
        # Now dispensing should fail — only breached stock available
        with pytest.raises(ValueError, match="Insufficient"):
            cc_engine.dispense_vaccine("Pentavalent", vials_needed=1)


# ── T6.5  Near-expiry alerts ──────────────────────────────────────────────────


def test_near_expiry_alerts_returned(app):
    """Batches expiring within threshold days should appear in alerts."""
    with app.app_context():
        cc_engine.receive_vaccine_batch(
            vaccine_name="BCG",
            batch_number="BCG-NEAR",
            manufacturer="KEMSA",
            quantity=5,
            doses_per_vial=1,
            expiry_date=NEAR_EXPIRY_DATE,
            storage_location="FRIDGE_A",
        )
        alerts = cc_engine.get_near_expiry_alerts(days_threshold=30)
        names = [a["batch_number"] for a in alerts]
        assert "BCG-NEAR" in names


def test_far_expiry_not_in_near_expiry_alerts(app):
    """Batches expiring far in the future should not appear in near-expiry alerts."""
    with app.app_context():
        cc_engine.receive_vaccine_batch(
            vaccine_name="IPV",
            batch_number="IPV-FAROUT",
            manufacturer="WHO Labs",
            quantity=5,
            doses_per_vial=1,
            expiry_date=date.today() + timedelta(days=365),
            storage_location="FRIDGE_A",
        )
        alerts = cc_engine.get_near_expiry_alerts(days_threshold=30)
        names = [a["batch_number"] for a in alerts]
        assert "IPV-FAROUT" not in names


# ── T6.6  Immunization cold-chain integration ─────────────────────────────────


def test_immunization_links_cold_chain_batch(app):
    """Recording a dose should deduct 1 vial from cold-chain stock and link the batch_id."""
    with app.app_context():
        _patient("CC-CHILD-01")
        cc_engine.receive_vaccine_batch(
            vaccine_name="BCG",
            batch_number="BCG-LINK-01",
            manufacturer="KEMSA",
            quantity=5,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_A",
        )
        before_qty = (
            VaccineBatch.query.filter_by(batch_number="BCG-LINK-01")
            .first()
            .quantity_remaining_vials
        )

        record = mch_engine.record_immunization(
            child_patient_id="CC-CHILD-01",
            vaccine_name="BCG",
            dose_number=1,
            deduct_from_cold_chain=True,
        )

        after_qty = (
            VaccineBatch.query.filter_by(batch_number="BCG-LINK-01")
            .first()
            .quantity_remaining_vials
        )

        assert (
            record.vaccine_batch_id is not None
        ), "ImmunizationRecord must link to VaccineBatch"
        assert after_qty == before_qty - 1, "Dispensing must deduct 1 vial"


def test_immunization_graceful_when_no_cold_chain_stock(app):
    """If no cold-chain stock exists, immunization should still be recorded."""
    with app.app_context():
        _patient("CC-CHILD-02")
        # No stock registered for this vaccine
        record = mch_engine.record_immunization(
            child_patient_id="CC-CHILD-02",
            vaccine_name="Yellow Fever",
            dose_number=1,
            deduct_from_cold_chain=True,  # no stock — should not crash
        )
        assert record is not None
        assert record.vaccine_batch_id is None  # no batch linked


def test_immunization_records_site_and_provider(app):
    """Extended immunization fields (site, provider, AEFI) should be persisted."""
    with app.app_context():
        _patient("CC-CHILD-03")
        record = mch_engine.record_immunization(
            child_patient_id="CC-CHILD-03",
            vaccine_name="OPV",
            dose_number=1,
            site_of_injection="LEFT_THIGH",
            administered_by="Nurse Jane",
            adverse_event_noted="Mild swelling",
            deduct_from_cold_chain=False,
        )
        saved = db.session.get(ImmunizationRecord, record.id)
        assert saved.site_of_injection == "LEFT_THIGH"
        assert saved.administered_by == "Nurse Jane"
        assert saved.adverse_event_noted == "Mild swelling"


# ── T6.7  Immunization without cold-chain (opt-out) ──────────────────────────


def test_immunization_skips_dispensing_when_disabled(app):
    """When deduct_from_cold_chain=False, no stock should be deducted."""
    with app.app_context():
        _patient("CC-CHILD-04")
        cc_engine.receive_vaccine_batch(
            vaccine_name="Rotavirus",
            batch_number="ROT-OPTOUT",
            manufacturer="GSK",
            quantity=5,
            doses_per_vial=1,
            expiry_date=FUTURE_DATE,
            storage_location="FRIDGE_A",
        )
        before = (
            VaccineBatch.query.filter_by(batch_number="ROT-OPTOUT")
            .first()
            .quantity_remaining_vials
        )

        mch_engine.record_immunization(
            child_patient_id="CC-CHILD-04",
            vaccine_name="Rotavirus",
            dose_number=1,
            deduct_from_cold_chain=False,
        )

        after = (
            VaccineBatch.query.filter_by(batch_number="ROT-OPTOUT")
            .first()
            .quantity_remaining_vials
        )
        assert after == before, "No vial deducted when deduct_from_cold_chain=False"


# ── T6.8  Enhanced ANC vitals ─────────────────────────────────────────────────


def test_anc_visit_records_clinical_vitals(app):
    """ANC visit should persist all optional clinical measurement fields."""
    with app.app_context():
        _patient("CC-MOM-01")
        visit = mch_engine.log_anc_visit(
            patient_id="CC-MOM-01",
            visit_number=2,
            gestation_weeks=28,
            blood_pressure_systolic=120,
            blood_pressure_diastolic=80,
            weight_kg=65.5,
            fundal_height_cm=28.0,
            foetal_heart_rate=144,
            haemoglobin_g_dl=11.2,
            urine_protein="NEGATIVE",
            hiv_status="NEGATIVE",
        )
        saved = db.session.get(AncVisit, visit.id)
        assert saved.blood_pressure_systolic == 120
        assert saved.blood_pressure_diastolic == 80
        assert saved.weight_kg == 65.5
        assert saved.fundal_height_cm == 28.0
        assert saved.foetal_heart_rate == 144
        assert saved.haemoglobin_g_dl == 11.2
        assert saved.urine_protein == "NEGATIVE"
        assert saved.hiv_status == "NEGATIVE"


def test_anc_visit_without_vitals_still_works(app):
    """ANC visit with no vitals should work fine (all optional)."""
    with app.app_context():
        _patient("CC-MOM-02")
        visit = mch_engine.log_anc_visit(
            patient_id="CC-MOM-02",
            visit_number=1,
            gestation_weeks=12,
        )
        assert visit.id is not None
        assert visit.blood_pressure_systolic is None


# ── T6.9  API smoke tests ─────────────────────────────────────────────────────


def test_api_receive_batch(client, app, nursing_user):
    """POST /mch/api/cold-chain/receive-batch should return 201."""
    client.post(
        "/login", data={"username": nursing_user.username, "password": "Password123!"}
    )
    resp = client.post(
        "/mch/api/cold-chain/receive-batch",
        json={
            "vaccine_name": "BCG",
            "batch_number": "BCG-API-01",
            "manufacturer": "KEMSA",
            "quantity": 30,
            "doses_per_vial": 1,
            "expiry_date": FUTURE_DATE.isoformat(),
            "storage_location": "FRIDGE_A",
            "supplied_by": "UNICEF",
        },
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["vaccine_name"] == "BCG"


def test_api_receive_batch_missing_fields(client, app, nursing_user):
    """POST /mch/api/cold-chain/receive-batch with missing fields returns 400."""
    client.post(
        "/login", data={"username": nursing_user.username, "password": "Password123!"}
    )
    resp = client.post(
        "/mch/api/cold-chain/receive-batch",
        json={"vaccine_name": "BCG"},  # missing required fields
    )
    assert resp.status_code == 400


def test_api_log_temperature(client, app, nursing_user):
    """POST /mch/api/cold-chain/temperature-log should return 201."""
    client.post(
        "/login", data={"username": nursing_user.username, "password": "Password123!"}
    )
    resp = client.post(
        "/mch/api/cold-chain/temperature-log",
        json={"storage_location": "FRIDGE_API", "temperature_celsius": 4.5},
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["is_breach"] is False


def test_api_log_temperature_breach(client, app, nursing_user):
    """POST temperature-log with a TOO_HOT temperature should return 201 and flag breach."""
    client.post(
        "/login", data={"username": nursing_user.username, "password": "Password123!"}
    )
    resp = client.post(
        "/mch/api/cold-chain/temperature-log",
        json={"storage_location": "FRIDGE_API_HOT", "temperature_celsius": 30.0},
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["is_breach"] is True
    assert data["breach_type"] == "TOO_HOT"


def test_api_stock_summary(client, app, nursing_user):
    """GET /mch/api/cold-chain/stock should return 200."""
    client.post(
        "/login", data={"username": nursing_user.username, "password": "Password123!"}
    )
    resp = client.get("/mch/api/cold-chain/stock")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "stock" in data


def test_api_temperature_history(client, app, nursing_user):
    """GET /mch/api/cold-chain/temperature-history/<location> should return 200."""
    client.post(
        "/login", data={"username": nursing_user.username, "password": "Password123!"}
    )
    resp = client.get("/mch/api/cold-chain/temperature-history/FRIDGE_A")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "logs" in data


def test_api_near_expiry_alerts(client, app, nursing_user):
    """GET /mch/api/cold-chain/alerts/near-expiry should return 200."""
    client.post(
        "/login", data={"username": nursing_user.username, "password": "Password123!"}
    )
    resp = client.get("/mch/api/cold-chain/alerts/near-expiry?days=60")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "alerts" in data
    assert data["threshold_days"] == 60


def test_api_missing_temperature_fields(client, app, nursing_user):
    """POST temperature-log missing required fields returns 400."""
    client.post(
        "/login", data={"username": nursing_user.username, "password": "Password123!"}
    )
    resp = client.post(
        "/mch/api/cold-chain/temperature-log",
        json={"storage_location": "FRIDGE_A"},  # missing temperature_celsius
    )
    assert resp.status_code == 400
