"""
tests/test_lims_westgard_qc.py
─────────────────────────────────────────────────────────────────────────────
Gap #8 — LIMS Specimen Barcode Tracking, Chain of Custody & Westgard Multi-Rule QC

Test coverage areas:
  T8.1  Specimen barcode generation & creation
  T8.2  Specimen lifecycle transitions (Ordered → Collected → Received)
  T8.3  Specimen rejection workflow
  T8.4  Westgard 1_2s warning rule
  T8.5  Westgard 1_3s rejection rule
  T8.6  Westgard 2_2s rejection rule
  T8.7  Westgard R_4s rejection rule
  T8.8  Westgard 4_1s rejection rule
  T8.9  Westgard 10_x rejection rule
  T8.10 QC control PASS scenario
  T8.11 LIMS service QC logging via DB
  T8.12 API endpoint smoke tests
"""

import json
from datetime import datetime

import pytest
from werkzeug.security import generate_password_hash

from departments.laboratory.lims_service import LIMSService, WestgardEngine
from departments.models.laboratory import LabQCSample, Specimen
from departments.models.records import Patient
from extensions import db

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_patient(patient_id: str) -> None:
    """Insert a minimal patient into the current db session."""
    p = Patient.query.filter_by(patient_id=patient_id).first()
    if not p:
        p = Patient(
            patient_id=patient_id,
            name=f"LIMS Test {patient_id}",
            sex="F",
            date_of_birth=datetime(1990, 1, 1),  # noqa: DTZ001
        )
        db.session.add(p)
        db.session.commit()


def _make_qc_sample(app, *, control_name="Bio-Rad Level 1", param="GLUCOSE",
                    mean=5.0, sd=0.2) -> LabQCSample:
    """Create and persist a LabQCSample inside the app context."""
    with app.app_context():
        sample = LabQCSample(
            control_name=control_name,
            lot_number="LOT-TEST-01",
            analyzer_name="Roche Cobas c501",
            parameter_name=param,
            target_mean=mean,
            target_sd=sd,
        )
        db.session.add(sample)
        db.session.commit()
        return sample


@pytest.fixture
def lab_user(app):
    """An authenticated lab staff user."""
    with app.app_context():
        u = db.session.query(
            __import__("departments.models.user", fromlist=["User"]).User
        ).filter_by(username="lab_staff_001").first()
        if not u:
            from departments.models.user import User
            u = User(
                username="lab_staff_001",
                password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
                role="laboratory",
            )
            db.session.add(u)
            db.session.commit()
        yield u


# ── T8.1  Specimen creation & barcode ────────────────────────────────────────


def test_specimen_barcode_format(app):
    """Generated barcode must follow SPEC-YYYYMMDD-XXXX format."""
    with app.app_context():
        _make_patient("LIMS-P001")
        specimen = LIMSService.create_specimen(patient_id="LIMS-P001")
        assert specimen.barcode.startswith("SPEC-")
        parts = specimen.barcode.split("-")
        assert len(parts) == 3, "Barcode must have 3 dash-separated segments"


def test_specimen_creation_persists_to_db(app):
    """Newly created specimen must be stored in DB with ORDERED status."""
    with app.app_context():
        _make_patient("LIMS-P002")
        specimen = LIMSService.create_specimen(
            patient_id="LIMS-P002",
            specimen_type="SERUM",
            container_type="SST_GOLD",
        )
        saved = db.session.get(Specimen, specimen.id)
        assert saved is not None
        assert saved.specimen_type == "SERUM"
        assert saved.container_type == "SST_GOLD"
        assert saved.status == "ORDERED"


def test_specimen_initial_chain_of_custody(app):
    """Chain of custody must start with a single ORDERED entry."""
    with app.app_context():
        _make_patient("LIMS-P003")
        specimen = LIMSService.create_specimen(patient_id="LIMS-P003")
        coc = json.loads(specimen.chain_of_custody)
        assert len(coc) == 1
        assert coc[0]["status"] == "ORDERED"


def test_specimen_barcodes_are_unique(app):
    """Creating two specimens for same patient must yield unique barcodes."""
    with app.app_context():
        _make_patient("LIMS-P004")
        s1 = LIMSService.create_specimen(patient_id="LIMS-P004")
        s2 = LIMSService.create_specimen(patient_id="LIMS-P004")
        assert s1.barcode != s2.barcode


# ── T8.2  Lifecycle transitions ───────────────────────────────────────────────


def test_specimen_collect_transition(app):
    """ORDERED → COLLECTED sets collected_at timestamp."""
    with app.app_context():
        _make_patient("LIMS-P005")
        specimen = LIMSService.create_specimen(patient_id="LIMS-P005")
        collected = LIMSService.update_specimen_status(
            specimen_id_or_barcode=specimen.barcode, new_status="COLLECTED"
        )
        assert collected.status == "COLLECTED"
        assert collected.collected_at is not None

        coc = json.loads(collected.chain_of_custody)
        assert coc[-1]["status"] == "COLLECTED"


def test_specimen_receive_transition(app):
    """COLLECTED → RECEIVED sets received_at timestamp."""
    with app.app_context():
        _make_patient("LIMS-P006")
        specimen = LIMSService.create_specimen(patient_id="LIMS-P006")
        LIMSService.update_specimen_status(
            specimen_id_or_barcode=specimen.barcode, new_status="COLLECTED"
        )
        received = LIMSService.update_specimen_status(
            specimen_id_or_barcode=specimen.barcode, new_status="RECEIVED"
        )
        assert received.status == "RECEIVED"
        assert received.received_at is not None

        coc = json.loads(received.chain_of_custody)
        statuses = [e["status"] for e in coc]
        assert statuses == ["ORDERED", "COLLECTED", "RECEIVED"]


def test_specimen_lookup_by_id(app):
    """update_specimen_status should work with numeric ID as well as barcode."""
    with app.app_context():
        _make_patient("LIMS-P007")
        specimen = LIMSService.create_specimen(patient_id="LIMS-P007")
        updated = LIMSService.update_specimen_status(
            specimen_id_or_barcode=str(specimen.id), new_status="COLLECTED"
        )
        assert updated.status == "COLLECTED"


# ── T8.3  Rejection workflow ──────────────────────────────────────────────────


def test_specimen_rejection_sets_reason(app):
    """Rejected specimen must have the rejection_reason stored."""
    with app.app_context():
        _make_patient("LIMS-P008")
        specimen = LIMSService.create_specimen(patient_id="LIMS-P008")
        rejected = LIMSService.update_specimen_status(
            specimen_id_or_barcode=specimen.barcode,
            new_status="REJECTED",
            rejection_reason="HEMOLYZED",
            notes="Gross hemolysis on visual inspection",
        )
        assert rejected.status == "REJECTED"
        assert rejected.rejection_reason == "HEMOLYZED"
        coc = json.loads(rejected.chain_of_custody)
        assert coc[-1]["rejection_reason"] == "HEMOLYZED"


def test_unknown_specimen_raises_value_error(app):
    """Trying to update a non-existent specimen should raise ValueError."""
    with app.app_context(), pytest.raises(ValueError, match="not found"):
        LIMSService.update_specimen_status(
            specimen_id_or_barcode="SPEC-INVALID-XXXX",
            new_status="COLLECTED",
        )


# ── T8.4–T8.9  Westgard Multi-Rule Engine (no DB required) ───────────────────


def test_westgard_pass_within_2sd():
    """Z = 0.5 → PASS, no rules violated."""
    status, z, rules = WestgardEngine.evaluate_qc_run(
        measured_value=5.10, mean=5.0, sd=0.2, history_z_scores=[]
    )
    assert status == "PASS"
    assert rules == []


def test_westgard_1_2s_warning():
    """Z = +2.2 (|Z| > 2.0) → WARNING, 1_2s triggered."""
    status, z, rules = WestgardEngine.evaluate_qc_run(
        measured_value=5.44, mean=5.0, sd=0.2, history_z_scores=[]
    )
    assert status == "WARNING"
    assert "1_2s" in rules
    assert "1_3s" not in rules


def test_westgard_1_3s_rejection():
    """Z = +3.3 (|Z| > 3.0) → REJECT, 1_3s triggered."""
    status, z, rules = WestgardEngine.evaluate_qc_run(
        measured_value=5.66, mean=5.0, sd=0.2, history_z_scores=[]
    )
    assert status == "REJECT"
    assert "1_3s" in rules
    assert z == pytest.approx(3.3)


def test_westgard_2_2s_rejection():
    """Two consecutive runs > +2.0 → REJECT, 2_2s triggered."""
    status, z, rules = WestgardEngine.evaluate_qc_run(
        measured_value=5.42,   # Z = +2.1
        mean=5.0, sd=0.2,
        history_z_scores=[2.3],  # previous run Z = +2.3
    )
    assert status == "REJECT"
    assert "2_2s" in rules


def test_westgard_r_4s_rejection():
    """Current Z = +2.1, previous Z = -2.1 → range = 4.2 SD → R_4s REJECT."""
    status, z, rules = WestgardEngine.evaluate_qc_run(
        measured_value=5.42,   # Z = +2.1
        mean=5.0, sd=0.2,
        history_z_scores=[-2.1],
    )
    assert status == "REJECT"
    assert "R_4s" in rules


def test_westgard_4_1s_rejection():
    """4 consecutive results all > +1.0 SD → 4_1s REJECT."""
    history = [1.2, 1.3, 1.1]   # previous 3 runs all > +1.0
    status, z, rules = WestgardEngine.evaluate_qc_run(
        measured_value=5.24,   # Z = +1.2
        mean=5.0, sd=0.2,
        history_z_scores=history,
    )
    assert status == "REJECT"
    assert "4_1s" in rules


def test_westgard_10_x_rejection():
    """10 consecutive results on same side of mean → 10_x REJECT."""
    history = [0.2, 0.4, 0.1, 0.3, 0.5, 0.2, 0.4, 0.1, 0.3]   # 9 positive runs
    status, z, rules = WestgardEngine.evaluate_qc_run(
        measured_value=5.10,   # Z = +0.5 (10th positive run)
        mean=5.0, sd=0.2,
        history_z_scores=history,
    )
    assert status == "REJECT"
    assert "10_x" in rules


def test_westgard_z_score_calculation():
    """Z-score arithmetic must be correct."""
    z = WestgardEngine.calculate_z_score(measured_value=5.4, mean=5.0, sd=0.2)
    assert z == pytest.approx(2.0)


def test_westgard_zero_sd_returns_zero():
    """SD = 0 must not divide by zero; returns Z = 0."""
    z = WestgardEngine.calculate_z_score(measured_value=10.0, mean=5.0, sd=0.0)
    assert z == 0.0


# ── T8.10 QC logging via LIMS service (DB) ───────────────────────────────────


def test_lims_qc_logging_pass(app):
    """QC run exactly on target mean (Z = 0) → status PASS persisted in DB."""
    with app.app_context():
        sample = LabQCSample(
            control_name="Bio-Rad PASS Test",
            lot_number="LOT-QC-PASS",
            analyzer_name="Siemens Atellica",
            parameter_name="CREATININE",
            target_mean=90.0,
            target_sd=5.0,
        )
        db.session.add(sample)
        db.session.commit()

        result = LIMSService.log_qc_result(
            qc_sample_id=sample.id, measured_value=90.0
        )
        assert result.id is not None
        assert result.z_score == pytest.approx(0.0)
        assert result.status == "PASS"


def test_lims_qc_logging_reject(app):
    """QC run far outside limits → status REJECT persisted with rules in DB."""
    with app.app_context():
        sample = LabQCSample(
            control_name="Bio-Rad REJECT Test",
            lot_number="LOT-QC-REJECT",
            analyzer_name="Siemens Atellica",
            parameter_name="ALT",
            target_mean=40.0,
            target_sd=3.0,
        )
        db.session.add(sample)
        db.session.commit()

        # Measured value far outside (Z ≈ +5.0)
        result = LIMSService.log_qc_result(
            qc_sample_id=sample.id, measured_value=55.0
        )
        assert result.status == "REJECT"
        violated = json.loads(result.violated_rules)
        assert "1_3s" in violated


def test_lims_qc_logging_nonexistent_sample(app):
    """Logging against non-existent QC sample should raise ValueError."""
    with app.app_context(), pytest.raises(ValueError, match="not found"):
        LIMSService.log_qc_result(qc_sample_id=99999, measured_value=5.0)


# ── T8.11 Dashboard metrics ───────────────────────────────────────────────────


def test_lims_dashboard_metrics_structure(app):
    """LIMS dashboard metrics should return expected keys."""
    with app.app_context():
        metrics = LIMSService.get_lims_dashboard_metrics()
        assert "total_specimens" in metrics
        assert "status_counts" in metrics
        assert "qc_metrics" in metrics
        assert "pass_rate_pct" in metrics["qc_metrics"]


# ── T8.12 API Endpoint smoke tests ───────────────────────────────────────────


def test_api_create_specimen(client, app, lab_user):
    """POST /laboratory/api/lims/specimens/create → 201 with barcode."""
    with app.app_context():
        _make_patient("LIMS-API-P001")

    client.post("/login", data={"username": lab_user.username, "password": "Password123!"})
    res = client.post(
        "/laboratory/api/lims/specimens/create",
        json={
            "patient_id": "LIMS-API-P001",
            "specimen_type": "URINE",
            "container_type": "STERILE_CUP",
        },
    )
    assert res.status_code == 201
    data = res.get_json()
    assert "barcode" in data
    assert data["barcode"].startswith("SPEC-")
    assert data["status"] == "ORDERED"


def test_api_create_specimen_missing_patient_id(client, app, lab_user):
    """POST /laboratory/api/lims/specimens/create without patient_id → 400."""
    client.post("/login", data={"username": lab_user.username, "password": "Password123!"})
    res = client.post(
        "/laboratory/api/lims/specimens/create", json={"specimen_type": "SERUM"}
    )
    assert res.status_code == 400


def test_api_track_specimen(client, app, lab_user):
    """GET /laboratory/api/lims/specimens/track/<barcode> → 200 with full CoC."""
    with app.app_context():
        _make_patient("LIMS-API-P002")
        specimen = LIMSService.create_specimen(patient_id="LIMS-API-P002")
        barcode = specimen.barcode

    client.post("/login", data={"username": lab_user.username, "password": "Password123!"})
    res = client.get(f"/laboratory/api/lims/specimens/track/{barcode}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["barcode"] == barcode
    assert data["status"] == "ORDERED"
    assert len(data["chain_of_custody"]) == 1


def test_api_track_specimen_not_found(client, app, lab_user):
    """GET /laboratory/api/lims/specimens/track/NONEXISTENT → 404."""
    client.post("/login", data={"username": lab_user.username, "password": "Password123!"})
    res = client.get("/laboratory/api/lims/specimens/track/SPEC-INVALID-XXXX")
    assert res.status_code == 404


def test_api_collect_specimen(client, app, lab_user):
    """POST /laboratory/api/lims/specimens/collect → 200, status COLLECTED."""
    with app.app_context():
        _make_patient("LIMS-API-P003")
        specimen = LIMSService.create_specimen(patient_id="LIMS-API-P003")
        barcode = specimen.barcode

    client.post("/login", data={"username": lab_user.username, "password": "Password123!"})
    res = client.post(
        "/laboratory/api/lims/specimens/collect", json={"barcode": barcode}
    )
    assert res.status_code == 200
    assert res.get_json()["status"] == "COLLECTED"


def test_api_reject_specimen(client, app, lab_user):
    """POST /laboratory/api/lims/specimens/reject → 200, status REJECTED."""
    with app.app_context():
        _make_patient("LIMS-API-P004")
        specimen = LIMSService.create_specimen(patient_id="LIMS-API-P004")
        barcode = specimen.barcode

    client.post("/login", data={"username": lab_user.username, "password": "Password123!"})
    res = client.post(
        "/laboratory/api/lims/specimens/reject",
        json={"barcode": barcode, "rejection_reason": "CLOTTED"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "REJECTED"
    assert data["rejection_reason"] == "CLOTTED"


def test_api_dashboard_metrics(client, app, lab_user):
    """GET /laboratory/api/lims/dashboard-metrics → 200 with proper structure."""
    client.post("/login", data={"username": lab_user.username, "password": "Password123!"})
    res = client.get("/laboratory/api/lims/dashboard-metrics")
    assert res.status_code == 200
    metrics = res.get_json()
    assert "total_specimens" in metrics
    assert "qc_metrics" in metrics
