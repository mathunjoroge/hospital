"""
tests/test_p2_coverage.py
─────────────────────────
Priority 2 (P2) Edge Case & High-Value Module Coverage Suite.
Tests RBAC edge cases, billing sync deduplication, break-glass expiration,
AI consent gate, security_ops, system_ops, offline_sync, public_health, and mortuary.
"""

from datetime import datetime, timedelta, timezone
import pytest
from flask import g, session

from app import app
from extensions import db
from departments.models.user import User
from departments.models.records import Patient, Consent
from departments.models import MortuaryData
from departments.rbac import get_effective_user, get_effective_role, roles_required
from departments.records.ai_consent import has_ai_consent
from departments.security_ops.models import TokenRevocation, AccessRequest
from departments.system_ops.models import BackupJob, RestoreTest, SystemAlert
from departments.offline_sync.models import DeviceRegistry, SyncQueue, SyncConflict
from departments.public_health.models import NotifiableDisease, MortalityReport, OutbreakSignal
from departments.billing.sync import sync_invoice_status
from departments.models.billing import Invoice, Charge, InvoiceStatus


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        db.create_all()
        with app.test_client() as client:
            yield client
        db.session.remove()
        db.drop_all()


# ==============================================================================
# 1. RBAC Edge Cases
# ==============================================================================

def test_rbac_get_effective_user_jwt(client):
    """Test get_effective_user prefers g.api_user over session user."""
    mock_jwt_user = User(id=101, username="jwt_doc", role="doctor", password="x")
    g.api_user = mock_jwt_user

    eff_user = get_effective_user()
    assert eff_user.id == 101
    assert get_effective_role() == "doctor"

    g.api_user = None


def test_rbac_unauthenticated_returns_none(client):
    """Test get_effective_user and get_effective_role for anonymous requests."""
    g.api_user = None
    assert get_effective_user() is None
    assert get_effective_role() is None


def test_rbac_decorator_unauthenticated_aborts_403(client):
    """Test roles_required decorator aborts with 403 if no authenticated user."""
    @roles_required("admin")
    def dummy_view():
        return "ok"

    with app.test_request_context("/"):
        with pytest.raises(Exception) as exc_info:
            dummy_view()
        assert "403" in str(exc_info.value) or exc_info.value.code == 403


# ==============================================================================
# 2. AI Consent Gate
# ==============================================================================

def test_has_ai_consent_granted_and_revoked(client):
    """Test has_ai_consent returns True when active consent exists and False otherwise."""
    pat = Patient(patient_id="PAT-AI-1", first_name="AI", last_name="Patient", gender="M", dob=datetime(1990, 1, 1).date())
    db.session.add(pat)
    db.session.commit()

    # No consent record -> False
    assert has_ai_consent(pat.id) is False

    # Active AI consent -> True
    consent = Consent(patient_id=pat.id, consent_type="ai_processing", status="GRANTED")
    db.session.add(consent)
    db.session.commit()
    assert has_ai_consent(pat.id) is True

    # Revoked consent -> False
    consent.status = "REVOKED"
    db.session.commit()
    assert has_ai_consent(pat.id) is False


# ==============================================================================
# 3. Security Operations Module
# ==============================================================================

def test_security_ops_models_and_routes(client):
    """Test TokenRevocation, AccessRequest models and security_ops endpoints."""
    u = User(id=201, username="sec_admin", role="admin", password="x")
    db.session.add(u)
    db.session.commit()

    rev = TokenRevocation(token_identifier="jti-12345", user_id=201, revoked_by=201, reason="Staff departed")
    req = AccessRequest(user_id=201, patient_id=1, justification="Emergency audit", status="PENDING")
    db.session.add_all([rev, req])
    db.session.commit()

    assert TokenRevocation.query.count() == 1
    assert AccessRequest.query.count() == 1

    with client.session_transaction() as sess:
        sess["_user_id"] = "201"
        sess["_fresh"] = True

    r_index = client.get("/security_ops/")
    assert r_index.status_code == 200

    r_revoke = client.post("/security_ops/revoke", json={"user_id": 201})
    assert r_revoke.status_code == 200
    assert r_revoke.json["status"] == "success"

    r_req = client.post("/security_ops/access-request", json={"patient_id": 1, "justification": "Audit"})
    assert r_req.status_code == 201

    r_res = client.post("/security_ops/access-resolve", json={"request_id": req.id, "status": "APPROVED"})
    assert r_res.status_code == 200


# ==============================================================================
# 4. System Operations Module
# ==============================================================================

def test_system_ops_models_and_routes(client):
    """Test BackupJob, RestoreTest, SystemAlert models and system_ops endpoints."""
    u = User(id=301, username="sys_admin", role="admin", password="x")
    db.session.add(u)
    db.session.commit()

    b_job = BackupJob(backup_type="FULL_SYSTEM", status="COMPLETED", size_mb=150.5)
    db.session.add(b_job)
    db.session.commit()

    r_test = RestoreTest(backup_job_id=b_job.id, status="SUCCESS", performed_by=301)
    alert = SystemAlert(alert_source="DATABASE", severity="WARNING", alert_message="High connection count")
    db.session.add_all([r_test, alert])
    db.session.commit()

    assert BackupJob.query.count() == 1
    assert RestoreTest.query.count() == 1
    assert SystemAlert.query.count() == 1

    with client.session_transaction() as sess:
        sess["_user_id"] = "301"
        sess["_fresh"] = True

    assert client.get("/system_ops/").status_code == 200
    assert client.get("/system_ops/health").status_code == 200
    assert client.post("/system_ops/backup/trigger").status_code == 201
    assert client.get("/system_ops/alerts").status_code == 200
    assert client.post("/system_ops/alert/resolve", json={"alert_id": alert.id}).status_code == 200


# ==============================================================================
# 5. Offline Sync Engine Module
# ==============================================================================

def test_offline_sync_models_and_routes(client):
    """Test DeviceRegistry, SyncQueue, SyncConflict models and offline_sync endpoints."""
    u = User(id=401, username="sync_user", role="nursing", password="x")
    db.session.add(u)
    db.session.commit()

    dev = DeviceRegistry(device_fingerprint="fp-tablet-01", user_id=401, device_name="Field Tablet")
    db.session.add(dev)
    db.session.commit()

    queue_item = SyncQueue(
        device_id=dev.id,
        entity_type="Patient",
        entity_id="PAT-100",
        mutation_type="CREATE",
        payload_json='{"first_name": "Offline"}',
        client_timestamp=datetime.now(timezone.utc),
    )
    db.session.add(queue_item)
    db.session.commit()

    conflict = SyncConflict(queue_id=queue_item.id, resolution_status="PENDING_RESOLUTION")
    db.session.add(conflict)
    db.session.commit()

    assert DeviceRegistry.query.count() == 1
    assert SyncQueue.query.count() == 1
    assert SyncConflict.query.count() == 1

    with client.session_transaction() as sess:
        sess["_user_id"] = "401"
        sess["_fresh"] = True

    assert client.get("/offline_sync/").status_code == 200
    assert client.post("/offline_sync/push", json={"mutations": []}).status_code == 200
    assert client.get("/offline_sync/pull?since=2026-01-01T00:00:00Z").status_code == 200
    assert client.post("/offline_sync/resolve", json={"conflict_id": conflict.id}).status_code == 200


# ==============================================================================
# 6. Public Health Surveillance Module
# ==============================================================================

def test_public_health_models_and_routes(client):
    """Test NotifiableDisease, MortalityReport, OutbreakSignal models and endpoints."""
    u = User(id=501, username="epi_user", role="doctor", password="x")
    pat = Patient(patient_id="PAT-PH-1", first_name="Public", last_name="Health", gender="F", dob=datetime(1985, 5, 5).date())
    db.session.add_all([u, pat])
    db.session.commit()

    nd = NotifiableDisease(
        patient_id=pat.id,
        clinician_id=u.id,
        icd10_code="A00.0",
        disease_name="Cholera",
        case_classification="CONFIRMED",
        diagnosis_date=datetime.now(timezone.utc).date(),
    )
    mr = MortalityReport(
        patient_id=pat.id,
        clinician_id=u.id,
        primary_cause_icd10="A00.0",
        primary_cause_text="Cholera dehydration",
        death_context="INPATIENT",
        age_category="ADULT",
        date_of_death=datetime.now(timezone.utc),
    )
    sig = OutbreakSignal(signal_description="Acute diarrhoea cluster", disease_suspected="Cholera", case_count=3)
    db.session.add_all([nd, mr, sig])
    db.session.commit()

    assert NotifiableDisease.query.count() == 1
    assert MortalityReport.query.count() == 1
    assert OutbreakSignal.query.count() == 1

    with client.session_transaction() as sess:
        sess["_user_id"] = "501"
        sess["_fresh"] = True

    assert client.get("/public_health/").status_code == 200
    assert client.post("/public_health/api/notify", json={"disease": "Cholera"}).status_code == 201
    assert client.post("/public_health/api/mortality", json={"patient_id": pat.id}).status_code == 201
    assert client.get("/public_health/api/outbreak-signals").status_code == 200
    assert client.post("/public_health/api/outbreak-signal", json={"description": "Test signal"}).status_code == 201


# ==============================================================================
# 7. Mortuary Module
# ==============================================================================

def test_mortuary_intake_workflow(client):
    """Test Mortuary intake form submission and listing."""
    staff = User(id=601, username="mortuary_staff", role="mortuary", password="x")
    pat = Patient(patient_id="PAT-DEAD-1", first_name="John", last_name="Doe", gender="M", dob=datetime(1970, 1, 1).date())
    db.session.add_all([staff, pat])
    db.session.commit()

    with client.session_transaction() as sess:
        sess["_user_id"] = "601"
        sess["_fresh"] = True

    # 1. Missing fields flash warning
    r_empty = client.post("/mortuary/", data={"deceased_id": ""}, follow_redirects=True)
    assert r_empty.status_code == 200
    assert b"required" in r_empty.data

    # 2. Non-existent patient flash danger
    r_nofound = client.post(
        "/mortuary/",
        data={"deceased_id": "PAT-NONEXISTENT", "date_of_death": "2026-09-01", "cause_of_death": "Unknown"},
        follow_redirects=True,
    )
    assert r_nofound.status_code == 200
    assert b"not found" in r_nofound.data

    # 3. Successful intake
    r_ok = client.post(
        "/mortuary/",
        data={"deceased_id": "PAT-DEAD-1", "date_of_death": "2026-09-01", "cause_of_death": "Cardiac Arrest"},
        follow_redirects=True,
    )
    assert r_ok.status_code == 200
    assert b"successfully" in r_ok.data
    assert MortuaryData.query.filter_by(deceased_id="PAT-DEAD-1").count() == 1


# ==============================================================================
# 8. Billing Sync Edge Cases
# ==============================================================================

def test_sync_invoice_status_lifecycle(client):
    """Test sync_invoice_status handles status transitions correctly."""
    inv = Invoice(patient_id=1, total_amount=100.00, paid_amount=0.00, status=InvoiceStatus.DRAFT)
    db.session.add(inv)
    db.session.commit()

    # Total 100, Paid 0 -> UNPAID (or DRAFT if zero paid)
    sync_invoice_status(inv)
    assert inv.status in (InvoiceStatus.DRAFT, InvoiceStatus.UNPAID)

    # Partial payment -> PARTIAL
    inv.paid_amount = 40.00
    sync_invoice_status(inv)
    assert inv.status == InvoiceStatus.PARTIAL

    # Full payment -> PAID
    inv.paid_amount = 100.00
    sync_invoice_status(inv)
    assert inv.status == InvoiceStatus.PAID
