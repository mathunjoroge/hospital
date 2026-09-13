"""
tests/test_hipaa_hitrust_compliance.py
───────────────────────────────────────
Unit & Integration tests for HIPAA / HITRUST CSF Certification Readiness & Cryptographic Audit Controls.
Tests:
  - Cryptographic SHA-256 hash chaining on audit logs (HIPAA § 164.312(b))
  - Tamper detection in verify_audit_log_chain()
  - Automated 5-domain HIPAAComplianceEngine evaluations
  - Compliance Dashboard view (/compliance/hipaa-dashboard)
  - Compliance API endpoints (/compliance/api/hipaa-status & /compliance/api/verify-audit-chain)
"""


from departments.audit import (
    GENESIS_HASH,
    compute_log_hash,
    log_audit_event,
    verify_audit_log_chain,
)
from departments.compliance.hipaa_engine import HIPAAComplianceEngine
from departments.models.admin import Log
from extensions import db


class TestCryptographicAuditLogChain:
    """Test cryptographic SHA-256 hash chaining and tamper detection."""

    def test_compute_log_hash(self):
        h1 = compute_log_hash("2026-09-13T22:00:00+00:00", "INFO", "Test log msg", "1", "audit", GENESIS_HASH)
        assert len(h1) == 64
        assert isinstance(h1, str)

        # Mutating any field changes hash
        h2 = compute_log_hash("2026-09-13T22:00:00+00:00", "INFO", "Mutated log msg", "1", "audit", GENESIS_HASH)
        assert h1 != h2

    def test_log_audit_event_chaining(self, app):
        with app.app_context():
            # Clear logs table for isolated test
            Log.query.delete()
            db.session.commit()

            # Insert 2 chained log entries
            log_audit_event(db.session, "INFO", "Action 1: Patient Created", user_id=1, source="test")
            db.session.commit()

            log_audit_event(db.session, "INFO", "Action 2: Prescribed Meds", user_id=1, source="test")
            db.session.commit()

            logs = Log.query.order_by(Log.id.asc()).all()
            assert len(logs) == 2
            assert logs[0].previous_hash == GENESIS_HASH
            assert logs[0].entry_hash is not None

            assert logs[1].previous_hash == logs[0].entry_hash
            assert logs[1].entry_hash is not None

            # Verify chain is intact
            chain_res = verify_audit_log_chain()
            assert chain_res["valid"] is True
            assert chain_res["total_logs"] == 2
            assert len(chain_res["tampered_logs"]) == 0

    def test_tamper_detection(self, app):
        with app.app_context():
            Log.query.delete()
            db.session.commit()

            log_audit_event(db.session, "INFO", "Original Untampered Entry", user_id=1, source="test")
            db.session.commit()

            # Tamper with the message in the database directly
            tampered_log = Log.query.first()
            tampered_log.message = "MALICIOUSLY TAMPERED ENTRY"
            db.session.commit()

            chain_res = verify_audit_log_chain()
            assert chain_res["valid"] is False
            assert len(chain_res["tampered_logs"]) >= 1
            assert chain_res["tampered_logs"][0]["log_id"] == tampered_log.id


class TestHIPAAComplianceEngine:
    """Test automated 5-domain HIPAA evaluation engine."""

    def test_run_full_hipaa_audit(self, app):
        with app.app_context():
            report = HIPAAComplianceEngine.run_full_hipaa_audit()
            assert "overall_status" in report
            assert report["compliance_score_pct"] >= 80.0
            assert len(report["domains"]) == 5


class TestComplianceRoutes:
    """Test Compliance UI & API Endpoints."""

    def test_hipaa_dashboard_view(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.get("/compliance/hipaa-dashboard")
        assert resp.status_code == 200
        assert b"HIPAA &amp; HITRUST CSF Certification Console" in resp.data or b"HIPAA & HITRUST CSF Certification Console" in resp.data

    def test_api_hipaa_status(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.get("/compliance/api/hipaa-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "report" in data

    def test_api_verify_audit_chain(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.get("/compliance/api/verify-audit-chain")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "verification" in data
