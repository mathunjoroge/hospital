import json
import unittest

from werkzeug.security import generate_password_hash

from app import app, db
from departments.api.audit import log_audit_event
from departments.models.user import User


class TestAuditAndPWA(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["RATELIMIT_ENABLED"] = False
        try:
            from app import limiter

            limiter.enabled = False
        except ImportError:
            pass
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

        db.create_all()

        # Create test users if they don't exist

        admin = User.query.filter_by(username="admin_audit_test").first()
        if not admin:
            admin = User(
                username="admin_audit_test",
                password=generate_password_hash(
                    "AdminTest123!", method="pbkdf2:sha256"
                ),
                role="admin",
            )
            db.session.add(admin)
            db.session.commit()

    def tearDown(self):
        self.client.get("/logout")
        db.session.remove()
        self.app_context.pop()

    def _login(self, username, password):
        return self.client.post(
            "/login",
            data={"username": username, "password": password},
            follow_redirects=True,
        )

    def test_audit_log_model_creation(self):
        """Test creating and serializing AuditLog entry."""
        with app.app_context():
            entry = log_audit_event(
                action="PATIENT_VIEW",
                resource_type="Patient",
                resource_id="P001",
                details={"reason": "routine checkup"},
                user_id=1,
                username="records_user",
            )
            self.assertIsNotNone(entry)
            self.assertIsNotNone(entry.id)
            self.assertEqual(entry.action, "PATIENT_VIEW")
            self.assertEqual(entry.resource_id, "P001")

            d = entry.to_dict()
            self.assertEqual(d["action"], "PATIENT_VIEW")
            self.assertEqual(d["resource_type"], "Patient")

    def test_admin_audit_trail_route(self):
        """Test admin audit trail web view and filtering."""
        with app.app_context():
            log_audit_event(
                action="BILL_PAYMENT",
                resource_type="Invoice",
                resource_id="INV-100",
                user_id=1,
                username="admin",
            )
            log_audit_event(
                action="PATIENT_CREATE",
                resource_type="Patient",
                resource_id="P100",
                user_id=1,
                username="admin",
            )

        # Login as admin
        self._login("admin_audit_test", "AdminTest123!")

        res = self.client.get("/admin/audit-trail")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"System Audit Trail", res.data)
        self.assertIn(b"BILL_PAYMENT", res.data)

        # Test JSON format
        res_json = self.client.get("/admin/audit-trail?format=json")
        self.assertEqual(res_json.status_code, 200)
        data = res_json.get_json()
        self.assertIn("logs", data)
        self.assertGreaterEqual(len(data["logs"]), 2)

    def test_export_audit_trail_siem(self):
        """Test exporting audit logs in SIEM JSON format."""
        with app.app_context():
            log_audit_event(
                action="LOGIN_SUCCESS",
                resource_type="User",
                resource_id="1",
                user_id=1,
                username="admin",
            )

        self._login("admin_audit_test", "AdminTest123!")

        res = self.client.get("/admin/audit-trail/export")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["system"], "HMIS")
        self.assertIn("audit_logs", data)
        self.assertGreaterEqual(data["count"], 1)

    def test_pwa_static_assets(self):
        """Test PWA manifest, service worker, and offline page accessibility."""
        res_manifest = self.client.get("/static/manifest.json")
        self.assertEqual(res_manifest.status_code, 200)
        manifest_data = json.loads(res_manifest.data)
        self.assertEqual(manifest_data["display"], "standalone")

        res_sw = self.client.get("/static/sw.js")
        self.assertEqual(res_sw.status_code, 200)
        self.assertIn(b"hims-pwa-v1", res_sw.data)

        res_offline = self.client.get("/static/offline.html")
        self.assertEqual(res_offline.status_code, 200)
        self.assertIn(b"Network Disconnected", res_offline.data)


if __name__ == "__main__":
    unittest.main()
