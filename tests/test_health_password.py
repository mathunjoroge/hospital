import os
import unittest

from app import app, db
from departments.api.security import validate_password_strength
from scripts.backup_db import perform_backup


class TestHealthAndPasswordPolicy(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

    def tearDown(self):
        db.session.remove()
        self.app_context.pop()

    def test_password_strength_validator(self):
        """Test password complexity validation rules."""
        # Short password
        valid, msg = validate_password_strength("Short1!")
        self.assertFalse(valid)
        self.assertIn("at least 8 characters", msg)

        # Missing uppercase
        valid, msg = validate_password_strength("lowercase1!")
        self.assertFalse(valid)
        self.assertIn("uppercase letter", msg)

        # Missing lowercase
        valid, msg = validate_password_strength("UPPERCASE1!")
        self.assertFalse(valid)
        self.assertIn("lowercase letter", msg)

        # Missing digit
        valid, msg = validate_password_strength("NoDigitsHere!")
        self.assertFalse(valid)
        self.assertIn("numeric digit", msg)

        # Missing special character
        valid, msg = validate_password_strength("NoSpecialChar123")
        self.assertFalse(valid)
        self.assertIn("special character", msg)

        # Valid strong password
        valid, msg = validate_password_strength("ValidP@ssw0rd2026")
        self.assertTrue(valid)
        self.assertIn("meets complexity", msg)

    def test_healthz_endpoint(self):
        """Test GET /healthz endpoint returns 200 OK and expected status metrics."""
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["system"], "HMIS")
        self.assertIn("checks", data)
        self.assertEqual(data["checks"]["database"], "connected")
        self.assertIn("disk", data["checks"])
        self.assertIn("free_gb", data["checks"]["disk"])

    def test_database_backup_script(self):
        """Test perform_backup script execution."""
        backup_file = perform_backup()
        self.assertIsNotNone(backup_file)
        self.assertTrue(os.path.exists(backup_file))
        self.assertGreater(os.path.getsize(backup_file), 0)

    def test_secret_key_raises_in_development_env(self):
        """SECRET_KEY must raise RuntimeError in development env (not just production)."""
        import importlib
        from unittest.mock import patch

        original_secret_key = os.environ.get("SECRET_KEY")
        original_flask_env = os.environ.get("FLASK_ENV")
        try:
            os.environ.pop("SECRET_KEY", None)
            os.environ["FLASK_ENV"] = "development"
            import app as app_module

            with patch("dotenv.load_dotenv"):
                with self.assertRaises(RuntimeError) as ctx:
                    importlib.reload(app_module)
            self.assertIn("SECRET_KEY", str(ctx.exception))
        finally:
            if original_secret_key:
                os.environ["SECRET_KEY"] = original_secret_key
            if original_flask_env:
                os.environ["FLASK_ENV"] = original_flask_env
            else:
                os.environ.pop("FLASK_ENV", None)
            import app as app_module

            importlib.reload(app_module)

    def test_secret_key_raises_when_env_unset(self):
        """SECRET_KEY must raise RuntimeError when FLASK_ENV is not set at all."""
        import importlib
        from unittest.mock import patch

        original_secret_key = os.environ.get("SECRET_KEY")
        original_flask_env = os.environ.get("FLASK_ENV")
        try:
            os.environ.pop("SECRET_KEY", None)
            os.environ.pop("FLASK_ENV", None)
            import app as app_module

            with patch("dotenv.load_dotenv"):
                with self.assertRaises(RuntimeError) as ctx:
                    importlib.reload(app_module)
            self.assertIn("SECRET_KEY", str(ctx.exception))
        finally:
            if original_secret_key:
                os.environ["SECRET_KEY"] = original_secret_key
            if original_flask_env:
                os.environ["FLASK_ENV"] = original_flask_env
            else:
                os.environ.pop("FLASK_ENV", None)
            import app as app_module

            importlib.reload(app_module)

    def test_secret_key_does_not_raise_in_testing_env(self):
        """SECRET_KEY must NOT raise RuntimeError when FLASK_ENV=testing."""
        import importlib

        original_secret_key = os.environ.get("SECRET_KEY")
        original_flask_env = os.environ.get("FLASK_ENV")
        try:
            os.environ.pop("SECRET_KEY", None)
            os.environ["FLASK_ENV"] = "testing"
            import app as app_module

            # Should not raise
            importlib.reload(app_module)
        finally:
            if original_secret_key:
                os.environ["SECRET_KEY"] = original_secret_key
            if original_flask_env:
                os.environ["FLASK_ENV"] = original_flask_env
            else:
                os.environ.pop("FLASK_ENV", None)
            import app as app_module

            importlib.reload(app_module)


if __name__ == "__main__":
    unittest.main()
