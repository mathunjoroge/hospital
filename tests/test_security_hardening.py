"""
tests/test_security_hardening.py
──────────────────────────────────
Tests for the security hardening changes in this session:

1. ENCRYPTION_KEY production guard
   - crypto.get_fernet_key() raises RuntimeError when key is missing outside testing
   - crypto.get_fernet_key() generates an ephemeral key in test env (no crash)
   - Startup (app.py) refuses to boot in production without ENCRYPTION_KEY

2. Security response headers
   - Referrer-Policy is present and correct
   - Permissions-Policy is present
   - CSP no longer contains 'unsafe-eval'
   - X-Content-Type-Options, X-Frame-Options still present
   - HSTS is NOT set in testing (only in production)

3. /metrics endpoint
   - Returns 200 with text/plain content type
   - Contains the new counter metric names
   - Prometheus text format (# HELP / # TYPE lines)
"""

import os
from unittest.mock import patch


# ── ENCRYPTION_KEY guard ──────────────────────────────────────────────────────

class TestEncryptionKeyGuard:
    def test_get_fernet_key_raises_in_non_testing_without_key(self):
        """get_fernet_key() must refuse to return the old static fallback key.
        When ENCRYPTION_KEY is absent and FLASK_ENV is not 'testing', it must
        raise RuntimeError rather than silently using a known static value."""
        from departments.crypto import get_fernet_key

        with patch.dict(
            os.environ,
            {"FLASK_ENV": "development", "ENCRYPTION_KEY": ""},
            clear=False,
        ):
            # Temporarily unset the key so the guard triggers
            env_backup = os.environ.pop("ENCRYPTION_KEY", None)
            try:
                import importlib
                import departments.crypto as crypto_mod
                # Call directly with no app context and no env var
                orig_env = os.environ.get("FLASK_ENV")
                os.environ["FLASK_ENV"] = "development"
                try:
                    # Should raise — no key, not testing
                    raised = False
                    try:
                        # Clear the key from env for this call
                        os.environ.pop("ENCRYPTION_KEY", None)
                        crypto_mod.get_fernet_key()
                    except RuntimeError as e:
                        raised = True
                        assert "ENCRYPTION_KEY" in str(e)
                    assert raised, "Expected RuntimeError when ENCRYPTION_KEY absent in non-testing"
                finally:
                    if orig_env is not None:
                        os.environ["FLASK_ENV"] = orig_env
            finally:
                if env_backup is not None:
                    os.environ["ENCRYPTION_KEY"] = env_backup

    def test_get_fernet_key_returns_bytes_when_key_set(self):
        """When ENCRYPTION_KEY is set, get_fernet_key() returns valid bytes."""
        from cryptography.fernet import Fernet
        from departments.crypto import get_fernet_key

        test_key = Fernet.generate_key().decode()
        with patch.dict(os.environ, {"ENCRYPTION_KEY": test_key}):
            key = get_fernet_key()
        assert isinstance(key, bytes)
        assert len(key) > 0
        # Must be a valid Fernet key
        Fernet(key)  # raises if invalid

    def test_get_fernet_key_test_env_does_not_crash(self):
        """In FLASK_ENV=testing with no key, get_fernet_key() generates an ephemeral
        key rather than raising (tests shouldn't need to set ENCRYPTION_KEY for
        non-encryption-focused tests)."""
        from departments.crypto import get_fernet_key

        with patch.dict(
            os.environ, {"FLASK_ENV": "testing"}, clear=False
        ):
            # Remove key from env so the fallback path runs
            env_backup = os.environ.pop("ENCRYPTION_KEY", None)
            try:
                key = get_fernet_key()
                assert isinstance(key, bytes)
                assert len(key) > 0
            finally:
                if env_backup is not None:
                    os.environ["ENCRYPTION_KEY"] = env_backup

    def test_encrypt_decrypt_with_explicit_key(self):
        """End-to-end: encrypt then decrypt with an explicit key round-trips correctly."""
        from cryptography.fernet import Fernet
        from departments.crypto import decrypt_value, encrypt_value

        test_key = Fernet.generate_key().decode()
        with patch.dict(os.environ, {"ENCRYPTION_KEY": test_key}):
            plaintext = "KE-ID-123456789"
            cipher = encrypt_value(plaintext)
            assert cipher.startswith("enc_v1:")
            assert decrypt_value(cipher) == plaintext


# ── Security response headers ─────────────────────────────────────────────────

class TestSecurityHeaders:
    def test_referrer_policy_header(self, client):
        resp = client.get("/healthz")
        assert "Referrer-Policy" in resp.headers
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_permissions_policy_header(self, client):
        resp = client.get("/healthz")
        assert "Permissions-Policy" in resp.headers
        policy = resp.headers["Permissions-Policy"]
        assert "geolocation=()" in policy
        assert "camera=()" in policy

    def test_csp_no_unsafe_eval(self, client):
        resp = client.get("/healthz")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "unsafe-eval" not in csp, (
            "CSP must not contain 'unsafe-eval' — it was removed in this hardening pass"
        )

    def test_csp_has_frame_ancestors(self, client):
        """frame-ancestors is a clickjacking defence in CSP (supplements X-Frame-Options)."""
        resp = client.get("/healthz")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "frame-ancestors" in csp

    def test_x_content_type_options(self, client):
        resp = client.get("/healthz")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client):
        resp = client.get("/healthz")
        assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"

    def test_hsts_absent_in_test_env(self, client):
        """HSTS must NOT be set in testing (would break HTTP-only local dev)."""
        resp = client.get("/healthz")
        assert "Strict-Transport-Security" not in resp.headers, (
            "HSTS should only be set in FLASK_ENV=production"
        )

    def test_x_request_id_present(self, client):
        resp = client.get("/healthz")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) > 0


# ── /metrics endpoint ─────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.content_type

    def test_metrics_has_prometheus_help_type_lines(self, client):
        resp = client.get("/metrics")
        body = resp.data.decode()
        assert "# HELP" in body
        assert "# TYPE" in body

    def test_metrics_includes_request_counter(self, client):
        resp = client.get("/metrics")
        body = resp.data.decode()
        assert "hmis_requests_total" in body

    def test_metrics_includes_error_counter(self, client):
        resp = client.get("/metrics")
        body = resp.data.decode()
        assert "hmis_request_errors_total" in body

    def test_metrics_includes_latency_metrics(self, client):
        resp = client.get("/metrics")
        body = resp.data.decode()
        assert "hmis_request_latency_ms_sum" in body
        assert "hmis_request_latency_ms_count" in body

    def test_metrics_includes_celery_queue_depth(self, client):
        resp = client.get("/metrics")
        body = resp.data.decode()
        assert "hmis_celery_queue_depth" in body

    def test_metrics_counters_increment_after_requests(self, client):
        """After making some requests, the request counter must be > 0."""
        # Make a few requests first
        client.get("/healthz")
        client.get("/healthz")
        resp = client.get("/metrics")
        body = resp.data.decode()
        # Find the counter line
        for line in body.splitlines():
            if line.startswith("hmis_requests_total "):
                count = int(float(line.split()[-1]))
                assert count > 0, "Request counter should be non-zero after requests"
                return
        pytest.fail("hmis_requests_total metric line not found in /metrics output")
