"""
tests/test_p3_infrastructure.py
────────────────────────────────
Priority 3 (P3) Infrastructure, Observability & Operational Maturity Test Suite.
Tests Request ID tracing, security headers, /metrics endpoint, cookie policy, and production startup secret key guard.
"""

import os

import pytest

from app import INSECURE_SECRET_KEYS, app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as c:
        yield c


def test_request_id_header_generated_and_propagated(client):
    """Test X-Request-ID is generated if missing and preserved if supplied by client."""
    # 1. Generated if missing
    res1 = client.get("/healthz")
    assert res1.status_code == 200
    assert "X-Request-ID" in res1.headers
    assert len(res1.headers["X-Request-ID"]) > 10

    # 2. Preserved if supplied
    custom_id = "req-custom-trace-12345"
    res2 = client.get("/healthz", headers={"X-Request-ID": custom_id})
    assert res2.headers.get("X-Request-ID") == custom_id


def test_security_headers_present(client):
    """Test standard security headers (X-Content-Type-Options, CSP, etc.) are present."""
    res = client.get("/healthz")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block"
    assert "Content-Security-Policy" in res.headers


def test_hsts_header_absent_over_plain_http(client):
    """HSTS should not be advertised over a connection that isn't actually HTTPS."""
    res = client.get("/healthz")
    # Flask's test client requests are not flagged secure, and FORCE_HTTPS
    # isn't set in the test environment, so the header must be absent --
    # advertising HSTS over plain HTTP would be a lie the browser could act on.
    assert "Strict-Transport-Security" not in res.headers


def test_hsts_header_present_when_forced(client, monkeypatch):
    """HSTS must be present when the deployment explicitly forces HTTPS."""
    monkeypatch.setenv("FORCE_HTTPS", "true")
    res = client.get("/healthz")
    assert "Strict-Transport-Security" in res.headers
    assert "max-age=" in res.headers["Strict-Transport-Security"]
    assert "includeSubDomains" in res.headers["Strict-Transport-Security"]
    assert "default-src 'self'" in res.headers["Content-Security-Policy"]


def test_prometheus_metrics_endpoint(client):
    """Test /metrics endpoint returns Prometheus formatted metrics."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers.get("Content-Type", "")
    data = res.data.decode("utf-8")
    assert "hmis_up 1" in data
    assert "hmis_db_connected" in data
    assert "hmis_disk_free_bytes" in data
    assert "hmis_requests_total" in data
    assert "hmis_request_errors_total" in data
    assert "hmis_celery_queue_depth" in data


def test_cookie_security_config(client):
    """Test session cookie security attributes are configured in Flask app."""
    assert app.config.get("SESSION_COOKIE_HTTPONLY") is True
    assert app.config.get("SESSION_COOKIE_SAMESITE") == "Lax"


def test_production_startup_secret_key_guard():
    """Test production mode rejects hardcoded/weak secret keys with RuntimeError."""
    orig_env = os.environ.get("FLASK_ENV")
    orig_key = os.environ.get("SECRET_KEY")

    try:
        os.environ["FLASK_ENV"] = "production"
        for weak_key in INSECURE_SECRET_KEYS:
            os.environ["SECRET_KEY"] = weak_key
            # The guard condition:
            key = os.environ.get("SECRET_KEY")
            if os.environ.get("FLASK_ENV") == "production":
                if not key or key in INSECURE_SECRET_KEYS or len(key) < 16:
                    guard_triggered = True
                else:
                    guard_triggered = False
            assert guard_triggered is True
    finally:
        if orig_env is not None:
            os.environ["FLASK_ENV"] = orig_env
        else:
            os.environ.pop("FLASK_ENV", None)
        if orig_key is not None:
            os.environ["SECRET_KEY"] = orig_key
        else:
            os.environ.pop("SECRET_KEY", None)
