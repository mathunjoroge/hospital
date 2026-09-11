"""Tests for OpenTelemetry observability setup (Phase 2, P2-02)."""
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.mark.forked
def test_flask_auto_instrumentation_emits_span():
    """setup_observability() auto-instruments Flask routes (fresh process)."""
    from flask import Flask

    from departments.observability import setup_observability

    app = Flask("otel_test")
    app.config["OTEL_ENABLED"] = True

    @app.route("/otel-ping")
    def ping():
        return "pong"

    exporter = InMemorySpanExporter()
    provider = setup_observability(app, exporter=exporter)
    assert provider is not None

    resp = app.test_client().get("/otel-ping")
    assert resp.status_code == 200

    spans = exporter.get_finished_spans()
    assert spans, "FlaskInstrumentor should emit a span"
    assert any("otel-ping" in s.name for s in spans)


@pytest.mark.forked
def test_slow_query_emits_span(app, monkeypatch):
    """A DB query over the threshold emits a db.slow_query span."""
    import departments.observability as obs
    from departments.observability import setup_observability
    from extensions import db

    monkeypatch.setattr(obs, "SLOW_QUERY_THRESHOLD_MS", 0.0)

    exporter = InMemorySpanExporter()
    app.config["OTEL_ENABLED"] = True
    provider = setup_observability(app, exporter=exporter)
    assert provider is not None

    with app.app_context():
        db.session.execute(db.text("SELECT 1"))

    spans = exporter.get_finished_spans()
    slow = [s for s in spans if s.name == "db.slow_query"]
    assert slow, f"Expected db.slow_query span, got: {[s.name for s in spans]}"


@pytest.mark.forked
def test_external_requests_instrumentation(app):
    """RequestsInstrumentor emits a span for outbound HTTP calls."""
    import requests

    from departments.observability import setup_observability

    exporter = InMemorySpanExporter()
    app.config["OTEL_ENABLED"] = True
    provider = setup_observability(app, exporter=exporter)
    assert provider is not None

    # Trigger a ConnectionError on a closed port. OTel still records the span.
    try:
        requests.get("http://127.0.0.1:1", timeout=0.1)
    except requests.exceptions.ConnectionError:
        pass

    spans = exporter.get_finished_spans()
    http_spans = [s for s in spans if s.name.startswith(("GET", "HTTP"))]
    assert http_spans, f"Expected HTTP span, got: {[s.name for s in spans]}"
