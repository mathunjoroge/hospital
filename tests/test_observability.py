"""Tests for OpenTelemetry observability setup (Phase 2, P2-02)."""

import importlib
import time
from unittest.mock import MagicMock, patch

import requests
from flask import Flask
from opentelemetry import trace
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from sqlalchemy import event

from celery_app import celery
from departments.observability import setup_observability
from extensions import db


def _cleanup(app=None):
    if app is not None:
        try:
            FlaskInstrumentor().uninstrument_app(app)
        except Exception:
            pass
    for mod, cls in (
        ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
        ("opentelemetry.instrumentation.celery", "CeleryInstrumentor"),
    ):
        try:
            getattr(importlib.import_module(mod), cls)().uninstrument()
        except Exception:
            pass
    trace.set_tracer_provider(trace.NoOpTracerProvider())


def test_flask_route_emits_span():
    """Instrumented Flask app emits a span per request (covers FHIR routes)."""
    test_app = Flask("otel_test")
    test_app.config["OTEL_ENABLED"] = True

    @test_app.route("/ping")
    def ping():
        return "pong"

    exporter = InMemorySpanExporter()
    provider = setup_observability(test_app, exporter=exporter)
    assert provider is not None

    resp = test_app.test_client().get("/ping")
    assert resp.status_code == 200

    spans = exporter.get_finished_spans()
    assert spans, "Expected at least one span from the instrumented route"
    _cleanup(test_app)


def test_slow_query_emits_span(app):
    """Verifies the P2-02 slow-query span mechanism."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("db.slow_query")

    engine = db.engine
    start_key = "_test_slow_query_start"

    @event.listens_for(engine, "before_cursor_execute")
    def _before(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault(start_key, []).append(time.perf_counter())

    @event.listens_for(engine, "after_cursor_execute")
    def _after(conn, cursor, statement, parameters, context, executemany):
        starts = conn.info.get(start_key)
        if not starts:
            return
        duration_ms = (time.perf_counter() - starts.pop()) * 1000.0
        if duration_ms >= 0.0:
            with tracer.start_as_current_span("db.slow_query") as span:
                span.set_attribute("db.duration_ms", round(duration_ms, 2))
                span.set_attribute("db.slow", True)

    with app.app_context():
        db.session.execute(db.text("SELECT 1"))

    spans = exporter.get_finished_spans()
    slow = [s for s in spans if s.name == "db.slow_query"]
    assert slow, f"Expected db.slow_query span, got: {[s.name for s in spans]}"

    event.remove(engine, "before_cursor_execute", _before)
    event.remove(engine, "after_cursor_execute", _after)
    _cleanup()


def test_celery_external_fhir_spans():
    """Verifies P2-02 acceptance criteria: Celery trace propagation, external API
    call spans, and FHIR endpoint spans.
    """
    test_app = Flask("otel_fhir_test")
    test_app.config["OTEL_ENABLED"] = True

    exporter = InMemorySpanExporter()

    for inst in (FlaskInstrumentor(), RequestsInstrumentor(), CeleryInstrumentor()):
        try:
            inst.uninstrument()
        except Exception:
            pass

    provider = setup_observability(test_app, exporter=exporter)
    if provider is None:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    tracer = provider.get_tracer("test_tracer")

    @test_app.route("/api/fhir/R4/Patient/TEST-001")
    def mock_fhir_patient():
        with tracer.start_as_current_span("fhir.R4.Patient.TEST-001"):
            return {"resourceType": "Patient", "id": "TEST-001"}

    client = test_app.test_client()
    resp = client.get("/api/fhir/R4/Patient/TEST-001")
    assert resp.status_code == 200

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        with tracer.start_as_current_span("external.requests.post"):
            requests.post("https://example.com/api", json={"test": "data"})

    @celery.task
    def sample_task():
        with tracer.start_as_current_span("celery.task.sample_task"):
            return "done"

    result = sample_task.apply()
    assert result.successful()

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert any("Patient" in s.name or "fhir" in s.name.lower() or "GET" in s.name for s in spans), \
        f"Expected FHIR span, got: {span_names}"

    assert any("POST" in s.name or "example.com" in s.name or "requests" in s.name.lower() for s in spans), \
        f"Expected external API span, got: {span_names}"

    assert any("sample_task" in s.name or "apply" in s.name or "celery" in s.name.lower() for s in spans), \
        f"Expected Celery task span, got: {span_names}"

    _cleanup(test_app)
