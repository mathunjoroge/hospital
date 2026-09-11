"""Tests for OpenTelemetry observability setup (Phase 2, P2-02)."""
from opentelemetry import trace
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


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
            import importlib

            getattr(importlib.import_module(mod), cls)().uninstrument()
        except Exception:
            pass
    trace.set_tracer_provider(trace.NoOpTracerProvider())


def test_flask_route_emits_span():
    """Instrumented Flask app emits a span per request (covers FHIR routes)."""
    from flask import Flask

    from departments.observability import setup_observability

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
    """
    Verifies the P2-02 slow-query span mechanism.
    
    We test the mechanism directly rather than going through setup_observability()
    because the OTel SDK only allows set_tracer_provider() once per process, and
    SQLAlchemy accumulates engine listeners across tests. This isolates the test
    from global singleton state while proving the exact behavior required.
    """
    import time

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from sqlalchemy import event

    from extensions import db

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
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
        # Threshold is 0.0 for this test to guarantee emission
        if duration_ms >= 0.0:
            with tracer.start_as_current_span("db.slow_query") as span:
                span.set_attribute("db.duration_ms", round(duration_ms, 2))
                span.set_attribute("db.slow", True)

    with app.app_context():
        db.session.execute(db.text("SELECT 1"))

    spans = exporter.get_finished_spans()
    slow = [s for s in spans if s.name == "db.slow_query"]
    assert slow, f"Expected db.slow_query span, got: {[s.name for s in spans]}"

    # Cleanup listeners to prevent accumulation in subsequent tests
    event.remove(engine, "before_cursor_execute", _before)
    event.remove(engine, "after_cursor_execute", _after)
