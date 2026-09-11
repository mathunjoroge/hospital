"""
OpenTelemetry observability setup (Phase 2, P2-02).

Instruments:
  - Flask app (all routes incl. FHIR endpoints)  -> FlaskInstrumentor
  - External HTTP calls (requests)               -> RequestsInstrumentor
  - Celery tasks (trace-context propagation)     -> CeleryInstrumentor
  - Slow DB queries (> SLOW_QUERY_THRESHOLD_MS)  -> SQLAlchemy event listener

Gracefully degrades when OTel packages are missing or no collector endpoint
is configured. Disabled by default under TESTING unless OTEL_ENABLED is set,
so the existing test suite is unaffected.
"""
import logging
import os
import time

logger = logging.getLogger(__name__)

SLOW_QUERY_THRESHOLD_MS = 50.0

# Engines that already have the slow-query listener attached (idempotency).
_REGISTERED_ENGINES = set()


def _is_enabled(app):
    explicit = app.config.get("OTEL_ENABLED")
    if explicit is not None:
        return bool(explicit)
    if app.config.get("TESTING", False):
        return False
    endpoint = app.config.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    return bool(endpoint)


def setup_observability(app, celery=None, exporter=None):
    """Initialize OpenTelemetry tracing for the app.

    Returns the TracerProvider, or None when observability is disabled or the
    OTel packages are unavailable.
    """
    if not _is_enabled(app):
        logger.info("OpenTelemetry disabled (OTEL_ENABLED/endpoint/TESTING gate)")
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.flask import FlaskInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
    except ImportError as exc:  # pragma: no cover - optional deps
        logger.warning("OpenTelemetry packages unavailable, skipping: %s", exc)
        return None

    service_name = app.config.get("OTEL_SERVICE_NAME", "hospital-hmis")
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )

    if exporter is not None:
        # Tests: synchronous processor so spans are captured immediately.
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        endpoint = app.config.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )

    trace.set_tracer_provider(provider)

    # Flask routes (covers FHIR endpoints) -> spans.
    try:
        FlaskInstrumentor().instrument_app(app)
    except Exception as exc:  # already instrumented  # noqa: BLE001
        logger.debug("FlaskInstrumentor skipped: %s", exc)

    # External API calls (M-Pesa, NLP, notifications) -> spans.
    try:
        RequestsInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        logger.debug("RequestsInstrumentor skipped: %s", exc)

    # Celery trace-context propagation.
    try:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        logger.debug("CeleryInstrumentor skipped: %s", exc)

    # Slow DB query spans.
    try:
        _register_slow_query_listener(app, provider)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Slow-query listener not registered: %s", exc)

    logger.info("OpenTelemetry initialized (service=%s)", service_name)
    return provider


def _register_slow_query_listener(app, provider):
    """Emit a span for any DB query taking >= SLOW_QUERY_THRESHOLD_MS."""
    from sqlalchemy import event

    from extensions import db

    tracer = provider.get_tracer("db.slow_query")

    with app.app_context():
        engine = db.engine

    if id(engine) in _REGISTERED_ENGINES:
        return
    _REGISTERED_ENGINES.add(id(engine))

    @event.listens_for(engine, "before_cursor_execute")
    def _before(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault("_otel_query_start", []).append(time.perf_counter())

    @event.listens_for(engine, "after_cursor_execute")
    def _after(conn, cursor, statement, parameters, context, executemany):
        starts = conn.info.get("_otel_query_start")
        if not starts:
            return
        duration_ms = (time.perf_counter() - starts.pop()) * 1000.0
        if duration_ms >= SLOW_QUERY_THRESHOLD_MS:
            with tracer.start_as_current_span("db.slow_query") as span:
                span.set_attribute("db.duration_ms", round(duration_ms, 2))
                span.set_attribute("db.statement", statement[:500])
                span.set_attribute("db.slow", True)
