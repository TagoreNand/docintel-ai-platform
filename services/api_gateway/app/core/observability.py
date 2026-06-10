"""Observability: Prometheus metrics + optional OpenTelemetry tracing.

All instrumentation is no-op-safe: if ``prometheus-client`` is missing or
``ENABLE_PROMETHEUS`` is false the observe_* helpers do nothing, and tracing only
activates when ``ENABLE_OTEL`` is true and the OpenTelemetry packages are present.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PROM_AVAILABLE = False


def prometheus_enabled() -> bool:
    return _PROM_AVAILABLE and settings.enable_prometheus


if prometheus_enabled():
    REQUEST_COUNT = Counter(
        "docintel_requests_total", "HTTP requests", ["method", "path", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "docintel_request_latency_seconds", "HTTP request latency", ["method", "path"]
    )
    PIPELINE_LATENCY = Histogram("docintel_pipeline_seconds", "Pipeline stage seconds", ["stage"])
    PIPELINE_COUNT = Counter("docintel_documents_processed_total", "Documents processed", ["status"])
    RETRIEVAL_LATENCY = Histogram("docintel_retrieval_seconds", "Retrieval latency seconds")
    G_DOCUMENTS = Gauge("docintel_documents", "Total documents")
    G_CHUNKS = Gauge("docintel_chunks", "Total chunks")
    G_REVIEWS_OPEN = Gauge("docintel_review_tasks_open", "Open review tasks")
    G_INDEX_VECTORS = Gauge("docintel_index_vectors", "Vectors in the index")
    G_AVG_ANOMALY = Gauge("docintel_avg_anomaly_score", "Average anomaly score")
else:  # pragma: no cover - exercised only when prometheus is absent
    REQUEST_COUNT = REQUEST_LATENCY = PIPELINE_LATENCY = PIPELINE_COUNT = RETRIEVAL_LATENCY = None
    G_DOCUMENTS = G_CHUNKS = G_REVIEWS_OPEN = G_INDEX_VECTORS = G_AVG_ANOMALY = None


def observe_request(method: str, path: str, status: int, seconds: float) -> None:
    if REQUEST_COUNT is not None:
        REQUEST_COUNT.labels(method, path, str(status)).inc()
        REQUEST_LATENCY.labels(method, path).observe(seconds)


def observe_pipeline(stage: str, seconds: float, status: str | None = None) -> None:
    if PIPELINE_LATENCY is not None:
        PIPELINE_LATENCY.labels(stage).observe(seconds)
        if status:
            PIPELINE_COUNT.labels(status).inc()


def observe_retrieval(seconds: float) -> None:
    if RETRIEVAL_LATENCY is not None:
        RETRIEVAL_LATENCY.observe(seconds)


def set_db_gauges(documents: int, chunks: int, open_reviews: int, vectors: int, avg_anomaly: float) -> None:
    if G_DOCUMENTS is not None:
        G_DOCUMENTS.set(documents)
        G_CHUNKS.set(chunks)
        G_REVIEWS_OPEN.set(open_reviews)
        G_INDEX_VECTORS.set(vectors)
        G_AVG_ANOMALY.set(avg_anomaly)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def setup_tracing(app) -> None:
    """Instrument the FastAPI app with OpenTelemetry when enabled."""
    if not settings.enable_otel:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": settings.otel_service_name}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry tracing enabled", extra={"endpoint": settings.otel_exporter_otlp_endpoint})
    except Exception as exc:  # noqa: BLE001
        logger.warning("OTel setup failed (%s); continuing without tracing", exc.__class__.__name__)
