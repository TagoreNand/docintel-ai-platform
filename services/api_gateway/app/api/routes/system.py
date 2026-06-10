"""Operational introspection endpoints.

``/system/info`` reports which AI backends are actually live (embedding model +
device, vector store, reranker, trained classifier/anomaly models).
``/metrics`` exposes Prometheus metrics (real client exposition when available,
a plain-text gauge fallback otherwise). ``/events/recent`` shows the latest
domain events from the event bus.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.db.models import Document, DocumentChunk, ReviewTask

router = APIRouter()

APP_VERSION = "3.0.0"


def _read_card(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _db_stats(db: Session) -> dict:
    from app.services.vector_store import total_indexed_vectors

    documents = db.query(Document).count()
    chunks = db.query(DocumentChunk).count()
    open_reviews = db.query(ReviewTask).filter(ReviewTask.status == "open").count()
    scores = [d.anomaly_score for d in db.query(Document).all() if d.anomaly_score is not None]
    avg_anomaly = round(sum(scores) / len(scores), 4) if scores else 0.0
    try:
        vectors = total_indexed_vectors()
    except Exception:  # noqa: BLE001
        vectors = 0
    return {
        "documents": documents,
        "chunks": chunks,
        "open_reviews": open_reviews,
        "vectors": vectors,
        "avg_anomaly": avg_anomaly,
    }


@router.get("/system/info")
def system_info(db: Session = Depends(get_db)) -> dict:
    from app.services.embeddings import get_embedder
    from app.services.events import get_event_bus
    from app.services.ocr import get_ocr_engine
    from app.services.reranker import get_reranker
    from app.services.vector_store import get_vector_store, known_tenants, total_indexed_vectors

    embedder = get_embedder()
    store = get_vector_store()  # default tenant
    reranker = get_reranker() if settings.enable_reranker else None

    return {
        "app": settings.app_name,
        "version": APP_VERSION,
        "environment": settings.environment,
        "embedding": {"backend": embedder.name, "dim": embedder.dim, "device": getattr(embedder, "device", "cpu")},
        "vector_store": {"backend": store.backend, "vectors": total_indexed_vectors(), "tenants": len(known_tenants())},
        "retrieval": {
            "rrf_k": settings.rrf_k,
            "dense_top_k": settings.dense_top_k,
            "sparse_top_k": settings.sparse_top_k,
            "reranker": reranker.name if reranker else None,
        },
        "ocr": {"enabled": settings.ocr_enabled, "engine": get_ocr_engine().name, "available": get_ocr_engine().available},
        "auth": {"enabled": settings.enable_auth, "tenants": sorted(set(settings.api_key_map.values())) or [settings.default_tenant]},
        "events": {"backend": get_event_bus().backend, "ingestion_mode": settings.ingestion_mode},
        "classifier": {"loaded": settings.classifier_path.exists(), "card": _read_card(settings.classifier_card_path)},
        "anomaly_model": {"loaded": settings.anomaly_model_path.exists(), "card": _read_card(settings.anomaly_card_path)},
    }


@router.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    from app.core.observability import prometheus_enabled, render_metrics, set_db_gauges

    stats = _db_stats(db)
    if prometheus_enabled():
        set_db_gauges(stats["documents"], stats["chunks"], stats["open_reviews"], stats["vectors"], stats["avg_anomaly"])
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    gauges = {
        "docintel_documents_total": stats["documents"],
        "docintel_chunks_total": stats["chunks"],
        "docintel_review_tasks_open": stats["open_reviews"],
        "docintel_index_vectors": stats["vectors"],
        "docintel_avg_anomaly_score": stats["avg_anomaly"],
    }
    lines: list[str] = []
    for name, value in gauges.items():
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")
    return PlainTextResponse("\n".join(lines) + "\n")


@router.get("/events/recent")
def events_recent(limit: int = 50) -> dict:
    from app.services.events import recent_events

    return {"events": recent_events(limit)}
