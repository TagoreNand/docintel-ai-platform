"""Ingestion dispatch: run the pipeline in-process or hand it to a Kafka worker.

Default (``INGESTION_MODE=inproc``): processing runs in a FastAPI background task.
Distributed (``INGESTION_MODE=kafka`` + ``ENABLE_KAFKA=true``): the upload emits a
``document.uploaded`` event that a separate worker process consumes — decoupling
ingestion from the API for horizontal scale.
"""

from __future__ import annotations

from fastapi import BackgroundTasks

from app.core.config import settings
from app.core.logging import get_logger
from app.services.events import publish_event
from app.services.pipeline import process_document

logger = get_logger(__name__)


def submit_document(background_tasks: BackgroundTasks | None, document_id: str, path: str) -> str:
    """Route a document for processing; returns the dispatch mode used."""
    if settings.ingestion_mode == "kafka" and settings.enable_kafka:
        publish_event("document.uploaded", {"document_id": document_id, "path": path})
        logger.info("Dispatched to Kafka worker", extra={"document_id": document_id})
        return "kafka"

    if background_tasks is not None:
        background_tasks.add_task(process_document, document_id, path)
    else:
        process_document(document_id, path)
    return "inproc"
