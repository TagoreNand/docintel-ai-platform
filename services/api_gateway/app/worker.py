"""Distributed ingestion worker.

Consumes ``<prefix>.document.uploaded`` events from Kafka and runs the ingestion
pipeline out-of-process, decoupling heavy document processing from the API.
Enable with ``INGESTION_MODE=kafka ENABLE_KAFKA=true`` and run this module as a
separate container/process (see ``scripts/worker.py`` and docker-compose).

This requires a running Kafka broker (e.g. ``docker compose up -d kafka``); in
local single-machine mode you do NOT need the worker — ingestion runs in-process.
"""

from __future__ import annotations

import json
import sys

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.database import init_db
from app.services.pipeline import process_document

logger = get_logger(__name__)


def run_worker() -> None:
    configure_logging()
    init_db()

    from kafka import KafkaConsumer
    from kafka.errors import KafkaError

    topic = f"{settings.kafka_topic_prefix}.document.uploaded"
    servers = [s.strip() for s in settings.kafka_bootstrap_servers.split(",")]

    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=servers,
            group_id=settings.kafka_consumer_group,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            api_version_auto_timeout_ms=5000,
        )
    except KafkaError as exc:
        logger.error(
            "Cannot connect to Kafka at %s (%s).", settings.kafka_bootstrap_servers, exc.__class__.__name__
        )
        print(
            "\nNo Kafka broker reachable at "
            f"{settings.kafka_bootstrap_servers}.\n"
            "  - Start one:        docker compose up -d kafka\n"
            "  - Or skip the worker: local mode uses INGESTION_MODE=inproc (the default),\n"
            "    which processes uploads in-process — no worker needed.\n"
        )
        sys.exit(1)

    logger.info("Ingestion worker started", extra={"topic": topic, "group": settings.kafka_consumer_group})

    for message in consumer:
        payload = (message.value or {}).get("payload", {})
        document_id = payload.get("document_id")
        path = payload.get("path")
        if not document_id or not path:
            logger.warning("Skipping malformed event", extra={"value": message.value})
            continue
        try:
            logger.info("Worker processing document", extra={"document_id": document_id})
            process_document(document_id, path)
        except Exception:  # noqa: BLE001
            logger.exception("Worker failed to process document", extra={"document_id": document_id})


if __name__ == "__main__":
    run_worker()
