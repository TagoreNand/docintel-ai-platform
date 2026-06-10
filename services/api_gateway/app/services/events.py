"""Domain event streaming: Kafka backend with an in-memory fallback.

The pipeline and review flow emit events (``document.uploaded``,
``document.processed``, ``document.needs_review``, ``review.resolved``). With
``ENABLE_KAFKA=true`` they are produced to Kafka topics (``<prefix>.<event>``);
otherwise they are kept in an in-process ring buffer. Either way a recent-events
feed is available for the dashboard, and publishing never raises into callers.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_RECENT: deque[dict] = deque(maxlen=200)
_LOCK = threading.Lock()


def _record(event_type: str, payload: dict) -> dict:
    event = {"type": event_type, "payload": payload, "ts": time.time()}
    with _LOCK:
        _RECENT.append(event)
    return event


class EventBus:
    backend = "memory"

    def publish(self, event_type: str, payload: dict) -> None:
        raise NotImplementedError

    def recent(self, limit: int = 50) -> list[dict]:
        with _LOCK:
            return list(_RECENT)[-limit:][::-1]


class MemoryEventBus(EventBus):
    backend = "memory"

    def publish(self, event_type: str, payload: dict) -> None:
        _record(event_type, payload)
        logger.info("event", extra={"event_type": event_type})


class KafkaEventBus(EventBus):
    backend = "kafka"

    def __init__(self) -> None:
        from kafka import KafkaProducer

        self._producer = KafkaProducer(
            bootstrap_servers=[s.strip() for s in settings.kafka_bootstrap_servers.split(",")],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=1,
            request_timeout_ms=3000,
            max_block_ms=3000,
        )

    def publish(self, event_type: str, payload: dict) -> None:
        event = _record(event_type, payload)
        topic = f"{settings.kafka_topic_prefix}.{event_type}"
        self._producer.send(topic, event)
        self._producer.flush(timeout=2)


@lru_cache(maxsize=1)
def get_event_bus() -> EventBus:
    if settings.enable_kafka:
        try:
            bus = KafkaEventBus()
            logger.info("Kafka event bus ready", extra={"servers": settings.kafka_bootstrap_servers})
            return bus
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kafka unavailable (%s); using in-memory event bus", exc.__class__.__name__)
    return MemoryEventBus()


def publish_event(event_type: str, payload: dict[str, Any]) -> None:
    """Publish an event; never raises into the caller."""
    try:
        get_event_bus().publish(event_type, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Event publish failed (%s)", exc.__class__.__name__)


def recent_events(limit: int = 50) -> list[dict]:
    try:
        return get_event_bus().recent(limit)
    except Exception:  # noqa: BLE001
        return []


def reset_event_bus() -> None:
    get_event_bus.cache_clear()
