"""Structured, JSON-capable logging for the platform.

A single ``configure_logging`` call wires the root logger to emit either
human-friendly console lines (local dev) or single-line JSON (production /
log-aggregation friendly). ``get_logger`` returns a namespaced logger and a
``request_id`` context variable lets middleware stamp every log line in a
request with a correlation id.

A ``SafeLogger`` guards against the classic ``extra={"name": ...}`` /
``extra={"filename": ...}`` foot-gun by renaming keys that collide with reserved
``LogRecord`` attributes instead of raising ``KeyError``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from app.core.config import settings

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

_CONFIGURED = False
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class SafeLogger(logging.Logger):
    """Logger that never raises when ``extra`` collides with reserved fields."""

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        if extra:
            extra = {(f"x_{k}" if k in _RESERVED else k): v for k, v in extra.items()}
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)


class JsonFormatter(logging.Formatter):
    """Render log records as compact single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_ctx.get()
        if rid:
            payload["request_id"] = rid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Readable console formatter that appends the request id when present."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        rid = request_id_ctx.get()
        return f"{base} [request_id={rid}]" if rid else base


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.setLoggerClass(SafeLogger)

    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            ConsoleFormatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    for noisy in ("httpx", "urllib3", "sentence_transformers", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


@contextmanager
def log_timing(logger: logging.Logger, stage: str, **fields: Any) -> Iterator[None]:
    """Context manager that logs the wall-clock duration of a stage."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("%s done", stage, extra={"stage": stage, "elapsed_ms": elapsed_ms, **fields})
