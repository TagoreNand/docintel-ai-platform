from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger, request_id_ctx
from app.core.observability import observe_request, setup_tracing
from app.db.database import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_logging()
    init_db()
    logger.info(
        "DocIntel API ready",
        extra={"environment": settings.environment, "version": "3.0.0"},
    )
    yield


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title=settings.app_name,
        version="3.0.0",
        description=(
            "Enterprise document intelligence: hybrid (dense + BM25) retrieval with "
            "RRF and cross-encoder reranking, calibrated ML classification, and "
            "unsupervised anomaly detection with human-in-the-loop review."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = str(elapsed_ms)
        route = request.scope.get("route")
        path_label = getattr(route, "path", request.url.path)
        observe_request(request.method, path_label, response.status_code, elapsed_ms / 1000.0)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response

    application.include_router(api_router, prefix=settings.api_v1_prefix)
    setup_tracing(application)
    return application


app = create_app()
