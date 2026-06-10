"""Optional cross-encoder reranking stage.

A bi-encoder (the embedding model) is great for fast candidate recall; a
cross-encoder that jointly reads (query, passage) pairs is far more precise for
final ordering. This module loads ``cross-encoder/ms-marco-MiniLM-L-6-v2`` when
available and otherwise reports itself as unavailable so the retriever simply
keeps the fused ordering. This is the classic retrieve-then-rerank pattern.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # heavy, lazy import

        from app.services.embeddings import resolve_device

        self.model_name = model_name
        self.device = resolve_device()
        self._model = CrossEncoder(model_name, device=self.device)
        self.name = f"cross_encoder:{model_name}@{self.device}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker | None:
    """Return a cross-encoder reranker, or ``None`` when disabled/unavailable."""
    if not settings.enable_reranker:
        return None
    try:
        reranker = CrossEncoderReranker(settings.reranker_model)
        logger.info("Reranker ready", extra={"model": reranker.name})
        return reranker
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Cross-encoder reranker unavailable (%s); using fused ranking",
            exc.__class__.__name__,
        )
        return None


def reset_reranker() -> None:
    get_reranker.cache_clear()
