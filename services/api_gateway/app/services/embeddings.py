"""Pluggable text embedding providers.

The platform's semantic retrieval is backed by a real neural sentence encoder
(``sentence-transformers/all-MiniLM-L6-v2`` by default). To guarantee the system
runs in any environment — CI, air-gapped laptops, this sandbox — it transparently
falls back to a deterministic, dependency-free hashing embedder when torch or the
model weights are unavailable.

All providers return L2-normalised ``float32`` matrices so cosine similarity
reduces to a dot product everywhere downstream.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def resolve_device() -> str:
    """Resolve the inference device, auto-detecting CUDA when configured."""
    configured = settings.inference_device
    if configured != "auto":
        return configured
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch absent / broken => CPU
        return "cpu"


class EmbeddingProvider(ABC):
    """Common interface for every embedding backend."""

    name: str = "base"
    dim: int = 0
    device: str = "cpu"

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` L2-normalised float32 matrix."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(EmbeddingProvider):
    """Deterministic feature-hashing embedder (no external dependencies).

    Uses the signed hashing trick over word unigrams + bigrams with sublinear
    term weighting, projected into a fixed ``dim`` space. It is not as powerful
    as a neural encoder, but it is fully deterministic across processes (unlike
    Python's salted ``hash``) and provides a meaningful lexical-semantic signal
    that keeps the whole retrieval stack functional offline.
    """

    name = "hashing"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = int(dim or settings.hashing_embedding_dim)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = _TOKEN_RE.findall(text.lower())
        if not words:
            return []
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def _hash(self, feature: str) -> tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in self._tokens(text or ""):
                counts[token] = counts.get(token, 0) + 1
            for token, tf in counts.items():
                bucket, sign = self._hash(token)
                matrix[row, bucket] += sign * (1.0 + math.log(tf))
        return _l2_normalize(matrix)


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Real neural encoder backed by sentence-transformers."""

    name = "sentence_transformer"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy, heavy import

        self.model_name = model_name
        self.device = resolve_device()
        self._model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda" and settings.inference_fp16:
            self._model = self._model.half()
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence_transformer:{model_name}@{self.device}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


def _build_embedder() -> EmbeddingProvider:
    backend = settings.embedding_backend

    if backend in ("auto", "sentence_transformer"):
        try:
            embedder = SentenceTransformerEmbedder(settings.embedding_model)
            logger.info(
                "Embedding backend ready",
                extra={"backend": embedder.name, "dim": embedder.dim},
            )
            return embedder
        except Exception as exc:  # noqa: BLE001 - any failure should degrade
            if backend == "sentence_transformer":
                raise
            logger.warning(
                "Neural embedder unavailable (%s); falling back to hashing embedder",
                exc.__class__.__name__,
                extra={"error": str(exc)},
            )

    embedder = HashingEmbedder()
    logger.info(
        "Embedding backend ready",
        extra={"backend": embedder.name, "dim": embedder.dim},
    )
    return embedder


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    """Return the process-wide singleton embedding provider."""
    return _build_embedder()


def reset_embedder() -> None:
    """Clear the cached embedder (used by tests that switch backends)."""
    get_embedder.cache_clear()
