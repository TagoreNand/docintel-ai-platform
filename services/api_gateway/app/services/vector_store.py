"""Persisted, tenant-scoped vector store with a pluggable backend.

Two interchangeable backends implement the same interface:

* ``QdrantVectorStore`` — a Qdrant collection (the production path). Per-tenant
  collections, plus configurable HNSW + scalar quantization, support multi-tenant
  isolation and scale.
* ``LocalVectorStore`` — a dependency-light, disk-persisted numpy store with
  brute-force cosine search (the always-available offline path), kept in a
  per-tenant subdirectory.

``get_vector_store(tenant)`` returns the (cached) store for a tenant, so document
vectors are physically isolated per tenant and survive restarts everywhere.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorRecord:
    """A single indexed chunk plus the metadata returned with search hits."""

    id: str
    document_id: str
    chunk_index: int
    filename: str
    text: str
    tenant: str = "default"


@dataclass
class VectorHit:
    record: VectorRecord
    score: float


class VectorStore:
    """Common interface for vector backends."""

    backend: str = "base"

    def upsert(self, records: list[VectorRecord], vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query: np.ndarray, top_k: int) -> list[VectorHit]:
        raise NotImplementedError

    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class LocalVectorStore(VectorStore):
    """Disk-persisted, brute-force cosine store backed by numpy."""

    backend = "local"

    def __init__(self, dim: int, index_dir=None) -> None:
        self.dim = int(dim)
        self.dir = Path(index_dir or settings.index_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self.dir / "vectors.npy"
        self._records_path = self.dir / "records.json"
        self._meta_path = self.dir / "meta.json"
        self._lock = threading.RLock()
        self._records: list[VectorRecord] = []
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not (self._vectors_path.exists() and self._records_path.exists()):
            return
        try:
            meta = json.loads(self._meta_path.read_text()) if self._meta_path.exists() else {}
            if int(meta.get("dim", self.dim)) != self.dim:
                logger.warning("Vector index dim mismatch; starting empty")
                return
            matrix = np.load(self._vectors_path)
            raw = json.loads(self._records_path.read_text())
            if matrix.shape[0] != len(raw):
                logger.warning("Vector index corrupt (size mismatch); starting empty")
                return
            self._matrix = matrix.astype(np.float32)
            self._records = [VectorRecord(**item) for item in raw]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load vector index (%s); starting empty", exc)

    def _persist(self) -> None:
        np.save(self._vectors_path, self._matrix)
        self._records_path.write_text(json.dumps([asdict(r) for r in self._records]))
        self._meta_path.write_text(json.dumps({"dim": self.dim, "count": len(self._records)}))

    def upsert(self, records: list[VectorRecord], vectors: np.ndarray) -> None:
        if not records:
            return
        with self._lock:
            existing = {r.id for r in records}
            keep = [i for i, r in enumerate(self._records) if r.id not in existing]
            self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
            self._records = [self._records[i] for i in keep]
            self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])
            self._records.extend(records)
            self._persist()

    def search(self, query: np.ndarray, top_k: int) -> list[VectorHit]:
        with self._lock:
            if not self._records:
                return []
            scores = self._matrix @ query.astype(np.float32)
            k = min(top_k, len(self._records))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [VectorHit(record=self._records[i], score=float(scores[i])) for i in top]

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            keep = [i for i, r in enumerate(self._records) if r.document_id != document_id]
            if len(keep) == len(self._records):
                return
            self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
            self._records = [self._records[i] for i in keep]
            self._persist()

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records = []
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            self._persist()


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store (production path) with HNSW + quantization."""

    backend = "qdrant"

    def __init__(self, dim: int, collection: str) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self.dim = int(dim)
        self._qmodels = qmodels
        self.collection = collection
        self.client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        qm = self._qmodels
        quantization = None
        if settings.qdrant_quantization == "scalar":
            quantization = qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True)
            )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            hnsw_config=qm.HnswConfigDiff(
                m=settings.qdrant_hnsw_m, ef_construct=settings.qdrant_hnsw_ef_construct
            ),
            quantization_config=quantization,
        )
        logger.info(
            "Created Qdrant collection",
            extra={"collection": self.collection, "hnsw_m": settings.qdrant_hnsw_m,
                   "quantization": settings.qdrant_quantization},
        )

    def upsert(self, records: list[VectorRecord], vectors: np.ndarray) -> None:
        if not records:
            return
        points = [
            self._qmodels.PointStruct(id=r.id, vector=vectors[i].tolist(), payload=asdict(r))
            for i, r in enumerate(records)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, top_k: int) -> list[VectorHit]:
        hits = self.client.search(
            collection_name=self.collection, query_vector=query.tolist(), limit=top_k, with_payload=True
        )
        results: list[VectorHit] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                VectorHit(
                    record=VectorRecord(
                        id=str(hit.id),
                        document_id=payload.get("document_id", ""),
                        chunk_index=payload.get("chunk_index", 0),
                        filename=payload.get("filename", ""),
                        text=payload.get("text", ""),
                        tenant=payload.get("tenant", "default"),
                    ),
                    score=float(hit.score),
                )
            )
        return results

    def delete_document(self, document_id: str) -> None:
        qm = self._qmodels
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
                )
            ),
        )

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection).count)

    def clear(self) -> None:
        self.client.delete_collection(self.collection)
        self._ensure_collection()


def _collection_for(tenant: str) -> str:
    if settings.qdrant_per_tenant:
        return f"{settings.qdrant_collection}__{tenant}"
    return settings.qdrant_collection


def _build_store(dim: int, tenant: str) -> VectorStore:
    backend = settings.vector_backend
    if backend in ("auto", "qdrant"):
        try:
            store = QdrantVectorStore(dim, _collection_for(tenant))
            logger.info("Vector backend ready", extra={"backend": "qdrant", "tenant": tenant, "dim": dim})
            return store
        except Exception as exc:  # noqa: BLE001
            if backend == "qdrant":
                raise
            logger.warning("Qdrant unavailable (%s); using local vector store", exc.__class__.__name__)
    store = LocalVectorStore(dim, index_dir=settings.index_path / tenant)
    logger.info("Vector backend ready", extra={"backend": "local", "tenant": tenant, "count": store.count()})
    return store


_STORES: dict[str, VectorStore] = {}
_STORE_LOCK = threading.Lock()


def get_vector_store(tenant: str = "default") -> VectorStore:
    """Return the (cached) vector store for a tenant."""
    from app.services.embeddings import get_embedder

    with _STORE_LOCK:
        if tenant not in _STORES:
            _STORES[tenant] = _build_store(get_embedder().dim, tenant)
        return _STORES[tenant]


def known_tenants() -> list[str]:
    with _STORE_LOCK:
        return list(_STORES.keys())


def total_indexed_vectors() -> int:
    with _STORE_LOCK:
        return sum(store.count() for store in _STORES.values())


def reset_vector_store() -> None:
    with _STORE_LOCK:
        _STORES.clear()
