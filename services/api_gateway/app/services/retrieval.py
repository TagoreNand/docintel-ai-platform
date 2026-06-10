"""Hybrid retrieval + grounded extractive QA (tenant-scoped).

Per query: dense recall (vector store) + sparse recall (BM25 Okapi) fused with
Reciprocal Rank Fusion, an optional cross-encoder rerank, then extractive answer
synthesis. Dense and sparse indexes are maintained per tenant so retrieval is
physically isolated between tenants.
"""

from __future__ import annotations

import re
import time
import threading
from collections import defaultdict

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import observe_retrieval
from app.db.models import Document, DocumentChunk, DocumentEntity
from app.schemas.search import EvidenceItem, RetrievalStrategy, SearchResponse
from app.services.embeddings import get_embedder
from app.services.reranker import get_reranker
from app.services.sparse_index import SparseIndex, tokenize
from app.services.vector_store import VectorRecord, get_vector_store

logger = get_logger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Per-tenant BM25 indexes.
_SPARSE: dict[str, SparseIndex] = {}
_SPARSE_LOCK = threading.Lock()


def _sparse_for(tenant: str) -> SparseIndex:
    with _SPARSE_LOCK:
        if tenant not in _SPARSE:
            _SPARSE[tenant] = SparseIndex(k1=settings.bm25_k1, b=settings.bm25_b)
        return _SPARSE[tenant]


def reset_sparse() -> None:
    with _SPARSE_LOCK:
        _SPARSE.clear()


def _tenant_or_default(tenant: str | None) -> str:
    return tenant or settings.default_tenant


# --------------------------------------------------------------------------- #
# Index lifecycle
# --------------------------------------------------------------------------- #
def _records_from_db(db: Session, tenant: str) -> list[VectorRecord]:
    rows = (
        db.query(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(Document.tenant_id == tenant)
        .order_by(DocumentChunk.created_at.asc(), DocumentChunk.id.asc())
        .all()
    )
    return [
        VectorRecord(
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            filename=document.filename,
            text=chunk.text,
            tenant=tenant,
        )
        for chunk, document in rows
    ]


def _signature(records: list[VectorRecord]) -> tuple:
    return (len(records), records[-1].id if records else None)


def reindex_document(db: Session, document_id: str) -> int:
    """Incrementally (re)index a single document's chunks into its tenant store."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return 0
    tenant = document.tenant_id
    store = get_vector_store(tenant)
    store.delete_document(document_id)
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )
    if chunks:
        records = [
            VectorRecord(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                filename=document.filename,
                text=chunk.text,
                tenant=tenant,
            )
            for chunk in chunks
        ]
        vectors = get_embedder().embed([r.text for r in records])
        store.upsert(records, vectors)
    _sparse_for(tenant).invalidate()
    logger.info("Reindexed document", extra={"document_id": document_id, "tenant": tenant, "chunks": len(chunks)})
    return len(chunks)


def ensure_index(db: Session, tenant: str) -> tuple:
    """Make a tenant's dense + sparse indexes consistent with the DB."""
    store = get_vector_store(tenant)
    records = _records_from_db(db, tenant)

    if store.count() != len(records):
        store.clear()
        if records:
            vectors = get_embedder().embed([r.text for r in records])
            store.upsert(records, vectors)
        logger.info("Rebuilt dense index from DB", extra={"tenant": tenant, "count": len(records)})

    signature = _signature(records)
    sparse = _sparse_for(tenant)
    if not sparse.is_current(signature):
        sparse.build(records, signature)

    return store, records


# --------------------------------------------------------------------------- #
# Fusion + answer synthesis
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(dense_hits, sparse_hits, k: int, w_dense: float, w_sparse: float) -> list[dict]:
    """Merge dense and sparse rankings via Reciprocal Rank Fusion."""
    fused: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        entry = fused.setdefault(
            hit.record.id, {"record": hit.record, "fused": 0.0, "dense": None, "sparse": None}
        )
        entry["fused"] += w_dense / (k + rank)
        entry["dense"] = hit.score

    for rank, (record, score) in enumerate(sparse_hits, start=1):
        entry = fused.setdefault(
            record.id, {"record": record, "fused": 0.0, "dense": None, "sparse": None}
        )
        entry["fused"] += w_sparse / (k + rank)
        entry["sparse"] = score

    return sorted(fused.values(), key=lambda item: item["fused"], reverse=True)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 0]


def _extractive_answer(question: str, records: list[VectorRecord], max_sentences: int) -> str:
    if not records:
        return "No relevant evidence found in the indexed documents."
    top_doc = records[0].document_id
    focus = [r for r in records if r.document_id == top_doc][:4]
    candidates: list[tuple[str, str]] = []
    for record in focus:
        for sentence in _split_sentences(record.text):
            if len(sentence.split()) >= 3:
                candidates.append((sentence, record.filename))
    if not candidates:
        return "No relevant evidence found in the indexed documents."

    embedder = get_embedder()
    query_vec = embedder.embed_one(question)
    sentence_vecs = embedder.embed([c[0] for c in candidates])
    sims = sentence_vecs @ query_vec

    n = min(max_sentences, len(candidates))
    best = np.argpartition(-sims, n - 1)[:n]
    best = sorted(best.tolist())
    chosen = [candidates[i][0] for i in best]
    source = candidates[int(np.argmax(sims))][1]
    return f"{' '.join(chosen)} [grounded in: {source}]"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def answer_question(
    db: Session,
    question: str,
    top_k: int = 5,
    rerank: bool | None = None,
    tenant: str | None = None,
) -> SearchResponse:
    started = time.perf_counter()
    tenant = _tenant_or_default(tenant)
    embedder = get_embedder()
    store, records = ensure_index(db, tenant)

    if not records:
        return SearchResponse(
            question=question,
            answer="No documents are indexed yet. Upload or ingest documents first.",
            evidence=[],
            related_entities={},
            strategy=RetrievalStrategy(embedding_backend=embedder.name, vector_backend=store.backend),
        )

    query_vec = embedder.embed_one(question)
    dense_hits = store.search(query_vec, settings.dense_top_k)
    sparse_hits = _sparse_for(tenant).search(question, settings.sparse_top_k)

    fused = reciprocal_rank_fusion(
        dense_hits, sparse_hits, settings.rrf_k, settings.dense_weight, settings.sparse_weight
    )

    use_reranker = settings.enable_reranker if rerank is None else rerank
    reranker = get_reranker() if use_reranker else None
    reranked = False
    if reranker and fused:
        candidates = fused[: settings.reranker_candidates]
        scores = reranker.score(question, [c["record"].text for c in candidates])
        for entry, score in zip(candidates, scores):
            entry["rerank"] = score
        candidates.sort(key=lambda item: item["rerank"], reverse=True)
        fused = candidates + fused[settings.reranker_candidates :]
        reranked = True

    top = fused[:top_k]
    evidence = [
        EvidenceItem(
            document_id=entry["record"].document_id,
            filename=entry["record"].filename,
            chunk_index=entry["record"].chunk_index,
            score=round(float(entry.get("rerank", entry["fused"]) if reranked else entry["fused"]), 6),
            dense_score=None if entry["dense"] is None else round(float(entry["dense"]), 6),
            sparse_score=None if entry["sparse"] is None else round(float(entry["sparse"]), 6),
            rerank_score=None if entry.get("rerank") is None else round(float(entry["rerank"]), 6),
            text=entry["record"].text[:450],
        )
        for entry in top
    ]

    answer = _extractive_answer(question, [entry["record"] for entry in top], settings.answer_max_sentences)

    related: dict[str, list[str]] = defaultdict(list)
    top_doc_ids = {item.document_id for item in evidence}
    if top_doc_ids:
        entities = db.query(DocumentEntity).filter(DocumentEntity.document_id.in_(top_doc_ids)).all()
        for entity in entities:
            if entity.field_value not in related[entity.field_name]:
                related[entity.field_name].append(entity.field_value)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    observe_retrieval(latency_ms / 1000.0)
    logger.info(
        "Answered query",
        extra={"tenant": tenant, "dense": len(dense_hits), "sparse": len(sparse_hits),
               "fused": len(fused), "reranked": reranked, "latency_ms": latency_ms},
    )

    return SearchResponse(
        question=question,
        answer=answer,
        evidence=evidence,
        related_entities=dict(related),
        strategy=RetrievalStrategy(
            embedding_backend=embedder.name,
            vector_backend=store.backend,
            reranker=(reranker.name if reranker else None),
            dense_candidates=len(dense_hits),
            sparse_candidates=len(sparse_hits),
            fused_candidates=len(fused),
            reranked=reranked,
            latency_ms=latency_ms,
        ),
    )
