"""End-to-end document ingestion pipeline.

Parse -> classify (hybrid ML+rules) -> extract entities -> chunk -> anomaly score
-> persist -> incrementally index into the vector store -> route to auto-approve
or human review based on confidence and anomaly thresholds.
"""

from __future__ import annotations

import time

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger, log_timing
from app.core.observability import observe_pipeline
from app.db.database import SessionLocal
from app.db.models import Document, DocumentChunk, DocumentEntity
from app.services.anomaly import score_anomalies
from app.services.chunking import chunk_text
from app.services.classification import classify_document
from app.services.events import publish_event
from app.services.extraction import extract_entities
from app.services.parser import parse_document
from app.services.retrieval import reindex_document
from app.services.review import create_review_task

logger = get_logger(__name__)


def build_summary(doc_type: str, entities: list[dict], text: str) -> str:
    entity_map = {entity["field_name"]: entity["field_value"] for entity in entities}
    if doc_type == "invoice":
        return (
            f"Invoice {entity_map.get('invoice_number', 'unknown')} from "
            f"{entity_map.get('vendor_name', 'unknown vendor')} totaling "
            f"{entity_map.get('total_amount', 'unknown')}."
        )
    if doc_type == "contract":
        return (
            f"Contract effective {entity_map.get('effective_date', 'unknown')} "
            f"with governing law {entity_map.get('governing_law', 'unspecified')}."
        )
    if doc_type == "claim_form":
        return (
            f"Claim {entity_map.get('claim_id', 'unknown')} filed by "
            f"{entity_map.get('claimant_name', 'unknown claimant')}."
        )
    return " ".join(text.split()[:25]) + ("..." if len(text.split()) > 25 else "")


def process_document(document_id: str, path: str) -> None:
    db: Session = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.warning("Document not found for processing", extra={"document_id": document_id})
            return

        document.status = "processing"
        db.commit()
        proc_start = time.perf_counter()
        logger.info("Processing document", extra={"document_id": document_id, "file": document.filename})

        with log_timing(logger, "ingest", document_id=document_id):
            text = parse_document(path)
            doc_type, confidence, _ = classify_document(text)
            entities = extract_entities(doc_type, text)
            chunks = chunk_text(text)
            anomaly_score, anomaly_reasons = score_anomalies(db, document, doc_type, entities, text)

        db.query(DocumentEntity).filter(DocumentEntity.document_id == document.id).delete()
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()

        for entity in entities:
            db.add(
                DocumentEntity(
                    document_id=document.id,
                    field_name=entity["field_name"],
                    field_value=entity["field_value"],
                    confidence=entity["confidence"],
                )
            )
        for idx, chunk in enumerate(chunks):
            db.add(DocumentChunk(document_id=document.id, chunk_index=idx, text=chunk))

        document.doc_type = doc_type
        document.confidence = confidence
        document.anomaly_score = anomaly_score
        document.processed_text = text[:10000]
        document.summary = build_summary(doc_type, entities, text)

        needs_review = (
            confidence < settings.human_review_threshold
            or anomaly_score >= settings.anomaly_review_threshold
        )
        if confidence >= settings.auto_approve_threshold and anomaly_score < settings.anomaly_review_threshold:
            document.status = "approved"
        elif needs_review:
            document.status = "needs_review"
            reasons: list[str] = []
            if confidence < settings.human_review_threshold:
                reasons.append(f"low_confidence={confidence}")
            if anomaly_score >= settings.anomaly_review_threshold:
                reasons.append(f"anomaly_score={anomaly_score}")
            if anomaly_reasons:
                reasons.append(",".join(anomaly_reasons))
            create_review_task(
                db,
                document.id,
                " | ".join(reasons),
                priority="high" if anomaly_score >= settings.anomaly_high_priority_threshold else "medium",
            )
        else:
            document.status = "processed"

        db.commit()

        # Keep the semantic/sparse indexes consistent with the new chunks.
        try:
            reindex_document(db, document.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Indexing failed (%s); search will self-heal later", exc.__class__.__name__)

        publish_event(
            "document.processed",
            {
                "document_id": document_id,
                "doc_type": doc_type,
                "status": document.status,
                "tenant_id": document.tenant_id,
            },
        )
        if document.status == "needs_review":
            publish_event(
                "document.needs_review",
                {"document_id": document_id, "anomaly_score": anomaly_score},
            )

        observe_pipeline("process", time.perf_counter() - proc_start, status=document.status)
        logger.info(
            "Document processed",
            extra={
                "document_id": document_id,
                "doc_type": doc_type,
                "confidence": confidence,
                "anomaly_score": anomaly_score,
                "status": document.status,
                "chunks": len(chunks),
            },
        )
    except Exception as exc:
        logger.exception("Processing failed", extra={"document_id": document_id})
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = "failed"
            document.summary = f"Processing failed: {exc}"
            db.commit()
        raise
    finally:
        db.close()
