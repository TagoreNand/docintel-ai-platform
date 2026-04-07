from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import Document, DocumentChunk, DocumentEntity
from app.services.anomaly import score_anomalies
from app.services.chunking import chunk_text
from app.services.classification import classify_document
from app.services.extraction import extract_entities
from app.services.parser import parse_document
from app.services.review import create_review_task


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
            return

        document.status = "processing"
        db.commit()

        text = parse_document(path)
        doc_type, confidence, _ = classify_document(text)
        entities = extract_entities(doc_type, text)
        chunks = chunk_text(text)
        anomaly_score, anomaly_reasons = score_anomalies(db, document, entities)

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

        if confidence >= settings.auto_approve_threshold and anomaly_score < 0.25:
            document.status = "approved"
        elif confidence < settings.human_review_threshold or anomaly_score >= 0.25:
            document.status = "needs_review"
            reason = []
            if confidence < settings.human_review_threshold:
                reason.append(f"low_confidence={confidence}")
            if anomaly_score >= 0.25:
                reason.append(f"anomaly_score={anomaly_score}")
            if anomaly_reasons:
                reason.append(",".join(anomaly_reasons))
            create_review_task(db, document.id, " | ".join(reason), priority="high" if anomaly_score >= 0.5 else "medium")
        else:
            document.status = "processed"

        db.commit()
    except Exception as exc:  # pragma: no cover
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = "failed"
            document.summary = f"Processing failed: {exc}"
            db.commit()
        raise
    finally:
        db.close()
