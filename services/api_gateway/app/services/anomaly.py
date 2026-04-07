from collections import Counter

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentEntity


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def score_anomalies(db: Session, document: Document, entities: list[dict]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    entity_map = {entity["field_name"]: entity["field_value"] for entity in entities}
    score = 0.05

    if document.doc_type == "invoice":
        subtotal = _to_float(entity_map.get("subtotal"))
        tax_amount = _to_float(entity_map.get("tax_amount"))
        total_amount = _to_float(entity_map.get("total_amount"))

        if subtotal is not None and tax_amount is not None and total_amount is not None:
            expected_total = round(subtotal + tax_amount, 2)
            if abs(expected_total - total_amount) > 0.01:
                score += 0.45
                reasons.append("invoice_total_mismatch")

        invoice_number = entity_map.get("invoice_number")
        if invoice_number:
            duplicate_ids = (
                db.query(DocumentEntity)
                .filter(DocumentEntity.field_name == "invoice_number", DocumentEntity.field_value == invoice_number)
                .all()
            )
            duplicate_docs = {entity.document_id for entity in duplicate_ids if entity.document_id != document.id}
            if duplicate_docs:
                score += 0.35
                reasons.append("duplicate_invoice_number")

    if document.doc_type == "claim_form":
        amount_claimed = _to_float(entity_map.get("amount_claimed"))
        if amount_claimed is not None and amount_claimed > 50000:
            score += 0.30
            reasons.append("high_claim_amount")

    # Tiny unsupervised-style heuristic using historic anomaly distribution.
    history = db.query(Document).filter(Document.doc_type == document.doc_type, Document.anomaly_score.isnot(None)).all()
    historic_scores = np.array([item.anomaly_score for item in history if item.anomaly_score is not None], dtype=float)
    if historic_scores.size >= 5:
        z_like = (score - historic_scores.mean()) / max(historic_scores.std(), 1e-6)
        if z_like > 1.5:
            reasons.append("score_outlier_vs_history")
            score += 0.10

    return round(min(score, 0.99), 4), reasons
