"""Anomaly scoring: deterministic business rules + an unsupervised ML model.

Two complementary signals are combined:

* **Rules** — precise, explainable checks (invoice arithmetic mismatch, duplicate
  invoice numbers, oversized claims). These dominate the score because they are
  high-precision and auditable.
* **IsolationForest** — an unsupervised model trained on the joint distribution of
  engineered document features. It catches "weird-looking" documents that break
  no single rule, contributing a smaller, weighted term.

Final score is in ``[0, 1]`` with human-readable reasons attached.
"""

from __future__ import annotations

import math
from functools import lru_cache

import joblib
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentEntity

logger = get_logger(__name__)

FEATURE_NAMES = [
    "log_text_len",
    "n_entities",
    "n_numeric",
    "log_max_value",
    "log_sum_value",
    "invoice_total_residual",
    "tax_ratio",
    "log_amount_claimed",
]


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def anomaly_features(doc_type: str, entities: list[dict], text: str) -> list[float]:
    """Build the fixed-length feature vector used by the IsolationForest."""
    entity_map = {e["field_name"]: e["field_value"] for e in entities}
    numeric = [v for v in (_to_float(e["field_value"]) for e in entities) if v is not None]

    subtotal = _to_float(entity_map.get("subtotal"))
    tax = _to_float(entity_map.get("tax_amount"))
    total = _to_float(entity_map.get("total_amount"))
    residual = 0.0
    if None not in (subtotal, tax, total):
        residual = abs(total - (subtotal + tax))
    tax_ratio = (tax / subtotal) if (subtotal and tax is not None and subtotal != 0) else 0.0
    amount_claimed = _to_float(entity_map.get("amount_claimed")) or 0.0

    return [
        math.log1p(len(text or "")),
        float(len(entities)),
        float(len(numeric)),
        math.log1p(max(numeric) if numeric else 0.0),
        math.log1p(sum(numeric) if numeric else 0.0),
        math.log1p(residual),
        float(tax_ratio),
        math.log1p(amount_claimed),
    ]


@lru_cache(maxsize=1)
def _load_model():
    path = settings.anomaly_model_path
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        logger.info("Loaded anomaly model", extra={"version": bundle.get("version")})
        return bundle
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load anomaly model (%s); using rules only", exc.__class__.__name__)
        return None


def reset_anomaly_model() -> None:
    _load_model.cache_clear()


def _ml_outlier(doc_type: str, entities: list[dict], text: str) -> float:
    """Return a normalised outlier probability in ``[0, 1]`` (0 when no model)."""
    bundle = _load_model()
    if bundle is None:
        return 0.0
    model = bundle["model"]
    features = np.array([anomaly_features(doc_type, entities, text)], dtype=np.float32)
    decision = float(model.decision_function(features)[0])
    # decision_function: > 0 => inlier (normal), < 0 => outlier (boundary at the
    # configured contamination). Map through a sigmoid so normal documents score
    # near 0 and only genuine outliers approach 1.
    scale = max(bundle.get("decision_std", 1.0) or 1.0, 1e-3)
    return float(1.0 / (1.0 + math.exp(decision / scale)))


def score_anomalies(
    db: Session,
    document: Document,
    doc_type: str,
    entities: list[dict],
    text: str,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    entity_map = {e["field_name"]: e["field_value"] for e in entities}
    score = 0.05

    if doc_type == "invoice":
        subtotal = _to_float(entity_map.get("subtotal"))
        tax = _to_float(entity_map.get("tax_amount"))
        total = _to_float(entity_map.get("total_amount"))
        if None not in (subtotal, tax, total) and abs(round(subtotal + tax, 2) - total) > 0.01:
            score += 0.45
            reasons.append("invoice_total_mismatch")

        invoice_number = entity_map.get("invoice_number")
        if invoice_number:
            duplicates = (
                db.query(DocumentEntity)
                .filter(
                    DocumentEntity.field_name == "invoice_number",
                    DocumentEntity.field_value == invoice_number,
                    DocumentEntity.document_id != document.id,
                )
                .count()
            )
            if duplicates:
                score += 0.35
                reasons.append("duplicate_invoice_number")

    if doc_type == "claim_form":
        amount_claimed = _to_float(entity_map.get("amount_claimed"))
        if amount_claimed is not None and amount_claimed > 50000:
            score += 0.30
            reasons.append("high_claim_amount")

    ml_outlier = _ml_outlier(doc_type, entities, text)
    if ml_outlier > 0:
        score += settings.anomaly_ml_weight * ml_outlier
        if ml_outlier >= 0.7:
            reasons.append(f"ml_outlier={round(ml_outlier, 2)}")

    return round(min(score, 0.99), 4), reasons
