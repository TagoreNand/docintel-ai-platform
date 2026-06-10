"""Hybrid document classification: calibrated ML model with a rule fallback.

When a trained model is present (``data/models/document_classifier.joblib``) the
calibrated logistic-regression probabilities drive the prediction. The keyword
rule engine is always evaluated too and blended in: agreement boosts confidence,
disagreement tempers it. If no model is available — or its top probability is
below ``classifier_min_confidence`` — the transparent rule engine takes over so
the system is never dependent on a trained artifact being present.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import joblib

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


DOCUMENT_PATTERNS = {
    "invoice": [
        "invoice number", "invoice #", "bill to", "due date", "subtotal",
        "tax", "total amount", "vendor", "po number",
    ],
    "contract": [
        "agreement", "effective date", "party", "termination", "renewal",
        "clause", "governing law", "confidentiality",
    ],
    "claim_form": [
        "claim id", "incident date", "claimant", "adjuster", "loss description",
        "amount claimed", "policy number",
    ],
    "bank_statement": [
        "account number", "statement period", "closing balance",
        "opening balance", "debit", "credit",
    ],
    "resume": ["experience", "education", "skills", "projects", "linkedin"],
    "compliance_report": [
        "audit", "control", "observation", "remediation", "risk rating", "finding",
    ],
}


def rule_scores(text: str) -> dict[str, int]:
    lowered = text.lower()
    scores: dict[str, int] = defaultdict(int)
    for doc_type, patterns in DOCUMENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in lowered:
                scores[doc_type] += 1
    return dict(scores)


def _rule_classify(scores: dict[str, int]) -> tuple[str, float, dict[str, int]]:
    if not scores:
        return "unknown", 0.35, {}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    total = sum(scores.values())
    confidence = round(min(0.99, 0.55 + (best_score / max(total, 1)) * 0.4), 4)
    return best_type, confidence, scores


@lru_cache(maxsize=1)
def _load_model():
    path = settings.classifier_path
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        logger.info("Loaded document classifier", extra={"version": bundle.get("version")})
        return bundle
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load classifier (%s); using rules", exc.__class__.__name__)
        return None


def reset_classifier() -> None:
    _load_model.cache_clear()


def classify_document(text: str) -> tuple[str, float, dict]:
    """Return ``(doc_type, confidence, details)``.

    ``details`` holds per-class probabilities when the ML model is used, or the
    raw rule hit-counts when the rule engine is used.
    """
    scores = rule_scores(text)
    rule_label = max(scores, key=scores.get) if scores else None
    bundle = _load_model()

    if bundle is not None:
        pipeline = bundle["pipeline"]
        classes = bundle["classes"]
        proba = pipeline.predict_proba([text])[0]
        best_idx = int(proba.argmax())
        ml_label = str(classes[best_idx])
        ml_conf = float(proba[best_idx])

        if ml_conf >= settings.classifier_min_confidence:
            blend = settings.classifier_rule_blend
            agreement = 1.0 if rule_label == ml_label else 0.0
            confidence = round(min(0.99, (1 - blend) * ml_conf + blend * agreement), 4)
            details = {str(cls): round(float(p), 4) for cls, p in zip(classes, proba)}
            return ml_label, confidence, details

    return _rule_classify(scores)
