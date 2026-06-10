"""Data/model drift monitoring.

Compares the live document population against a baseline captured at training time:

* **doc-type PSI** — Population Stability Index over predicted document types.
* **confidence PSI** — PSI over the classifier's max-probability histogram.
* **embedding drift** — cosine distance between the baseline and current embedding
  centroids.

PSI thresholds (``DRIFT_PSI_WARN`` / ``DRIFT_PSI_ALERT``) map the result to
``ok`` / ``warn`` / ``alert``.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Document

logger = get_logger(__name__)


def population_stability_index(expected: dict, actual: dict, eps: float = 1e-6) -> float:
    psi = 0.0
    for key in set(expected) | set(actual):
        e = max(expected.get(key, 0.0), eps)
        a = max(actual.get(key, 0.0), eps)
        psi += (a - e) * math.log(a / e)
    return float(psi)


def _normalize(counts: dict) -> dict:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _confidence_hist(confidences) -> dict:
    hist: dict[str, int] = {}
    for c in confidences:
        bucket = str(min(int(float(c) * 10), 9))
        hist[bucket] = hist.get(bucket, 0) + 1
    return _normalize(hist)


def capture_classifier_baseline(pipeline, texts: list[str], labels: list[str]) -> dict:
    """Persist the training-time distribution as the drift baseline."""
    from app.services.embeddings import get_embedder

    confidences = pipeline.predict_proba(texts).max(axis=1)
    centroid = get_embedder().embed(texts).mean(axis=0)
    norm = float(np.linalg.norm(centroid)) or 1.0

    baseline = {
        "label_distribution": _normalize(dict(Counter(labels))),
        "confidence_hist": _confidence_hist(confidences),
        "embedding_centroid": (centroid / norm).tolist(),
        "n": len(texts),
    }
    settings.model_path.mkdir(parents=True, exist_ok=True)
    settings.drift_baseline_path.write_text(json.dumps(baseline))
    logger.info("Captured drift baseline", extra={"n": len(texts)})
    return baseline


def _status(psi_values: list[float]) -> str:
    worst = max(psi_values) if psi_values else 0.0
    if worst >= settings.drift_psi_alert:
        return "alert"
    if worst >= settings.drift_psi_warn:
        return "warn"
    return "ok"


def compute_drift(db: Session, tenant: str | None = None) -> dict:
    if not settings.drift_baseline_path.exists():
        return {"status": "no_baseline", "detail": "Train the classifier to capture a baseline."}
    baseline = json.loads(settings.drift_baseline_path.read_text())

    query = db.query(Document).filter(Document.doc_type.isnot(None))
    if tenant is not None:
        query = query.filter(Document.tenant_id == tenant)
    docs = query.all()
    if not docs:
        return {"status": "no_data", "detail": "No processed documents yet."}

    current_labels = _normalize(dict(Counter(d.doc_type or "unknown" for d in docs)))
    label_psi = population_stability_index(baseline["label_distribution"], current_labels)

    confidences = [d.confidence for d in docs if d.confidence is not None]
    confidence_psi = (
        population_stability_index(baseline["confidence_hist"], _confidence_hist(confidences))
        if confidences
        else 0.0
    )

    embedding_drift = None
    texts = [d.processed_text for d in docs if d.processed_text]
    base_centroid = baseline.get("embedding_centroid")
    if texts and base_centroid:
        from app.services.embeddings import get_embedder

        centroid = get_embedder().embed(texts).mean(axis=0)
        norm = float(np.linalg.norm(centroid)) or 1.0
        centroid = centroid / norm
        base = np.asarray(base_centroid, dtype=float)
        if base.shape == centroid.shape:
            embedding_drift = round(1.0 - float(np.dot(centroid, base)), 4)

    return {
        "status": _status([label_psi, confidence_psi]),
        "doc_type_psi": round(label_psi, 4),
        "confidence_psi": round(confidence_psi, 4),
        "embedding_drift": embedding_drift,
        "n_documents": len(docs),
        "baseline_n": baseline.get("n"),
        "thresholds": {"warn": settings.drift_psi_warn, "alert": settings.drift_psi_alert},
    }
