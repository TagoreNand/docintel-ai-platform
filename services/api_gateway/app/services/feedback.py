"""Reviewer-feedback capture for continual learning.

Human reviewers occasionally correct a document's type. Those corrections are the
highest-quality labels available, so we append them to a JSONL feedback store that
``scripts/retrain_from_feedback.py`` folds back into the training set.
"""

from __future__ import annotations

import json

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def append_feedback(text: str, label: str, source: str = "review") -> None:
    if not text or not label:
        return
    settings.model_path.mkdir(parents=True, exist_ok=True)
    record = {"text": text[:5000], "label": label, "source": source}
    with settings.feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    logger.info("Captured reviewer feedback", extra={"label": label, "source": source})


def load_feedback() -> tuple[list[str], list[str]]:
    path = settings.feedback_path
    if not path.exists():
        return [], []
    texts: list[str] = []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("text") and record.get("label"):
            texts.append(record["text"])
            labels.append(record["label"])
    return texts, labels


def feedback_count() -> int:
    texts, _ = load_feedback()
    return len(texts)
