"""Train and persist the document-type classifier.

Pipeline: TF-IDF (word 1-2 grams) -> calibrated multinomial logistic regression.
Calibration (Platt scaling) yields trustworthy probability estimates, which the
serving layer uses directly as the document-type confidence that drives the
auto-approve / human-review routing thresholds.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.datasets import DOC_TYPES, build_dataset
from app.services.tracking import start_run

logger = get_logger(__name__)


def build_pipeline() -> Pipeline:
    base = LogisticRegression(max_iter=2000, C=6.0, class_weight="balanced")
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            ("clf", calibrated),
        ]
    )


def train(samples_per_class: int = 160, seed: int = 13, extra: tuple[list[str], list[str]] | None = None) -> dict:
    texts, labels = build_dataset(samples_per_class=samples_per_class, seed=seed)
    n_feedback = 0
    if extra and extra[0]:
        texts = texts + list(extra[0])
        labels = labels + list(extra[1])
        n_feedback = len(extra[0])
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    cv_scores = cross_val_score(build_pipeline(), texts, labels, cv=4, scoring="accuracy")

    card = {
        "model": "document_classifier",
        "version": settings.classifier_version,
        "algorithm": "tfidf + calibrated logistic regression",
        "classes": sorted(DOC_TYPES),
        "n_samples": len(texts),
        "n_feedback": n_feedback,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test_accuracy": round(accuracy, 4),
        "test_macro_f1": round(macro_f1, 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    settings.model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "classes": list(pipeline.classes_), "version": settings.classifier_version},
        settings.classifier_path,
    )
    settings.classifier_card_path.write_text(json.dumps(card, indent=2))

    from app.services.drift import capture_classifier_baseline

    capture_classifier_baseline(pipeline, texts, labels)

    with start_run("train_classifier") as run:
        run.log_params({"algorithm": card["algorithm"], "n_samples": card["n_samples"], "C": 6.0})
        run.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_macro_f1": macro_f1,
                "cv_accuracy_mean": float(cv_scores.mean()),
            }
        )
        run.log_artifact(str(settings.classifier_card_path))
        if settings.mlflow_register_models:
            run.log_model(pipeline, "model", registered_model_name="docintel-document-classifier")

    logger.info(
        "Trained document classifier",
        extra={"accuracy": card["test_accuracy"], "macro_f1": card["test_macro_f1"]},
    )
    return card


if __name__ == "__main__":
    summary = train()
    print(json.dumps(summary, indent=2))
