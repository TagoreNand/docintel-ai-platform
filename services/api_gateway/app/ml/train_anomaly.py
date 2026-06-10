"""Train and persist the unsupervised anomaly model (IsolationForest).

The model is fit on engineered features extracted from the synthetic "normal"
document population. At serving time a document's features are scored against this
learned manifold; documents that fall outside it raise the anomaly score even
when they trip no explicit business rule. We also inject a few synthetic
anomalies purely to *report* held-out separation in the model card.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.datasets import build_dataset
from app.services.anomaly import FEATURE_NAMES, anomaly_features
from app.services.extraction import extract_entities
from app.services.tracking import start_run

logger = get_logger(__name__)


def _features_for(texts: list[str], labels: list[str]) -> np.ndarray:
    rows = []
    for text, label in zip(texts, labels):
        entities = extract_entities(label, text)
        rows.append(anomaly_features(label, entities, text))
    return np.array(rows, dtype=np.float32)


def train(samples_per_class: int = 160, seed: int = 13) -> dict:
    texts, labels = build_dataset(samples_per_class=samples_per_class, seed=seed)
    features = _features_for(texts, labels)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features)

    decision = model.decision_function(features)
    decision_mean = float(decision.mean())
    decision_std = float(decision.std()) or 1.0

    # Report separation on a few obviously-broken invoices (total mismatch).
    rng = np.random.default_rng(seed)
    anomalies = features.copy()[:50]
    anomalies[:, FEATURE_NAMES.index("invoice_total_residual")] = np.log1p(
        rng.uniform(5000, 50000, size=50)
    )
    normal_flags = model.predict(features)  # 1 normal, -1 anomaly
    anomaly_flags = model.predict(anomalies)
    detected = float((anomaly_flags == -1).mean())
    normal_rate = float((normal_flags == 1).mean())

    card = {
        "model": "anomaly_iforest",
        "version": settings.anomaly_model_version,
        "algorithm": "IsolationForest (unsupervised)",
        "feature_names": FEATURE_NAMES,
        "n_samples": int(features.shape[0]),
        "n_estimators": 200,
        "contamination": 0.02,
        "decision_mean": round(decision_mean, 4),
        "decision_std": round(decision_std, 4),
        "normal_recall": round(normal_rate, 4),
        "injected_anomaly_detection_rate": round(detected, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    settings.model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "decision_mean": decision_mean,
            "decision_std": decision_std,
            "feature_names": FEATURE_NAMES,
            "version": settings.anomaly_model_version,
        },
        settings.anomaly_model_path,
    )
    settings.anomaly_card_path.write_text(json.dumps(card, indent=2))

    with start_run("train_anomaly") as run:
        run.log_params({"algorithm": "IsolationForest", "n_estimators": 200, "contamination": 0.02})
        run.log_metrics(
            {"normal_recall": normal_rate, "injected_anomaly_detection_rate": detected}
        )
        run.log_artifact(str(settings.anomaly_card_path))
        if settings.mlflow_register_models:
            run.log_model(model, "model", registered_model_name="docintel-anomaly-iforest")

    logger.info("Trained anomaly model", extra={"detection_rate": card["injected_anomaly_detection_rate"]})
    return card


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
