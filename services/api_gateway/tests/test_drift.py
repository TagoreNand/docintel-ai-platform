import json

from app.core.config import settings
from app.services.drift import compute_drift, population_stability_index
from tests.conftest import make_document


def test_psi_identical_is_zero():
    dist = {"invoice": 0.5, "contract": 0.5}
    assert population_stability_index(dist, dist) < 1e-6


def test_psi_positive_when_shifted():
    base = {"invoice": 0.9, "contract": 0.1}
    current = {"invoice": 0.1, "contract": 0.9}
    assert population_stability_index(base, current) > 0.5


def test_compute_drift_no_baseline(db, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "models_empty"))
    report = compute_drift(db, tenant="default")
    assert report["status"] == "no_baseline"


def test_compute_drift_with_baseline(db, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    settings.model_path.mkdir(parents=True, exist_ok=True)
    settings.drift_baseline_path.write_text(
        json.dumps(
            {
                "label_distribution": {"invoice": 1.0},
                "confidence_hist": {"9": 1.0},
                "embedding_centroid": None,
                "n": 10,
            }
        )
    )
    make_document(db, "a.txt", "Invoice total amount due.", doc_type="invoice")
    make_document(db, "b.txt", "Invoice number and vendor.", doc_type="invoice")

    report = compute_drift(db, tenant="default")
    assert report["status"] in {"ok", "warn", "alert"}
    assert "doc_type_psi" in report
    assert report["n_documents"] == 2
