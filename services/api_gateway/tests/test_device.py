from app.core.config import settings
from app.services.embeddings import resolve_device


def test_forced_device(monkeypatch):
    monkeypatch.setattr(settings, "inference_device", "cpu")
    assert resolve_device() == "cpu"
    monkeypatch.setattr(settings, "inference_device", "cuda")
    assert resolve_device() == "cuda"


def test_auto_device_resolves_valid(monkeypatch):
    monkeypatch.setattr(settings, "inference_device", "auto")
    assert resolve_device() in ("cpu", "cuda")
