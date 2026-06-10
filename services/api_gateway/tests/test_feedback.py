from app.core.config import settings
from app.services.feedback import append_feedback, feedback_count, load_feedback


def test_append_and_load(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    append_feedback("an invoice for services", "invoice")
    append_feedback("a service agreement contract", "contract")
    texts, labels = load_feedback()
    assert labels == ["invoice", "contract"]
    assert len(texts) == 2
    assert feedback_count() == 2


def test_empty_feedback_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    append_feedback("", "invoice")
    append_feedback("text", "")
    assert feedback_count() == 0
