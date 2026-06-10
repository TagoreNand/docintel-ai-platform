from pathlib import Path

from app.services import classification as C
from app.services.classification import classify_document, rule_scores


def test_rule_fallback_when_no_model(monkeypatch):
    monkeypatch.setattr(C, "_load_model", lambda: None)
    text = ("Invoice Number: INV-1024\nDue Date: 2026-04-20\nVendor: Nova\n"
            "Subtotal: 1250\nTax: 125\nTotal Amount: 1375")
    label, conf, _ = classify_document(text)
    assert label == "invoice"
    assert conf > 0.7


def test_rule_fallback_unknown(monkeypatch):
    monkeypatch.setattr(C, "_load_model", lambda: None)
    label, conf, _ = classify_document("the quick brown fox jumps over the lazy dog")
    assert label == "unknown"
    assert conf == 0.35


def test_rule_scores_counts_patterns():
    scores = rule_scores("Effective Date and Governing Law and termination clause")
    assert scores.get("contract", 0) >= 2


def test_ml_classifier_trains_and_predicts():
    from app.ml.train_classifier import train

    card = train(samples_per_class=40, seed=1)
    assert card["test_accuracy"] >= 0.9
    assert set(card["classes"]) >= {"invoice", "contract", "claim_form"}

    C.reset_classifier()
    text = ("Insurance Claim Form\nClaim ID: CLM-77\nClaimant: Jane Doe\n"
            "Policy Number: POL-9\nAmount Claimed: 4200")
    label, conf, details = classify_document(text)
    assert label == "claim_form"
    assert 0.0 < conf <= 0.99
    assert isinstance(details, dict) and details
    C.reset_classifier()
