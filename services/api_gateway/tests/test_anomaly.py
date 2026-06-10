import uuid

from app.db.models import Document
from app.services import anomaly as A
from app.services.anomaly import anomaly_features, score_anomalies
from app.services.extraction import extract_entities


def _doc(db):
    d = Document(id=str(uuid.uuid4()), filename="x", stored_path="x", status="processing")
    db.add(d)
    db.commit()
    return d


def test_invoice_total_mismatch_flagged(db, monkeypatch):
    monkeypatch.setattr(A, "_load_model", lambda: None)
    text = "Invoice Number: INV-1\nSubtotal: 1000.00\nTax: 100.00\nTotal Amount: 9999.00"
    entities = extract_entities("invoice", text)
    score, reasons = score_anomalies(db, _doc(db), "invoice", entities, text)
    assert "invoice_total_mismatch" in reasons
    assert score >= 0.45


def test_clean_invoice_not_flagged(db, monkeypatch):
    monkeypatch.setattr(A, "_load_model", lambda: None)
    text = "Invoice Number: INV-2\nSubtotal: 1000.00\nTax: 100.00\nTotal Amount: 1100.00"
    entities = extract_entities("invoice", text)
    score, reasons = score_anomalies(db, _doc(db), "invoice", entities, text)
    assert "invoice_total_mismatch" not in reasons
    assert score < 0.25


def test_duplicate_invoice_number_flagged(db, monkeypatch):
    monkeypatch.setattr(A, "_load_model", lambda: None)
    from app.db.models import DocumentEntity

    other = _doc(db)
    db.add(DocumentEntity(document_id=other.id, field_name="invoice_number", field_value="INV-DUP", confidence=0.9))
    db.commit()

    text = "Invoice Number: INV-DUP\nSubtotal: 100.00\nTax: 10.00\nTotal Amount: 110.00"
    entities = extract_entities("invoice", text)
    score, reasons = score_anomalies(db, _doc(db), "invoice", entities, text)
    assert "duplicate_invoice_number" in reasons


def test_high_claim_amount_flagged(db, monkeypatch):
    monkeypatch.setattr(A, "_load_model", lambda: None)
    text = "Claim ID: CLM-1\nClaimant: J\nPolicy Number: P1\nAmount Claimed: 90000.00"
    entities = extract_entities("claim_form", text)
    score, reasons = score_anomalies(db, _doc(db), "claim_form", entities, text)
    assert "high_claim_amount" in reasons


def test_feature_vector_is_fixed_length():
    feats = anomaly_features("invoice", extract_entities("invoice", "Subtotal: 10\nTax: 1\nTotal Amount: 11"), "x")
    assert len(feats) == len(A.FEATURE_NAMES)
