from app.services.classification import classify_document
from app.services.extraction import extract_entities


def test_invoice_classification():
    text = """
    Invoice Number: INV-1024
    Invoice Date: 2026-04-01
    Due Date: 2026-04-20
    Vendor: Nova Industrial Supplies
    Subtotal: 1250.00
    Tax: 125.00
    Total Amount: 1375.00
    """
    doc_type, confidence, _ = classify_document(text)
    assert doc_type == "invoice"
    assert confidence > 0.7


def test_invoice_extraction():
    text = """
    Invoice Number: INV-1024
    Invoice Date: 2026-04-01
    Due Date: 2026-04-20
    Vendor: Nova Industrial Supplies
    Subtotal: 1250.00
    Tax: 125.00
    Total Amount: 1375.00
    """
    entities = extract_entities("invoice", text)
    fields = {entity["field_name"]: entity["field_value"] for entity in entities}
    assert fields["invoice_number"] == "INV-1024"
    assert fields["total_amount"] == "1375.00"
