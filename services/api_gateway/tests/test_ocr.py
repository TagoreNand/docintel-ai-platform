import pytest

from app.services.ocr import OcrEngine, get_ocr_engine, reset_ocr_engine
from app.services.parser import parse_document


def test_null_engine_returns_empty():
    engine = OcrEngine()
    assert engine.available is False
    assert engine.image_to_text("anything") == ""
    assert engine.pdf_to_text("anything") == ""


def test_image_ocr_or_skip(tmp_path):
    reset_ocr_engine()
    engine = get_ocr_engine()
    if not engine.available:
        pytest.skip("Tesseract OCR not available in this environment")
    from PIL import Image, ImageDraw

    path = tmp_path / "doc.png"
    image = Image.new("RGB", (440, 90), "white")
    ImageDraw.Draw(image).text((12, 32), "Hello OCR World", fill="black")
    image.save(path)

    text = parse_document(str(path))
    assert "OCR" in text or "Hello" in text
    reset_ocr_engine()
