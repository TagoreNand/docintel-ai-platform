"""OCR with a layered fallback chain.

Engine precedence (highest first), each degrading gracefully to the next:

1. ``LayoutOcrEngine`` — an OCR-free, layout-aware document-understanding model
   (Donut / LayoutLM family via transformers). Enabled with ``ENABLE_LAYOUT_OCR``.
2. ``TesseractOcrEngine`` — classic OCR via the Tesseract binary.
3. ``OcrEngine`` (null) — used when nothing is available; the parser then relies on
   any embedded text layer only.

This keeps scanned-document support working everywhere while allowing a powerful
visual model in production.
"""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OcrEngine:
    """Null OCR engine: used when OCR is disabled or unavailable."""

    name = "none"
    available = False

    def image_to_text(self, path: Path) -> str:
        return ""

    def pdf_to_text(self, path: Path) -> str:
        return ""


class TesseractOcrEngine(OcrEngine):
    name = "tesseract"
    available = True

    def __init__(self) -> None:
        import pytesseract  # noqa: F401

        self._pytesseract = pytesseract

    def image_to_text(self, path: Path) -> str:
        from PIL import Image

        with Image.open(path) as image:
            return self._pytesseract.image_to_string(image, lang=settings.ocr_languages)

    def pdf_to_text(self, path: Path) -> str:
        from pdf2image import convert_from_path

        pages = convert_from_path(str(path))
        return "\n\n".join(
            self._pytesseract.image_to_string(page, lang=settings.ocr_languages) for page in pages
        )


class LayoutOcrEngine(OcrEngine):
    """Layout-aware, OCR-free document understanding (Donut / LayoutLM).

    Donut is a vision-encoder-decoder that reads a page image and emits structured
    content without a separate OCR step, capturing layout context that line-based
    OCR loses. Best results need a task-fine-tuned checkpoint; ``LAYOUT_MODEL`` is
    configurable.
    """

    name = "layout"
    available = True

    def __init__(self) -> None:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        from app.services.embeddings import resolve_device

        self.device = resolve_device()
        self.name = f"layout:{settings.layout_model}@{self.device}"
        self._processor = DonutProcessor.from_pretrained(settings.layout_model)
        self._model = VisionEncoderDecoderModel.from_pretrained(settings.layout_model).to(self.device)

    @staticmethod
    def _flatten(value) -> str:
        if isinstance(value, dict):
            return " ".join(LayoutOcrEngine._flatten(v) for v in value.values())
        if isinstance(value, list):
            return " ".join(LayoutOcrEngine._flatten(v) for v in value)
        return str(value)

    def _read_image(self, image) -> str:
        prompt = "<s_cord-v2>"
        pixel_values = self._processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self._processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        outputs = self._model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=768)
        sequence = self._processor.batch_decode(outputs)[0]
        sequence = sequence.replace(self._processor.tokenizer.eos_token, "").replace(
            self._processor.tokenizer.pad_token, ""
        )
        try:
            return self._flatten(self._processor.token2json(sequence))
        except Exception:  # noqa: BLE001
            return sequence.strip()

    def image_to_text(self, path: Path) -> str:
        from PIL import Image

        with Image.open(path) as image:
            return self._read_image(image)

    def pdf_to_text(self, path: Path) -> str:
        from pdf2image import convert_from_path

        return "\n\n".join(self._read_image(page) for page in convert_from_path(str(path)))


def _build_engine() -> OcrEngine:
    if not settings.ocr_enabled:
        return OcrEngine()

    if settings.enable_layout_ocr:
        try:
            engine = LayoutOcrEngine()
            logger.info("OCR engine ready", extra={"engine": engine.name})
            return engine
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Layout OCR unavailable (%s); falling back to Tesseract", exc.__class__.__name__
            )

    if shutil.which("tesseract") is None:
        logger.warning("Tesseract binary not found; OCR disabled")
        return OcrEngine()
    try:
        engine = TesseractOcrEngine()
        logger.info("OCR engine ready", extra={"engine": engine.name, "lang": settings.ocr_languages})
        return engine
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR unavailable (%s); continuing without OCR", exc.__class__.__name__)
        return OcrEngine()


@lru_cache(maxsize=1)
def get_ocr_engine() -> OcrEngine:
    return _build_engine()


def reset_ocr_engine() -> None:
    get_ocr_engine.cache_clear()
