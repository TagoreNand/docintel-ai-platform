"""Document parsing & normalisation, with OCR for scanned PDFs and images."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pypdf import PdfReader

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ocr import get_ocr_engine

logger = get_logger(__name__)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # Scanned / image-only PDFs yield little or no text -> fall back to OCR.
    if len(text.strip()) < settings.ocr_min_chars:
        ocr = get_ocr_engine()
        if ocr.available:
            logger.info("PDF has little extractable text; running OCR", extra={"file": path.name})
            ocr_text = ocr.pdf_to_text(path)
            if len(ocr_text.strip()) > len(text.strip()):
                return ocr_text
    return text


def parse_document(path: str) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return _read_text(file_path)

    if suffix == ".json":
        return json.dumps(json.loads(_read_text(file_path)), indent=2)

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            return "\n".join(", ".join(row) for row in csv.reader(handle))

    if suffix in settings.image_extensions:
        ocr = get_ocr_engine()
        if not ocr.available:
            logger.warning("Image uploaded but OCR is unavailable", extra={"file": file_path.name})
            return ""
        return ocr.image_to_text(file_path)

    if suffix == ".pdf":
        return _parse_pdf(file_path)

    raise ValueError(f"Unsupported format: {suffix}")
