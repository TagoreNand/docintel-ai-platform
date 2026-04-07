import csv
import json
from pathlib import Path

from pypdf import PdfReader


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_document(path: str) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return _read_text(file_path)

    if suffix == ".json":
        data = json.loads(_read_text(file_path))
        return json.dumps(data, indent=2)

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            return "\n".join([", ".join(row) for row in reader])

    if suffix == ".pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    raise ValueError(f"Unsupported format: {suffix}")
