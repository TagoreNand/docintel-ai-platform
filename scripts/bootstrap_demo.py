"""Bootstrap a full local demo: train models, ingest samples, index them."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api_gateway"
sys.path.insert(0, str(API_ROOT))

from app.core.config import settings  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402
from app.db.models import Document  # noqa: E402
from app.services.pipeline import process_document  # noqa: E402
from app.services.vector_store import get_vector_store  # noqa: E402


def _ensure_models() -> None:
    if settings.classifier_path.exists() and settings.anomaly_model_path.exists():
        return
    print("Models not found; training them first ...")
    from app.ml.train_anomaly import train as train_anomaly
    from app.ml.train_classifier import train as train_classifier

    train_classifier()
    train_anomaly()


def main() -> None:
    init_db()
    _ensure_models()
    db = SessionLocal()
    sample_dir = PROJECT_ROOT / "sample_data"
    created = 0
    for path in sorted(sample_dir.glob("*")):
        if path.suffix.lower() not in settings.supported_extensions:
            continue
        document = Document(
            id=path.stem,
            filename=path.name,
            stored_path=str(path),
            source="sample",
            status="queued",
        )
        if not db.query(Document).filter(Document.id == document.id).first():
            db.add(document)
            db.commit()
        process_document(document.id, str(path))
        created += 1

    print(f"Bootstrapped {created} sample documents.")
    print(f"Vector index now holds {get_vector_store().count()} chunks ({get_vector_store().backend}).")
    db.close()


if __name__ == "__main__":
    main()
