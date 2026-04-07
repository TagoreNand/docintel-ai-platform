import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api_gateway"
sys.path.insert(0, str(API_ROOT))

from app.db.database import SessionLocal, init_db  # noqa: E402
from app.db.models import Document  # noqa: E402
from app.services.pipeline import process_document  # noqa: E402


def main() -> None:
    init_db()
    db = SessionLocal()
    sample_dir = PROJECT_ROOT / "sample_data"
    created = 0
    for path in sorted(sample_dir.glob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".json", ".csv", ".pdf"}:
            continue

        document = Document(
            id=path.stem,
            filename=path.name,
            stored_path=str(path),
            source="sample",
            status="queued",
        )
        existing = db.query(Document).filter(Document.id == document.id).first()
        if not existing:
            db.add(document)
            db.commit()
        process_document(document.id, str(path))
        created += 1

    print(f"Bootstrapped {created} sample documents into the local database.")
    db.close()


if __name__ == "__main__":
    main()
