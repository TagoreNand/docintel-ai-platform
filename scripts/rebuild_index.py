"""Rebuild the vector index from the database, for every tenant."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "api_gateway"))

from app.core.config import settings  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402
from app.db.models import Document  # noqa: E402
from app.services.retrieval import ensure_index  # noqa: E402
from app.services.vector_store import get_vector_store, total_indexed_vectors  # noqa: E402


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        tenants = [row[0] for row in db.query(Document.tenant_id).distinct().all()]
        if not tenants:
            tenants = [settings.default_tenant]
        for tenant in tenants:
            get_vector_store(tenant).clear()
            store, records = ensure_index(db, tenant)
            print(f"  tenant={tenant!r}: {store.count()} vectors ({store.backend}) from {len(records)} chunks")
        print(f"Rebuilt index: {total_indexed_vectors()} vectors across {len(tenants)} tenant(s).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
