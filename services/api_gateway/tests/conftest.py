"""Pytest configuration: force the offline-safe ML backends and isolate state.

Environment variables are set *before* the app config is imported so the whole
suite runs deterministically with the hashing embedder, the local vector store,
and no reranker / MLflow — no network, no model downloads, no external services.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="docintel_test_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("INDEX_DIR", f"{_TMP}/index")
os.environ.setdefault("MODEL_DIR", f"{_TMP}/models")
os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("VECTOR_BACKEND", "local")
os.environ.setdefault("ENABLE_RERANKER", "false")
os.environ.setdefault("ENABLE_MLFLOW", "false")

import pytest  # noqa: E402

from app.db.database import Base, SessionLocal, engine, init_db  # noqa: E402


@pytest.fixture()
def db():
    """Fresh database + empty indexes for each test."""
    init_db()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    from app.core.config import settings
    from app.services.retrieval import reset_sparse
    from app.services.vector_store import get_vector_store, reset_vector_store

    reset_vector_store()
    reset_sparse()
    try:
        get_vector_store(settings.default_tenant).clear()
    except Exception:
        pass

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def make_document(session, filename, text, doc_type="invoice"):
    """Helper: insert a Document plus naive sentence chunks."""
    import uuid

    from app.db.models import Document, DocumentChunk

    doc_id = str(uuid.uuid4())
    session.add(
        Document(id=doc_id, filename=filename, stored_path=filename, status="processed", doc_type=doc_type)
    )
    for i, sentence in enumerate(s for s in text.split(". ") if s.strip()):
        session.add(DocumentChunk(id=str(uuid.uuid4()), document_id=doc_id, chunk_index=i, text=sentence))
    session.commit()
    return doc_id
