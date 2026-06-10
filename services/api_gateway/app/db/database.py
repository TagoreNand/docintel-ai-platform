from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings


def _normalized_database_url() -> str:
    """Resolve a relative SQLite path against the project root and ensure its
    parent directory exists, so the app runs from any working directory."""
    url = settings.database_url
    prefix = "sqlite:///"
    if url.startswith(prefix):
        raw = url[len(prefix):]
        path = Path(raw)
        if not path.is_absolute():
            path = (settings.project_root / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"{prefix}{path.as_posix()}"
    return url


DATABASE_URL = _normalized_database_url()
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()


def init_db() -> None:
    from app.db.models import Document, DocumentChunk, DocumentEntity, ReviewTask  # noqa: F401

    data_dir = Path(settings.project_root) / "data"
    for sub in ("", "uploads", "models", "index"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
