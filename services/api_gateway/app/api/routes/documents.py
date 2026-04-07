from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.db.models import Document, DocumentChunk, DocumentEntity
from app.schemas.document import (
    DocumentChunkRead,
    DocumentDetailRead,
    DocumentRead,
    EntityRead,
)
from app.services.pipeline import process_document

router = APIRouter()


@router.post("/upload", response_model=DocumentRead)
async def upload_document(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
) -> Document:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.supported_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    document_id = str(uuid4())
    target_dir = Path(settings.upload_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{document_id}{suffix}"

    content = await file.read()
    target_path.write_bytes(content)

    document = Document(
        id=document_id,
        filename=file.filename,
        stored_path=str(target_path),
        status="queued",
        source="upload",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    background_tasks.add_task(process_document, document_id, str(target_path))
    return document


@router.post("/ingest-sample", response_model=List[DocumentRead])
def ingest_sample_documents(db: Session = Depends(get_db)) -> list[Document]:
    sample_dir = Path(settings.project_root) / "sample_data"
    created: list[Document] = []
    for path in sorted(sample_dir.glob("*")):
        if path.suffix.lower() not in settings.supported_extensions:
            continue
        document = Document(
            id=str(uuid4()),
            filename=path.name,
            stored_path=str(path),
            status="queued",
            source="sample",
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        process_document(document.id, str(path))
        db.refresh(document)
        created.append(document)
    return created


@router.get("", response_model=List[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return db.query(Document).order_by(Document.created_at.desc()).all()


@router.get("/{document_id}", response_model=DocumentDetailRead)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentDetailRead:
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    entities = db.query(DocumentEntity).filter(DocumentEntity.document_id == document_id).all()
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )

    return DocumentDetailRead(
        **document.to_dict(),
        entities=[EntityRead.model_validate(entity.to_dict()) for entity in entities],
        chunks=[DocumentChunkRead.model_validate(chunk.to_dict()) for chunk in chunks],
    )
