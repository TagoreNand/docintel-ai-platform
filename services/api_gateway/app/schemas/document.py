from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EntityRead(BaseModel):
    id: str
    document_id: str
    field_name: str
    field_value: str
    confidence: float
    extractor_version: str
    created_at: datetime


class DocumentChunkRead(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    text: str
    created_at: datetime


class DocumentRead(BaseModel):
    id: str
    filename: str
    status: str
    source: str
    doc_type: str | None = None
    confidence: float | None = None
    anomaly_score: float | None = None
    summary: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailRead(DocumentRead):
    entities: list[EntityRead]
    chunks: list[DocumentChunkRead]
