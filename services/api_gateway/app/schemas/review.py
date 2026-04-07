from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ReviewResolution(BaseModel):
    outcome: str
    notes: str | None = None


class ReviewTaskRead(BaseModel):
    id: str
    document_id: str
    reason: str
    status: str
    priority: str
    notes: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
