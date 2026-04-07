from pydantic import BaseModel


class SearchRequest(BaseModel):
    question: str
    top_k: int = 5


class EvidenceItem(BaseModel):
    document_id: str
    filename: str
    score: float
    text: str


class SearchResponse(BaseModel):
    question: str
    answer: str
    evidence: list[EvidenceItem]
    related_entities: dict[str, list[str]]
