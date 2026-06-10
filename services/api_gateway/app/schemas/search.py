from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool | None = None  # override the configured reranker behaviour


class EvidenceItem(BaseModel):
    document_id: str
    filename: str
    chunk_index: int = 0
    score: float  # final score (rerank score if reranked, else fused RRF score)
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    text: str


class RetrievalStrategy(BaseModel):
    embedding_backend: str
    vector_backend: str
    reranker: str | None = None
    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused_candidates: int = 0
    reranked: bool = False
    latency_ms: float = 0.0


class SearchResponse(BaseModel):
    question: str
    answer: str
    evidence: list[EvidenceItem]
    related_entities: dict[str, list[str]]
    strategy: RetrievalStrategy | None = None
