from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk, DocumentEntity
from app.schemas.search import EvidenceItem, SearchResponse


def _keyword_overlap(query: str, text: str) -> float:
    q_tokens = {token for token in query.lower().split() if len(token) > 2}
    t_tokens = set(text.lower().split())
    return len(q_tokens & t_tokens) / max(len(q_tokens), 1)


def answer_question(db: Session, question: str, top_k: int = 5) -> SearchResponse:
    chunks = (
        db.query(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(Document.created_at.desc())
        .all()
    )

    if not chunks:
        return SearchResponse(
            question=question,
            answer="No documents are indexed yet. Upload or ingest documents first.",
            evidence=[],
            related_entities={},
        )

    texts = [chunk.text for chunk, _ in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([question])
    cosine_scores = linear_kernel(query_vec, matrix).flatten()

    scored = []
    for idx, ((chunk, document), cosine_score) in enumerate(zip(chunks, cosine_scores)):
        score = float(cosine_score) * 0.8 + _keyword_overlap(question, chunk.text) * 0.2
        scored.append((score, chunk, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_k]

    evidence = [
        EvidenceItem(
            document_id=document.id,
            filename=document.filename,
            score=round(score, 4),
            text=chunk.text[:450],
        )
        for score, chunk, document in top
    ]

    related_entities_map: dict[str, list[str]] = defaultdict(list)
    top_doc_ids = {item.document_id for item in evidence}
    entities = db.query(DocumentEntity).filter(DocumentEntity.document_id.in_(top_doc_ids)).all()
    for entity in entities:
        related_entities_map[entity.field_name].append(entity.field_value)

    if evidence:
        answer = (
            f"Top evidence suggests the answer is most strongly supported by {evidence[0].filename}. "
            f"The highest-ranked passage is: {evidence[0].text[:220].strip()}..."
        )
    else:
        answer = "No relevant evidence found."

    return SearchResponse(
        question=question,
        answer=answer,
        evidence=evidence,
        related_entities=dict(related_entities_map),
    )
