from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.retrieval import answer_question

router = APIRouter()


@router.post("/query", response_model=SearchResponse)
def query_documents(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    return answer_question(db, payload.question, payload.top_k)
