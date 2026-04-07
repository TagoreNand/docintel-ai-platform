from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.models import Document, ReviewTask

router = APIRouter()


@router.get("/overview")
def analytics_overview(db: Session = Depends(get_db)) -> dict:
    documents = db.query(Document).all()
    reviews = db.query(ReviewTask).all()

    by_type = Counter(document.doc_type or "unknown" for document in documents)
    by_status = Counter(document.status for document in documents)
    review_status = Counter(review.status for review in reviews)

    anomalies = [document.anomaly_score for document in documents if document.anomaly_score is not None]
    avg_anomaly_score = round(sum(anomalies) / len(anomalies), 4) if anomalies else 0.0

    return {
        "documents_total": len(documents),
        "review_tasks_total": len(reviews),
        "documents_by_type": dict(by_type),
        "documents_by_status": dict(by_status),
        "review_status": dict(review_status),
        "average_anomaly_score": avg_anomaly_score,
        "documents_auto_approved": sum(1 for document in documents if document.status == "approved"),
        "documents_pending_review": sum(1 for document in documents if document.status == "needs_review"),
    }
