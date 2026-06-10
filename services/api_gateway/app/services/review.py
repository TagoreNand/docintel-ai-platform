from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import Document, ReviewTask
from app.services.events import publish_event


def create_review_task(db: Session, document_id: str, reason: str, priority: str = "medium") -> ReviewTask:
    task = ReviewTask(document_id=document_id, reason=reason, priority=priority, status="open")
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def resolve_review_task(
    db: Session,
    task_id: str,
    outcome: str,
    notes: str | None = None,
    corrected_doc_type: str | None = None,
) -> ReviewTask:
    task = db.query(ReviewTask).filter(ReviewTask.id == task_id).first()
    task.status = outcome
    task.notes = notes
    task.resolved_at = datetime.now(timezone.utc)

    document = db.query(Document).filter(Document.id == task.document_id).first()
    if document:
        document.status = "approved" if outcome == "approved" else "reviewed"
        if corrected_doc_type:
            document.doc_type = corrected_doc_type

    db.commit()
    db.refresh(task)
    publish_event(
        "review.resolved",
        {
            "task_id": task_id,
            "document_id": task.document_id,
            "outcome": outcome,
            "corrected_doc_type": corrected_doc_type,
        },
    )
    # Capture the human correction as a high-quality training label.
    if corrected_doc_type and document and document.processed_text:
        from app.services.feedback import append_feedback

        append_feedback(document.processed_text, corrected_doc_type, source="review")
    return task
