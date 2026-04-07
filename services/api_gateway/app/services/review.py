from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import Document, ReviewTask


def create_review_task(db: Session, document_id: str, reason: str, priority: str = "medium") -> ReviewTask:
    task = ReviewTask(document_id=document_id, reason=reason, priority=priority, status="open")
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def resolve_review_task(db: Session, task_id: str, outcome: str, notes: str | None = None) -> ReviewTask:
    task = db.query(ReviewTask).filter(ReviewTask.id == task_id).first()
    task.status = outcome
    task.notes = notes
    task.resolved_at = datetime.now(timezone.utc)

    document = db.query(Document).filter(Document.id == task.document_id).first()
    if document:
        document.status = "approved" if outcome == "approved" else "reviewed"

    db.commit()
    db.refresh(task)
    return task
