from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.models import ReviewTask
from app.schemas.review import ReviewResolution, ReviewTaskRead
from app.services.review import resolve_review_task

router = APIRouter()


@router.get("/tasks", response_model=List[ReviewTaskRead])
def list_review_tasks(db: Session = Depends(get_db)) -> list[ReviewTask]:
    return db.query(ReviewTask).order_by(ReviewTask.created_at.desc()).all()


@router.post("/tasks/{task_id}/resolve", response_model=ReviewTaskRead)
def resolve_task(task_id: str, payload: ReviewResolution, db: Session = Depends(get_db)) -> ReviewTask:
    task = db.query(ReviewTask).filter(ReviewTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Review task not found")
    return resolve_review_task(db, task_id, payload.outcome, payload.notes)
