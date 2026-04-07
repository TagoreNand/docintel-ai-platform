from fastapi import APIRouter

from app.api.routes import analytics, documents, health, review, search

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(review.router, prefix="/review", tags=["review"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
