"""Health check route."""

import os
from datetime import datetime, timezone
from fastapi import APIRouter
from app.routes.predict import is_model_ready
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System Health Status")
async def health_check() -> HealthResponse:
    """
    Returns API health and service availability.
    This endpoint operates reliably regardless of whether a machine learning model is loaded.
    """
    return HealthResponse(
        status="healthy",
        app_name="Smart Elderly Fall Detection Backend",
        version="0.1.0",
        model_loaded=is_model_ready(),
        environment=os.getenv("ENV", "development"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
