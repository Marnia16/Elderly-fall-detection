"""Health check route."""

import os
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_FILES = ["random_forest_model.joblib", "random_forest_model.pkl", "model.joblib", "model.pkl"]


def is_model_available() -> bool:
    """Check if a trained model file exists on disk."""
    if not MODEL_DIR.exists():
        return False
    return any((MODEL_DIR / filename).exists() for filename in MODEL_FILES)


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
        model_loaded=is_model_available(),
        environment=os.getenv("ENV", "development"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
