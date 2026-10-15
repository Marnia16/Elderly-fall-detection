"""Health check route with database and notification status."""

import os
from datetime import datetime, timezone
from fastapi import APIRouter

from app import database
from app.notifications import notification_service
from app.routes.predict import is_model_ready
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System Health Status")
async def health_check() -> HealthResponse:
    """
    Returns API health and service availability.
    Includes persistent SQLite database status, ML model availability, and notification service status.
    """
    # Verify SQLite connectivity and row counts
    try:
        summary = database.get_analytics_summary()
        db_status = {
            "connected": True,
            "engine": "SQLite",
            "path": str(database.DB_PATH),
            "total_incidents": summary.get("total_incidents", 0),
            "active_incidents": summary.get("active_incidents", 0),
        }
    except Exception as e:
        db_status = {
            "connected": False,
            "error": str(e),
        }

    # Notification status
    try:
        notif_status = notification_service.get_status()
    except Exception as e:
        notif_status = {
            "mode": "error",
            "error": str(e),
        }

    return HealthResponse(
        status="healthy",
        app_name="Smart Elderly Fall Detection Backend",
        version="0.2.0",
        model_loaded=is_model_ready(),
        environment=os.getenv("ENV", "development"),
        database=db_status,
        notifications=notif_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
