"""In-memory incident and alert management endpoints."""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.schemas import AlertCreate, AlertResponse, AlertStatus, AlertUpdate

router = APIRouter(tags=["Alerts"])

# In-memory storage for incidents/alerts
_alerts_db: Dict[str, dict] = {}


def _seed_initial_alert() -> None:
    """Pre-populate a sample incident for frontend dashboard demonstration."""
    sample_id = "inc-001"
    _alerts_db[sample_id] = {
        "id": sample_id,
        "patient_id": "patient-101",
        "location": "Living Room",
        "severity": "HIGH",
        "message": "Rapid vertical deceleration detected — possible fall near couch.",
        "status": "ACTIVE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensor_snapshot": {
            "accel_x": 0.35,
            "accel_y": -1.82,
            "accel_z": 18.45,
            "gyro_magnitude": 142.3,
        },
        "notes": "Automated alert logged by telemetry stream.",
    }


# Seed the storage on startup
_seed_initial_alert()


@router.get(
    "/alerts",
    response_model=List[AlertResponse],
    summary="List all incident alerts",
)
async def list_alerts(
    status_filter: Optional[AlertStatus] = Query(
        None,
        alias="status",
        description="Filter alerts by status (ACTIVE, ACKNOWLEDGED, RESOLVED)",
    ),
    limit: int = Query(50, ge=1, le=100, description="Max number of items to return"),
) -> List[AlertResponse]:
    """Retrieve all alerts stored in memory, ordered with the newest first."""
    alerts = list(_alerts_db.values())

    if status_filter:
        alerts = [a for a in alerts if a.get("status") == status_filter.value]

    # Sort descending by timestamp
    alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
    return [AlertResponse(**a) for a in alerts[:limit]]


@router.post(
    "/alerts",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new fall incident alert",
)
async def create_alert(payload: AlertCreate) -> AlertResponse:
    """Create a new fall incident alert and store it in memory."""
    alert_id = f"inc-{uuid.uuid4().hex[:8]}"
    alert_record = {
        "id": alert_id,
        "patient_id": payload.patient_id or "patient-001",
        "location": payload.location or "Unknown",
        "severity": payload.severity.value,
        "message": payload.message,
        "status": AlertStatus.ACTIVE.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensor_snapshot": payload.sensor_snapshot,
        "notes": None,
    }
    _alerts_db[alert_id] = alert_record
    return AlertResponse(**alert_record)


@router.get(
    "/alerts/{alert_id}",
    response_model=AlertResponse,
    summary="Get incident details by ID",
)
async def get_alert(alert_id: str) -> AlertResponse:
    """Retrieve details for a specific incident alert."""
    if alert_id not in _alerts_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )
    return AlertResponse(**_alerts_db[alert_id])


@router.patch(
    "/alerts/{alert_id}",
    response_model=AlertResponse,
    summary="Update incident alert status or notes",
)
async def update_alert(alert_id: str, payload: AlertUpdate) -> AlertResponse:
    """Update status (e.g. ACKNOWLEDGED or RESOLVED) or caregiver notes."""
    if alert_id not in _alerts_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )

    alert = _alerts_db[alert_id]
    if payload.status is not None:
        alert["status"] = payload.status.value
    if payload.notes is not None:
        alert["notes"] = payload.notes

    return AlertResponse(**alert)


@router.delete(
    "/alerts/{alert_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Dismiss or delete an incident alert",
)
async def delete_alert(alert_id: str):
    """Delete an alert from memory."""
    if alert_id not in _alerts_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )
    del _alerts_db[alert_id]
    return None


@router.delete(
    "/alerts",
    status_code=status.HTTP_200_OK,
    summary="Clear all alerts (for testing/reset)",
)
async def clear_all_alerts() -> dict:
    """Clear all in-memory alerts."""
    count = len(_alerts_db)
    _alerts_db.clear()
    return {"message": f"Cleared {count} alerts from memory."}
