"""Persistent SQLite incident and alert management endpoints."""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from app import database
from app.notifications import notification_service
from app.schemas import (
    AlertAnalyticsResponse,
    AlertCreate,
    AlertResponse,
    AlertSeverity,
    AlertStatus,
    AlertUpdate,
)

router = APIRouter(tags=["Alerts"])


class CaregiverActionPayload(BaseModel):
    notes: Optional[str] = Field(
        default=None,
        description="Optional caregiver notes or resolution comment",
        examples=["Assisted resident to chair. No injuries reported."],
    )


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
    severity_filter: Optional[AlertSeverity] = Query(
        None,
        alias="severity",
        description="Filter alerts by severity (LOW, MEDIUM, HIGH, CRITICAL)",
    ),
    start_date: Optional[str] = Query(
        None,
        description="Filter created_at after ISO date string",
    ),
    end_date: Optional[str] = Query(
        None,
        description="Filter created_at before ISO date string",
    ),
    search: Optional[str] = Query(
        None,
        description="Search query across message, location, notes, and patient_id",
    ),
    limit: int = Query(50, ge=1, le=200, description="Max number of items to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> List[AlertResponse]:
    """Retrieve persistent incident records from SQLite, ordered with newest first."""
    incidents, _ = database.list_incidents(
        status_filter=status_filter.value if status_filter else None,
        severity_filter=severity_filter.value if severity_filter else None,
        start_date=start_date,
        end_date=end_date,
        search=search,
        limit=limit,
        offset=offset,
    )
    return [AlertResponse(**inc) for inc in incidents]


@router.get(
    "/alerts/analytics/summary",
    response_model=AlertAnalyticsResponse,
    summary="Get incident analytics summary and 7-day trend",
)
async def get_analytics_summary() -> AlertAnalyticsResponse:
    """Computes real incident metrics and daily frequency trend from persistent SQLite data."""
    summary = database.get_analytics_summary()
    return AlertAnalyticsResponse(**summary)


@router.get(
    "/alerts/export/csv",
    summary="Export incidents as CSV report",
)
async def export_alerts_csv(
    status_filter: Optional[AlertStatus] = Query(
        None,
        alias="status",
        description="Filter exported incidents by status",
    ),
    start_date: Optional[str] = Query(
        None,
        description="Filter start ISO date",
    ),
    end_date: Optional[str] = Query(
        None,
        description="Filter end ISO date",
    ),
) -> Response:
    """Generates downloadable CSV containing incident history for caregiver reporting."""
    csv_content = database.export_incidents_to_csv(
        status_filter=status_filter.value if status_filter else None,
        start_date=start_date,
        end_date=end_date,
    )
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"elderly_fall_incidents_{now_str}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/alerts",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new fall incident alert",
)
async def create_alert(payload: AlertCreate) -> AlertResponse:
    """Create a new fall incident alert and persist it in SQLite."""
    record_dict = payload.model_dump()
    # Normalize enum to string for database layer
    if isinstance(record_dict.get("severity"), AlertSeverity):
        record_dict["severity"] = record_dict["severity"].value
    if isinstance(record_dict.get("status"), AlertStatus):
        record_dict["status"] = record_dict["status"].value

    created = database.create_incident(record_dict)

    # Dispatch notification for high/critical active alerts
    try:
        notification_service.send_fall_alert(created)
    except Exception:
        pass

    return AlertResponse(**created)


@router.get(
    "/alerts/{alert_id}",
    response_model=AlertResponse,
    summary="Get incident details by ID",
)
async def get_alert(alert_id: str) -> AlertResponse:
    """Retrieve details for a specific incident alert from SQLite."""
    incident = database.get_incident(alert_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )
    return AlertResponse(**incident)


@router.patch(
    "/alerts/{alert_id}",
    response_model=AlertResponse,
    summary="Update incident alert status or notes",
)
async def update_alert(alert_id: str, payload: AlertUpdate) -> AlertResponse:
    """Update status (e.g. ACKNOWLEDGED or RESOLVED), notes, or escalation flag."""
    existing = database.get_incident(alert_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )

    updated = database.update_incident(
        incident_id=alert_id,
        status=payload.status.value if payload.status else None,
        notes=payload.notes,
        escalated=payload.escalated,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )
    return AlertResponse(**updated)


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertResponse,
    summary="Acknowledge an active incident",
)
async def acknowledge_alert(
    alert_id: str,
    payload: Optional[CaregiverActionPayload] = None,
) -> AlertResponse:
    """Mark an incident as acknowledged by a caregiver, recording timestamp and optional notes."""
    existing = database.get_incident(alert_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )

    notes = payload.notes if payload and payload.notes else existing.get("notes")
    updated = database.update_incident(
        incident_id=alert_id,
        status="ACKNOWLEDGED",
        notes=notes,
        acknowledged_at=datetime.now(timezone.utc).isoformat(),
    )
    return AlertResponse(**updated)


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertResponse,
    summary="Resolve an incident",
)
async def resolve_alert(
    alert_id: str,
    payload: Optional[CaregiverActionPayload] = None,
) -> AlertResponse:
    """Mark an incident as resolved, recording resolution timestamp and caregiver notes."""
    existing = database.get_incident(alert_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )

    combined_notes = existing.get("notes") or ""
    if payload and payload.notes:
        if combined_notes:
            combined_notes += f" | Resolution: {payload.notes}"
        else:
            combined_notes = payload.notes

    updated = database.update_incident(
        incident_id=alert_id,
        status="RESOLVED",
        notes=combined_notes or None,
        resolved_at=datetime.now(timezone.utc).isoformat(),
    )
    return AlertResponse(**updated)


@router.delete(
    "/alerts/{alert_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Dismiss or delete an incident alert",
)
async def delete_alert(alert_id: str):
    """Delete an incident permanently from the SQLite database."""
    success = database.delete_incident(alert_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident alert with ID '{alert_id}' not found.",
        )
    return None


@router.delete(
    "/alerts",
    status_code=status.HTTP_200_OK,
    summary="Clear all alerts (for testing/reset)",
)
async def clear_all_alerts() -> dict:
    """Clear all alerts from SQLite database."""
    count = database.clear_all_incidents()
    return {"message": f"Cleared {count} alerts from database."}
