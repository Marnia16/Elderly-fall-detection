"""Request and response schemas for the Camera-based Fall Detection API."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


# --- Health Schemas ---
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    app_name: str = Field(..., examples=["Smart Elderly Fall Detection Backend"])
    version: str = Field(..., examples=["0.2.0"])
    model_loaded: bool = Field(
        ...,
        description="Indicates if the trained ML model artifact is loaded and active",
        examples=[False],
    )
    environment: str = Field(..., examples=["development"])
    database: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Persistent SQLite database connectivity and metrics",
    )
    notifications: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Notification service operational status (demo/smtp)",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        examples=["2026-10-12T06:00:00Z"],
    )


# --- MediaPipe Pose Landmark & Prediction Schemas ---
class PoseLandmark(BaseModel):
    x: float = Field(
        ...,
        description="Normalized horizontal coordinate in [0.0, 1.0] from camera frame",
        examples=[0.485],
    )
    y: float = Field(
        ...,
        description="Normalized vertical coordinate in [0.0, 1.0] from camera frame (0 is top)",
        examples=[0.552],
    )
    z: float = Field(
        ...,
        description="Depth coordinate relative to midpoint of hips (smaller values closer to camera)",
        examples=[-0.124],
    )
    visibility: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence likelihood that landmark is visible within the frame (0.0 to 1.0)",
        examples=[0.98],
    )


class PosePredictionRequest(BaseModel):
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of the video frame capture",
        examples=["2026-10-12T06:30:00Z"],
    )
    landmarks: List[PoseLandmark] = Field(
        ...,
        description="Array of exactly 33 MediaPipe Pose landmarks",
    )
    frame_id: Optional[int] = Field(
        default=None,
        description="Sequential frame index from the client camera feed",
        examples=[1024],
    )
    patient_id: Optional[str] = Field(
        default="patient-001",
        description="Identifier of the monitored individual",
        examples=["patient-001"],
    )

    @field_validator("landmarks")
    @classmethod
    def validate_landmark_count(cls, v: List[PoseLandmark]) -> List[PoseLandmark]:
        if len(v) != 33:
            raise ValueError(f"MediaPipe Pose requires exactly 33 landmarks, received {len(v)}.")
        return v


class PredictionResponse(BaseModel):
    model_loaded: bool = Field(
        ...,
        description="Flag indicating if the trained Random Forest model is loaded",
        examples=[False],
    )
    fall_detected: Optional[bool] = Field(
        default=None,
        description="Raw instantaneous frame-level fall detection verdict",
        examples=[None],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Winning-class confidence score between 0.0 and 1.0",
        examples=[0.96],
    )
    fall_probability: Optional[float] = Field(
        default=None,
        description="Estimated probability of fall between 0.0 and 1.0",
        examples=[0.96],
    )
    features_extracted: Optional[int] = Field(
        default=None,
        description="Number of geometric and spatial features extracted from the 33 landmarks",
        examples=[47],
    )
    # Temporal confirmation & active event tracker fields
    fall_confirmed: Optional[bool] = Field(
        default=None,
        description="True if temporal confirmation window confirmed an active fall event",
        examples=[False],
    )
    active_incident_id: Optional[str] = Field(
        default=None,
        description="Active persistent incident identifier if an event is currently ongoing",
        examples=["inc-001"],
    )
    tracker_state: Optional[str] = Field(
        default="MONITORING",
        description="Current state machine mode: MONITORING, CONFIRMING, FALL_ACTIVE, or COOLDOWN",
        examples=["MONITORING"],
    )
    near_fall: Optional[bool] = Field(
        default=False,
        description="Warning indicator for substantial postural instability or near-fall tilt",
        examples=[False],
    )
    post_fall_immobility: Optional[bool] = Field(
        default=False,
        description="Indicator flag when prolonged post-fall lack of movement is detected",
        examples=[False],
    )
    message: str = Field(
        ...,
        description="Status message regarding prediction availability or result",
        examples=["Fall detection inference completed successfully."],
    )
    status: str = Field(
        ...,
        description="Machine-readable status code: model_not_ready, ok, or error",
        examples=["ok"],
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        examples=["2026-10-12T06:30:00Z"],
    )


# --- Alert / Incident Schemas ---
class AlertCreate(BaseModel):
    patient_id: Optional[str] = Field(
        default="patient-001",
        description="Identifier of the monitored individual",
        examples=["patient-001"],
    )
    location: Optional[str] = Field(
        default="Living Room",
        description="Physical location where the incident occurred",
        examples=["Living Room"],
    )
    severity: AlertSeverity = Field(
        default=AlertSeverity.HIGH,
        description="Severity level of the incident",
        examples=[AlertSeverity.HIGH],
    )
    prediction_type: Optional[str] = Field(
        default="MANUAL_ALERT",
        description="Classification or detection trigger type",
        examples=["CONFIRMED_FALL"],
    )
    message: str = Field(
        ...,
        description="Human-readable description of the incident",
        examples=["High-impact fall detected near armchair"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Model confidence score (0.0 to 1.0)",
    )
    fall_probability: Optional[float] = Field(
        default=None,
        description="Model fall probability score (0.0 to 1.0)",
    )
    duration_seconds: Optional[float] = Field(
        default=0.0,
        description="Duration in seconds of the active fall event",
    )
    sensor_snapshot: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional sensor or pose telemetry snapshot captured at the moment of incident",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Caregiver notes or resolution details",
    )


class AlertUpdate(BaseModel):
    status: Optional[AlertStatus] = Field(
        default=None,
        description="Updated lifecycle state of the incident",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Caregiver notes or resolution details",
    )
    escalated: Optional[bool] = Field(
        default=None,
        description="Escalation flag status",
    )


class AlertResponse(BaseModel):
    id: str = Field(..., description="Unique incident identifier")
    patient_id: str = Field(...)
    location: str = Field(...)
    severity: AlertSeverity = Field(...)
    prediction_type: Optional[str] = Field(default="CONFIRMED_FALL")
    message: str = Field(...)
    status: AlertStatus = Field(...)
    confidence: Optional[float] = None
    fall_probability: Optional[float] = None
    duration_seconds: Optional[float] = 0.0
    timestamp: str = Field(...)
    created_at: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    escalated: Optional[bool] = False
    is_demo: Optional[bool] = False
    sensor_snapshot: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class DailyTrendPoint(BaseModel):
    date: str
    label: str
    count: int


class AlertAnalyticsResponse(BaseModel):
    total_incidents: int
    active_incidents: int
    acknowledged_incidents: int
    resolved_incidents: int
    average_confidence: float
    average_fall_probability: float
    daily_trend: List[DailyTrendPoint]


class TrackerConfigResponse(BaseModel):
    confirmation_frames_needed: int
    confirmation_window_s: float
    cooldown_period_s: float
    min_fall_probability: float
    immobility_seconds_threshold: float
    escalation_delay_seconds: float
