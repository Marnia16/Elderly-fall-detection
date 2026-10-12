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
    version: str = Field(..., examples=["0.1.0"])
    model_loaded: bool = Field(
        ...,
        description="Indicates if the trained ML model artifact is loaded and active",
        examples=[False],
    )
    environment: str = Field(..., examples=["development"])
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
        description="Fall detection verdict (None when model is not yet loaded)",
        examples=[None],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score between 0.0 and 1.0 (None when model is not yet loaded)",
        examples=[None],
    )
    features_extracted: Optional[int] = Field(
        default=None,
        description="Number of geometric and spatial features extracted from the 33 landmarks",
        examples=[47],
    )
    message: str = Field(
        ...,
        description="Status message regarding prediction availability or result",
        examples=["Random Forest model is not yet integrated. Feature extraction succeeded."],
    )
    status: str = Field(
        ...,
        description="Machine-readable status code: model_not_ready, ok, or error",
        examples=["model_not_ready"],
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
    message: str = Field(
        ...,
        description="Human-readable description of the incident",
        examples=["High-impact fall detected near armchair"],
    )
    sensor_snapshot: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional sensor or pose telemetry snapshot captured at the moment of incident",
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


class AlertResponse(BaseModel):
    id: str = Field(..., description="Unique incident identifier")
    patient_id: str = Field(...)
    location: str = Field(...)
    severity: AlertSeverity = Field(...)
    message: str = Field(...)
    status: AlertStatus = Field(...)
    timestamp: str = Field(...)
    sensor_snapshot: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
