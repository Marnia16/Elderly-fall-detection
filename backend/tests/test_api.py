"""Automated test suite for Camera-based Fall Detection API endpoints."""

import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app  # noqa: E402

client = TestClient(app)


def make_landmarks(count: int = 33, visibility: float = 0.95):
    """Generate mock MediaPipe landmark list of specified length."""
    landmarks = []
    for i in range(count):
        # Realistic upright posture coordinate ladder
        y_val = 0.5
        if i == 0:  # nose
            y_val = 0.18
        elif i in (11, 12):  # shoulders
            y_val = 0.28
        elif i in (13, 14):  # elbows
            y_val = 0.40
        elif i in (15, 16):  # wrists
            y_val = 0.50
        elif i in (23, 24):  # hips
            y_val = 0.55
        elif i in (25, 26):  # knees
            y_val = 0.75
        elif i in (27, 28):  # ankles
            y_val = 0.92

        landmarks.append({
            "x": round(0.5 + (0.08 if i % 2 == 1 else -0.08), 3),
            "y": round(y_val, 3),
            "z": 0.0,
            "visibility": visibility,
        })
    return landmarks


def test_root_endpoint():
    """Verify the root index provides service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/api/health" in data["health_check"]
    assert "/docs" in data["documentation"]


def test_health_check_without_model():
    """Verify that health check succeeds when no trained model exists."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False
    assert "timestamp" in data

    # Verify root-level health alias
    alias_response = client.get("/health")
    assert alias_response.status_code == 200
    assert alias_response.json()["model_loaded"] is False


def test_predict_endpoint_valid_landmarks_model_not_ready():
    """
    Verify that predict endpoint accepts 33 valid MediaPipe landmarks,
    extracts all 47 features, and explicitly returns model_not_ready.
    """
    payload = {
        "timestamp": "2026-10-12T12:00:00Z",
        "frame_id": 101,
        "patient_id": "patient-001",
        "landmarks": make_landmarks(33),
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Requirements: No fake predictions or accuracy, explicit model_not_ready status
    assert data["model_loaded"] is False
    assert data["fall_detected"] is None
    assert data["confidence"] is None
    assert data["features_extracted"] == 47
    assert data["status"] == "model_not_ready"
    assert "Random Forest model is not yet integrated" in data["message"]
    assert "33/33" in data["message"]


def test_predict_endpoint_rejects_insufficient_landmarks():
    """Verify that sending fewer than 33 landmarks returns 422 Unprocessable Entity."""
    payload = {
        "timestamp": "2026-10-12T12:00:00Z",
        "landmarks": make_landmarks(10),  # Only 10 landmarks
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("33 landmarks" in err["msg"] for err in errors)


def test_predict_endpoint_rejects_excess_landmarks():
    """Verify that sending more than 33 landmarks returns 422 Unprocessable Entity."""
    payload = {
        "timestamp": "2026-10-12T12:00:00Z",
        "landmarks": make_landmarks(34),  # 34 landmarks
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("33 landmarks" in err["msg"] for err in errors)


def test_predict_endpoint_rejects_invalid_visibility():
    """Verify that landmark visibility outside [0.0, 1.0] triggers 422 validation error."""
    invalid_landmarks = make_landmarks(33)
    invalid_landmarks[0]["visibility"] = 1.5  # Invalid: > 1.0
    payload = {
        "timestamp": "2026-10-12T12:00:00Z",
        "landmarks": invalid_landmarks,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_status():
    """Verify prediction pipeline status reporting."""
    response = client.get("/api/predict/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False
    assert data["status"] == "model_not_ready"
    assert data["expected_landmarks"] == 33
    assert data["features_count"] == 47


def test_feature_schema_endpoint():
    """Verify that the feature schema coordination endpoint exposes 47 features."""
    response = client.get("/api/predict/feature-schema")
    assert response.status_code == 200
    data = response.json()
    assert data["total_features"] == 47
    assert data["expected_landmarks"] == 33
    assert len(data["feature_names"]) == 47
    assert "torso_vertical_angle_deg" in data["feature_names"]
    assert "rel_nose_x" in data["feature_names"]


def test_alerts_lifecycle():
    """Test full lifecycle of alerts: list, create, get, patch, delete."""
    # 1. List initial seeded alerts
    list_res = client.get("/api/alerts")
    assert list_res.status_code == 200
    alerts = list_res.json()
    assert isinstance(alerts, list)
    assert len(alerts) >= 1

    # 2. Create new incident alert with pose telemetry snapshot
    new_alert_payload = {
        "patient_id": "patient-202",
        "location": "Bedroom",
        "severity": "CRITICAL",
        "message": "Rapid vertical torso collapse detected from camera feed",
        "sensor_snapshot": {
            "torso_vertical_angle_deg": 84.2,
            "bbox_aspect_ratio": 1.45,
            "source": "mediapipe_pose",
        },
    }
    create_res = client.post("/api/alerts", json=new_alert_payload)
    assert create_res.status_code == 201
    created_alert = create_res.json()
    alert_id = created_alert["id"]
    assert alert_id.startswith("inc-")
    assert created_alert["patient_id"] == "patient-202"
    assert created_alert["severity"] == "CRITICAL"
    assert created_alert["status"] == "ACTIVE"

    # 3. Retrieve specific alert by ID
    get_res = client.get(f"/api/alerts/{alert_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == alert_id

    # 4. Update status and caregiver notes
    update_res = client.patch(
        f"/api/alerts/{alert_id}",
        json={"status": "ACKNOWLEDGED", "notes": "Caregiver en route to room"},
    )
    assert update_res.status_code == 200
    updated_alert = update_res.json()
    assert updated_alert["status"] == "ACKNOWLEDGED"
    assert updated_alert["notes"] == "Caregiver en route to room"

    # 5. Delete the alert
    delete_res = client.delete(f"/api/alerts/{alert_id}")
    assert delete_res.status_code == 204

    # 6. Verify alert is gone
    not_found_res = client.get(f"/api/alerts/{alert_id}")
    assert not_found_res.status_code == 404


def test_cors_preflight():
    """Verify CORS preflight handling for Vite dev frontend."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/api/predict", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
