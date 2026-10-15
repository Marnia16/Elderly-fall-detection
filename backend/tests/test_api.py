"""Automated test suite for Camera-based Fall Detection API endpoints."""

import sys
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pytest

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app  # noqa: E402
from app.features import (  # noqa: E402
    FEATURE_NAMES,
    LM_LEFT_HIP,
    LM_LEFT_SHOULDER,
    LM_RIGHT_HIP,
    LM_RIGHT_SHOULDER,
    extract_feature_vector,
    extract_features,
)

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


def test_health_check_without_model(monkeypatch, tmp_path):
    """Verify that health check succeeds when no trained model exists."""
    from app.routes import predict
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path / "empty_models")
    predict._cached_model = None
    predict._cached_model_path = None
    predict._cached_model_mtime = None

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


def test_predict_endpoint_valid_landmarks_model_not_ready(monkeypatch, tmp_path):
    """
    Verify that predict endpoint accepts 33 valid MediaPipe landmarks,
    extracts all 47 features, and explicitly returns model_not_ready.
    """
    from app.routes import predict
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path / "empty_models")
    predict._cached_model = None
    predict._cached_model_path = None
    predict._cached_model_mtime = None

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


def test_predict_status(monkeypatch, tmp_path):
    """Verify prediction pipeline status reporting."""
    from app.routes import predict
    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path / "empty_models")
    predict._cached_model = None
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    response = client.get("/api/predict/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False
    assert data["status"] == "model_not_ready"
    assert data["expected_landmarks"] == 33
    assert data["features_count"] == 47


def test_predict_endpoint_with_trained_model():
    """Verify prediction pipeline with real trained RandomForest model artifact."""
    from app.routes import predict
    predict._cached_model = None
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    payload = {
        "timestamp": "2026-10-15T12:00:00Z",
        "frame_id": 202,
        "patient_id": "patient-001",
        "landmarks": make_landmarks(33),
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["features_extracted"] == 47
    assert data["status"] == "ok"
    assert isinstance(data["fall_detected"], bool)
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["fall_probability"] <= 1.0


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


def test_feature_extraction_mid_hip_z_accuracy():
    """
    Verify that mid_hip_z correctly uses z coordinates and not y coordinates.
    When hip y = 0.55 and hip z = -0.3, mid_hip_z must be -0.3.
    """
    landmarks = make_landmarks(33)
    # Set distinct y and z values on hips
    landmarks[LM_LEFT_HIP]["y"] = 0.55
    landmarks[LM_LEFT_HIP]["z"] = -0.20
    landmarks[LM_RIGHT_HIP]["y"] = 0.55
    landmarks[LM_RIGHT_HIP]["z"] = -0.40

    # Shoulders at z = 0.0, y = 0.28
    landmarks[LM_LEFT_SHOULDER]["y"] = 0.28
    landmarks[LM_LEFT_SHOULDER]["z"] = 0.0
    landmarks[LM_RIGHT_SHOULDER]["y"] = 0.28
    landmarks[LM_RIGHT_SHOULDER]["z"] = 0.0

    features = extract_features(landmarks)

    # mid_hip_z should be (-0.20 + -0.40) / 2 = -0.30
    # mid_shoulder_z = 0.0
    # torso_dz = mid_hip_z - mid_shoulder_z = -0.30
    # scale = sqrt(0^2 + (0.55 - 0.28)^2 + (-0.30)^2) = sqrt(0.27^2 + 0.09) = sqrt(0.0729 + 0.09) = sqrt(0.1629) ~= 0.4036
    # rel_mid_shoulder_z = round((0.0 - (-0.30)) / scale, 4) = round(0.30 / 0.4036, 4) ~= 0.7433
    # If mid_hip_z had erroneously used y (0.55), torso_dz would have been 0.55 and rel_mid_shoulder_z would be negative!
    assert features["rel_mid_shoulder_z"] > 0.0
    # Verify exact 47 features returned
    assert len(features) == 47


def test_feature_vector_dimensions_and_keys():
    """Verify that extract_feature_vector returns exactly 47 ordered floats matching FEATURE_NAMES."""
    landmarks = make_landmarks(33)
    features_dict = extract_features(landmarks)
    assert set(features_dict.keys()) == set(FEATURE_NAMES)

    vec = extract_feature_vector(landmarks)
    assert isinstance(vec, list)
    assert len(vec) == 47
    assert all(isinstance(v, (int, float)) for v in vec)


class DummyNoPredict:
    """Mock model artifact without a predict method."""
    pass


class DummyMismatchModel:
    """Mock model with feature dimension mismatch (6 instead of 47)."""
    n_features_in_ = 6

    def predict(self, X):
        return [1]


class DummyValidModel:
    """Mock model matching the 47-feature schema with valid predict/predict_proba."""
    n_features_in_ = 47
    classes_ = [0, 1]

    def predict(self, X):
        assert len(X[0]) == 47
        return [1]

    def predict_proba(self, X):
        return [[0.04, 0.96]]


def test_model_artifact_validation_and_inference(monkeypatch, tmp_path):
    """
    Test model loading, validation, and inference paths:
    1. Rejection of corrupted file
    2. Rejection of object without predict method
    3. Rejection of model with feature dimension mismatch (e.g. 6 instead of 47)
    4. Successful inference execution with a valid 47-feature model
    """
    from app.routes import predict
    import joblib

    mock_model_dir = tmp_path / "model"
    mock_model_dir.mkdir()
    monkeypatch.setattr(predict, "MODEL_DIR", mock_model_dir)

    payload = {
        "timestamp": "2026-10-15T12:00:00Z",
        "landmarks": make_landmarks(33),
    }

    # 1. Corrupted file test
    corrupt_file = mock_model_dir / "random_forest_model.joblib"
    corrupt_file.write_bytes(b"corrupt non-pickle binary content")
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    state = predict.get_model_status_and_instance()
    assert state["status"] == "model_error"
    assert "Failed to load model artifact" in state["error"]

    res_corrupt = client.post("/api/predict", json=payload)
    assert res_corrupt.status_code == 200
    corrupt_data = res_corrupt.json()
    assert corrupt_data["model_loaded"] is False
    assert corrupt_data["fall_detected"] is None
    assert corrupt_data["confidence"] is None
    assert corrupt_data["status"] == "model_not_ready"
    assert "Model artifact error" in corrupt_data["message"]

    # 2. Object without predict method
    nopred_file = mock_model_dir / "random_forest_model.joblib"
    joblib.dump(DummyNoPredict(), nopred_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    state = predict.get_model_status_and_instance()
    assert state["status"] == "model_error"
    assert "missing callable 'predict' method" in state["error"]

    res_nopred = client.post("/api/predict", json=payload)
    assert res_nopred.json()["model_loaded"] is False
    assert res_nopred.json()["fall_detected"] is None

    # 3. Model with feature count mismatch (e.g. 6 features)
    mismatch_file = mock_model_dir / "random_forest_model.joblib"
    joblib.dump(DummyMismatchModel(), mismatch_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    state = predict.get_model_status_and_instance()
    assert state["status"] == "model_error"
    assert "expects 6 features" in state["error"]

    res_mismatch = client.post("/api/predict", json=payload)
    mismatch_data = res_mismatch.json()
    assert mismatch_data["model_loaded"] is False
    assert mismatch_data["fall_detected"] is None
    assert mismatch_data["status"] == "model_not_ready"
    assert "Model feature mismatch" in mismatch_data["message"]

    # 4. Valid model with 47 features
    valid_file = mock_model_dir / "random_forest_model.joblib"
    joblib.dump(DummyValidModel(), valid_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    state = predict.get_model_status_and_instance()
    assert state["status"] == "ready"
    assert state["error"] is None

    res_valid = client.post("/api/predict", json=payload)
    assert res_valid.status_code == 200
    valid_data = res_valid.json()
    assert valid_data["model_loaded"] is True
    assert valid_data["fall_detected"] is True
    assert valid_data["confidence"] == 0.96
    assert valid_data["fall_probability"] == 0.96
    assert valid_data["features_extracted"] == 47
    assert valid_data["status"] == "ok"

    # Status endpoint should also report ready
    res_status = client.get("/api/predict/status")
    assert res_status.json()["model_loaded"] is True
    assert res_status.json()["status"] == "ready"

    # Health endpoint should report model_loaded: true
    res_health = client.get("/api/health")
    assert res_health.json()["model_loaded"] is True


class DummyCustomReturnModel:
    """Mock model returning a configurable prediction label."""
    n_features_in_ = 47

    def __init__(self, return_label, classes=None, probas=None):
        self.return_label = return_label
        if classes != "NO_CLASSES":
            self.classes_ = classes if classes is not None else [0, 1]
        self.probas = probas if probas is not None else [[0.1, 0.9]]

    def predict(self, X):
        assert len(X[0]) == 47
        return [self.return_label]

    def predict_proba(self, X):
        return self.probas


def test_interpret_fall_prediction_unit():
    """Unit tests for interpret_fall_prediction covering all branches and errors."""
    from app.routes.predict import interpret_fall_prediction

    # Numeric tests
    assert interpret_fall_prediction(1) is True
    assert interpret_fall_prediction(0) is False
    assert interpret_fall_prediction(1.0) is True
    assert interpret_fall_prediction(0.0) is False
    assert interpret_fall_prediction(True) is True
    assert interpret_fall_prediction(False) is False

    # NumPy scalar tests (unwrapped via .item())
    assert interpret_fall_prediction(np.int64(1)) is True
    assert interpret_fall_prediction(np.int64(0)) is False
    assert interpret_fall_prediction(np.int32(1)) is True
    assert interpret_fall_prediction(np.int32(0)) is False
    assert interpret_fall_prediction(np.bool_(True)) is True
    assert interpret_fall_prediction(np.bool_(False)) is False
    assert interpret_fall_prediction(np.float64(1.0)) is True
    assert interpret_fall_prediction(np.float64(0.0)) is False

    # String tests
    assert interpret_fall_prediction("fall") is True
    assert interpret_fall_prediction("FALL") is True
    assert interpret_fall_prediction("normal") is False
    assert interpret_fall_prediction("Normal") is False
    assert interpret_fall_prediction("adl") is False
    assert interpret_fall_prediction("ADL") is False
    assert interpret_fall_prediction("not_fall") is False

    # Rejected -1 and -1.0 as ambiguous numeric/string labels
    with pytest.raises(ValueError, match="Ambiguous numeric"):
        interpret_fall_prediction(-1)

    with pytest.raises(ValueError, match="Ambiguous numeric"):
        interpret_fall_prediction(-1.0)

    with pytest.raises(ValueError, match="Ambiguous string"):
        interpret_fall_prediction("-1")

    with pytest.raises(ValueError, match="Ambiguous string"):
        interpret_fall_prediction("-1.0")

    # Ambiguous or unexpected values must raise ValueError
    with pytest.raises(ValueError, match="Ambiguous numeric"):
        interpret_fall_prediction(99)

    with pytest.raises(ValueError, match="Ambiguous string"):
        interpret_fall_prediction("unrecognized_class")

    with pytest.raises(ValueError, match="Unsupported prediction label type"):
        interpret_fall_prediction([1, 2, 3])


def test_class_label_interpretation_numeric_and_strings(monkeypatch, tmp_path):
    """
    Verify class-label interpretation via POST /api/predict:
    - Numeric 0 -> fall_detected: False (normal/ADL)
    - Numeric 1 -> fall_detected: True (fall)
    - NumPy scalars np.int64, np.bool_ handled transparently
    - String "normal" -> fall_detected: False (NOT truthy!)
    - String "fall" -> fall_detected: True
    - String "adl" -> fall_detected: False
    - Rejection of -1 -> reports ambiguity without crashing
    - Ambiguous label -> reports ambiguity without fabricating a verdict
    """
    from app.routes import predict
    import joblib

    mock_model_dir = tmp_path / "model"
    mock_model_dir.mkdir()
    model_file = mock_model_dir / "random_forest_model.joblib"
    monkeypatch.setattr(predict, "MODEL_DIR", mock_model_dir)

    payload = {
        "timestamp": "2026-10-15T12:00:00Z",
        "landmarks": make_landmarks(33),
    }

    test_cases = [
        (0, False, "ok"),
        (1, True, "ok"),
        (0.0, False, "ok"),
        (1.0, True, "ok"),
        (np.int64(0), False, "ok"),
        (np.int64(1), True, "ok"),
        (np.bool_(True), True, "ok"),
        (np.bool_(False), False, "ok"),
        ("normal", False, "ok"),
        ("Normal", False, "ok"),
        ("fall", True, "ok"),
        ("FALL", True, "ok"),
        ("adl", False, "ok"),
        ("ADL", False, "ok"),
        ("not_fall", False, "ok"),
    ]

    for label, expected_verdict, expected_status in test_cases:
        joblib.dump(DummyCustomReturnModel(label), model_file)
        predict._cached_model_path = None
        predict._cached_model_mtime = None

        res = client.post("/api/predict", json=payload)
        assert res.status_code == 200, f"Failed on label: {label}"
        data = res.json()
        assert data["model_loaded"] is True
        assert data["fall_detected"] is expected_verdict, f"Label {label} expected {expected_verdict}, got {data['fall_detected']}"
        assert data["status"] == expected_status

    # Test rejection of -1 and -1.0 labels
    for invalid_label in (-1, -1.0, "-1", "-1.0"):
        joblib.dump(DummyCustomReturnModel(invalid_label), model_file)
        predict._cached_model_path = None
        predict._cached_model_mtime = None

        res_neg = client.post("/api/predict", json=payload)
        assert res_neg.status_code == 200
        neg_data = res_neg.json()
        assert neg_data["fall_detected"] is None
        assert neg_data["fall_probability"] is None
        assert neg_data["status"] == "model_not_ready"
        assert "Model output ambiguity" in neg_data["message"]

    # Test ambiguous / undocumented label
    joblib.dump(DummyCustomReturnModel("unrecognized_ambiguous_label"), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    res_ambiguous = client.post("/api/predict", json=payload)
    assert res_ambiguous.status_code == 200
    ambiguous_data = res_ambiguous.json()
    assert ambiguous_data["fall_detected"] is None
    assert ambiguous_data["fall_probability"] is None
    assert ambiguous_data["status"] == "model_not_ready"
    assert "Model output ambiguity" in ambiguous_data["message"]


def test_fall_probability_and_confidence_semantics(monkeypatch, tmp_path):
    """
    Verify winning-class confidence vs. fall_probability:
    1. Normal prediction: winning confidence = 0.88, fall_probability = 0.12
    2. Fall prediction: winning confidence = 0.95, fall_probability = 0.95
    3. Inverted classes [1, 0]: fall_probability derived from index of class 1, not naive index 1
    4. Missing classes_ attribute: fall_probability is None (not fabricated)
    5. Ambiguous classes_ (no fall class): fall_probability is None
    """
    from app.routes import predict
    import joblib

    mock_model_dir = tmp_path / "model"
    mock_model_dir.mkdir()
    model_file = mock_model_dir / "random_forest_model.joblib"
    monkeypatch.setattr(predict, "MODEL_DIR", mock_model_dir)

    payload = {
        "timestamp": "2026-10-15T12:00:00Z",
        "landmarks": make_landmarks(33),
    }

    # 1. Normal prediction: classes_ = [0, 1], probas = [[0.88, 0.12]]
    joblib.dump(DummyCustomReturnModel(0, classes=[0, 1], probas=[[0.88, 0.12]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None
    res_normal = client.post("/api/predict", json=payload)
    data_normal = res_normal.json()
    assert data_normal["fall_detected"] is False
    assert data_normal["confidence"] == 0.88
    assert data_normal["fall_probability"] == 0.12

    # 2. Fall prediction: classes_ = [0, 1], probas = [[0.05, 0.95]]
    joblib.dump(DummyCustomReturnModel(1, classes=[0, 1], probas=[[0.05, 0.95]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None
    res_fall = client.post("/api/predict", json=payload)
    data_fall = res_fall.json()
    assert data_fall["fall_detected"] is True
    assert data_fall["confidence"] == 0.95
    assert data_fall["fall_probability"] == 0.95

    # 3. Inverted classes: classes_ = [1, 0], probas = [[0.91, 0.09]]
    # Index 0 is Fall (1), so fall_probability must be 0.91!
    joblib.dump(DummyCustomReturnModel(1, classes=[1, 0], probas=[[0.91, 0.09]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None
    res_inv = client.post("/api/predict", json=payload)
    data_inv = res_inv.json()
    assert data_inv["fall_detected"] is True
    assert data_inv["confidence"] == 0.91
    assert data_inv["fall_probability"] == 0.91

    # 4. Missing classes_ attribute: fall_probability must be None
    joblib.dump(DummyCustomReturnModel(1, classes="NO_CLASSES", probas=[[0.1, 0.9]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None
    res_noclass = client.post("/api/predict", json=payload)
    data_noclass = res_noclass.json()
    assert data_noclass["fall_detected"] is True
    assert data_noclass["confidence"] == 0.9
    assert data_noclass["fall_probability"] is None

    # 5. Ambiguous classes_ (no fall class): fall_probability must be None
    joblib.dump(DummyCustomReturnModel(0, classes=["walking", "sitting"], probas=[[0.7, 0.3]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None
    res_amb = client.post("/api/predict", json=payload)
    data_amb = res_amb.json()
    assert data_amb["fall_probability"] is None


def test_sqlite_persistence_and_crud():
    """Verify SQLite database persistence across alerts endpoints."""
    # 1. Create a test incident
    create_payload = {
        "patient_id": "test-patient-42",
        "location": "Kitchen",
        "severity": "HIGH",
        "prediction_type": "RAPID_COLLAPSE",
        "message": "Sudden descent detected near stove.",
        "confidence": 0.93,
        "fall_probability": 0.95,
        "notes": "Initial telemetry capture",
    }
    res = client.post("/api/alerts", json=create_payload)
    assert res.status_code == 201
    created = res.json()
    alert_id = created["id"]
    assert created["patient_id"] == "test-patient-42"
    assert created["status"] == "ACTIVE"
    assert created["confidence"] == 0.93

    # 2. Retrieve the incident by ID
    res_get = client.get(f"/api/alerts/{alert_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == alert_id

    # 3. Patch the incident
    patch_payload = {"notes": "Caregiver en route", "escalated": True}
    res_patch = client.patch(f"/api/alerts/{alert_id}", json=patch_payload)
    assert res_patch.status_code == 200
    assert res_patch.json()["notes"] == "Caregiver en route"
    assert res_patch.json()["escalated"] is True

    # 4. Filter list alerts
    res_list = client.get("/api/alerts?status=ACTIVE&search=Kitchen")
    assert res_list.status_code == 200
    items = res_list.json()
    assert any(i["id"] == alert_id for i in items)

    # 5. Delete incident
    res_del = client.delete(f"/api/alerts/{alert_id}")
    assert res_del.status_code == 204
    assert client.get(f"/api/alerts/{alert_id}").status_code == 404


def test_acknowledge_and_resolve_endpoints():
    """Verify acknowledge and resolve transitions with timestamps and notes."""
    # Create an alert
    res = client.post("/api/alerts", json={
        "patient_id": "ack-patient",
        "location": "Bathroom",
        "severity": "CRITICAL",
        "message": "Slip detected",
    })
    assert res.status_code == 201
    alert_id = res.json()["id"]

    # Acknowledge
    ack_res = client.post(f"/api/alerts/{alert_id}/acknowledge", json={"notes": "Nurse checked room"})
    assert ack_res.status_code == 200
    ack_data = ack_res.json()
    assert ack_data["status"] == "ACKNOWLEDGED"
    assert ack_data["acknowledged_at"] is not None
    assert ack_data["notes"] == "Nurse checked room"

    # Resolve
    res_resolve = client.post(f"/api/alerts/{alert_id}/resolve", json={"notes": "Patient safe, vitals normal"})
    assert res_resolve.status_code == 200
    res_data = res_resolve.json()
    assert res_data["status"] == "RESOLVED"
    assert res_data["resolved_at"] is not None
    assert "Patient safe" in res_data["notes"]

    # Clean up
    client.delete(f"/api/alerts/{alert_id}")


def test_analytics_summary_endpoint():
    """Verify analytics summary endpoint aggregates counts and returns 7-day trend."""
    res = client.get("/api/alerts/analytics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_incidents" in data
    assert "active_incidents" in data
    assert "acknowledged_incidents" in data
    assert "resolved_incidents" in data
    assert "daily_trend" in data
    assert len(data["daily_trend"]) == 7
    for point in data["daily_trend"]:
        assert "date" in point
        assert "label" in point
        assert "count" in point


def test_alerts_csv_export():
    """Verify CSV export returns valid CSV content with expected headers."""
    res = client.get("/api/alerts/export/csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "Content-Disposition" in res.headers
    text = res.text
    assert "Incident ID,Patient ID,Location,Severity" in text


def test_enhanced_health_endpoint():
    """Verify health endpoint includes database connectivity and notification status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.2.0"
    assert "database" in data
    assert data["database"]["connected"] is True
    assert "notifications" in data
    assert data["notifications"]["mode"] in ("demo", "smtp")


def test_temporal_fall_confirmation_and_tracking(monkeypatch, tmp_path):
    """
    Verify fall tracker state progression through /api/predict:
    Frame 1 positive -> CONFIRMING state
    Frame 2 positive within 2.5s -> FALL_ACTIVE, fall_confirmed=True, creates active incident in DB.
    """
    from app.routes import predict
    import joblib

    mock_model_dir = tmp_path / "model"
    mock_model_dir.mkdir()
    model_file = mock_model_dir / "random_forest_model.joblib"
    monkeypatch.setattr(predict, "MODEL_DIR", mock_model_dir)

    # Reset tracker before testing
    client.post("/api/predict/tracker/reset")

    # Set up dummy fall model (1 = Fall, confidence = 0.95)
    joblib.dump(DummyCustomReturnModel(1, classes=[0, 1], probas=[[0.05, 0.95]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    payload = {
        "timestamp": "2026-10-15T12:00:00Z",
        "landmarks": make_landmarks(33),
        "patient_id": "temp-test-patient",
    }

    # Frame 1: first detection -> tracker enters CONFIRMING
    res1 = client.post("/api/predict", json=payload)
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["fall_detected"] is True
    assert d1["tracker_state"] == "CONFIRMING"
    assert d1["fall_confirmed"] is False

    # Frame 2: second positive detection in close proximity -> FALL_ACTIVE
    res2 = client.post("/api/predict", json=payload)
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["fall_detected"] is True
    assert d2["fall_confirmed"] is True
    assert d2["tracker_state"] == "FALL_ACTIVE"
    assert d2["active_incident_id"] is not None

    # Clean up created incident and reset tracker
    incident_id = d2["active_incident_id"]
    client.delete(f"/api/alerts/{incident_id}")
    client.post("/api/predict/tracker/reset")


def test_is_demo_origin_tracking_and_csv():
    """Verify demo seed incidents vs live incidents are explicitly distinguished in API and CSV."""
    # 1. Check seeded demo incidents
    res = client.get("/api/alerts")
    assert res.status_code == 200
    alerts = res.json()
    demo_alerts = [a for a in alerts if a["id"] in ("inc-001", "inc-002", "inc-003")]
    for a in demo_alerts:
        assert a.get("is_demo") is True

    # 2. Create a live telemetry incident
    res_live = client.post("/api/alerts", json={
        "patient_id": "live-pat-1",
        "location": "Corridor",
        "severity": "HIGH",
        "message": "Real-time fall detected by pose telemetry",
    })
    assert res_live.status_code == 201
    live_alert = res_live.json()
    assert live_alert["is_demo"] is False

    # 3. Check CSV export includes Origin column
    csv_res = client.get("/api/alerts/export/csv")
    assert csv_res.status_code == 200
    text = csv_res.text
    assert "Origin (Live/Demo)" in text
    assert "Demo Seed" in text
    assert "Live Telemetry" in text

    # Cleanup live alert
    client.delete(f"/api/alerts/{live_alert['id']}")


def test_notification_duplicate_dispatch_suppression():
    """Verify NotificationService guards against duplicate alert and escalation dispatches."""
    import uuid
    from app.notifications import notification_service

    unique_id = f"test-dup-{uuid.uuid4().hex[:8]}"
    test_incident = {
        "id": unique_id,
        "patient_id": "patient-dup",
        "location": "Living Room",
        "severity": "HIGH",
        "confidence": 0.95,
        "message": "Test duplicate suppression",
    }

    # First dispatch -> logs notification
    res1 = notification_service.send_fall_alert(test_incident)
    assert res1["success"] is True
    assert res1["status"] in ("DEMO_LOGGED", "DELIVERED")

    # Second dispatch for same incident -> suppressed duplicate
    res2 = notification_service.send_fall_alert(test_incident)
    assert res2["success"] is True
    assert res2.get("already_dispatched") is True
    assert res2["status"] == "ALREADY_DISPATCHED"

    # Escalation duplicate protection
    esc1 = notification_service.send_escalation_alert(test_incident)
    assert esc1["success"] is True
    assert esc1["status"] == "ESCALATED"

    esc2 = notification_service.send_escalation_alert(test_incident)
    assert esc2["success"] is True
    assert esc2.get("already_dispatched") is True
    assert esc2["status"] == "ALREADY_ESCALATED"


def test_tracker_cooldown_rearm_and_separate_falls(monkeypatch, tmp_path):
    """
    Verify complete lifecycle across separate falls:
    Fall 1 -> Recovery -> Cooldown -> Re-arm -> Fall 2 (creates separate incident).
    """
    from app.routes import predict
    from app.fall_tracker import fall_tracker
    import joblib

    mock_model_dir = tmp_path / "model"
    mock_model_dir.mkdir()
    model_file = mock_model_dir / "random_forest_model.joblib"
    monkeypatch.setattr(predict, "MODEL_DIR", mock_model_dir)

    client.post("/api/predict/tracker/reset")

    # Set up dummy model returning 1 (Fall)
    joblib.dump(DummyCustomReturnModel(1, classes=[0, 1], probas=[[0.05, 0.95]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    payload = {
        "timestamp": "2026-10-15T12:00:00Z",
        "landmarks": make_landmarks(33),
        "patient_id": "multi-fall-test",
    }

    # 1. Fall 1: Confirming -> Active
    client.post("/api/predict", json=payload)  # Frame 1: Confirming
    res_fall1 = client.post("/api/predict", json=payload)  # Frame 2: Confirmed Fall 1
    d_fall1 = res_fall1.json()
    assert d_fall1["fall_confirmed"] is True
    incident_id_1 = d_fall1["active_incident_id"]
    assert incident_id_1 is not None

    # 2. Resident recovers: Switch dummy model to normal posture (0 = Normal)
    joblib.dump(DummyCustomReturnModel(0, classes=[0, 1], probas=[[0.95, 0.05]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    # Send 4 normal frames to transition from FALL_ACTIVE to COOLDOWN
    for _ in range(4):
        client.post("/api/predict", json=payload)

    # Cooldown should now be active
    state_after_rec = fall_tracker._get_patient_state("multi-fall-test")
    assert state_after_rec["tracker_state"] == "COOLDOWN"

    # 3. Reset or re-arm tracker
    client.post("/api/predict/tracker/reset?patient_id=multi-fall-test")
    assert fall_tracker._get_patient_state("multi-fall-test")["tracker_state"] == "MONITORING"

    # 4. Fall 2: Switch dummy model back to Fall
    joblib.dump(DummyCustomReturnModel(1, classes=[0, 1], probas=[[0.05, 0.95]]), model_file)
    predict._cached_model_path = None
    predict._cached_model_mtime = None

    client.post("/api/predict", json=payload)  # Frame 1
    res_fall2 = client.post("/api/predict", json=payload)  # Frame 2
    d_fall2 = res_fall2.json()
    assert d_fall2["fall_confirmed"] is True
    incident_id_2 = d_fall2["active_incident_id"]

    # Verify that Fall 1 and Fall 2 are two distinct incidents in SQLite!
    assert incident_id_1 != incident_id_2
    assert client.get(f"/api/alerts/{incident_id_1}").status_code == 200
    assert client.get(f"/api/alerts/{incident_id_2}").status_code == 200

    # Clean up
    client.delete(f"/api/alerts/{incident_id_1}")
    client.delete(f"/api/alerts/{incident_id_2}")
    client.post("/api/predict/tracker/reset?patient_id=multi-fall-test")


def test_missing_or_invalid_landmarks_handling():
    """Verify predictive endpoint gracefully rejects invalid or non-33 landmark inputs."""
    # 1. Missing landmarks array
    res_empty = client.post("/api/predict", json={"landmarks": []})
    assert res_empty.status_code == 422

    # 2. Fewer than 33 landmarks
    res_short = client.post("/api/predict", json={"landmarks": make_landmarks(15)})
    assert res_short.status_code == 422
    assert "33" in res_short.text

    # 3. Greater than 33 landmarks
    res_long = client.post("/api/predict", json={"landmarks": make_landmarks(34)})
    assert res_long.status_code == 422
