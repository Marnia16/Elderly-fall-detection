"""Fall detection prediction endpoint for camera-based MediaPipe Pose landmarks."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter
from app.features import FEATURE_NAMES, extract_feature_vector, extract_features
from app.schemas import PosePredictionRequest, PredictionResponse

router = APIRouter(tags=["Prediction"])

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_FILES = ["random_forest_model.joblib", "random_forest_model.pkl", "model.joblib", "model.pkl"]

# In-memory cached model reference and metadata
_cached_model: Optional[Any] = None
_cached_model_path: Optional[Path] = None
_cached_model_mtime: Optional[float] = None
_cached_model_error: Optional[str] = None


def get_model_status_and_instance() -> Dict[str, Any]:
    """
    Locates, loads, and validates the trained Random Forest model artifact.

    Validations performed:
    1. Checks if a recognized model artifact exists in MODEL_DIR.
    2. Safely loads via joblib without crashing on corrupt files.
    3. Validates that the loaded object has a callable 'predict' method.
    4. Validates that the model's expected input feature count matches the
       standardized 47-feature schema (via model.n_features_in_).

    Returns a dict with:
        status: 'ready', 'model_not_ready', or 'model_error'
        model: loaded model instance (if ready) or None
        error: error detail string (if model_error) or None
        path: Path to detected artifact or None
    """
    global _cached_model, _cached_model_path, _cached_model_mtime, _cached_model_error

    if not MODEL_DIR.exists():
        _cached_model = None
        _cached_model_path = None
        _cached_model_mtime = None
        _cached_model_error = None
        return {"status": "model_not_ready", "model": None, "error": None, "path": None}

    # Locate first matching model artifact
    detected_path: Optional[Path] = None
    for model_filename in MODEL_FILES:
        candidate = MODEL_DIR / model_filename
        if candidate.exists() and candidate.is_file():
            detected_path = candidate
            break

    if detected_path is None:
        _cached_model = None
        _cached_model_path = None
        _cached_model_mtime = None
        _cached_model_error = None
        return {"status": "model_not_ready", "model": None, "error": None, "path": None}

    # Check cache validity using file modification time
    current_mtime = detected_path.stat().st_mtime
    if _cached_model_path == detected_path and _cached_model_mtime == current_mtime:
        if _cached_model_error is not None:
            return {
                "status": "model_error",
                "model": None,
                "error": _cached_model_error,
                "path": str(detected_path),
            }
        return {
            "status": "ready",
            "model": _cached_model,
            "error": None,
            "path": str(detected_path),
        }

    # Load and validate new or updated model artifact
    _cached_model_path = detected_path
    _cached_model_mtime = current_mtime
    _cached_model = None
    _cached_model_error = None

    try:
        import joblib
        loaded = joblib.load(detected_path)
    except Exception as exc:
        _cached_model_error = f"Failed to load model artifact ({type(exc).__name__}): {exc}"
        return {
            "status": "model_error",
            "model": None,
            "error": _cached_model_error,
            "path": str(detected_path),
        }

    # Validate predict interface
    if not hasattr(loaded, "predict") or not callable(getattr(loaded, "predict")):
        _cached_model_error = "Loaded model artifact is invalid: missing callable 'predict' method."
        return {
            "status": "model_error",
            "model": None,
            "error": _cached_model_error,
            "path": str(detected_path),
        }

    # Validate feature count against the 47-feature pipeline contract
    n_features = getattr(loaded, "n_features_in_", None)
    if n_features is not None and n_features != len(FEATURE_NAMES):
        _cached_model_error = (
            f"Model feature mismatch: loaded model expects {n_features} features, "
            f"but the pose feature extraction pipeline produces {len(FEATURE_NAMES)} features."
        )
        return {
            "status": "model_error",
            "model": None,
            "error": _cached_model_error,
            "path": str(detected_path),
        }

    _cached_model = loaded
    return {
        "status": "ready",
        "model": _cached_model,
        "error": None,
        "path": str(detected_path),
    }


def get_trained_model() -> Optional[Any]:
    """Returns the validated model instance if ready, otherwise None."""
    state = get_model_status_and_instance()
    return state["model"] if state["status"] == "ready" else None


def is_model_ready() -> bool:
    """Returns True if the model artifact is present, valid, and ready for inference."""
    return get_model_status_and_instance()["status"] == "ready"


# Documented label conventions matching FEATURE_SCHEMA.md & INTEGRATION_GUIDE.md:
# Primary convention: 1 = Fall, 0 = ADL / Normal Activity.
FALL_LABELS = {"1", "1.0", "fall", "falling", "fallen", "true", "yes"}
NORMAL_LABELS = {
    "0",
    "0.0",
    "normal",
    "adl",
    "not_fall",
    "not-fall",
    "no_fall",
    "no-fall",
    "non_fall",
    "non-fall",
    "false",
    "no",
}


def interpret_fall_prediction(raw_pred: Any) -> bool:
    """
    Explicitly interprets model prediction output into a boolean fall detection verdict.

    Handles numeric (1/0, True/False) and string ('fall', 'normal', 'adl') representations.
    Supports NumPy scalar types (np.int64, np.float64, np.bool_) via .item() without hard dependencies.
    Prevents truthy string bugs (e.g. bool('normal') == True).
    Raises ValueError on ambiguous or undocumented labels (-1, unrecognized strings, etc.)
    to prevent silent misclassification.
    """
    # 0. Unwrap NumPy scalar types (e.g. np.int64, np.float64, np.bool_) if supported
    if hasattr(raw_pred, "item") and callable(getattr(raw_pred, "item")):
        try:
            raw_pred = raw_pred.item()
        except (ValueError, TypeError):
            pass

    # 1. Pure boolean
    if isinstance(raw_pred, bool):
        return raw_pred

    # 2. Numeric types (int, float)
    if isinstance(raw_pred, (int, float)):
        int_val = int(raw_pred)
        if int_val == 1:
            return True
        if int_val == 0:
            return False
        if raw_pred == 1.0:
            return True
        if raw_pred == 0.0:
            return False
        raise ValueError(
            f"Ambiguous numeric prediction label '{raw_pred}'. "
            "Expected 1 (fall) or 0 (normal/ADL) according to project training convention."
        )

    # 3. String representations
    if isinstance(raw_pred, str):
        normalized = raw_pred.strip().lower()
        if normalized in FALL_LABELS:
            return True
        if normalized in NORMAL_LABELS:
            return False
        raise ValueError(
            f"Ambiguous string prediction label '{raw_pred}'. "
            "Expected documented label such as 'fall', 'normal', or 'adl'."
        )

    # 4. Unknown/Unsupported type
    raise ValueError(
        f"Unsupported prediction label type '{type(raw_pred).__name__}' with value '{raw_pred}'. "
        "Expected int, float, bool, or str."
    )


def extract_fall_probability(model: Any, probas: Any) -> Optional[float]:
    """
    Derives the probability of the fall class from predict_proba output.

    Uses model.classes_ to identify which probability index corresponds to
    the documented fall class (e.g. 1, 1.0, 'fall', True).
    Never assumes index 1 is fall without inspecting model.classes_.
    Returns None if:
    - model lacks classes_ attribute
    - classes_ count does not match probas length
    - fall class cannot be identified unambiguously (0 matches or >1 matches)
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None

    try:
        classes_list = list(classes)
        probas_list = list(probas)
    except Exception:
        return None

    if len(classes_list) != len(probas_list):
        return None

    fall_indices = []
    for idx, cls in enumerate(classes_list):
        try:
            if interpret_fall_prediction(cls) is True:
                fall_indices.append(idx)
        except Exception:
            continue

    if len(fall_indices) == 1:
        try:
            return round(float(probas_list[fall_indices[0]]), 4)
        except Exception:
            return None

    return None


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Fall from 33 MediaPipe Pose Landmarks",
)
async def predict_fall(request: PosePredictionRequest) -> PredictionResponse:
    """
    Evaluates 33 camera-extracted MediaPipe Pose landmarks to detect falls.

    1. Landmarks undergo feature extraction (47 biomechanical and spatial features).
    2. Validates loaded model presence and feature compatibility.
    3. When model artifact is missing or invalid, explicitly returns status='model_not_ready'
       with fall_detected=None, confidence=None, and fall_probability=None.
    4. When a verified model is loaded, executes real inference using the 47-feature vector,
       coercing numeric/string prediction labels safely without truthy string bugs.
    """
    features: Dict[str, float] = extract_features(request.landmarks)
    model_state = get_model_status_and_instance()
    state_status = model_state["status"]
    model = model_state["model"]
    error_msg = model_state["error"]

    if state_status == "model_not_ready":
        return PredictionResponse(
            model_loaded=False,
            fall_detected=None,
            confidence=None,
            fall_probability=None,
            features_extracted=len(features),
            message=(
                "Random Forest model is not yet integrated. MediaPipe landmarks validated (33/33) "
                f"and {len(features)} features successfully extracted. Model training on Colab "
                "(e.g., URFD dataset) is currently in progress."
            ),
            status="model_not_ready",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    if state_status == "model_error":
        return PredictionResponse(
            model_loaded=False,
            fall_detected=None,
            confidence=None,
            fall_probability=None,
            features_extracted=len(features),
            message=f"Model artifact error: {error_msg}. Inference halted safely.",
            status="model_not_ready",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Valid model is loaded: execute inference with the 47-feature vector
    feature_vector = extract_feature_vector(request.landmarks)
    try:
        raw_pred = model.predict([feature_vector])[0]
        fall_detected = interpret_fall_prediction(raw_pred)

        confidence: Optional[float] = None
        fall_probability: Optional[float] = None
        if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
            try:
                probas = model.predict_proba([feature_vector])[0]
                confidence = round(float(max(probas)), 4)
                fall_probability = extract_fall_probability(model, probas)
            except Exception:
                confidence = None
                fall_probability = None

        return PredictionResponse(
            model_loaded=True,
            fall_detected=fall_detected,
            confidence=confidence,
            fall_probability=fall_probability,
            features_extracted=len(features),
            message="Fall detection inference completed successfully.",
            status="ok",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except ValueError as val_err:
        return PredictionResponse(
            model_loaded=True,
            fall_detected=None,
            confidence=None,
            fall_probability=None,
            features_extracted=len(features),
            message=f"Model output ambiguity: {val_err}. Please align model class labels with training convention.",
            status="model_not_ready",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        return PredictionResponse(
            model_loaded=False,
            fall_detected=None,
            confidence=None,
            fall_probability=None,
            features_extracted=len(features),
            message=f"Inference execution failed: {exc}",
            status="model_not_ready",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


@router.get(
    "/predict/status",
    summary="Check Model Integration Readiness",
)
async def get_prediction_status() -> dict:
    """Returns the integration status of the machine learning pipeline."""
    model_state = get_model_status_and_instance()
    is_ready = model_state["status"] == "ready"
    return {
        "model_loaded": is_ready,
        "model_dir": str(MODEL_DIR),
        "expected_artifacts": MODEL_FILES,
        "expected_landmarks": 33,
        "features_count": len(FEATURE_NAMES),
        "status": "ready" if is_ready else "model_not_ready",
        "detail": model_state["error"],
        "message": (
            "Model is active and ready for inference."
            if is_ready
            else (
                f"Model error: {model_state['error']}"
                if model_state["error"]
                else "Awaiting trained Random Forest model artifact from teammate."
            )
        ),
    }


@router.get(
    "/predict/feature-schema",
    summary="Feature Schema for Random Forest Training Pipeline",
)
async def get_feature_schema() -> dict:
    """
    Returns the exact 47 feature names and ordering expected by the model.
    Used by teammate in Google Colab to ensure 100% consistency between training and inference.
    """
    return {
        "total_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "expected_landmarks": 33,
        "description": (
            "47 features extracted from 33 MediaPipe Pose landmarks: "
            "11 biomechanical/geometric features (torso angle, bounding box aspect ratio, "
            "normalized vertical distances) + 36 normalized relative spatial coordinates "
            "(12 key body joints relative to hip center and scaled by torso length)."
        ),
    }
