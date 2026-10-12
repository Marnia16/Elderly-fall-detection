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

# In-memory reference for the loaded model once integrated
_trained_model: Optional[Any] = None


def get_trained_model() -> Optional[Any]:
    """
    Attempt to locate and load the trained Random Forest model artifact.
    Returns None if the teammate's model is not yet present.
    Strictly avoids generating fake models or dummy predictions.
    """
    global _trained_model
    if _trained_model is not None:
        return _trained_model

    if not MODEL_DIR.exists():
        return None

    for model_filename in MODEL_FILES:
        model_path = MODEL_DIR / model_filename
        if model_path.exists():
            try:
                import joblib  # lazy import
                _trained_model = joblib.load(model_path)
                return _trained_model
            except Exception:
                # If loading fails or joblib is unavailable, leave as None
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

    Landmarks undergo feature extraction (47 biomechanical and spatial features).
    When the teammate's trained Random Forest model artifact is not yet present,
    this endpoint explicitly returns status='model_not_ready', model_loaded=False,
    fall_detected=None, and confidence=None.
    """
    # Extract features using the standardized training-inference feature pipeline
    features: Dict[str, float] = extract_features(request.landmarks)

    model = get_trained_model()

    if model is None:
        return PredictionResponse(
            model_loaded=False,
            fall_detected=None,
            confidence=None,
            features_extracted=len(features),
            message=(
                "Random Forest model is not yet integrated. MediaPipe landmarks validated (33/33) "
                f"and {len(features)} features successfully extracted. Model training on Colab "
                "(e.g., URFD dataset) is currently in progress."
            ),
            status="model_not_ready",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Future integration path once teammate provides real trained model artifact:
    # feature_vector = extract_feature_vector(request.landmarks)
    # prediction = bool(model.predict([feature_vector])[0])
    # proba = float(model.predict_proba([feature_vector])[0][1]) if hasattr(model, "predict_proba") else 1.0

    return PredictionResponse(
        model_loaded=True,
        fall_detected=None,
        confidence=None,
        features_extracted=len(features),
        message="Model artifact detected, awaiting final feature alignment with teammate training pipeline.",
        status="model_ready",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/predict/status",
    summary="Check Model Integration Readiness",
)
async def get_prediction_status() -> dict:
    """Returns the integration status of the machine learning pipeline."""
    model = get_trained_model()
    return {
        "model_loaded": model is not None,
        "model_dir": str(MODEL_DIR),
        "expected_artifacts": MODEL_FILES,
        "expected_landmarks": 33,
        "features_count": len(FEATURE_NAMES),
        "status": "ready" if model is not None else "model_not_ready",
        "message": (
            "Model is active and ready for inference."
            if model is not None
            else "Awaiting trained Random Forest model artifact from teammate."
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
