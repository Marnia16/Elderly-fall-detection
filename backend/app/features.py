"""Pose feature extraction module for MediaPipe landmarks.

Maintains strict consistency between training-time (Google Colab) and
inference-time (FastAPI backend) feature engineering for the Random Forest model.
Uses pure Python standard library for maximum portability across Windows, macOS,
Linux, and Google Colab environments.
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

# MediaPipe Pose Landmark Indices (0 to 32)
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM_NOSE = 0
LM_LEFT_EYE_INNER = 1
LM_LEFT_EYE = 2
LM_LEFT_EYE_OUTER = 3
LM_RIGHT_EYE_INNER = 4
LM_RIGHT_EYE = 5
LM_RIGHT_EYE_OUTER = 6
LM_LEFT_EAR = 7
LM_RIGHT_EAR = 8
LM_MOUTH_LEFT = 9
LM_MOUTH_RIGHT = 10
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_PINKY = 17
LM_RIGHT_PINKY = 18
LM_LEFT_INDEX = 19
LM_RIGHT_INDEX = 20
LM_LEFT_THUMB = 21
LM_RIGHT_THUMB = 22
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28
LM_LEFT_HEEL = 29
LM_RIGHT_HEEL = 30
LM_LEFT_FOOT_INDEX = 31
LM_RIGHT_FOOT_INDEX = 32

# Key landmark subsets used for normalized relative coordinates
KEY_LANDMARK_MAP: Dict[str, int] = {
    "nose": LM_NOSE,
    "left_shoulder": LM_LEFT_SHOULDER,
    "right_shoulder": LM_RIGHT_SHOULDER,
    "left_elbow": LM_LEFT_ELBOW,
    "right_elbow": LM_RIGHT_ELBOW,
    "left_wrist": LM_LEFT_WRIST,
    "right_wrist": LM_RIGHT_WRIST,
    "left_hip": LM_LEFT_HIP,
    "right_hip": LM_RIGHT_HIP,
    "left_knee": LM_LEFT_KNEE,
    "right_knee": LM_RIGHT_KNEE,
    "left_ankle": LM_LEFT_ANKLE,
    "right_ankle": LM_RIGHT_ANKLE,
}

# Explicit, ordered list of all 47 features expected by the Random Forest model
FEATURE_NAMES: Tuple[str, ...] = (
    # Biomechanical & Geometric features (11 features)
    "torso_vertical_angle_deg",
    "torso_length",
    "bbox_aspect_ratio",
    "bbox_height_to_torso_ratio",
    "hip_to_ankle_vertical_dist",
    "head_to_hip_vertical_dist",
    "shoulder_tilt_deg",
    "hip_tilt_deg",
    "knee_to_ankle_vertical_dist",
    "head_to_ankle_vertical_dist",
    "mean_landmark_visibility",
    # Normalized relative spatial coordinates (36 features: 12 key points * x, y, z)
    # Positions are relative to hip center and scaled by torso length
    "rel_nose_x",
    "rel_nose_y",
    "rel_nose_z",
    "rel_left_shoulder_x",
    "rel_left_shoulder_y",
    "rel_left_shoulder_z",
    "rel_right_shoulder_x",
    "rel_right_shoulder_y",
    "rel_right_shoulder_z",
    "rel_mid_shoulder_x",
    "rel_mid_shoulder_y",
    "rel_mid_shoulder_z",
    "rel_left_elbow_x",
    "rel_left_elbow_y",
    "rel_left_elbow_z",
    "rel_right_elbow_x",
    "rel_right_elbow_y",
    "rel_right_elbow_z",
    "rel_left_wrist_x",
    "rel_left_wrist_y",
    "rel_left_wrist_z",
    "rel_right_wrist_x",
    "rel_right_wrist_y",
    "rel_right_wrist_z",
    "rel_left_knee_x",
    "rel_left_knee_y",
    "rel_left_knee_z",
    "rel_right_knee_x",
    "rel_right_knee_y",
    "rel_right_knee_z",
    "rel_left_ankle_x",
    "rel_left_ankle_y",
    "rel_left_ankle_z",
    "rel_right_ankle_x",
    "rel_right_ankle_y",
    "rel_right_ankle_z",
)


def _get_coords(landmark: Any) -> Tuple[float, float, float, float]:
    """Helper to extract (x, y, z, visibility) from either dict or object."""
    if isinstance(landmark, dict):
        return (
            float(landmark.get("x", 0.0)),
            float(landmark.get("y", 0.0)),
            float(landmark.get("z", 0.0)),
            float(landmark.get("visibility", 1.0)),
        )
    return (
        float(getattr(landmark, "x", 0.0)),
        float(getattr(landmark, "y", 0.0)),
        float(getattr(landmark, "z", 0.0)),
        float(getattr(landmark, "visibility", 1.0)),
    )


def extract_features(landmarks: Sequence[Any]) -> Dict[str, float]:
    """
    Extract a dictionary of 47 biomechanical and normalized spatial features
    from 33 MediaPipe Pose landmarks.

    Parameters:
        landmarks: Ordered list of 33 landmarks containing x, y, z, visibility.

    Returns:
        Dict mapping each feature name in FEATURE_NAMES to its computed float value.
    """
    if len(landmarks) != 33:
        raise ValueError(f"Expected exactly 33 landmarks, received {len(landmarks)}.")

    # Parse all 33 landmarks
    parsed = [_get_coords(lm) for lm in landmarks]
    xs = [p[0] for p in parsed]
    ys = [p[1] for p in parsed]
    zs = [p[2] for p in parsed]
    visibilities = [p[3] for p in parsed]

    # Key anatomical points
    ls_x, ls_y, ls_z, _ = parsed[LM_LEFT_SHOULDER]
    rs_x, rs_y, rs_z, _ = parsed[LM_RIGHT_SHOULDER]
    lh_x, lh_y, lh_z, _ = parsed[LM_LEFT_HIP]
    rh_x, rh_y, rh_z, _ = parsed[LM_RIGHT_HIP]
    lk_x, lk_y, lk_z, _ = parsed[LM_LEFT_KNEE]
    rk_x, rk_y, rk_z, _ = parsed[LM_RIGHT_KNEE]
    la_x, la_y, la_z, _ = parsed[LM_LEFT_ANKLE]
    ra_x, ra_y, ra_z, _ = parsed[LM_RIGHT_ANKLE]
    nose_x, nose_y, nose_z, _ = parsed[LM_NOSE]

    # Midpoints
    mid_shoulder_x = (ls_x + rs_x) / 2.0
    mid_shoulder_y = (ls_y + rs_y) / 2.0
    mid_shoulder_z = (ls_z + rs_z) / 2.0

    mid_hip_x = (lh_x + rh_x) / 2.0
    mid_hip_y = (lh_y + rh_y) / 2.0
    mid_hip_z = (lh_z + rh_z) / 2.0

    mid_knee_y = (lk_y + rk_y) / 2.0
    mid_ankle_y = (la_y + ra_y) / 2.0

    # 1. Torso vector from mid-shoulder to mid-hip
    torso_dx = mid_hip_x - mid_shoulder_x
    torso_dy = mid_hip_y - mid_shoulder_y
    torso_dz = mid_hip_z - mid_shoulder_z

    torso_length = math.sqrt(torso_dx**2 + torso_dy**2 + torso_dz**2)
    # Safe scale factor to avoid zero division
    scale = torso_length if torso_length > 1e-4 else 1e-4

    # 2. Torso vertical angle (in degrees relative to vertical Y axis)
    # In standing pose, mid-shoulder is above mid-hip (torso_dy > 0 in screen coords, angle ~0 deg)
    # In fallen/horizontal pose, angle approaches 70-90 deg
    horizontal_dist = math.sqrt(torso_dx**2 + torso_dz**2)
    torso_vertical_angle_deg = math.degrees(math.atan2(horizontal_dist, abs(torso_dy) + 1e-6))

    # 3. Bounding box features
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    bbox_aspect_ratio = bbox_w / (bbox_h + 1e-6)
    bbox_height_to_torso_ratio = bbox_h / scale

    # 4. Normalized vertical distances (scaled by torso length)
    hip_to_ankle_vertical_dist = (mid_ankle_y - mid_hip_y) / scale
    head_to_hip_vertical_dist = (mid_hip_y - nose_y) / scale
    knee_to_ankle_vertical_dist = (mid_ankle_y - mid_knee_y) / scale
    head_to_ankle_vertical_dist = (mid_ankle_y - nose_y) / scale

    # 5. Segment tilts relative to horizontal (degrees)
    shoulder_tilt_deg = math.degrees(math.atan2(abs(ls_y - rs_y), abs(ls_x - rs_x) + 1e-6))
    hip_tilt_deg = math.degrees(math.atan2(abs(lh_y - rh_y), abs(lh_x - rh_x) + 1e-6))

    # 6. Mean visibility
    mean_visibility = sum(visibilities) / len(visibilities)

    features: Dict[str, float] = {
        "torso_vertical_angle_deg": round(torso_vertical_angle_deg, 4),
        "torso_length": round(torso_length, 4),
        "bbox_aspect_ratio": round(bbox_aspect_ratio, 4),
        "bbox_height_to_torso_ratio": round(bbox_height_to_torso_ratio, 4),
        "hip_to_ankle_vertical_dist": round(hip_to_ankle_vertical_dist, 4),
        "head_to_hip_vertical_dist": round(head_to_hip_vertical_dist, 4),
        "shoulder_tilt_deg": round(shoulder_tilt_deg, 4),
        "hip_tilt_deg": round(hip_tilt_deg, 4),
        "knee_to_ankle_vertical_dist": round(knee_to_ankle_vertical_dist, 4),
        "head_to_ankle_vertical_dist": round(head_to_ankle_vertical_dist, 4),
        "mean_landmark_visibility": round(mean_visibility, 4),
    }

    # 7. Normalized relative coordinates for key body landmarks
    # Origin centered at mid-hip, distances scaled by torso length
    def _rel(x: float, y: float, z: float) -> Tuple[float, float, float]:
        return (
            round((x - mid_hip_x) / scale, 4),
            round((y - mid_hip_y) / scale, 4),
            round((z - mid_hip_z) / scale, 4),
        )

    # Key points relative coordinates
    rx, ry, rz = _rel(nose_x, nose_y, nose_z)
    features["rel_nose_x"], features["rel_nose_y"], features["rel_nose_z"] = rx, ry, rz

    rx, ry, rz = _rel(ls_x, ls_y, ls_z)
    features["rel_left_shoulder_x"], features["rel_left_shoulder_y"], features["rel_left_shoulder_z"] = rx, ry, rz

    rx, ry, rz = _rel(rs_x, rs_y, rs_z)
    features["rel_right_shoulder_x"], features["rel_right_shoulder_y"], features["rel_right_shoulder_z"] = rx, ry, rz

    rx, ry, rz = _rel(mid_shoulder_x, mid_shoulder_y, mid_shoulder_z)
    features["rel_mid_shoulder_x"], features["rel_mid_shoulder_y"], features["rel_mid_shoulder_z"] = rx, ry, rz

    le_x, le_y, le_z, _ = parsed[LM_LEFT_ELBOW]
    rx, ry, rz = _rel(le_x, le_y, le_z)
    features["rel_left_elbow_x"], features["rel_left_elbow_y"], features["rel_left_elbow_z"] = rx, ry, rz

    re_x, re_y, re_z, _ = parsed[LM_RIGHT_ELBOW]
    rx, ry, rz = _rel(re_x, re_y, re_z)
    features["rel_right_elbow_x"], features["rel_right_elbow_y"], features["rel_right_elbow_z"] = rx, ry, rz

    lw_x, lw_y, lw_z, _ = parsed[LM_LEFT_WRIST]
    rx, ry, rz = _rel(lw_x, lw_y, lw_z)
    features["rel_left_wrist_x"], features["rel_left_wrist_y"], features["rel_left_wrist_z"] = rx, ry, rz

    rw_x, rw_y, rw_z, _ = parsed[LM_RIGHT_WRIST]
    rx, ry, rz = _rel(rw_x, rw_y, rw_z)
    features["rel_right_wrist_x"], features["rel_right_wrist_y"], features["rel_right_wrist_z"] = rx, ry, rz

    rx, ry, rz = _rel(lk_x, lk_y, lk_z)
    features["rel_left_knee_x"], features["rel_left_knee_y"], features["rel_left_knee_z"] = rx, ry, rz

    rx, ry, rz = _rel(rk_x, rk_y, rk_z)
    features["rel_right_knee_x"], features["rel_right_knee_y"], features["rel_right_knee_z"] = rx, ry, rz

    rx, ry, rz = _rel(la_x, la_y, la_z)
    features["rel_left_ankle_x"], features["rel_left_ankle_y"], features["rel_left_ankle_z"] = rx, ry, rz

    rx, ry, rz = _rel(ra_x, ra_y, ra_z)
    features["rel_right_ankle_x"], features["rel_right_ankle_y"], features["rel_right_ankle_z"] = rx, ry, rz

    return features


def extract_feature_vector(landmarks: Sequence[Any]) -> List[float]:
    """
    Extract feature vector as an ordered list of floats matching FEATURE_NAMES exactly.
    Ready for model.predict() or model.predict_proba() input: `[vector]`.
    """
    feat_dict = extract_features(landmarks)
    return [feat_dict[name] for name in FEATURE_NAMES]
