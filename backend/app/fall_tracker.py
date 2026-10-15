"""Temporal fall confirmation, duplicate suppression, cooldown, and near-fall tracking engine."""

import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from app.database import create_incident, update_incident, get_incident
from app.notifications import notification_service


class FallTracker:
    """
    State machine that processes per-frame model inference results to provide:
    1. Temporal confirmation (prevents single-frame false positives).
    2. Active incident management and duplicate suppression (prevents 100 alerts for 1 fall).
    3. Cooldown and re-arming once the subject stands up or recovers.
    4. Near-fall detection (postural instability warning without triggering emergency alarm).
    5. Post-fall immobility monitoring (upgrades severity if no movement detected post-fall).
    6. Alert escalation tracking for prolonged unacknowledged active falls.
    """

    def __init__(self):
        # Configurable thresholds
        self.confirmation_frames_needed = int(os.getenv("FALL_CONFIRMATION_FRAMES", 2))
        self.confirmation_window_s = float(os.getenv("FALL_WINDOW_SECONDS", 2.5))
        self.cooldown_period_s = float(os.getenv("FALL_COOLDOWN_SECONDS", 10.0))
        self.min_fall_probability = float(os.getenv("MIN_FALL_PROBABILITY", 0.60))
        self.immobility_seconds_threshold = float(os.getenv("IMMOBILITY_SECONDS_THRESHOLD", 5.0))
        self.escalation_delay_seconds = float(os.getenv("ALERT_ESCALATION_DELAY_SECONDS", 60.0))

        # Per-patient state dictionary
        self._states: Dict[str, Dict[str, Any]] = {}

    def _get_patient_state(self, patient_id: str) -> Dict[str, Any]:
        """Initializes or retrieves tracking state for a specific patient."""
        if patient_id not in self._states:
            self._states[patient_id] = {
                "tracker_state": "MONITORING",  # MONITORING | CONFIRMING | FALL_ACTIVE | COOLDOWN
                "consecutive_fall_frames": 0,
                "first_fall_time": None,
                "active_incident_id": None,
                "fall_start_time": None,
                "cooldown_until": None,
                "last_seen_time": time.time(),
                "recent_landmark_coords": [],
                "immobile_start_time": None,
                "post_fall_immobility": False,
                "escalated": False,
                "near_fall_detected": False,
            }
        return self._states[patient_id]

    def reset_patient(self, patient_id: str = "patient-001") -> None:
        """Resets the state for a patient (e.g. after incident resolution)."""
        if patient_id in self._states:
            self._states[patient_id] = {
                "tracker_state": "MONITORING",
                "consecutive_fall_frames": 0,
                "first_fall_time": None,
                "active_incident_id": None,
                "fall_start_time": None,
                "cooldown_until": None,
                "last_seen_time": time.time(),
                "recent_landmark_coords": [],
                "immobile_start_time": None,
                "post_fall_immobility": False,
                "escalated": False,
                "near_fall_detected": False,
            }

    def detect_near_fall(
        self,
        landmarks: List[Dict[str, Any]],
        features: Optional[Dict[str, float]] = None,
        fall_probability: Optional[float] = None,
    ) -> bool:
        """
        Computes near-fall heuristic:
        Subject shows substantial torso tilt (40° to 65°) or rapid level descent
        with moderate instability, without full ground impact collapse.
        """
        if not features or not landmarks or len(landmarks) != 33:
            return False

        torso_angle = features.get("torso_vertical_angle_deg", 0.0)
        aspect_ratio = features.get("bbox_aspect_ratio", 0.0)

        # Torso angled between 40° and 65° and aspect ratio starting to widen
        is_tilting = (40.0 <= torso_angle <= 65.0) and (aspect_ratio >= 0.85)

        # Probability in warning range (0.35 to 0.60)
        prob = fall_probability if fall_probability is not None else 0.0
        is_instability_prob = 0.35 <= prob < self.min_fall_probability

        return bool(is_tilting or is_instability_prob)

    def _calculate_motion(self, current_landmarks: List[Dict[str, Any]], prev_landmarks: List[Dict[str, Any]]) -> float:
        """Calculates mean displacement of key joints across consecutive frames."""
        if not current_landmarks or not prev_landmarks or len(current_landmarks) != len(prev_landmarks):
            return 1.0

        key_indices = [0, 11, 12, 23, 24, 25, 26]  # nose, shoulders, hips, knees
        total_disp = 0.0
        for idx in key_indices:
            dx = current_landmarks[idx].get("x", 0.0) - prev_landmarks[idx].get("x", 0.0)
            dy = current_landmarks[idx].get("y", 0.0) - prev_landmarks[idx].get("y", 0.0)
            total_disp += math.sqrt(dx * dx + dy * dy)
        return total_disp / len(key_indices)

    def process_frame(
        self,
        landmarks: List[Dict[str, Any]],
        features: Dict[str, float],
        fall_detected: Optional[bool],
        confidence: Optional[float],
        fall_probability: Optional[float],
        patient_id: str = "patient-001",
        location: str = "Living Area",
    ) -> Dict[str, Any]:
        """
        Processes single inference frame through temporal state machine.

        Returns:
            Dict containing:
                - fall_confirmed (bool): True if temporal confirmation criteria met
                - tracker_state (str): MONITORING | CONFIRMING | FALL_ACTIVE | COOLDOWN
                - active_incident_id (str | None): ID of current persistent incident if any
                - near_fall (bool): Near-fall instability detected
                - post_fall_immobility (bool): Immobility detected post-fall
                - escalated (bool): True if alert escalated due to lack of acknowledgement
                - duration_seconds (float): Duration of current active fall event
        """
        now = time.time()
        st = self._get_patient_state(patient_id)
        st["last_seen_time"] = now

        # 1. Evaluate near-fall warning
        near_fall = self.detect_near_fall(landmarks, features, fall_probability)
        st["near_fall_detected"] = near_fall

        # 2. Check if active incident in DB was resolved out-of-band by caregiver
        if st["active_incident_id"]:
            inc = get_incident(st["active_incident_id"])
            if inc and inc.get("status") == "RESOLVED":
                # Caregiver resolved incident; transition to COOLDOWN
                st["tracker_state"] = "COOLDOWN"
                st["cooldown_until"] = now + self.cooldown_period_s
                st["active_incident_id"] = None
                st["fall_start_time"] = None
                st["consecutive_fall_frames"] = 0
                st["post_fall_immobility"] = False
                st["immobile_start_time"] = None

        # 3. Model output evaluation
        prob = fall_probability if fall_probability is not None else (1.0 if fall_detected else 0.0)
        is_raw_fall = bool(fall_detected is True and prob >= self.min_fall_probability)

        current_state = st["tracker_state"]

        # --- STATE: COOLDOWN ---
        if current_state == "COOLDOWN":
            if st["cooldown_until"] and now >= st["cooldown_until"]:
                # Cooldown completed, re-arm detector
                st["tracker_state"] = "MONITORING"
                st["consecutive_fall_frames"] = 0
                st["first_fall_time"] = None
            else:
                return {
                    "fall_confirmed": False,
                    "tracker_state": "COOLDOWN",
                    "active_incident_id": None,
                    "near_fall": False,
                    "post_fall_immobility": False,
                    "escalated": False,
                    "duration_seconds": 0.0,
                }

        # --- STATE: FALL_ACTIVE ---
        if st["tracker_state"] == "FALL_ACTIVE":
            duration = now - (st["fall_start_time"] or now)

            # Motion & Post-fall immobility tracking
            motion = 1.0
            if st["recent_landmark_coords"]:
                motion = self._calculate_motion(landmarks, st["recent_landmark_coords"])
            st["recent_landmark_coords"] = landmarks

            if motion < 0.035:  # Low motion threshold
                if st["immobile_start_time"] is None:
                    st["immobile_start_time"] = now
                elif (now - st["immobile_start_time"]) >= self.immobility_seconds_threshold:
                    st["post_fall_immobility"] = True
            else:
                st["immobile_start_time"] = now
                st["post_fall_immobility"] = False

            # Escalation check
            if duration >= self.escalation_delay_seconds and not st["escalated"]:
                inc = get_incident(st["active_incident_id"])
                if inc and inc.get("status") == "ACTIVE":
                    st["escalated"] = True
                    update_incident(st["active_incident_id"], escalated=True)
                    notification_service.send_escalation_alert(inc)

            # Update active incident duration and metrics in database
            if st["active_incident_id"]:
                severity_to_set = "CRITICAL" if st["post_fall_immobility"] or st["escalated"] else "HIGH"
                update_incident(
                    st["active_incident_id"],
                    duration_seconds=round(duration, 1),
                    confidence=confidence,
                    fall_probability=fall_probability,
                )

            # Check if subject stood up / normal posture resumed consistently
            if not is_raw_fall:
                st["consecutive_fall_frames"] = max(0, st["consecutive_fall_frames"] - 1)
                # If normal for 4 consecutive frames (~2.5s), transition to COOLDOWN
                if st["consecutive_fall_frames"] <= 0:
                    st["tracker_state"] = "COOLDOWN"
                    st["cooldown_until"] = now + self.cooldown_period_s
                    st["active_incident_id"] = None
                    st["fall_start_time"] = None
            else:
                st["consecutive_fall_frames"] = min(10, st["consecutive_fall_frames"] + 1)

            return {
                "fall_confirmed": True,
                "tracker_state": "FALL_ACTIVE",
                "active_incident_id": st["active_incident_id"],
                "near_fall": False,
                "post_fall_immobility": st["post_fall_immobility"],
                "escalated": st["escalated"],
                "duration_seconds": round(duration, 1),
            }

        # --- STATE: CONFIRMING ---
        if st["tracker_state"] == "CONFIRMING":
            if is_raw_fall:
                st["consecutive_fall_frames"] += 1
                elapsed = now - (st["first_fall_time"] or now)

                # Has confirmed within the window?
                if st["consecutive_fall_frames"] >= self.confirmation_frames_needed and elapsed <= self.confirmation_window_s:
                    # CONFIRMED FALL EVENT!
                    st["tracker_state"] = "FALL_ACTIVE"
                    st["fall_start_time"] = now
                    st["immobile_start_time"] = now

                    # Create persistent incident in SQLite
                    new_inc = create_incident({
                        "patient_id": patient_id,
                        "location": location,
                        "severity": "HIGH",
                        "prediction_type": "CONFIRMED_FALL",
                        "message": (
                            f"Confirmed fall event detected in {location} "
                            f"(confidence: {confidence:.2f}, probability: {prob:.2f})."
                        ),
                        "status": "ACTIVE",
                        "confidence": confidence,
                        "fall_probability": fall_probability,
                        "duration_seconds": 1.0,
                        "sensor_snapshot": {
                            "torso_vertical_angle_deg": features.get("torso_vertical_angle_deg"),
                            "bbox_aspect_ratio": features.get("bbox_aspect_ratio"),
                            "near_fall_precursor": near_fall,
                        },
                        "notes": "Automated incident created by temporal confirmation tracker.",
                    })

                    st["active_incident_id"] = new_inc["id"]

                    # Dispatch notification
                    notification_service.send_fall_alert(new_inc)

                    return {
                        "fall_confirmed": True,
                        "tracker_state": "FALL_ACTIVE",
                        "active_incident_id": st["active_incident_id"],
                        "near_fall": False,
                        "post_fall_immobility": False,
                        "escalated": False,
                        "duration_seconds": 1.0,
                    }
            else:
                # Dropped below threshold during confirmation window
                elapsed = now - (st["first_fall_time"] or now)
                if elapsed > self.confirmation_window_s:
                    # Confirmation window expired without meeting consecutive threshold
                    st["tracker_state"] = "MONITORING"
                    st["consecutive_fall_frames"] = 0
                    st["first_fall_time"] = None

            return {
                "fall_confirmed": False,
                "tracker_state": "CONFIRMING",
                "active_incident_id": None,
                "near_fall": near_fall,
                "post_fall_immobility": False,
                "escalated": False,
                "duration_seconds": 0.0,
            }

        # --- STATE: MONITORING (Default Idle) ---
        if is_raw_fall:
            st["tracker_state"] = "CONFIRMING"
            st["consecutive_fall_frames"] = 1
            st["first_fall_time"] = now
            return {
                "fall_confirmed": False,
                "tracker_state": "CONFIRMING",
                "active_incident_id": None,
                "near_fall": near_fall,
                "post_fall_immobility": False,
                "escalated": False,
                "duration_seconds": 0.0,
            }

        return {
            "fall_confirmed": False,
            "tracker_state": "MONITORING",
            "active_incident_id": None,
            "near_fall": near_fall,
            "post_fall_immobility": False,
            "escalated": False,
            "duration_seconds": 0.0,
        }


# Global tracker singleton
fall_tracker = FallTracker()
