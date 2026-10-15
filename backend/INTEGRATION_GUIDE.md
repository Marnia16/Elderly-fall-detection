# Backend Readiness and Integration Guide

**Smart Elderly Fall Detection & Alert System**
**Role:** Backend / API & ML Integration (Keerthana)
**Collaborator:** Frontend & ML Training (Marnia)
**Branch:** `feature/backend-api`

---

## 1. How to Launch the Backend

### Primary Command (Recommended from `backend/` directory)
```bash
# Navigate to the backend directory
cd backend

# Launch the FastAPI server with auto-reload
python -m uvicorn app.main:app --reload
```

### Alternative Cross-Platform Runner (From repository root)
```bash
python backend/run.py
```

* **Default Host & Port:** `http://127.0.0.1:8000` (or `http://localhost:8000`)
* **Auto-Reload:** Enabled by default in development mode.

---

## 2. Core Service URLs & Documentation

| Resource | URL | Method | Description |
| :--- | :--- | :--- | :--- |
| **API Root** | `http://localhost:8000/` | `GET` | Service overview, metadata, and routing index |
| **Swagger Interactive Docs** | `http://localhost:8000/docs` | `GET` | Interactive OpenAPI / Swagger UI |
| **ReDoc Documentation** | `http://localhost:8000/redoc` | `GET` | Clean technical API reference |
| **Health Check (API)** | `http://localhost:8000/api/health` | `GET` | System health and model availability check |
| **Health Check (Root Alias)** | `http://localhost:8000/health` | `GET` | Deployment health probe for Render |
| **Prediction Endpoint** | `http://localhost:8000/api/predict` | `POST` | Camera-based fall prediction from 33 pose landmarks |
| **Prediction Pipeline Status** | `http://localhost:8000/api/predict/status` | `GET` | Status of model artifact integration |
| **Feature Schema Definition** | `http://localhost:8000/api/predict/feature-schema` | `GET` | Detailed 47-feature schema contract for Colab training |
| **Alerts Management** | `http://localhost:8000/api/alerts` | `GET`, `POST`, `DELETE` | Incident logging and querying |

---

## 3. Camera-Based Prediction API Specification

### Endpoint: `POST /api/predict`
Evaluates 33 body landmarks produced by **MediaPipe Pose** on the client webcam feed.

#### Validation Rules
1. **Landmark Count:** Must contain **exactly 33 landmarks** (indices 0 to 32). Requests with $<33$ or $>33$ landmarks are rejected with HTTP `422 Unprocessable Entity`.
2. **Coordinates:** `x` and `y` normalized to $[0.0, 1.0]$. `z` represents depth relative to the hips.
3. **Visibility Score:** Must be a float between $0.0$ and $1.0$ (`ge=0.0, le=1.0`).

#### Exact Request Example
```json
{
  "timestamp": "2026-10-15T12:00:00Z",
  "frame_id": 1024,
  "patient_id": "patient-001",
  "landmarks": [
    {"x": 0.502, "y": 0.182, "z": -0.045, "visibility": 0.99},
    {"x": 0.510, "y": 0.170, "z": -0.042, "visibility": 0.98},
    {"x": 0.515, "y": 0.170, "z": -0.042, "visibility": 0.98},
    {"x": 0.520, "y": 0.171, "z": -0.042, "visibility": 0.98},
    {"x": 0.490, "y": 0.170, "z": -0.042, "visibility": 0.98},
    {"x": 0.485, "y": 0.170, "z": -0.042, "visibility": 0.98},
    {"x": 0.480, "y": 0.171, "z": -0.042, "visibility": 0.98},
    {"x": 0.535, "y": 0.190, "z": 0.012, "visibility": 0.97},
    {"x": 0.465, "y": 0.190, "z": 0.010, "visibility": 0.97},
    {"x": 0.515, "y": 0.215, "z": -0.035, "visibility": 0.98},
    {"x": 0.488, "y": 0.215, "z": -0.035, "visibility": 0.98},
    {"x": 0.585, "y": 0.285, "z": -0.050, "visibility": 0.99},
    {"x": 0.415, "y": 0.285, "z": -0.050, "visibility": 0.99},
    {"x": 0.630, "y": 0.410, "z": -0.080, "visibility": 0.95},
    {"x": 0.370, "y": 0.410, "z": -0.080, "visibility": 0.95},
    {"x": 0.655, "y": 0.515, "z": -0.120, "visibility": 0.94},
    {"x": 0.345, "y": 0.515, "z": -0.120, "visibility": 0.94},
    {"x": 0.670, "y": 0.530, "z": -0.130, "visibility": 0.90},
    {"x": 0.330, "y": 0.530, "z": -0.130, "visibility": 0.90},
    {"x": 0.665, "y": 0.535, "z": -0.135, "visibility": 0.91},
    {"x": 0.335, "y": 0.535, "z": -0.135, "visibility": 0.91},
    {"x": 0.650, "y": 0.520, "z": -0.125, "visibility": 0.92},
    {"x": 0.350, "y": 0.520, "z": -0.125, "visibility": 0.92},
    {"x": 0.550, "y": 0.550, "z": -0.020, "visibility": 0.99},
    {"x": 0.450, "y": 0.550, "z": -0.020, "visibility": 0.99},
    {"x": 0.565, "y": 0.740, "z": -0.010, "visibility": 0.97},
    {"x": 0.435, "y": 0.740, "z": -0.010, "visibility": 0.97},
    {"x": 0.575, "y": 0.915, "z": 0.085, "visibility": 0.96},
    {"x": 0.425, "y": 0.915, "z": 0.085, "visibility": 0.96},
    {"x": 0.580, "y": 0.935, "z": 0.090, "visibility": 0.94},
    {"x": 0.420, "y": 0.935, "z": 0.090, "visibility": 0.94},
    {"x": 0.585, "y": 0.950, "z": 0.050, "visibility": 0.95},
    {"x": 0.415, "y": 0.950, "z": 0.050, "visibility": 0.95}
  ]
}
```

#### Exact Response (Current `model_not_ready` State)
Because the trained Random Forest model has not yet been placed in `backend/app/model/`, the endpoint **never fabricates predictions or confidence scores**:
```json
{
  "model_loaded": false,
  "fall_detected": null,
  "confidence": null,
  "features_extracted": 47,
  "message": "Random Forest model is not yet integrated. MediaPipe landmarks validated (33/33) and 47 features successfully extracted. Model training on Colab (e.g., URFD dataset) is currently in progress.",
  "status": "model_not_ready",
  "timestamp": "2026-10-15T12:00:00.000000Z"
}
```

#### Exact Response (Post-Model Integration State)
Once the real model artifact is saved and loaded:
```json
{
  "model_loaded": true,
  "fall_detected": false,
  "confidence": 0.96,
  "features_extracted": 47,
  "message": "Inference completed successfully.",
  "status": "ok",
  "timestamp": "2026-10-15T12:00:00.000000Z"
}
```

---

## 4. The 47-Feature Schema & Training Alignment Contract

To avoid train/inference feature mismatch, `backend/app/features.py` extracts exactly **47 numerical features** in strict order:

### Expected Model Path
* `backend/app/model/random_forest_model.joblib` (or `.pkl`, `model.joblib`)

### Ordered Feature List
1. `torso_vertical_angle_deg`: Angle (degrees) between mid-shoulder to mid-hip vector and vertical Y axis.
2. `torso_length`: Euclidean length $L_{torso} = \sqrt{(H_x - S_x)^2 + (H_y - S_y)^2 + (H_z - S_z)^2}$.
3. `bbox_aspect_ratio`: Bounding box width / height.
4. `bbox_height_to_torso_ratio`: Bounding box height / $L_{torso}$.
5. `hip_to_ankle_vertical_dist`: Vertical distance from mid-hip to mid-ankle / $L_{torso}$.
6. `head_to_hip_vertical_dist`: Vertical distance from nose to mid-hip / $L_{torso}$.
7. `shoulder_tilt_deg`: Angular tilt of shoulder axis relative to horizontal.
8. `hip_tilt_deg`: Angular tilt of hip axis relative to horizontal.
9. `knee_to_ankle_vertical_dist`: Vertical distance from knee to ankle / $L_{torso}$.
10. `head_to_ankle_vertical_dist`: Vertical distance from nose to ankle / $L_{torso}$.
11. `mean_landmark_visibility`: Average visibility score across all 33 landmarks.
12â€“47. **Normalized Relative Spatial Coordinates (36 features):**
   12 key body joints normalized by centering at Mid-Hip $(H_x, H_y, H_z)$ and scaling by $L_{torso}$:
   $\tilde{x} = \frac{x - H_x}{L_{torso}}, \quad \tilde{y} = \frac{y - H_y}{L_{torso}}, \quad \tilde{z} = \frac{z - H_z}{L_{torso}}$
   Joints: `nose`, `left_shoulder`, `right_shoulder`, `mid_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.

> **Notice on `mid_hip_z`:**
> The midpoint of hips for $z$ coordinates is computed as $H_z = \frac{\text{left\_hip}_z + \text{right\_hip}_z}{2}$. An audited typo where hip $y$ coordinates were mistakenly used for $H_z$ was verified and corrected in `backend/app/features.py`.

---

## 5. Current Alert Functionality & Architectural Limitations

The backend includes REST endpoints for logging and managing fall incidents:

### Endpoints
* `GET /api/alerts`: List alerts. Supports optional query parameters `status=ACTIVE|ACKNOWLEDGED|RESOLVED` and `limit=50`.
* `POST /api/alerts`: Register an incident (status code `201 Created`).
* `GET /api/alerts/{alert_id}`: Retrieve incident details.
* `PATCH /api/alerts/{alert_id}`: Update incident status or append caregiver notes.
* `DELETE /api/alerts/{alert_id}`: Delete an incident (`204 No Content`).
* `DELETE /api/alerts`: Reset and clear all alerts.

### Current Limitations
1. **In-Memory Storage Only:** Incidents are stored in a Python dictionary in memory. If the backend process restarts or redeploys on Render, all incidents reset to the seeded default incident (`inc-001`).
2. **No Automated External Notifications:** There is **no** active SMS (Twilio), email (SendGrid), push notifications, or WebSocket broadcast. The system does not currently notify caregivers automatically; all incident retrieval relies on the frontend polling or fetching via the REST API.

---

## 6. Environment Variables and CORS Configuration

### Environment Variables
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | IP interface to bind to |
| `PORT` | `8000` | Port for the Uvicorn server |
| `RELOAD` | `true` | Enables auto-reload on code changes |
| `ENV` | `development` | Environment label returned by `/api/health` |
| `CORS_ORIGINS` | `""` | Comma-separated list of extra allowed origins or `*` |

### CORS Configuration
Configured in `backend/app/main.py`:
* **Default Allowed Origins:**
  * `http://localhost:5173` (React + Vite default)
  * `http://127.0.0.1:5173`
  * `http://localhost:3000`
  * `http://127.0.0.1:3000`
* **Regex Matching:** `https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.vercel\.app`
* **Security Guard:** `allow_credentials` is set to `True` for designated origins and Vercel domains, but automatically disabled if `CORS_ORIGINS=*` to prevent browser credential vulnerabilities.

---

## 7. Steps to Integrate the Trained Model and Frontend

### Step A: Machine Learning Model Integration (Marnia & Keerthana)
1. **Google Colab Training:** Use `features.py` inside Colab to extract the exact 47 features from each labeled video frame of the URFD or similar fall dataset:
   ```python
   from features import extract_feature_vector, FEATURE_NAMES

   # For each frame with 33 MediaPipe landmarks:
   feature_vector = extract_feature_vector(landmarks)
   X.append(feature_vector)
   y.append(label)  # 1 for Fall, 0 for Normal Activity (ADL)
   ```
2. **Train Classifier:** Train a `RandomForestClassifier` on the 47-feature vectors:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
   clf.fit(X_train, y_train)
   ```
3. **Save Artifact:**
   ```python
   import joblib
   joblib.dump(clf, "random_forest_model.joblib")
   ```
4. **Deploy Artifact:** Place `random_forest_model.joblib` directly into:
   `backend/app/model/random_forest_model.joblib`
   *(Create the `backend/app/model/` directory if not present).*
5. **Automated Loading & Validation:**
   - The backend automatically detects the artifact via file timestamp.
   - It validates that the model provides a callable `.predict()` method.
   - It validates that `model.n_features_in_ == 47`. If there is a feature count mismatch or file corruption, the backend halts inference safely and returns `model_not_ready` with an informative error message instead of crashing.
   - **No code modification or server reconfiguration is required.**
6. **Verify Model Readiness:**
   - Query `GET http://localhost:8000/api/health` to confirm `"model_loaded": true`.
   - Query `GET http://localhost:8000/api/predict/status` to confirm `"status": "ready"`.
   - Send a sample 33-landmark payload to `POST http://localhost:8000/api/predict` to receive real model inference (`fall_detected: false|true`, `confidence: float`, `status: "ok"`).

### Step B: Frontend Dashboard Integration (React + Vite)
1. **Client-Side Pose Estimation:** Frontend loads `@mediapipe/pose` or `@mediapipe/tasks-vision` and tracks 33 landmarks on the webcam stream.
2. **Stream Telemetry:** When webcam frames are processed (e.g. at 10â€“15 FPS), send landmark array to:
   `POST http://localhost:8000/api/predict`
3. **Trigger Incident Alert:** If response indicates `fall_detected: true`, frontend issues a:
   `POST http://localhost:8000/api/alerts` with the incident location, severity, and pose snapshot.
4. **Caregiver Dashboard:** Polls or fetches `GET http://localhost:8000/api/alerts?status=ACTIVE` to display real-time fall alerts.
