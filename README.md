# Smart Elderly Fall Detection & Alert System

An end-to-end, privacy-preserving, AI-powered elderly fall detection telemetry platform. Combining real-time browser-based computer vision (MediaPipe Tasks Vision), a standardized 47-feature spatial geometry pipeline, a trained Random Forest classifier, a persistent SQLite incident store, automated caregiver notifications, and a responsive React monitoring dashboard.

---

## Architecture Overview

```
                                      +---------------------------------------------+
                                      |                Web Browser                  |
                                      |  (React 18 + Vite 5 Caregiver Dashboard)    |
                                      +---------------------+-----------------------+
                                                            |
                                        Webcam Video (1280x720 @ 30 FPS)
                                                            v
                                      +---------------------------------------------+
                                      |    MediaPipe Pose Landmarker (WASM v1.1.0)  |
                                      |         33 Normalized 3D Keypoints          |
                                      +---------------------+-----------------------+
                                                            |
                                     POST /api/predict (JSON: 33 landmarks, ~650ms)
                                                            v
+---------------------------------------------------------------------------------------------------------+
|                                      FastAPI Telemetry Backend                                          |
|                                                                                                         |
|   1. 47-Feature Extraction Pipeline                                                                     |
|      - Bounding box aspect ratios, centroid coordinates, bounding box area                              |
|      - Torso vertical orientation angle (deg), hip midpoint, shoulder midpoint                          |
|      - Raw 3D normalized coordinates (33 x 1 = 33 features) + geometric spatial indicators              |
|                                                                                                         |
|   2. Random Forest Classification (scikit-learn)                                                        |
|      - 100 Estimators, trained on URFD dataset features                                                 |
|      - Winning-class confidence & exact fall probability derived from class index                       |
|                                                                                                         |
|   3. Temporal Fall Confirmation & State Machine Tracker                                                 |
|      - States: MONITORING -> CONFIRMING -> FALL_ACTIVE -> COOLDOWN                                      |
|      - Temporal Confirmation: Requires >= 2 consecutive positive detections within <= 2.5s window       |
|      - Near-Fall Detection: Postural tilt heuristic (40 deg <= theta <= 65 deg)                         |
|      - Post-Fall Immobility: Lack of significant centroid motion (< 0.035) for > 5s -> Severity: CRITICAL|
|      - Escalation: Unacknowledged active falls escalate after 60s                                       |
|                                                                                                         |
|   4. Persistent Incident Storage (SQLite WAL Mode)                                                      |
|      - Automatic database creation at backend/app/data/incidents.db                                     |
|      - Tables: incidents, notification_logs                                                             |
|                                                                                                         |
|   5. Notification Service                                                                               |
|      - Modes: demo (local audit log) or smtp (dispatch emergency emails to caregiver)                   |
+---------------------------------------------------+-----------------------------------------------------+
                                                    |
                                                    v
+---------------------------------------------------------------------------------------------------------+
|                                    Caregiver Dashboard UI Features                                      |
|                                                                                                         |
|   - Active Emergency Banner: High-visibility alert with persistent audio beeper (Web Audio API)        |
|   - One-Click Actions: Acknowledge alert, Resolve incident with clinical audit notes, Siren Mute/Unmute |
|   - System Health Pills: Real-time indicators for FastAPI, Model, SQLite Database, and Notifications    |
|   - Real-Time Analytics & 7-Day Trend: Dynamic SVG trend chart computed from real SQLite timestamps     |
|   - Historical Audit Table: Multi-criteria filtering (status, severity, search), details modal         |
|   - Export & Reporting: One-click CSV export stream and printable caregiver audit view                  |
+---------------------------------------------------------------------------------------------------------+
```

---

## Key Features

1. **Camera-Based Pose Estimation (Zero Wearables Required)**
   - Uses `@mediapipe/tasks-vision@1.1.0` in the browser.
   - Pinned WASM binary ensures zero initialization failures.
   - Front-facing camera captures 33 keypoints without uploading video frames to servers, preserving resident privacy.

2. **Standardized 47-Feature Server Pipeline**
   - The backend owns feature extraction, guaranteeing identical feature engineering between ML training and real-time live inference.
   - Evaluates vertical torso angle, aspect ratio collapse, vertical centroid descent, and 33 normalized coordinates.

3. **Temporal Confirmation & False Alarm Rejection**
   - Single-frame anomalies (e.g. resident quickly sitting on a sofa or picking up an item) are filtered by a sliding temporal confirmation window ($\ge 2$ positive frames in $\le 2.5$s).
   - Duplicate suppression ensures a single ongoing incident record is updated with duration rather than flooding the database with hundreds of alerts.

4. **Near-Fall & Post-Fall Immobility Detection**
   - **Near-Fall Warning**: Detects significant torso tilt ($40^\circ \le \theta \le 65^\circ$) without triggering a full emergency dispatch.
   - **Post-Fall Immobility**: Monitors resident motion after collapse; if stationary for $> 5$ seconds, automatically escalates incident severity to `CRITICAL`.

5. **Persistent SQLite Incident Management**
   - Stored in `backend/app/data/incidents.db` with Write-Ahead Logging (`PRAGMA journal_mode=WAL`).
   - Retains full incident audit history, sensor snapshots, resolution notes, and timestamps across backend restarts.

6. **Audio Alert Synthesizer**
   - Pure Web Audio API two-tone emergency beeper synthesized directly in the browser without external MP3 dependencies.
   - Persists mute preference in browser local storage.

7. **Caregiver Reporting & Analytics**
   - Pure React SVG 7-day trend chart showing daily incident frequency.
   - Direct CSV export stream (`/api/alerts/export/csv`) formatted for medical audit.
   - Printable report view (`window.print()`).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API root index and service discovery |
| `GET` | `/api/health` | System health check (Model, SQLite DB, Notifications) |
| `POST` | `/api/predict` | Run fall detection on 33 MediaPipe pose landmarks |
| `GET` | `/api/predict/status` | Model readiness and feature count contract |
| `GET` | `/api/predict/tracker-config` | View temporal confirmation engine thresholds |
| `POST` | `/api/predict/tracker/reset` | Reset temporal tracker state to `MONITORING` |
| `GET` | `/api/alerts` | List incidents (supports `status`, `severity`, `search`, `limit`, `offset`) |
| `POST` | `/api/alerts` | Create incident record and trigger notification |
| `GET` | `/api/alerts/analytics/summary` | Real incident counts and 7-day daily trend |
| `GET` | `/api/alerts/export/csv` | Download filtered incidents as CSV report |
| `GET` | `/api/alerts/{id}` | Get specific incident details and sensor snapshot |
| `PATCH` | `/api/alerts/{id}` | Update incident status, notes, or escalation flag |
| `POST` | `/api/alerts/{id}/acknowledge` | Mark incident as acknowledged by caregiver |
| `POST` | `/api/alerts/{id}/resolve` | Mark incident as resolved with caregiver notes |
| `DELETE` | `/api/alerts/{id}` | Delete incident record |
| `DELETE` | `/api/alerts` | Reset/clear all incident records |

---

## Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ and npm
- Webcam / Camera access

### 1. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run automated test suite (23 tests)
pytest tests/

# Start FastAPI development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend is now live at `http://localhost:8000` with interactive Swagger docs at `http://localhost:8000/docs`.

### 2. Frontend Setup

```bash
cd frontend

# Install npm dependencies
npm install

# Build production bundle to verify compilation
npm run build

# Start Vite development server
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Environment Variables

Copy `.env.example` to configure custom behavior:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_PATH` | `app/data/incidents.db` | Custom path for SQLite database file |
| `NOTIFICATION_MODE` | `demo` | `demo` (local logging) or `smtp` (dispatches real emails) |
| `ALERT_RECIPIENT_EMAIL` | `caregiver-team@hospital.org` | Recipient email for emergency dispatches |
| `SMTP_HOST` | `smtp.gmail.com` | SMTP relay server host |
| `SMTP_PORT` | `587` | SMTP relay server port |
| `SMTP_USER` | `""` | SMTP authentication user |
| `SMTP_PASSWORD` | `""` | SMTP authentication password / app key |
| `CORS_ORIGINS` | Localhost ports | Comma-separated list of allowed origins |
| `VITE_API_BASE_URL` | `http://localhost:8000/api` | Frontend base URL for FastAPI backend |

---

## Running Automated Verification

### Backend Tests (pytest)
```bash
cd backend
.venv/bin/pytest tests/ -v
```
Verifies:
- Root and health status responses
- 47-feature vector extraction mathematical constraints
- Random Forest model inference and probability derivation
- Persistent SQLite incident CRUD across requests
- Temporal confirmation and duplicate suppression state machine
- Acknowledge and Resolve endpoints with clinical notes
- Analytics summary and 7-day trend calculation
- CSV export formatting and HTTP headers

### Frontend Build
```bash
cd frontend
npm run build
```
Verifies:
- All JSX components, hooks, Web Audio API utils, and API endpoints compile cleanly.

---

## Git Branch Information
- Active Development Branch: `feature/project-enhancements`
- Integrated from: `feature/frontend-dashboard` (with MediaPipe fix, model artifact, and working webcam).
