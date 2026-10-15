"""SQLite database layer for persistent fall incident management."""

import csv
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default database file in backend/app/data/incidents.db
DB_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DB_PATH = DB_DIR / "incidents.db"
DB_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for concurrency and foreign keys
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    """Initializes the database schema and indexes."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                location TEXT NOT NULL,
                severity TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                fall_probability REAL,
                duration_seconds REAL DEFAULT 0,
                sensor_snapshot TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                acknowledged_at TEXT,
                resolved_at TEXT,
                escalated INTEGER DEFAULT 0,
                is_demo INTEGER DEFAULT 0
            );
        """)
        # Safe migration if table already exists without is_demo column
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN is_demo INTEGER DEFAULT 0;")
        except sqlite3.OperationalError:
            pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notification_logs (
                id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                recipient TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL
            );
        """)
        # Indexes for fast querying and filtering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_patient ON incidents(patient_id);")
        conn.commit()

    # Seed initial demonstration incidents if database is completely empty
    seed_if_empty()


def seed_if_empty() -> None:
    """Populates a few demonstration records if the table is freshly created."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) as count FROM incidents;")
        count = cursor.fetchone()["count"]
        if count == 0:
            now = datetime.now(timezone.utc)
            sample_records = [
                {
                    "id": "inc-001",
                    "patient_id": "patient-101",
                    "location": "Living Room",
                    "severity": "HIGH",
                    "prediction_type": "RAPID_COLLAPSE",
                    "message": "Rapid vertical deceleration detected — possible fall near couch.",
                    "status": "RESOLVED",
                    "confidence": 0.94,
                    "fall_probability": 0.94,
                    "duration_seconds": 18.0,
                    "sensor_snapshot": json.dumps({"torso_vertical_angle_deg": 82.5, "bbox_aspect_ratio": 1.48}),
                    "notes": "Caregiver attended. Resident verified safe and assisted to chair.",
                    "created_at": (now - timedelta(hours=3, minutes=15)).isoformat(),
                    "acknowledged_at": (now - timedelta(hours=3, minutes=13)).isoformat(),
                    "resolved_at": (now - timedelta(hours=3, minutes=5)).isoformat(),
                    "escalated": 0,
                    "is_demo": 1,
                },
                {
                    "id": "inc-002",
                    "patient_id": "patient-101",
                    "location": "Bedroom",
                    "severity": "MEDIUM",
                    "prediction_type": "POSTURAL_INSTABILITY",
                    "message": "Near-fall postural tilt and abrupt level drop detected.",
                    "status": "RESOLVED",
                    "confidence": 0.72,
                    "fall_probability": 0.48,
                    "duration_seconds": 6.5,
                    "sensor_snapshot": json.dumps({"torso_vertical_angle_deg": 52.1, "bbox_aspect_ratio": 1.15}),
                    "notes": "Bedside transition wobble; recovered balance autonomously.",
                    "created_at": (now - timedelta(days=1, hours=4)).isoformat(),
                    "acknowledged_at": (now - timedelta(days=1, hours=3, minutes=58)).isoformat(),
                    "resolved_at": (now - timedelta(days=1, hours=3, minutes=50)).isoformat(),
                    "escalated": 0,
                    "is_demo": 1,
                },
                {
                    "id": "inc-003",
                    "patient_id": "patient-101",
                    "location": "Hallway",
                    "severity": "CRITICAL",
                    "prediction_type": "CONFIRMED_FALL",
                    "message": "High-impact fall detected with post-fall immobility.",
                    "status": "RESOLVED",
                    "confidence": 0.98,
                    "fall_probability": 0.99,
                    "duration_seconds": 45.0,
                    "sensor_snapshot": json.dumps({"torso_vertical_angle_deg": 88.0, "bbox_aspect_ratio": 1.62}),
                    "notes": "Emergency response team dispatched. Resident treated for minor bruising.",
                    "created_at": (now - timedelta(days=2, hours=6)).isoformat(),
                    "acknowledged_at": (now - timedelta(days=2, hours=5, minutes=59)).isoformat(),
                    "resolved_at": (now - timedelta(days=2, hours=5, minutes=20)).isoformat(),
                    "escalated": 1,
                    "is_demo": 1,
                },
            ]
            for r in sample_records:
                conn.execute(
                    """
                    INSERT INTO incidents (
                        id, patient_id, location, severity, prediction_type, message, status,
                        confidence, fall_probability, duration_seconds, sensor_snapshot, notes,
                        created_at, acknowledged_at, resolved_at, escalated, is_demo
                    ) VALUES (
                        :id, :patient_id, :location, :severity, :prediction_type, :message, :status,
                        :confidence, :fall_probability, :duration_seconds, :sensor_snapshot, :notes,
                        :created_at, :acknowledged_at, :resolved_at, :escalated, :is_demo
                    );
                    """,
                    r,
                )
            conn.commit()


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a SQLite row into a clean dictionary with deserialized JSON sensor_snapshot."""
    d = dict(row)
    if d.get("sensor_snapshot") and isinstance(d["sensor_snapshot"], str):
        try:
            d["sensor_snapshot"] = json.loads(d["sensor_snapshot"])
        except Exception:
            pass
    # Map created_at to timestamp for backward compatibility with AlertResponse schema
    if "created_at" in d and "timestamp" not in d:
        d["timestamp"] = d["created_at"]
    # Explicitly determine demo vs live origin
    d["is_demo"] = bool(d.get("is_demo") or d.get("id") in ("inc-001", "inc-002", "inc-003"))
    return d


def create_incident(data: Dict[str, Any]) -> Dict[str, Any]:
    """Inserts a new incident into SQLite and returns the full record."""
    incident_id = data.get("id") or f"inc-{uuid.uuid4().hex[:8]}"
    now_iso = datetime.now(timezone.utc).isoformat()
    created_at = data.get("created_at") or data.get("timestamp") or now_iso

    sensor_str = None
    if data.get("sensor_snapshot"):
        if isinstance(data["sensor_snapshot"], str):
            sensor_str = data["sensor_snapshot"]
        else:
            sensor_str = json.dumps(data["sensor_snapshot"])

    is_demo = int(
        data.get("is_demo", 0)
        or (1 if str(incident_id).startswith("demo-") or incident_id in ("inc-001", "inc-002", "inc-003") else 0)
    )

    record = {
        "id": incident_id,
        "patient_id": data.get("patient_id") or "patient-001",
        "location": data.get("location") or "Living Room",
        "severity": data.get("severity") or "HIGH",
        "prediction_type": data.get("prediction_type") or "CONFIRMED_FALL",
        "message": data.get("message") or "Fall event detected by pose telemetry.",
        "status": data.get("status") or "ACTIVE",
        "confidence": data.get("confidence"),
        "fall_probability": data.get("fall_probability"),
        "duration_seconds": float(data.get("duration_seconds", 0.0) or 0.0),
        "sensor_snapshot": sensor_str,
        "notes": data.get("notes"),
        "created_at": created_at,
        "acknowledged_at": data.get("acknowledged_at"),
        "resolved_at": data.get("resolved_at"),
        "escalated": int(data.get("escalated", 0) or 0),
        "is_demo": is_demo,
    }

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO incidents (
                id, patient_id, location, severity, prediction_type, message, status,
                confidence, fall_probability, duration_seconds, sensor_snapshot, notes,
                created_at, acknowledged_at, resolved_at, escalated, is_demo
            ) VALUES (
                :id, :patient_id, :location, :severity, :prediction_type, :message, :status,
                :confidence, :fall_probability, :duration_seconds, :sensor_snapshot, :notes,
                :created_at, :acknowledged_at, :resolved_at, :escalated, :is_demo
            );
            """,
            record,
        )
        conn.commit()
        conn.commit()

    return get_incident(incident_id)


def get_incident(incident_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an incident by its unique ID."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM incidents WHERE id = ?;", (incident_id,))
        row = cursor.fetchone()
        return row_to_dict(row) if row else None


def list_incidents(
    status_filter: Optional[str] = None,
    severity_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Lists incidents with optional filtering, sorting by created_at DESC."""
    query = "SELECT * FROM incidents WHERE 1=1"
    count_query = "SELECT COUNT(*) as total FROM incidents WHERE 1=1"
    params = []

    if status_filter:
        query += " AND status = ?"
        count_query += " AND status = ?"
        params.append(status_filter.upper())

    if severity_filter:
        query += " AND severity = ?"
        count_query += " AND severity = ?"
        params.append(severity_filter.upper())

    if start_date:
        query += " AND created_at >= ?"
        count_query += " AND created_at >= ?"
        params.append(start_date)

    if end_date:
        query += " AND created_at <= ?"
        count_query += " AND created_at <= ?"
        params.append(end_date)

    if search:
        search_pattern = f"%{search}%"
        query += " AND (message LIKE ? OR location LIKE ? OR patient_id LIKE ? OR notes LIKE ? OR id LIKE ?)"
        count_query += " AND (message LIKE ? OR location LIKE ? OR patient_id LIKE ? OR notes LIKE ? OR id LIKE ?)"
        params.extend([search_pattern, search_pattern, search_pattern, search_pattern, search_pattern])

    with get_db_connection() as conn:
        count_cursor = conn.execute(count_query, params)
        total_count = count_cursor.fetchone()["total"]

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?;"
        cursor = conn.execute(query, params + [limit, offset])
        rows = cursor.fetchall()
        return [row_to_dict(r) for r in rows], total_count


def update_incident(
    incident_id: str,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    acknowledged_at: Optional[str] = None,
    resolved_at: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    escalated: Optional[bool] = None,
    confidence: Optional[float] = None,
    fall_probability: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Updates fields on an existing incident."""
    existing = get_incident(incident_id)
    if not existing:
        return None

    updates = []
    params = []

    if status is not None:
        updates.append("status = ?")
        params.append(status.upper())
        if status.upper() == "ACKNOWLEDGED" and not existing.get("acknowledged_at") and not acknowledged_at:
            updates.append("acknowledged_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
        elif status.upper() == "RESOLVED" and not existing.get("resolved_at") and not resolved_at:
            updates.append("resolved_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())

    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)

    if acknowledged_at is not None:
        updates.append("acknowledged_at = ?")
        params.append(acknowledged_at)

    if resolved_at is not None:
        updates.append("resolved_at = ?")
        params.append(resolved_at)

    if duration_seconds is not None:
        updates.append("duration_seconds = ?")
        params.append(duration_seconds)

    if escalated is not None:
        updates.append("escalated = ?")
        params.append(1 if escalated else 0)

    if confidence is not None:
        updates.append("confidence = ?")
        params.append(confidence)

    if fall_probability is not None:
        updates.append("fall_probability = ?")
        params.append(fall_probability)

    if not updates:
        return existing

    params.append(incident_id)
    with get_db_connection() as conn:
        conn.execute(f"UPDATE incidents SET {', '.join(updates)} WHERE id = ?;", params)
        conn.commit()

    return get_incident(incident_id)


def delete_incident(incident_id: str) -> bool:
    """Deletes an incident by ID. Returns True if deleted, False if not found."""
    with get_db_connection() as conn:
        cursor = conn.execute("DELETE FROM incidents WHERE id = ?;", (incident_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_incidents() -> int:
    """Clears all incidents from database (for test resets). Returns count of deleted rows."""
    with get_db_connection() as conn:
        cursor = conn.execute("DELETE FROM incidents;")
        conn.commit()
        return cursor.rowcount


def get_analytics_summary() -> Dict[str, Any]:
    """Computes total, active, acknowledged, resolved, and 7-day trend metrics from real records."""
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status = 'ACKNOWLEDGED' THEN 1 ELSE 0 END) as acknowledged,
                SUM(CASE WHEN status = 'RESOLVED' THEN 1 ELSE 0 END) as resolved,
                AVG(CASE WHEN confidence IS NOT NULL THEN confidence ELSE NULL END) as avg_confidence,
                AVG(CASE WHEN fall_probability IS NOT NULL THEN fall_probability ELSE NULL END) as avg_fall_probability
            FROM incidents;
        """)
        summary_row = cursor.fetchone()

        # 7-day trend aggregation
        now = datetime.now(timezone.utc)
        days = []
        for i in range(6, -1, -1):
            day_dt = now - timedelta(days=i)
            day_str = day_dt.strftime("%Y-%m-%d")
            day_label = day_dt.strftime("%b %d")
            days.append((day_str, day_label))

        trend_query = """
            SELECT substr(created_at, 1, 10) as day, COUNT(*) as count
            FROM incidents
            WHERE created_at >= ?
            GROUP BY day;
        """
        seven_days_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        trend_cursor = conn.execute(trend_query, (seven_days_ago,))
        trend_map = {r["day"]: r["count"] for r in trend_cursor.fetchall()}

        daily_counts = [
            {"date": day_str, "label": day_label, "count": trend_map.get(day_str, 0)}
            for day_str, day_label in days
        ]

        total = summary_row["total"] or 0
        active = summary_row["active"] or 0
        acknowledged = summary_row["acknowledged"] or 0
        resolved = summary_row["resolved"] or 0
        avg_conf = round(summary_row["avg_confidence"], 3) if summary_row["avg_confidence"] else 0.0
        avg_fall_prob = round(summary_row["avg_fall_probability"], 3) if summary_row["avg_fall_probability"] else 0.0

        return {
            "total_incidents": total,
            "active_incidents": active,
            "acknowledged_incidents": acknowledged,
            "resolved_incidents": resolved,
            "average_confidence": avg_conf,
            "average_fall_probability": avg_fall_prob,
            "daily_trend": daily_counts,
        }


def export_incidents_to_csv(
    status_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Exports filtered incidents to a CSV string."""
    incidents, _ = list_incidents(
        status_filter=status_filter,
        start_date=start_date,
        end_date=end_date,
        limit=1000,
        offset=0,
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Incident ID",
        "Patient ID",
        "Location",
        "Severity",
        "Prediction Type",
        "Origin (Live/Demo)",
        "Status",
        "Confidence",
        "Fall Probability",
        "Duration (Seconds)",
        "Created At (UTC)",
        "Acknowledged At (UTC)",
        "Resolved At (UTC)",
        "Caregiver Notes",
        "Message",
    ])
    for inc in incidents:
        origin_str = "Demo Seed" if inc.get("is_demo") else "Live Telemetry"
        writer.writerow([
            inc.get("id"),
            inc.get("patient_id"),
            inc.get("location"),
            inc.get("severity"),
            inc.get("prediction_type"),
            origin_str,
            inc.get("status"),
            f"{inc.get('confidence', 0):.2f}" if inc.get("confidence") is not None else "N/A",
            f"{inc.get('fall_probability', 0):.2f}" if inc.get("fall_probability") is not None else "N/A",
            f"{inc.get('duration_seconds', 0):.1f}",
            inc.get("created_at"),
            inc.get("acknowledged_at") or "",
            inc.get("resolved_at") or "",
            inc.get("notes") or "",
            inc.get("message") or "",
        ])
    return output.getvalue()


# Automatically initialize schema on module import
init_db()
