"""Notification service interface supporting demo/local mode and optional SMTP email dispatch."""

import os
import smtplib
import uuid
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from app.database import get_db_connection


class NotificationService:
    """
    Manages alert notifications for confirmed falls and escalation warnings.

    Operates in 'demo' (local simulation) mode by default.
    If SMTP credentials are provided, attempts real email dispatch while
    recording honest delivery or simulated statuses in SQLite logs.
    """

    def __init__(self):
        self.mode = os.getenv("NOTIFICATION_MODE", "demo").lower()
        self.smtp_host = os.getenv("SMTP_HOST", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_from = os.getenv("SMTP_FROM", "alerts@elderly-care.local")
        self.recipient_email = os.getenv("ALERT_RECIPIENT_EMAIL", "caregiver@elderly-care.local")

    @property
    def is_smtp_configured(self) -> bool:
        """Returns True only when required SMTP settings are populated."""
        return bool(self.smtp_host and self.smtp_user and self.smtp_password and self.recipient_email)

    def log_notification(
        self,
        incident_id: str,
        channel: str,
        recipient: str,
        status: str,
        details: str,
    ) -> None:
        """Stores a notification event in the database for caregiver audit trails."""
        log_id = f"notif-{uuid.uuid4().hex[:8]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO notification_logs (id, incident_id, channel, recipient, status, details, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (log_id, incident_id, channel, recipient, status, details, now_iso),
                )
                conn.commit()
        except Exception as e:
            print(f"[Notifications] Error logging notification: {e}")

    def send_fall_alert(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches a critical fall notification for a newly confirmed incident.
        Uses SMTP if configured, otherwise logs a demo notification.
        Protected against duplicate dispatches for the same incident ID.
        """
        incident_id = incident.get("id", "unknown")

        # Duplicate dispatch protection: verify no previous fall alert log exists for this incident
        try:
            with get_db_connection() as conn:
                cur = conn.execute(
                    "SELECT COUNT(*) as c FROM notification_logs WHERE incident_id = ? AND channel IN ('EMAIL_SMTP', 'DEMO_LOCAL');",
                    (incident_id,),
                )
                if cur.fetchone()["c"] > 0:
                    return {
                        "success": True,
                        "already_dispatched": True,
                        "channel": "DUPLICATE_SUPPRESSED",
                        "status": "ALREADY_DISPATCHED",
                        "details": f"Alert already logged for incident {incident_id}",
                    }
        except Exception:
            pass

        location = incident.get("location", "Living Area")
        confidence = incident.get("confidence")
        conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
        timestamp = incident.get("created_at") or incident.get("timestamp", "Just now")

        subject = f"URGENT: Fall Detected in {location} [{incident_id}]"
        body_text = (
            f"Smart Elderly Fall Detection Alert\n"
            f"------------------------------------\n"
            f"Incident ID: {incident_id}\n"
            f"Patient ID:  {incident.get('patient_id', 'patient-001')}\n"
            f"Location:    {location}\n"
            f"Severity:    {incident.get('severity', 'HIGH')}\n"
            f"Confidence:  {conf_str}\n"
            f"Timestamp:   {timestamp}\n\n"
            f"Message: {incident.get('message', 'High-confidence fall detected.')}\n\n"
            f"Caregiver Action Required: Please verify resident safety in dashboard immediately.\n"
        )

        if self.is_smtp_configured and self.mode != "demo":
            try:
                msg = MIMEMultipart()
                msg["From"] = self.smtp_from
                msg["To"] = self.recipient_email
                msg["Subject"] = subject
                msg.attach(MIMEText(body_text, "plain"))

                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=8) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)

                detail_msg = f"Email dispatched to {self.recipient_email}"
                self.log_notification(incident_id, "EMAIL_SMTP", self.recipient_email, "DELIVERED", detail_msg)
                return {"success": True, "channel": "EMAIL_SMTP", "status": "DELIVERED", "details": detail_msg}
            except Exception as exc:
                err_msg = f"SMTP dispatch failed: {exc}"
                print(f"[Notifications] {err_msg}")
                self.log_notification(incident_id, "EMAIL_SMTP", self.recipient_email, "FAILED", err_msg)
                return {"success": False, "channel": "EMAIL_SMTP", "status": "FAILED", "details": err_msg}

        # Demo / Local notification mode (honest reporting, no fake delivery claimed)
        demo_msg = (
            f"Demo notification logged locally — not delivered to external inbox (SMTP unconfigured). "
            f"Subject: '{subject}'"
        )
        print(f"[Notifications] {demo_msg}")
        self.log_notification(incident_id, "DEMO_LOCAL", self.recipient_email, "DEMO_LOGGED", demo_msg)
        return {
            "success": True,
            "channel": "DEMO_LOCAL",
            "status": "DEMO_LOGGED",
            "details": demo_msg,
        }

    def send_escalation_alert(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an escalation warning when an incident remains unacknowledged past threshold."""
        incident_id = incident.get("id", "unknown")

        # Duplicate escalation protection
        try:
            with get_db_connection() as conn:
                cur = conn.execute(
                    "SELECT COUNT(*) as c FROM notification_logs WHERE incident_id = ? AND channel = 'ESCALATION';",
                    (incident_id,),
                )
                if cur.fetchone()["c"] > 0:
                    return {
                        "success": True,
                        "already_dispatched": True,
                        "channel": "ESCALATION",
                        "status": "ALREADY_ESCALATED",
                        "details": f"Incident {incident_id} already escalated previously.",
                    }
        except Exception:
            pass

        duration = incident.get("duration_seconds", 60.0)
        detail_msg = f"Incident {incident_id} escalated: Unacknowledged after {duration:.0f}s."
        print(f"[Notifications] [ESCALATION] {detail_msg}")
        self.log_notification(incident_id, "ESCALATION", self.recipient_email, "ESCALATED", detail_msg)
        return {"success": True, "channel": "ESCALATION", "status": "ESCALATED", "details": detail_msg}

    def get_status(self) -> Dict[str, Any]:
        """Reports the live configuration status of the notification system."""
        return {
            "mode": self.mode,
            "smtp_configured": self.is_smtp_configured,
            "smtp_host": self.smtp_host or None,
            "recipient_email": self.recipient_email,
            "status_message": (
                "Real SMTP Email delivery active"
                if self.is_smtp_configured and self.mode != "demo"
                else "Demo / Local notification mode active (external email delivery disabled)"
            ),
        }

    def get_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent notification logs from the database."""
        try:
            with get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM notification_logs ORDER BY timestamp DESC LIMIT ?;",
                    (limit,),
                )
                return [dict(r) for r in cursor.fetchall()]
        except Exception:
            return []


# Global singleton instance
notification_service = NotificationService()
