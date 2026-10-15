/**
 * API Configuration
 * Configurable base URL supporting local FastAPI backend and deployed Render URL.
 */
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
).replace(/\/$/, '');

export const API_ENDPOINTS = {
  health: `${API_BASE_URL}/health`,
  predict: `${API_BASE_URL}/predict`,
  predictStatus: `${API_BASE_URL}/predict/status`,
  trackerConfig: `${API_BASE_URL}/predict/tracker-config`,
  trackerReset: `${API_BASE_URL}/predict/tracker/reset`,
  alerts: `${API_BASE_URL}/alerts`,
  alertsAnalytics: `${API_BASE_URL}/alerts/analytics/summary`,
  alertsExportCsv: `${API_BASE_URL}/alerts/export/csv`,
};

/**
 * Checks connectivity and model status with the FastAPI backend.
 * Returns { online: boolean, status: string, modelLoaded?: boolean, database?: object, notifications?: object, latencyMs?: number }
 */
export async function checkBackendHealth() {
  const start = performance.now();
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3500);

    const res = await fetch(API_ENDPOINTS.health, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    clearTimeout(timeoutId);

    const latencyMs = Math.round(performance.now() - start);

    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      return {
        online: true,
        status: data.status || 'healthy',
        modelLoaded: Boolean(data.model_loaded),
        version: data.version || '0.2.0',
        database: data.database || null,
        notifications: data.notifications || null,
        latencyMs,
      };
    }
    return { online: false, status: `HTTP ${res.status}`, latencyMs };
  } catch (err) {
    return {
      online: false,
      status: err.name === 'AbortError' ? 'timeout' : 'offline',
    };
  }
}

/**
 * Sends exactly 33 MediaPipe Pose landmarks to the FastAPI backend prediction endpoint.
 * Backend owns feature extraction (47 features) and Random Forest model inference.
 */
export async function sendPosePrediction(landmarks, frameId = null, patientId = 'patient-001', signal = null) {
  if (!landmarks || landmarks.length !== 33) {
    throw new Error(`MediaPipe Pose requires exactly 33 landmarks, received ${landmarks ? landmarks.length : 0}`);
  }

  const payload = {
    timestamp: new Date().toISOString(),
    landmarks: landmarks.map((lm) => ({
      x: Number(lm.x ?? 0),
      y: Number(lm.y ?? 0),
      z: Number(lm.z ?? 0),
      visibility: Math.max(0, Math.min(1, Number(lm.visibility ?? 1.0))),
    })),
    frame_id: frameId,
    patient_id: patientId,
  };

  const res = await fetch(API_ENDPOINTS.predict, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) {
    const errorText = await res.text().catch(() => '');
    throw new Error(`Backend error (${res.status}): ${errorText || res.statusText}`);
  }

  return await res.json();
}

/**
 * Fetches list of persistent incidents from the SQLite database.
 */
export async function fetchAlerts(params = {}) {
  const query = new URLSearchParams();
  if (params.status && params.status !== 'ALL') query.append('status', params.status);
  if (params.severity && params.severity !== 'ALL') query.append('severity', params.severity);
  if (params.search) query.append('search', params.search);
  if (params.limit) query.append('limit', params.limit);
  if (params.offset) query.append('offset', params.offset);

  const url = `${API_ENDPOINTS.alerts}${query.toString() ? `?${query.toString()}` : ''}`;
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    throw new Error(`Failed to fetch alerts: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Fetches 7-day incident analytics summary from SQLite backend.
 */
export async function fetchAlertAnalytics() {
  const res = await fetch(API_ENDPOINTS.alertsAnalytics, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch alert analytics: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Creates a new incident alert.
 */
export async function createAlert(payload) {
  const res = await fetch(API_ENDPOINTS.alerts, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Failed to create alert: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Acknowledges an active incident alert.
 */
export async function acknowledgeAlert(alertId, notes = null) {
  const res = await fetch(`${API_ENDPOINTS.alerts}/${alertId}/acknowledge`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ notes }),
  });
  if (!res.ok) {
    throw new Error(`Failed to acknowledge alert: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Resolves an incident alert.
 */
export async function resolveAlert(alertId, notes = null) {
  const res = await fetch(`${API_ENDPOINTS.alerts}/${alertId}/resolve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ notes }),
  });
  if (!res.ok) {
    throw new Error(`Failed to resolve alert: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Deletes an incident alert permanently.
 */
export async function deleteAlert(alertId) {
  const res = await fetch(`${API_ENDPOINTS.alerts}/${alertId}`, {
    method: 'DELETE',
  });
  if (!res.ok && res.status !== 204) {
    throw new Error(`Failed to delete alert: ${res.statusText}`);
  }
  return true;
}

/**
 * Gets the direct CSV export download URL.
 */
export function getAlertsExportCsvUrl(status = null) {
  if (status && status !== 'ALL') {
    return `${API_ENDPOINTS.alertsExportCsv}?status=${encodeURIComponent(status)}`;
  }
  return API_ENDPOINTS.alertsExportCsv;
}
