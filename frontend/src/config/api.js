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
  alerts: `${API_BASE_URL}/alerts`,
};

/**
 * Checks connectivity and model status with the FastAPI backend.
 * Returns { online: boolean, status: string, modelLoaded?: boolean, latencyMs?: number }
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
 *
 * @param {Array<{x: number, y: number, z: number, visibility?: number}>} landmarks - 33 pose landmarks
 * @param {number|null} frameId - Optional frame sequence counter
 * @param {string} patientId - Patient identifier
 * @param {AbortSignal} signal - Optional abort signal for request cancellation
 * @returns {Promise<{
 *   model_loaded: boolean,
 *   fall_detected: boolean|null,
 *   confidence: number|null,
 *   fall_probability: number|null,
 *   features_extracted: number|null,
 *   message: string,
 *   status: string,
 *   timestamp: string
 * }>}
 */
export async function sendPosePrediction(landmarks, frameId = null, patientId = 'patient-001', signal = null) {
  if (!landmarks || landmarks.length !== 33) {
    throw new Error(`MediaPipe Pose requires exactly 33 landmarks, received ${landmarks ? landmarks.length : 0}`);
  }

  // Preserve normalized coordinates and visibility without calculating inconsistent features in frontend
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
