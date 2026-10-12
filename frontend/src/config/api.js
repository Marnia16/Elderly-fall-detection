/**
 * API Configuration
 * Configurable base URL supporting local FastAPI backend and deployed Render URL.
 */
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
).replace(/\/$/, '');

export const API_ENDPOINTS = {
  health: `${API_BASE_URL}/health`,
  predictLandmarks: `${API_BASE_URL}/predict/landmarks`,
  predictFrame: `${API_BASE_URL}/predict/frame`,
  alerts: `${API_BASE_URL}/alerts`,
};

/**
 * Checks connectivity with the FastAPI backend.
 * Returns { online: boolean, status: string, latencyMs?: number }
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
