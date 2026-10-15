"""FastAPI Application entrypoint with CORS configuration and route registration."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.alerts import router as alerts_router
from app.routes.health import router as health_router
from app.routes.predict import router as predict_router

app = FastAPI(
    title="Smart Elderly Fall Detection & Alert System API",
    description=(
        "Backend REST API for real-time elderly fall detection telemetry, "
        "incident alerting, and caregiver monitoring. "
        "Built with FastAPI, designed for integration with a Random Forest ML model."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS Configuration ---
# Configured for local React + Vite development and cloud production (Vercel/Render)
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

cors_env = os.getenv("CORS_ORIGINS", "")
if cors_env.strip() == "*":
    origins = ["*"]
    allow_credentials = False
elif cors_env.strip():
    origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()]
    allow_credentials = True
else:
    origins = default_origins
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.vercel\.app",
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Route Inclusions ---
# Include routers with standard /api prefix
app.include_router(health_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")

# Also include health_router at root level for deployment health checks (e.g. Render /health)
app.include_router(health_router)


@app.get("/", summary="API Root Index")
async def root():
    """Root endpoint providing service overview and documentation links."""
    return {
        "service": "Smart Elderly Fall Detection & Alert System API",
        "version": "0.1.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/api/health",
        "predict_endpoint": "/api/predict",
        "alerts_endpoint": "/api/alerts",
    }
