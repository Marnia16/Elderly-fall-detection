"""Cross-platform local runner for the FastAPI application."""

import os
import sys
from pathlib import Path
import uvicorn

# Ensure the backend directory is in the Python module search path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() in ("true", "1", "yes")

    print(f"============================================================")
    print(f"  Smart Elderly Fall Detection Backend API")
    print(f"  Running at: http://{host}:{port}")
    print(f"  API Docs:   http://{host}:{port}/docs")
    print(f"  Health:     http://{host}:{port}/api/health")
    print(f"============================================================")

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
