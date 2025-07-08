#!/usr/bin/env python3
"""
Startup script for the Shadow Trainer FastAPI application.
"""
import sys
from pathlib import Path
API_INFERENCE_DIR = Path(__file__).parent
sys.path.insert(0, str(API_INFERENCE_DIR))

from app.core.config import ensure_directories

if __name__ == "__main__":
    import uvicorn

    # Ensure all required directories exist
    ensure_directories()

    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
