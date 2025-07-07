#!/usr/bin/env python3
"""
Startup script for the Shadow Trainer FastAPI application.
"""
import sys
import os
from pathlib import Path

# Add the api_inference directory to Python path
API_INFERENCE_DIR = Path(__file__).parent
sys.path.insert(0, str(API_INFERENCE_DIR))

# Import and run the FastAPI app
from app.main import app
from app.core.config import ensure_directories

if __name__ == "__main__":
    import uvicorn
    
    # Ensure all required directories exist
    ensure_directories()
    
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
    # # Run the application
    # uvicorn.run(
    #     "app.main:app",
    #     host="0.0.0.0",
    #     port=8000,
    #     reload=True,
    #     log_level="info"
    # )
