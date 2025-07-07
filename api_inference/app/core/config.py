"""
Core configuration and utilities for the Shadow Trainer application.
"""
import os
from pathlib import Path

# Application directories
APP_DIR = Path(__file__).parent.parent
API_INFERENCE_DIR = APP_DIR.parent
ROOT_DIR = API_INFERENCE_DIR.parent

# Static files and templates
STATIC_DIR = APP_DIR / "frontend" / "static"
TEMPLATES_DIR = APP_DIR / "frontend" / "templates"
VIDEOS_DIR = STATIC_DIR / "videos"

# API configuration
API_BASE_URL = os.environ.get("SHADOW_TRAINER_API_URL", "http://localhost:8000")
DEFAULT_MODEL_SIZE = "xs"

# Ensure directories exist
def ensure_directories():
    """Ensure all required directories exist."""
    VIDEOS_DIR.mkdir(exist_ok=True)
    (STATIC_DIR / "images").mkdir(exist_ok=True)
    (STATIC_DIR / "css").mkdir(exist_ok=True)
    (STATIC_DIR / "js").mkdir(exist_ok=True)
