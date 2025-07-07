"""
Main FastAPI application that serves both frontend UI and backend API endpoints.
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Get the app directory path
APP_DIR = Path(__file__).parent
API_INFERENCE_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "frontend" / "static"
TEMPLATES_DIR = APP_DIR / "frontend" / "templates"

# Add api_inference to path for imports
sys.path.insert(0, str(API_INFERENCE_DIR))

from app.api.routes import api_router
from app.frontend.routes import frontend_router
from app.core.config import ensure_directories

# Initialize FastAPI app
app = FastAPI(
    title="Shadow Trainer",
    description="Complete Shadow Trainer application with frontend UI and backend API for 3D human pose estimation from video",
    version="1.0.0"
)

# Ensure required directories exist
ensure_directories()

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include API routes with prefix
app.include_router(api_router, prefix="/api/v1", tags=["API"])

# Include frontend routes
app.include_router(frontend_router, tags=["Frontend"])

# Root endpoint redirects to frontend
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint that serves the main frontend page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ui", response_class=HTMLResponse)
async def frontend_ui_redirect(request: Request):
    """Alternative frontend endpoint."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
