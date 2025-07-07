"""
Frontend routes for serving the web UI.
"""
import os
from pathlib import Path
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional, List

# Get template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

frontend_router = APIRouter()

@frontend_router.get("/ui", response_class=HTMLResponse)
async def frontend_ui(request: Request):
    """Serve the main frontend UI page."""
    return templates.TemplateResponse("index.html", {"request": request})

@frontend_router.post("/upload_video")
async def upload_video(
    request: Request,
    video_file: UploadFile = File(...),
    model_size: str = Form("xs"),
    handedness: str = Form("Right-handed"),
    pitch_types: Optional[str] = Form("")
):
    """Handle video upload and processing via the frontend."""
    import sys
    from pathlib import Path
    
    # Add parent directories to Python path to import existing modules
    API_INFERENCE_DIR = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(API_INFERENCE_DIR))
    
    # Save uploaded file temporarily
    upload_dir = API_INFERENCE_DIR / "tmp_upload"
    upload_dir.mkdir(exist_ok=True)
    
    temp_file_path = upload_dir / video_file.filename
    
    try:
        # Save uploaded file
        content = await video_file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        return JSONResponse({
            "message": "Video uploaded successfully",
            "filename": str(temp_file_path),
            "model_size": model_size,
            "handedness": handedness,
            "pitch_types": pitch_types
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to upload video: {str(e)}"}
        )
