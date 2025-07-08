"""
Frontend routes for serving the Shadow Trainer web UI.
"""
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import logging

from ..services.video_service import VideoService
from ..core.utils.file_utils import FileManager
from ..config import get_settings
from ..schemas.requests import VideoProcessRequest
from ..schemas.common import ModelSize, Handedness, PitchType

# Setup logging
logger = logging.getLogger(__name__)

# Get template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize router
frontend_router = APIRouter()

# Dependencies
settings = get_settings()
file_manager = FileManager()


@frontend_router.get("/", response_class=HTMLResponse)
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
    try:
        # Validate file type
        if not video_file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload MP4, MOV, or AVI files."
            )
        
        # Save uploaded file
        temp_file_path = await file_manager.save_uploaded_file(video_file)
        logger.info(f"Video uploaded successfully: {temp_file_path}")
        
        # Initialize video service and process video
        video_service = VideoService()
        
        # Create request object
        request = VideoProcessRequest(
            file_path=str(temp_file_path),
            model_size=ModelSize(model_size),
            handedness=Handedness(handedness),
            pitch_types=[PitchType(pt) for pt in (pitch_types.split(",") if pitch_types else [])]
        )
        
        result = await video_service.process_video(request)
        
        # Extract filename from output path for download URL
        output_filename = Path(result.output_video_local_path).name if result.output_video_local_path else None
        
        return JSONResponse({
            "success": True,
            "message": "Video processed successfully",
            "result": result.dict(),
            "download_url": f"/frontend/download/{output_filename}" if output_filename else None
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to process video: {str(e)}"
            }
        )


@frontend_router.post("/process_s3_video")
async def process_s3_video(
    s3_path: str = Form(...),
    model_size: str = Form("xs"),
    handedness: str = Form("Right-handed"),
    pitch_types: Optional[str] = Form("")
):
    """Process a video from S3 path or asset path."""
    try:
        # Handle asset paths (sample videos)
        if s3_path.startswith('/assets/') or s3_path.startswith('/static/'):
            # Convert asset URL to actual file path
            if s3_path.startswith('/assets/'):
                actual_path = settings.assets_dir / s3_path[8:]  # Remove '/assets/' prefix
            else:  # /static/
                static_dir = Path(__file__).parent / "static"
                actual_path = static_dir / s3_path[8:]  # Remove '/static/' prefix
            
            video_path = str(actual_path)
            logger.info(f"Processing sample video: {video_path}")
        else:
            # Handle actual S3 paths or other external URLs
            video_path = s3_path
            logger.info(f"Processing S3/external video: {video_path}")
        
        # Initialize video service and process video
        video_service = VideoService()
        
        # Create request object
        request = VideoProcessRequest(
            file_path=video_path,
            model_size=ModelSize(model_size),
            handedness=Handedness(handedness),
            pitch_types=[PitchType(pt) for pt in (pitch_types.split(",") if pitch_types else [])]
        )
        
        result = await video_service.process_video(request)
        
        # Extract filename from output path for download URL
        output_filename = Path(result.output_video_local_path).name if result.output_video_local_path else None
        
        return JSONResponse({
            "success": True,
            "message": "Video processed successfully",
            "result": result.dict(),
            "download_url": f"/frontend/download/{output_filename}" if output_filename else None
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to process video: {str(e)}"
            }
        )


@frontend_router.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed video file."""
    try:
        output_dir = settings.output_dir
        file_path = output_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='video/mp4'
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download file")


@frontend_router.get("/sample_videos")
async def get_sample_videos():
    """Get list of sample videos available for testing."""
    try:
        # Check both potential locations for sample videos
        sample_videos_dir = settings.assets_dir / "videos"
        static_videos_dir = Path(__file__).parent / "static" / "videos"
        
        videos = []
        seen_videos = set()  # Track videos we've already added
        
        # Check assets videos directory first (preferred location)
        if sample_videos_dir.exists():
            for extension in ["*.mp4", "*.mov"]:
                for video_file in sample_videos_dir.glob(extension):
                    if video_file.name not in seen_videos:
                        videos.append({
                            "name": video_file.stem,
                            "filename": video_file.name,
                            "url": f"/assets/videos/{video_file.name}"
                        })
                        seen_videos.add(video_file.name)
        
        # Check static videos directory only if we don't have any videos yet
        if not videos and static_videos_dir.exists():
            for extension in ["*.mp4", "*.mov"]:
                for video_file in static_videos_dir.glob(extension):
                    if video_file.name not in seen_videos:
                        videos.append({
                            "name": video_file.stem,
                            "filename": video_file.name,
                            "url": f"/static/videos/{video_file.name}"
                        })
                        seen_videos.add(video_file.name)
        
        # If no videos found, add some placeholder entries
        if not videos:
            videos = [
                {
                    "name": "Sample Video 1",
                    "filename": "sample1.mp4",
                    "url": "/static/videos/placeholder.mp4",
                    "placeholder": True
                },
                {
                    "name": "Sample Video 2", 
                    "filename": "sample2.mp4",
                    "url": "/static/videos/placeholder.mp4",
                    "placeholder": True
                }
            ]
        
        return JSONResponse({"videos": videos})
        
    except Exception as e:
        logger.error(f"Error getting sample videos: {str(e)}")
        return JSONResponse({"videos": []})


@frontend_router.get("/status/{task_id}")
async def get_processing_status(task_id: str):
    """Get the status of a video processing task."""
    try:
        # This would integrate with a task queue system like Celery
        # For now, return a simple response
        return JSONResponse({
            "task_id": task_id,
            "status": "completed",
            "progress": 100
        })
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get task status"}
        )
