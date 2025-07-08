"""
Video processing endpoints for Shadow Trainer API.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any

from shadow_trainer.schemas.requests import VideoProcessRequest
from shadow_trainer.schemas.responses import VideoProcessResponse
from shadow_trainer.schemas.common import ModelSize, Handedness, PitchType
from shadow_trainer.services.video_service import VideoService
from shadow_trainer.api.dependencies import video_service_dependency

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/process", response_model=VideoProcessResponse)
async def process_video(
    request: VideoProcessRequest,
    video_service: VideoService = Depends(video_service_dependency)
):
    """
    Process a video file from a given path (local or S3).
    
    Args:
        request: Video processing request with file path and parameters
        video_service: Injected video service
        
    Returns:
        Video processing response with results
    """
    try:
        logger.info(f"Processing video request: {request.dict()}")
        result = await video_service.process_video(request)
        return result
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@router.post("/upload-and-process", response_model=VideoProcessResponse)
async def upload_and_process_video(
    video_file: UploadFile = File(..., description="Video file to process"),
    model_size: ModelSize = Form(ModelSize.XS, description="Model size for processing"),
    handedness: Handedness = Form(Handedness.RIGHT, description="User's dominant hand"),
    pitch_types: str = Form("", description="Comma-separated list of pitch types"),
    video_service: VideoService = Depends(video_service_dependency)
):
    """
    Upload and process a video file in one request.
    
    Args:
        video_file: Uploaded video file
        model_size: Model size to use for processing
        handedness: User's dominant hand
        pitch_types: Comma-separated pitch types
        video_service: Injected video service
        
    Returns:
        Video processing response with results
    """
    try:
        # Validate file type
        if not video_file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Parse pitch types
        pitch_types_list = []
        if pitch_types.strip():
            for pt in pitch_types.split(","):
                pt = pt.strip().upper()
                if pt in [e.value for e in PitchType]:
                    pitch_types_list.append(PitchType(pt))
        
        # Save uploaded file
        logger.info(f"Uploading file: {video_file.filename}")
        file_path = await video_service.save_uploaded_file(video_file)
        
        # Create processing request
        request = VideoProcessRequest(
            file_path=str(file_path),
            model_size=model_size,
            handedness=handedness,
            pitch_types=pitch_types_list
        )
        
        # Process video
        result = await video_service.process_video(request)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and process error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload and processing failed: {str(e)}")

@router.get("/sample-videos")
async def get_sample_videos(
    video_service: VideoService = Depends(video_service_dependency)
) -> Dict[str, Any]:
    """
    Get list of available sample videos.
    
    Returns:
        List of sample videos with metadata
    """
    try:
        return video_service.get_sample_videos()
    except Exception as e:
        logger.error(f"Error getting sample videos: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sample videos: {str(e)}")

@router.post("/cleanup")
async def cleanup_temp_files(
    max_age_hours: int = 24,
    video_service: VideoService = Depends(video_service_dependency)
) -> Dict[str, Any]:
    """
    Clean up old temporary files.
    
    Args:
        max_age_hours: Maximum age of files to keep in hours
        
    Returns:
        Cleanup results
    """
    try:
        if max_age_hours < 1:
            raise HTTPException(status_code=400, detail="max_age_hours must be at least 1")
        
        return video_service.cleanup_old_files(max_age_hours)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, Any]:
    """Get information about supported video formats and limits."""
    from shadow_trainer.config import settings
    from shadow_trainer.core.utils.file_utils import FileUtils
    
    return {
        "supported_extensions": settings.supported_video_extensions,
        "max_file_size_bytes": settings.max_file_size,
        "max_file_size_formatted": FileUtils.format_file_size(settings.max_file_size),
        "max_processing_time_seconds": settings.max_processing_time,
        "supported_model_sizes": settings.supported_model_sizes,
        "default_model_size": settings.default_model_size
    }
