"""
Shadow Trainer API Service
Professional video processing API for pose estimation and motion analysis.
"""
from pathlib import Path

from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from api_videos import sub_app_videos
from tasks import (
    celery_app, 
    add_task,
    list_s3_pro_keypoints,
)

logger = get_task_logger(__name__)

# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(
    title="Shadow Trainer API",
    description="AI-Powered Motion Analysis for Athletic Performance",
    version="2.0.0"
)
app.include_router(sub_app_videos, prefix="/videos", tags=["Videos"])

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Video Processing API. Use /upload-and-process/ to upload a video."}


@app.get("/health")
def health_check():
    try:
        # Check Redis connection via Celery
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        redis_healthy = stats is not None
        
        return {
            "status": "healthy",
            "service": "shadow-trainer-api",
            "redis_connection": "ok" if redis_healthy else "error",
            "celery_workers": len(stats) if stats else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail={
            "status": "unhealthy",
            "service": "shadow-trainer-api",
            "error": str(e)
        })


@app.get("/status/{task_id}")
def get_processing_status(task_id: str):
    result = AsyncResult(id=task_id, app=celery_app)
    response = {
        "task_id": result.task_id,
        "status": result.status,
        "progress": None,
        "error": None
    }
    
    if result.ready():
        if result.successful():
            response["result"] = result.result
            response["download_ready"] = True
        else:
            # Handle failed tasks properly
            response["error"] = str(result.info) if result.info else "Unknown error"
    else:
        # Check for progress updates if your task supports it
        if hasattr(result, 'info') and isinstance(result.info, dict):
            response["progress"] = result.info.get('progress', 0)
    
    return response



@app.get("/pro_keypoints/list")
async def list_pro_keypoints():
    """List all professional keypoints files available in S3. Used for dropdown menu in frontend."""
    files = list_s3_pro_keypoints()
    return {"files": files}    


@app.get("/test_add")
def test_add(x: int = Query(5), y: int = Query(1)):
    result = add_task.delay(x, y)
    return {"task_id": result.id}



@app.get("/files/{task_id}/download")
async def download_file(task_id: str):
    """
    Generalized download endpoint that can handle any file type based on task result.
    """
    result = AsyncResult(task_id, app=celery_app)
    if not result.ready() or not result.successful():
        raise HTTPException(status_code=400, detail="Job not completed yet")

    # Handle different result formats
    file_path = None
    if isinstance(result.result, dict):
        file_path = result.result.get("output_path")
    elif isinstance(result.result, str):
        file_path = result.result

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Determine media type based on file extension
    file_extension = Path(file_path).suffix.lower()
    media_type_map = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska',
        '.npy': 'application/octet-stream',
        '.json': 'application/json',
        '.txt': 'text/plain',
        '.csv': 'text/csv'
    }
    
    media_type = media_type_map.get(file_extension, 'application/octet-stream')
    filename = Path(file_path).name
    
    logger.info(f"Serving file: {file_path} with media type: {media_type}")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
