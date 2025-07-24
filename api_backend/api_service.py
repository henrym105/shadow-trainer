"""
Shadow Trainer API Service
Professional video processing API for pose estimation and motion analysis.
"""
from pathlib import Path
import uuid

from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from constants import TMP_PRO_KEYPOINTS_FILE, TMP_PRO_KEYPOINTS_FILE_S3
from tasks import (
    celery_app, 
    process_video_task,
    process_video_task_small, 
    save_uploaded_file, 
    validate_video_file, 
    list_s3_pro_keypoints, 
    add_task,
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


@app.get("/test_add")
def test_add(x: int = Query(5), y: int = Query(1)):
    result = add_task.delay(x, y)
    return {"task_id": result.id}


@app.post("/videos/upload_test")
async def upload_and_process_video(
    file: UploadFile = File(...),
    model_size: str = Query("xs", description="Model size: xs, s, m, l"),
    is_lefty: bool = Query(False, description="Whether the user is left-handed"),
    pro_keypoints_filename: str = Query(TMP_PRO_KEYPOINTS_FILE_S3, description="Professional keypoints filename from S3")
):
    """Upload video file and start processing task"""
    if not validate_video_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a video file (.mp4, .mov, .avi, .mkv)"
        )

    file_id = str(uuid.uuid4())
    try:
        input_path = save_uploaded_file(file, file_id)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Start processing task after confirming file is ready
    task = process_video_task_small.delay(
        input_video_path=input_path,
        model_size=model_size,
        is_lefty=is_lefty,
        pro_keypoints_filename=pro_keypoints_filename
    )

    return {
        "task_id": task.id,
        "file_id": file_id,
        "original_filename": file.filename,
        "status": "queued"
    }


@app.post("/videos/upload")
async def upload_and_process_video(
    file: UploadFile = File(...),
    model_size: str = Query("xs", description="Model size: xs, s, m, l"),
    is_lefty: bool = Query(False, description="Whether the user is left-handed"),
    pro_keypoints_filename: str = Query(TMP_PRO_KEYPOINTS_FILE_S3, description="Professional keypoints filename from S3")
):
    """Upload video file and start processing task"""
    if not validate_video_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a video file (.mp4, .mov, .avi, .mkv)"
        )

    file_id = str(uuid.uuid4())
    try:
        input_path = save_uploaded_file(file, file_id)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Start processing task after confirming file is ready
    try:
        task = process_video_task.delay(
            input_video_path=input_path,
            model_size=model_size,
            is_lefty=is_lefty,
            pro_keypoints_filename=pro_keypoints_filename
        )
    except Exception as e:
        logger.error(f"Error starting Celery task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing task: {str(e)}")

    return {
        "task_id": task.id,
        "file_id": file_id,
        "original_filename": file.filename,
        "status": "queued"
    }



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


@app.get("/videos/{task_id}/status")
def get_video_task_status(task_id: str):
    """
    Get processing status for a Celery video processing task.
    """
    result = AsyncResult(id=task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": result.status,
        "progress": None,
        "error": None,
        "result_url": None
    }
    # Celery status harmonization
    if result.ready():
        if result.successful():
            response["status"] = "completed"
            response["result"] = result.result
            response["download_ready"] = True
            output_path = result.result if isinstance(result.result, str) else None
            if output_path and Path(output_path).exists():
                response["result_url"] = f"/videos/{task_id}/download"
        else:
            response["status"] = "failed"
            response["error"] = str(result.info) if result.info else "Unknown error"
    else:
        # Celery status: PENDING, STARTED, PROGRESS
        if hasattr(result, 'info') and isinstance(result.info, dict):
            response["progress"] = result.info.get('progress', 0)
        if result.status in ["PROGRESS", "STARTED", "PENDING"]:
            response["status"] = "processing"
    return response


@app.get("/videos/{task_id}/download")
async def download_processed_video(task_id: str):
    """
    Download the processed video file from Celery task result.
    """
    result = AsyncResult(task_id, app=celery_app)
    if not result.ready() or not result.successful():
        raise HTTPException(status_code=400, detail="Job not completed yet")

    output_path = result.result['output_path'] if 'output_path' in result.result else None
    logger.info(f"{output_path = }")

    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    filename = f"processed_{Path(output_path).name}"
    return FileResponse(
        path=output_path,
        filename=filename,
        media_type="video/mp4"
    )


@app.get("/videos/{task_id}/preview")
async def get_video_preview(task_id: str):
    """
    Stream video for preview in browser from Celery task result.
    """
    result = AsyncResult(task_id, app=celery_app)
    if not result.ready() or not result.successful():
        raise HTTPException(status_code=400, detail="Job not completed yet")

    # logger.info(f"Result of task {task_id}: {result.__dict__ = }")
    # logger.info(f"result: {result = }")
    # logger.info(f"{result['output_path'] = }")
    output_path = result.result['output_path'] if 'output_path' in result.result else None

    logger.info(f"{output_path = }")

    if not output_path or not Path(output_path).exists():
        logger.info(f"Output Path not found at result.result.output_path.")
        logger.info(f"Result of task {task_id}:")
        logger.info(f"\t{result.__dict__ = }")
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": "inline; filename=preview.mp4"
        }
    )



@app.post("/videos/sample-lefty")
async def process_sample_lefty_video(
    model_size: str = Query("xs", description="Model size: xs, s, m, l"),
    pro_keypoints_filename: str = Query(TMP_PRO_KEYPOINTS_FILE_S3, description="Professional keypoints filename from S3")
):
    """Process the sample lefty video with specified parameters"""
    from constants import SAMPLE_VIDEO_PATH
    
    if not Path(SAMPLE_VIDEO_PATH).exists():
        raise HTTPException(
            status_code=404, 
            detail="Sample video not found"
        )

    try:
        # Start processing task with sample video
        task = process_video_task.delay(
            input_video_path=str(SAMPLE_VIDEO_PATH),
            model_size=model_size,
            is_lefty=True,  # Sample is specifically for lefty
            pro_keypoints_filename=pro_keypoints_filename
        )
    except Exception as e:
        logger.error(f"Error starting sample video processing task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing task: {str(e)}")

    return {
        "task_id": task.id,
        "file_id": "sample_lefty",
        "original_filename": "Left_Hand_Friend_Side.MOV",
        "status": "queued"
    }


@app.get("/pro_keypoints/list")
async def list_pro_keypoints():
    files = list_s3_pro_keypoints()
    return {"files": files}


@app.post("/videos/{task_id}/terminate")
async def terminate_video_task(task_id: str):
    """Terminate a running video processing task"""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if result.state in ['SUCCESS', 'FAILURE']:
            return {
                "task_id": task_id,
                "status": "terminated",
                "message": "Task has been terminated"
            }
        
        # Revoke (terminate) the task
        celery_app.control.revoke(task_id, terminate=True, signal='SIGKILL')
        
        logger.info(f"Task {task_id} has been terminated by user request")
        
        return {
            "task_id": task_id,
            "status": "terminated",
            "message": "Task has been terminated"
        }
        
    except Exception as e:
        logger.error(f"Error terminating task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to terminate task: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
