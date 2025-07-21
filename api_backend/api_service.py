"""
Shadow Trainer API Service
Professional video processing API for pose estimation and motion analysis.
"""
import os
from pathlib import Path
import shutil
import uuid

from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from tasks import (
    celery_app, 
    process_video_task, 
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


@app.get("/test_add")
def test_add(x: int = Query(5), y: int = Query(1)):
    result = add_task.delay(x, y)
    return {"task_id": result.id}


@app.post("/videos/upload")
async def upload_and_process_video(
    file: UploadFile = File(...),
    model_size: str = Query("xs", description="Model size: xs, s, m, l"),
    is_lefty: bool = Query(False, description="Whether the user is left-handed"),
    pro_keypoints_filename: Optional[str] = Query(None, description="Professional keypoints filename from S3")
):
    """Upload video file and start processing task"""
    if not validate_video_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a video file (.mp4, .mov, .avi, .mkv)"
        )

    try:
        file_id = str(uuid.uuid4())
        input_path = save_uploaded_file(file, file_id)
    except Exception as e:
        raise e

    # Verify file exists and is readable before starting task
    if not os.path.exists(input_path):
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Additional verification - ensure file has content
    if os.stat(input_path).st_size == 0:
        raise HTTPException(status_code=500, detail="Uploaded file is empty")
    
    # Start processing task after confirming file is ready
    task = process_video_task.delay(
        str(input_path), 
        model_size=model_size, 
        is_lefty=is_lefty, 
        pro_keypoints_filename=pro_keypoints_filename
    )

    return {
        "task_id": task.id,
        "file_id": file_id,
        "original_filename": file.filename,
        "status": "processing"
    }



@app.get("/status/{task_id}")
def get_processing_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
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
    files = list_s3_pro_keypoints()
    return {"files": files}



# @app.get("/status/{task_id}")
# def get_processing_status(task_id: str):
#     result = AsyncResult(task_id, app=celery_app)
    
#     response = {
#         "task_id": task_id,
#         "status": result.status,
#         "progress": None,
#         "error": None
#     }
    
#     if result.ready():
#         if result.successful():
#             response["result"] = result.result
#             response["download_ready"] = True
#         else:
#             # Handle failed tasks properly
#             response["error"] = str(result.info) if result.info else "Unknown error"
#     else:
#         # Check for progress updates if your task supports it
#         if hasattr(result, 'info') and isinstance(result.info, dict):
#             response["progress"] = result.info.get('progress', 0)
    
#     return response



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
