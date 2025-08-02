"""
Shadow Trainer API Service
Professional video processing API for pose estimation and motion analysis.
"""
import json
from pathlib import Path
import uuid

from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from tasks import generate_3d_keypoints_from_video_task
import numpy as np

from src.kpts_analysis import evaluate_all_joints_text
from constants import TMP_PRO_KEYPOINTS_FILE_S3, OUTPUT_DIR, SAMPLE_VIDEO_PATH
from tasks import (
    celery_app, 
    process_video_task,
    process_video_task_small, 
    save_uploaded_file, 
    validate_video_file, 
    list_s3_pro_keypoints, 
    add_task,
    generate_joint_evaluation_task,
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
    pro_keypoints_filename: str = Query(TMP_PRO_KEYPOINTS_FILE_S3, description="Professional keypoints filename from S3"),
    visualization_type: str = Query("combined", description="Output video format: combined (2D+3D), 3d_only, or dynamic_3d_animation")
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
        pro_keypoints_filename=pro_keypoints_filename,
        visualization_type=visualization_type
    )

    return {
        "task_id": task.id,
        "file_id": file_id,
        "original_filename": file.filename,
        "status": "queued"
    }



@app.post("/videos/get_3d_keypoints")
async def generate_3d_keypoints(
    file: UploadFile = File(...),
    model_size: str = Query("xs", description="Model size: xs, s, m, l")
):
    """ Upload a video and return the 3D keypoints as a downloadable .npy file. 
    This endpoint processes the uploaded video to extract 3D keypoints using a pre-trained model.
    """
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

    try:
        task = generate_3d_keypoints_from_video_task.delay(
            input_video_path=input_path,
            model_size=model_size
        )
    except Exception as e:
        logger.error(f"Error starting 3D keypoints task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start 3D keypoints task: {str(e)}")

    return {
        "task_id": task.id,
        "file_id": file_id,
        "original_filename": file.filename,
        "status": "queued"
    }


@app.get("/videos/{task_id}/3d_keypoints_download")
async def download_3d_keypoints(task_id: str):
    """
    Download the generated 3D keypoints .npy file from Celery task result.
    """
    result = AsyncResult(task_id, app=celery_app)
    if not result.ready() or not result.successful():
        raise HTTPException(status_code=400, detail="Job not completed yet")

    # The celery task returns a dict with the key "output_path"
    keypoints_path = None
    if isinstance(result.result, dict):
        keypoints_path = result.result.get("output_path")
    elif isinstance(result.result, str):
        keypoints_path = result.result

    if not keypoints_path or not Path(keypoints_path).exists():
        raise HTTPException(status_code=404, detail="3D keypoints file not found")
    filename = f"3d_keypoints_{Path(keypoints_path).stem}.npy"
    return FileResponse(
        path=keypoints_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.post("/videos/upload")
async def upload_and_process_video(
    file: UploadFile = File(...),
    model_size: str = Query("xs", description="Model size: xs, s, m, l"),
    is_lefty: bool = Query(False, description="Whether the user is left-handed"),
    pro_keypoints_filename: str = Query(TMP_PRO_KEYPOINTS_FILE_S3, description="Professional keypoints filename from S3"),
    visualization_type: str = Query("combined", description="Output video format: combined (2D+3D) or 3d_only")
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
            pro_keypoints_filename=pro_keypoints_filename,
            visualization_type=visualization_type
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


@app.post("/videos/sample-lefty")
async def process_sample_lefty_video(
    model_size: str = Query("xs", description="Model size: xs, s, m, l"),
    pro_keypoints_filename: str = Query(TMP_PRO_KEYPOINTS_FILE_S3, description="Professional keypoints filename from S3"),
    visualization_type: str = Query("combined", description="Output video format: combined (2D+3D) or 3d_only")
):
    """Process the sample lefty video with specified parameters"""
    
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
            pro_keypoints_filename=pro_keypoints_filename,
            visualization_type=visualization_type
        )
    except Exception as e:
        logger.error(f"Error starting sample video processing task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing task: {str(e)}")

    return {
        "task_id": task.id,
        "file_id": "sample_lefty",
        "original_filename": SAMPLE_VIDEO_PATH.name,
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
            response["message"] = result.info.get('message', None)
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


@app.get("/videos/{task_id}/preview/processed")
async def get_processed_video_preview(task_id: str):
    """
    Stream processed video for preview in browser from Celery task result.
    """
    result = AsyncResult(task_id, app=celery_app)
    if not result.ready() or not result.successful():
        raise HTTPException(status_code=400, detail="Job not completed yet")

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
            "Content-Disposition": "inline; filename=processed_preview.mp4"
        }
    )


@app.get("/videos/{task_id}/preview/original")
async def get_original_video_preview(task_id: str):
    """Stream original uploaded video for preview in browser."""
    result = AsyncResult(task_id, app=celery_app)
    
    # Get original video path from completed task result
    original_video_path = None
    if result.result and isinstance(result.result, dict) and 'original_video_path' in result.result:
        original_video_path = result.result['original_video_path']
    else:
        # For running/pending tasks, look in expected location: {task_id}_output/original.{ext}
        task_output_dir = OUTPUT_DIR / f"{task_id}_output"
        for ext in ['.mp4', '.mov', '.avi', '.mkv']:
            potential_path = task_output_dir / f"original{ext}"
            if potential_path.exists():
                original_video_path = str(potential_path)
                break
    
    if not original_video_path or not Path(original_video_path).exists():
        raise HTTPException(status_code=404, detail="Original video file not found")
    
    # Determine media type from file extension
    file_extension = Path(original_video_path).suffix.lower()
    media_type_map = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime', 
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska'
    }
    media_type = media_type_map.get(file_extension, 'video/mp4')
    
    return FileResponse(
        path=original_video_path,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename=original_preview{file_extension}"
        }
    )


# Keep the old endpoint for backward compatibility
@app.get("/videos/{task_id}/preview")
async def get_video_preview(task_id: str):
    """
    Stream video for preview in browser from Celery task result.
    (Backward compatibility - returns processed video)
    """
    return await get_processed_video_preview(task_id)



@app.get("/pro_keypoints/list")
async def list_pro_keypoints():
    files = list_s3_pro_keypoints()
    return {"files": files}


@app.get("/videos/{task_id}/keypoints/user")
async def get_user_keypoints(task_id: str, format: str = Query("npy", description="Output format: npy (default) or flattened for SkeletonViewer")):
    """Get user 3D keypoints data from processed video task"""    
    try:
        result = AsyncResult(task_id, app=celery_app)
        if not result.ready() or not result.successful():
            raise HTTPException(status_code=400, detail="Task not completed successfully")

        # Get the output directory from task result
        task_result = result.result
        if isinstance(task_result, dict) and 'output_dir' in task_result:
            output_dir = Path(task_result['output_dir'])
        elif isinstance(task_result, dict) and 'output_path' in task_result:
            # Fallback: infer output directory from output_path for older tasks
            output_path = Path(task_result['output_path'])
            output_dir = output_path.parent  # e.g., /path/to/taskid_output/file.mp4 -> /path/to/taskid_output
        else:
            raise HTTPException(status_code=404, detail="Task output directory not found")

        # Look for user 3D keypoints file in raw_keypoints subdirectory
        user_keypoints_path = output_dir / 'raw_keypoints' / 'user_3D_keypoints.npy'
        
        if not user_keypoints_path.exists():
            raise HTTPException(status_code=404, detail="User 3D keypoints file not found")

        # Load and return the keypoints data
        keypoints_data = np.load(user_keypoints_path)
        
        if format == "flattened":
            # Check if data needs to be flattened for SkeletonViewer component
            if keypoints_data.ndim == 3 and keypoints_data.shape[1:] == (17, 3):
                # Flatten each frame to a 1D list of 51 floats (same as npy_to_json.py)
                keypoints_list = keypoints_data.reshape((keypoints_data.shape[0], -1)).tolist()
            else:
                raise HTTPException(status_code=400, detail=f"Expected shape (nframes, 17, 3) for flattened format, got {keypoints_data.shape}")
        else:
            keypoints_list = keypoints_data.tolist()  # Convert numpy array to Python list for JSON

        return {
            "task_id": task_id,
            "keypoints": keypoints_list,
            "shape": keypoints_data.shape,
            "dtype": str(keypoints_data.dtype),
            "format": format
        }

    except Exception as e:
        logger.error(f"Error getting user keypoints for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user keypoints: {str(e)}")


@app.get("/videos/{task_id}/info")
async def get_task_info(task_id: str):
    """Get task info including pro name from info.json file"""
    
    try:
        result = AsyncResult(task_id, app=celery_app)
        if not result.ready() or not result.successful():
            raise HTTPException(status_code=400, detail="Task not completed successfully")

        # Get the output directory from task result
        task_result = result.result
        if isinstance(task_result, dict) and 'output_dir' in task_result:
            output_dir = Path(task_result['output_dir'])
        elif isinstance(task_result, dict) and 'output_path' in task_result:
            # Fallback: infer output directory from output_path for older tasks
            output_path = Path(task_result['output_path'])
            output_dir = output_path.parent
        else:
            raise HTTPException(status_code=404, detail="Task output directory not found")

        # Look for info.json file
        info_file_path = output_dir / 'info.json'
        
        if not info_file_path.exists():
            # Return default info if file doesn't exist (for backwards compatibility)
            # For existing tasks, default to right-handed (is_lefty = False)
            return {
                "task_id": task_id,
                "pro_name": "Professional Pitcher",
                "is_lefty": False
            }

        # Load and return the info data
        with open(info_file_path, 'r') as f:
            info_data = json.load(f)
        
        return {
            "task_id": task_id,
            **info_data
        }

    except Exception as e:
        logger.error(f"Error getting task info for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task info: {str(e)}")


@app.get("/videos/{task_id}/keypoints/pro")
async def get_pro_keypoints(task_id: str, format: str = Query("npy", description="Output format: npy (default) or flattened for SkeletonViewer")):
    """Get professional 3D keypoints data from processed video task"""    
    try:
        result = AsyncResult(task_id, app=celery_app)
        if not result.ready() or not result.successful():
            raise HTTPException(status_code=400, detail="Task not completed successfully")

        # Get the output directory from task result
        task_result = result.result
        if isinstance(task_result, dict) and 'output_dir' in task_result:
            output_dir = Path(task_result['output_dir'])
        elif isinstance(task_result, dict) and 'output_path' in task_result:
            # Fallback: infer output directory from output_path for older tasks
            output_path = Path(task_result['output_path'])
            output_dir = output_path.parent  # e.g., /path/to/taskid_output/file.mp4 -> /path/to/taskid_output
        else:
            raise HTTPException(status_code=404, detail="Task output directory not found")

        # Look for pro 3D keypoints file in raw_keypoints subdirectory
        pro_keypoints_path = output_dir / 'raw_keypoints' / 'pro_3D_keypoints.npy'
        
        if not pro_keypoints_path.exists():
            raise HTTPException(status_code=404, detail="Professional 3D keypoints file not found")

        # Load and return the keypoints data
        keypoints_data = np.load(pro_keypoints_path)
        
        if format == "flattened":
            # Check if data needs to be flattened for SkeletonViewer component
            if keypoints_data.ndim == 3 and keypoints_data.shape[1:] == (17, 3):
                # Flatten each frame to a 1D list of 51 floats (same as npy_to_json.py)
                keypoints_list = keypoints_data.reshape((keypoints_data.shape[0], -1)).tolist()
            else:
                raise HTTPException(status_code=400, detail=f"Expected shape (nframes, 17, 3) for flattened format, got {keypoints_data.shape}")
        else:
            keypoints_list = keypoints_data.tolist()  # Convert numpy array to Python list for JSON

        return {
            "task_id": task_id,
            "keypoints": keypoints_list,
            "shape": keypoints_data.shape,
            "dtype": str(keypoints_data.dtype),
            "format": format
        }

    except Exception as e:
        logger.error(f"Error getting pro keypoints for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pro keypoints: {str(e)}")


@app.get("/videos/{task_id}/joint_evaluation")
async def get_joint_evaluation(task_id: str):
    """Get joint evaluation text from processed video task"""
    try:
        result = AsyncResult(task_id, app=celery_app)
        if not result.ready() or not result.successful():
            raise HTTPException(status_code=400, detail="Task not completed successfully")

        # Get the output directory from task result
        task_result = result.result
        if isinstance(task_result, dict) and 'output_dir' in task_result:
            output_dir = Path(task_result['output_dir'])
        elif isinstance(task_result, dict) and 'output_path' in task_result:
            # Fallback: infer output directory from output_path for older tasks
            output_path = Path(task_result['output_path'])
            output_dir = output_path.parent
        else:
            raise HTTPException(status_code=404, detail="Task output directory not found")

        # Look for keypoints files
        user_keypoints_path = output_dir / 'raw_keypoints' / 'user_3D_keypoints.npy'
        pro_keypoints_path = output_dir / 'raw_keypoints' / 'pro_3D_keypoints.npy'
        
        if not user_keypoints_path.exists() or not pro_keypoints_path.exists():
            raise HTTPException(status_code=404, detail="Keypoints files not found")

        # Load keypoints data
        user_kps = np.load(user_keypoints_path)
        pro_kps = np.load(pro_keypoints_path)
        
        # Run joint evaluation and get text
        joint_text = evaluate_all_joints_text(user_kps, pro_kps)
        
        return {
            "task_id": task_id,
            "joint_evaluation_text": joint_text
        }

    except Exception as e:
        logger.error(f"Error getting joint evaluation for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get joint evaluation: {str(e)}")


@app.post("/videos/{task_id}/generate-evaluation")
async def generate_evaluation(task_id: str):
    """Generate joint evaluation analysis for a processed video task"""
    try:
        # Start the joint evaluation task
        evaluation_task = generate_joint_evaluation_task.delay(task_id)
        
        return {
            "evaluation_task_id": evaluation_task.id,
            "original_task_id": task_id,
            "status": "queued",
            "message": "Joint evaluation generation started"
        }
        
    except Exception as e:
        logger.error(f"Error starting joint evaluation for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start joint evaluation: {str(e)}")


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
