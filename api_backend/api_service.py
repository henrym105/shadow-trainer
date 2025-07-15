"""
Shadow Trainer API Service
Professional video processing API for pose estimation and motion analysis.
"""
import logging
import uuid
import shutil
from pathlib import Path
from typing import Optional
import gc
import json
import os
import time
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from pydantic_models import (
    VideoUploadResponse, 
    ProcessingStatusResponse, 
    JobStatus,
)
from job_manager import job_manager
from src.inference import (
    create_3d_pose_images_from_array,
    generate_output_combined_frames, 
    get_pose2D, 
    get_pose3D_no_vis, 
    img2video, 
    get_pytorch_device,
    create_2D_images,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_ROOT_DIR = Path(__file__).parent.absolute()
TMP_DIR = API_ROOT_DIR / "tmp_api_output"
TMP_DIR.mkdir(exist_ok=True)

# S3 config for pro keypoints
S3_BUCKET = "shadow-trainer-dev"
S3_PRO_PREFIX = "test/professional/"

TMP_PRO_KEYPOINTS_FILE = API_ROOT_DIR / "checkpoint" / "example_SnellBlake.npy"
# ==================== UTILITY FUNCTIONS ====================

def list_s3_pro_keypoints():
    """List available professional keypoints files in S3."""
    import boto3
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PRO_PREFIX)
    files = [
        obj["Key"].replace(S3_PRO_PREFIX, "")
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".npy")
    ]
    return files

def download_pro_keypoints_from_s3(filename, dest_path):
    import boto3
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_PRO_PREFIX + filename, str(dest_path))

# FastAPI app
app = FastAPI(
    title="Shadow Trainer API",
    description="AI-Powered Motion Analysis for Athletic Performance",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output videos
app.mount("/output", StaticFiles(directory=str(TMP_DIR)), name="output")


# ==================== UTILITY FUNCTIONS ====================

def validate_video_file(file: UploadFile) -> bool:
    """Validate uploaded video file"""
    if not file.filename:
        return False
    
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
    file_ext = Path(file.filename).suffix.lower()
    
    return file_ext in allowed_extensions


def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to temporary directory"""
    # Create unique filename
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix.lower()
    temp_filename = f"{file_id}{file_ext}"
    temp_filepath = TMP_DIR / temp_filename
    
    # Save file
    with open(temp_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"Saved uploaded file: {temp_filepath}")
    return str(temp_filepath)


def get_model_config_path(model_size: str = "xs") -> str:
    """Load model configuration from config file"""
    config_path = API_ROOT_DIR / "model_config_map.json"
    
    try:
        with open(config_path, 'r') as f:
            config_map = json.load(f)
            config_yaml_path = config_map.get(model_size, "")
    except FileNotFoundError:
        logger.warning(f"Model config file not found: {config_path}")
        return "src/configs/h36m/MotionAGFormer-small.yaml"

    # Ensure absolute path is returned for YAML config
    if not os.path.isabs(config_yaml_path):
        config_yaml_path = str(API_ROOT_DIR / config_yaml_path)

    return config_yaml_path
        


def cleanup_old_files(retention_minutes: int = 60):
    """Clean up old temporary files (older than retention_minutes)"""
    current_time = time.time()
    file_retention_cutoff_time = current_time - (retention_minutes * 60)
    for item in TMP_DIR.iterdir():
        is_past_retention = (item.stat().st_mtime < file_retention_cutoff_time)
        if item.is_file() and is_past_retention:
            try:
                item.unlink()
                logger.info(f"Cleaned up old file: {item}")
            except OSError as e:
                logger.warning(f"Failed to clean up file {item}: {e}")
        if item.is_dir() and is_past_retention:
            try:
                shutil.rmtree(item)
                logger.info(f"Cleaned up old directory: {item}")
            except OSError as e:
                logger.warning(f"Failed to clean up directory {item}: {e}")


# ==================== VIDEO PROCESSING ====================

def process_video_pipeline(job_id: str, input_video_path: str, model_size: str = "xs", is_lefty: bool = False, pro_keypoints_filename: Optional[str] = None) -> str:
    """
    Process video with pose estimation and keypoint overlays
    
    Args:
        job_id: Unique job identifier
        input_video_path: Path to input video file
        model_size: Model size to use for processing
        is_lefty: Whether the user is left-handed
    
    Returns:
        Path to output video file
    """
    # Establish output directory constants for this job
    DIR_OUTPUT_BASE = TMP_DIR / f"{job_id}_output"

    DIR_POSE2D = DIR_OUTPUT_BASE / "pose2D"
    DIR_POSE3D = DIR_OUTPUT_BASE / "pose3D"
    DIR_COMBINED_FRAMES = DIR_OUTPUT_BASE / "combined_frames"
    DIR_KEYPOINTS = DIR_OUTPUT_BASE / "raw_keypoints"

    FILE_POSE2D = DIR_KEYPOINTS / "2D_keypoints.npy"
    FILE_POSE3D = DIR_KEYPOINTS / "user_3D_keypoints.npy"
    FILE_POSE3D_PRO = DIR_KEYPOINTS / "pro_3D_keypoints.npy"

    DIR_OUTPUT_BASE.mkdir(exist_ok=True)
    DIR_POSE2D.mkdir(exist_ok=True)
    DIR_POSE3D.mkdir(exist_ok=True)
    DIR_COMBINED_FRAMES.mkdir(exist_ok=True)
    DIR_KEYPOINTS.mkdir(exist_ok=True)

    try:
        logger.info(f"Starting video processing for job {job_id}")
        logger.info(f"User handedness preference: {'Left-handed' if is_lefty else 'Right-handed'}")
        cleanup_old_files(retention_minutes = 20)
        
        # Get device for processing
        device = get_pytorch_device()

        # Update job status to processing
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=5)
        
        # Load model configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        model_config_path = get_model_config_path(model_size)
        logger.info(f"Using model config: {model_config_path}")

        # Step 1: Extract 2D poses (20% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=20, message="Extracting 2D poses...")
        get_pose2D(video_path=input_video_path, output_file=FILE_POSE2D, device=device)

        # Step 2: Create 2D visualization frames (35% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=35, message="Creating 2D visualization frames...")
        cap = cv2.VideoCapture(input_video_path)
        keypoints_2d = np.load(FILE_POSE2D)
        create_2D_images(cap, keypoints_2d, DIR_POSE2D, is_lefty)
        cap.release()

        # Step 3: Generate 3D poses (50% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=50, message="Generating 3D poses...")
        get_pose3D_no_vis(
            user_2d_kpts_filepath = FILE_POSE2D,
            output_keypoints_path = FILE_POSE3D,
            video_path=input_video_path,
            device=device,
            model_size=model_size,
            yaml_path=model_config_path
        )

        # Step 4: Download pro keypoints if specified
        pro_keypoints_path = TMP_PRO_KEYPOINTS_FILE
        if pro_keypoints_filename:
            logger.info(f"Downloading pro keypoints file from S3: {pro_keypoints_filename}")
            download_pro_keypoints_from_s3(pro_keypoints_filename, FILE_POSE3D_PRO)
            pro_keypoints_path = FILE_POSE3D_PRO

        # Step 4: Create 3D visualization frames (70% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=70, message="Creating visualization frames...")
        create_3d_pose_images_from_array(
            user_3d_keypoints_filepath = FILE_POSE3D,
            output_dir = DIR_POSE3D,
            pro_keypoints_filepath = pro_keypoints_path,
            is_lefty = is_lefty
        )

        # Step 5: Generate combined frames (85% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=85, message="Combining frames with original video...")
        generate_output_combined_frames(
            output_dir_2D=DIR_POSE2D,
            output_dir_3D=DIR_POSE3D,
            output_dir_combined=DIR_COMBINED_FRAMES
        )

        # Step 6: Create final video (95% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=95, message="Generating final video...")
        output_video_path = img2video(
            video_path=input_video_path,
            input_frames_dir=DIR_COMBINED_FRAMES
        )

        # Complete job
        job_manager.update_job_status(
            job_id, JobStatus.COMPLETED, progress=100, 
            message="Video processing completed successfully!", output_path=output_video_path
        )
        logger.info(f"Video processing completed for job {job_id}: {output_video_path}")

        # Final garbage collection before completion
        gc.collect()

        return output_video_path
        
    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_msg)
        raise
# ==================== API ENDPOINTS ====================

# Endpoint to list available pro keypoints files from S3
from fastapi import APIRouter

@app.get("/api/pro_keypoints/list")
async def list_pro_keypoints():
    files = list_s3_pro_keypoints()
    return {"files": files}

# ==================== API ENDPOINTS ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if GPU is available
        device = get_pytorch_device()
        
        # Clean up old files
        cleanup_old_files()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "device": str(device),
            "active_jobs": len(job_manager.jobs)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/api/videos/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_size: str = Query("xs", description="Model size: xs, s, m, l"),
    is_lefty: bool = Query(False, description="Whether the user is left-handed"),
    pro_keypoints_filename: Optional[str] = Query(None, description="Professional keypoints filename from S3")
):
    """
    Upload a video file and start processing
    
    Args:
        file: Video file to process
        model_size: Model size to use for processing
        is_lefty: Whether the user is left-handed
    
    Returns:
        Upload response with job_id
    """
    # Validate file
    if not validate_video_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a video file (.mp4, .mov, .avi, .mkv)"
        )
    
    # Check file size (100MB limit)
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 100MB"
        )
    
    try:
        # Save uploaded file
        input_path = save_uploaded_file(file)
        
        # Create processing job
        job = job_manager.create_job(file.filename, input_path)
        
        # Start background processing
        background_tasks.add_task(
            process_video_pipeline,
            job.job_id,
            input_path,
            model_size,
            is_lefty,
            pro_keypoints_filename,
        )
        
        logger.info(f"Started processing job {job.job_id} for file {file.filename}")
        
        return VideoUploadResponse(
            job_id=job.job_id,
            message="Video uploaded successfully. Processing started.",
            estimated_time=120  # 2 minutes estimate
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/videos/{job_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(job_id: str):
    """
    Get processing status for a job
    
    Args:
        job_id: Job identifier
    
    Returns:
        Processing status response
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Build result URL if job is completed
    result_url = None
    if job.status == JobStatus.COMPLETED and job.output_path:
        result_url = f"/api/videos/{job_id}/download"

    return ProcessingStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message or (job.error_message if job.status == JobStatus.FAILED else f"Job is {job.status.value}"),
        result_url=result_url,
        error=job.error_message if job.status == JobStatus.FAILED else None
    )


@app.get("/api/videos/{job_id}/download")
async def download_processed_video(job_id: str):
    """
    Download the processed video file
    
    Args:
        job_id: Job identifier
    
    Returns:
        Video file download
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Return file download
    filename = f"processed_{job.original_filename}"
    return FileResponse(
        path=job.output_path,
        filename=filename,
        media_type="video/mp4"
    )


@app.get("/api/videos/{job_id}/preview")
async def get_video_preview(job_id: str):
    """
    Stream video for preview in browser
    Args:
        job_id: Job identifier
    Returns:
        Video file for streaming
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    # Ensure correct Content-Disposition for browser preview (do not force download)
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": "inline; filename=preview.mp4"
        }
    )



# ==================== 3D KEYPOINTS API ENDPOINTS ====================

@app.get("/api/videos/{job_id}/keypoints3d/user")
async def get_user_3d_keypoints(job_id: str):
    """
    Return user 3D keypoints as JSON (for three.js rendering)
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    keypoints_path = Path(job.input_path).parent / f"{job_id}_output" / "raw_keypoints" / "user_3D_keypoints.npy"
    if not keypoints_path.exists():
        raise HTTPException(status_code=404, detail="User 3D keypoints not found")
    arr = np.load(keypoints_path)
    return JSONResponse(content=arr.tolist())


@app.get("/api/videos/{job_id}/keypoints3d/pro")
async def get_pro_3d_keypoints(job_id: str):
    """
    Return pro 3D keypoints as JSON (for three.js rendering)
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    keypoints_path = Path(job.input_path).parent / f"{job_id}_output" / "raw_keypoints" / "pro_3D_keypoints.npy"
    if not keypoints_path.exists():
        raise HTTPException(status_code=404, detail="Pro 3D keypoints not found")
    arr = np.load(keypoints_path)
    return JSONResponse(content=arr.tolist())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
