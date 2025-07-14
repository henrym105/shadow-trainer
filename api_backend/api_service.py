"""
Shadow Trainer API Service
Professional video processing API for pose estimation and motion analysis.
"""
import logging
import uuid
import shutil
from pathlib import Path
from typing import Optional
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

TMP_PRO_KEYPOINTS_FILE = API_ROOT_DIR / "checkpoint" / "example_SnellBlake.npy"

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
        


def cleanup_old_files():
    """Clean up old temporary files (older than 1 hour)"""
    current_time = time.time()
    one_hour_ago = current_time - 3600
    
    for filepath in TMP_DIR.iterdir():
        if filepath.is_file() and filepath.stat().st_mtime < one_hour_ago:
            try:
                filepath.unlink()
                logger.info(f"Cleaned up old file: {filepath}")
            except OSError as e:
                logger.warning(f"Failed to clean up file {filepath}: {e}")


# ==================== VIDEO PROCESSING ====================

def process_video_pipeline(job_id: str, input_path: str, model_size: str = "xs") -> str:
    """
    Process video with pose estimation and keypoint overlays
    
    Args:
        job_id: Unique job identifier
        input_path: Path to input video file
        model_size: Model size to use for processing
    
    Returns:
        Path to output video file
    """
    try:
        logger.info(f"Starting video processing for job {job_id}")
        
        # Get device for processing
        device = get_pytorch_device()

        # Update job status to processing
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=5)
        
        # Load model configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        model_config_path = get_model_config_path(model_size)
        logger.info(f"Using model config: {model_config_path}")
        
        # Set up output paths
        input_path_obj = Path(input_path)
        base_name = input_path_obj.stem
        output_dir = TMP_DIR / f"{job_id}_output"
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Extract 2D poses (20% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=20, 
                                    message="Extracting 2D poses...")
        
        pose2d_file = output_dir / "raw_keypoints" / "2D_keypoints.npy"
        pose2d_file.parent.mkdir(exist_ok=True)
        
        get_pose2D(
            video_path=input_path,
            output_file=pose2d_file,
            device=device
        )
        
        # Step 2: Generate 3D poses (50% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=50,
                                    message="Generating 3D poses...")
        
        pose3d_file = output_dir / "raw_keypoints" / "user_3D_keypoints.npy"
        pose3d_file.parent.mkdir(exist_ok=True)
        
        get_pose3D_no_vis(
            user_keypoints_path = pose2d_file,
            output_keypoints_path = pose3d_file,
            video_path=input_path,
            device=str(device),
            model_size=model_size,
            yaml_path=model_config_path
        )
        

        # ---------------------------

        # Step 3: Create visualization frames (70% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=70,
                                    message="Creating visualization frames...")
        
        frames_dir = output_dir / "pose"
        frames_dir.mkdir(exist_ok=True)
        
        create_3d_pose_images_from_array(
            user_3d_keypoints_filepath = str(pose3d_file),
            output_dir = str(frames_dir),
            pro_keypoints_filepath = TMP_PRO_KEYPOINTS_FILE,
        )
        
        # Step 4: Create 2D visualization frames (75% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=75,
                                    message="Creating 2D visualization frames...")
        
        # Create 2D images from video and keypoints
        cap = cv2.VideoCapture(input_path)
        keypoints_2d = np.load(pose2d_file)
        frames_2d_dir = create_2D_images(cap, keypoints_2d, str(output_dir))
        cap.release()

        # Step 5: Generate combined frames (85% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=85,
                                    message="Combining frames with original video...")
        
        combined_frames_dir = output_dir / "combined_frames"
        combined_frames_dir.mkdir(exist_ok=True)
        
        generate_output_combined_frames(
            output_dir_2D=frames_2d_dir,
            output_dir_3D=str(frames_dir),
            output_dir=str(combined_frames_dir)
        )
        
        # Step 6: Create final video (95% progress)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=95,
                                    message="Generating final video...")
        
        output_video_path = img2video(
            video_path=input_path,
            output_dir=str(combined_frames_dir)
        )
        
        # Complete job
        job_manager.update_job_status(job_id, JobStatus.COMPLETED, progress=100,
                                    message="Video processing completed successfully!",
                                    output_path=output_video_path)
        
        logger.info(f"Video processing completed for job {job_id}: {output_video_path}")
        return output_video_path
        
    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_msg)
        raise


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
    model_size: str = Query("xs", description="Model size: xs, s, m, l")
):
    """
    Upload a video file and start processing
    
    Args:
        file: Video file to process
        model_size: Model size to use for processing
    
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
            model_size
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
    
    # Return file for streaming
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
