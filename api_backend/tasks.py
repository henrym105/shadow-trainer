import gc
import json
import os
from pathlib import Path
import shutil
import time
from typing import Optional
import uuid

from celery import Celery
from celery.utils.log import get_task_logger
import cv2
from fastapi import HTTPException, UploadFile
import numpy as np

from src.utils import get_pytorch_device
from src.yolo2d import rotate_video_until_upright
from src.inference import (
    create_2D_images, 
    create_3d_pose_images_from_array,
    crop_align_3d_keypoints, 
    flip_rgb_to_bgr, 
    generate_output_combined_frames, 
    get_pose2D, 
    get_pose3D_no_vis, 
    img2video
)
from constants import (
    API_ROOT_DIR,
    PRO_TEAMS_MAP,
    S3_BUCKET,
    S3_PRO_PREFIX,
    RESULT_EXPIRES,
    OUTPUT_DIR,
    TMP_PRO_KEYPOINTS_FILE,
)

logger = get_task_logger(__name__)

# ----------------------------------------------------
# Define Celery app 
# ----------------------------------------------------
# Default result expiration time inside of redis and celery (seconds)

celery_app = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

celery_app.conf.update(
    result_expires=RESULT_EXPIRES,
    worker_concurrency=2,
    task_track_started=True,  # Enable progress tracking
)

# ----------------------------------------------------
# Define Tasks
# ----------------------------------------------------

@celery_app.task(bind=True)
def process_video_task_small(
        self, input_video_path: str, model_size: str = "xs", 
        is_lefty: bool = False, pro_keypoints_filename: Optional[str] = None,
        visualization_type: str = "combined"
    ) -> str:
    """Process video with progress updates"""
    logger.info(f" ---> [ process_video_task_small ]")
    logger.info(f"{input_video_path=}")
    logger.info(f"{model_size=}")
    logger.info(f"{is_lefty=}")
    logger.info(f"{pro_keypoints_filename=}")

    try:
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 10})
        
        # Generate output path
        input_path = Path(input_video_path)
        output_filename = f"processed_{uuid.uuid4()}.mp4"
        output_path = Path("/app/output") / output_filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 50})

        flip_rgb_to_bgr(input_video_path, str(output_path))

        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 100})

        return {
            "input_path": input_video_path,
            "output_path": str(output_path),
            "original_filename": input_path.name,
            "file_size": os.path.getsize(output_path),
            "status": "completed"
        }

    except Exception as e:
        raise e



@celery_app.task(bind=True)
def process_video_task(
        self, 
        input_video_path: str, 
        model_size: str = "xs", 
        is_lefty: bool = False, 
        pro_keypoints_filename: str = None,
        visualization_type: str = "dynamic_3d_animation"
    ) -> str:
    """Process video with pose estimation and keypoint overlays
    
    Args:
        input_video_path: Path to input video file
        model_size: Model size to use for processing
        is_lefty: Whether the user is left-handed
        pro_keypoints_filename: str
        visualization_type: str ("combined", "3d_only", "dynamic_3d_animation")
    
    Returns:
        Path to output video file
    """
    task_id = self.request.id

    # Establish output directory constants for this task
    DIR_OUTPUT_BASE = OUTPUT_DIR / f"{task_id}_output"

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
        logger.info(f"Starting video processing for task {task_id}")
        logger.info(f"User handedness preference: {'Left-handed' if is_lefty else 'Right-handed'}")
        cleanup_old_files(OUTPUT_DIR, retention_seconds=RESULT_EXPIRES)

        # Get device for processing
        device = get_pytorch_device()

        # Update job status to processing
        self.update_state(state='PROGRESS', meta={'progress': 10}, message="Initializing video processing...")

        # Load model configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        model_config_path = get_model_config_path(model_size)
        logger.info(f"Using model config: {model_config_path}")

        # First, ensure the user uploaded video is upright
        rotate_video_until_upright(input_video_path)

        # Step 1: Extract 2D poses (20% progress)
        self.update_state(state='PROGRESS', meta={'progress': 10})
        self.update_state(state='PROGRESS', meta={'progress': 20}, message="Extracting 2D poses...")
        get_pose2D(video_path=input_video_path, output_file=FILE_POSE2D, device=device)

        # Step 2: Create 2D visualization frames (35% progress)
        self.update_state(state='PROGRESS', meta={'progress': 35}, message="Creating 2D visualization frames...")
        if visualization_type == "combined":
            cap = cv2.VideoCapture(input_video_path)
            keypoints_2d = np.load(FILE_POSE2D)
            create_2D_images(cap, keypoints_2d, DIR_POSE2D, is_lefty)
            cap.release()

        # Step 3: Generate 3D poses (50% progress)
        self.update_state(state='PROGRESS', meta={'progress': 50}, message="Generating 3D poses...")
        get_pose3D_no_vis(
            user_2d_kpts_filepath = FILE_POSE2D,
            output_keypoints_path = FILE_POSE3D,
            video_path=input_video_path,
            device=device,
            model_size=model_size,
            yaml_path=model_config_path
        )

        # Step 4: Download pro keypoints if specified, otherwise use temporary file (Blake Snell)
        pro_keypoints_path = TMP_PRO_KEYPOINTS_FILE
        pro_player_name = None
        if pro_keypoints_filename:
            logger.info(f"Downloading pro keypoints file from S3: {pro_keypoints_filename}")
            download_pro_keypoints_from_s3(pro_keypoints_filename, FILE_POSE3D_PRO)
            pro_keypoints_path = FILE_POSE3D_PRO

            # Extract player name from filename
            player_key = pro_keypoints_filename.replace("_median.npy", "").replace(".npy", "")
            pro_info = PRO_TEAMS_MAP.get(player_key, {})
            pro_player_name = pro_info.get("name", player_key)
            logger.info(f"Professional player: {pro_player_name}")

        # Create info.json file with pro_name
        import json
        info_data = {
            "pro_name": pro_player_name if pro_player_name else "Blake Snell"
        }
        info_file_path = DIR_OUTPUT_BASE / "info.json"
        with open(info_file_path, 'w') as f:
            json.dump(info_data, f, indent=2)
        logger.info(f"Created info.json with pro_name: {info_data['pro_name']}")

        # step 4.1: crop align the user keypoints to the same n_frames as the pro keypoints: 
        self.update_state(state='PROGRESS', meta={'progress': 60}, message="Aligning user keypoints with pro keypoints...")
        pro_kpts_path, user_kpts_path = crop_align_3d_keypoints(
            user_3d_keypoints_filepath=FILE_POSE3D,
            pro_keypoints_filepath=pro_keypoints_path,
            is_lefty=is_lefty
        )

        # Step 4.5: Create 3D visualization frames (70% progress)
        self.update_state(state='PROGRESS', meta={'progress': 70}, message="Creating 3D visualization frames...")
        if visualization_type in ["3d_only", "combined"]:
            create_3d_pose_images_from_array(
                user_3d_keypoints_filepath = user_kpts_path,
                pro_keypoints_filepath = pro_kpts_path,
                output_dir = DIR_POSE3D,
                is_lefty = is_lefty,
                pro_player_name = pro_player_name
            )

        # # Step 5: Generate combined frames (85% progress)
        self.update_state(state='PROGRESS', meta={'progress': 85}, message="Combining frames with original video...")
        if visualization_type == "combined":
            generate_output_combined_frames(
                output_dir_2D=DIR_POSE2D,
                output_dir_3D=DIR_POSE3D,
                output_dir_combined=DIR_COMBINED_FRAMES
            )

        # Step 6: Create final video (95% progress)
        self.update_state(state='PROGRESS', meta={'progress': 95}, message="Generating final video...")
        
        # Skip video generation for dynamic_3d_animation
        if visualization_type in ["3d_only", "combined"]:
            # Select output directory based on video format
            if visualization_type == "3d_only":
                input_frames_dir = DIR_POSE3D
                logger.info(f"Creating 3D-only video from {input_frames_dir}")
            else:  # "combined"
                input_frames_dir = DIR_COMBINED_FRAMES
                logger.info(f"Creating combined video format from {input_frames_dir}")
                
            output_video_path = img2video(
                video_path = input_video_path,
                input_frames_dir = input_frames_dir,
            )
        else:
            # For dynamic_3d_animation, we don't create a video file
            output_video_path = None
            logger.info("Skipping video generation for dynamic_3d_animation format")

        # Complete job
        self.update_state(state='PROGRESS', meta={'progress': 100}, message="Video processing completed.")
        logger.info(f"Video processing completed for job {task_id}: {output_video_path}")

        # Final garbage collection before completion
        gc.collect()
        
        return {
            "input_path": input_video_path,
            "output_path": str(output_video_path) if output_video_path else None,
            "output_dir": str(DIR_OUTPUT_BASE),
            "original_filename": os.path.basename(input_video_path),
            "file_size": os.path.getsize(output_video_path) if output_video_path else 0,
            "status": "completed"
        }

    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        logger.error(f"Job {task_id} failed: {error_msg}")
        self.update_state(state='FAILED', meta={'error': error_msg})
        raise e


@celery_app.task(bind=True)
def generate_3d_keypoints_from_video_task(self, input_video_path: str, model_size: str = "xs") -> str:
    """
    Celery task to generate 3D keypoints from a video file and return the path to the 3D keypoints file.

    Args:
        input_video_path: Path to the input video file.
        model_size: Model size to use for 3D pose estimation.

    Returns:
        str: Path to the generated 3D keypoints .npy file.
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10}, message="Initializing 3D keypoints extraction...")
        
        device = get_pytorch_device()
        model_config_path = get_model_config_path(model_size)
        # Temporary output paths
        temp_dir = OUTPUT_DIR / f"3d_keypoints_{uuid.uuid4()}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_2d_path = temp_dir / "2d_keypoints.npy"
        temp_3d_path = temp_dir / "3d_keypoints.npy"

        # Step 1: Extract 2D keypoints from video
        self.update_state(state='PROGRESS', meta={'progress': 50}, message="Extracting 2D poses...")
        get_pose2D(video_path=input_video_path, output_file=temp_2d_path, device=device)

        # Step 2: Generate 3D keypoints from 2D keypoints
        self.update_state(state='PROGRESS', meta={'progress': 90}, message="Generating 3D keypoints...")
        get_pose3D_no_vis(
            user_2d_kpts_filepath=temp_2d_path,
            output_keypoints_path=temp_3d_path,
            video_path=input_video_path,
            device=device,
            model_size=model_size,
            yaml_path=model_config_path
        )

        self.update_state(state='PROGRESS', meta={'progress': 100}, message="3D keypoints extraction completed!")
        return str(temp_3d_path)
    except Exception as e:
        logger.error(f"Failed to generate 3D keypoints: {e}")
        raise e



@celery_app.task(bind=True)
def generate_joint_evaluation_task(self, task_id: str) -> str:
    """Generate joint evaluation analysis for a processed video task"""
    try:
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 10}, message="Initializing joint evaluation...")
        
        # Construct paths based on task_id
        DIR_OUTPUT_BASE = OUTPUT_DIR / f"{task_id}_output"
        DIR_KEYPOINTS = DIR_OUTPUT_BASE / "raw_keypoints"
        FILE_USER_3D = DIR_KEYPOINTS / "user_3D_keypoints.npy"
        FILE_PRO_3D = DIR_KEYPOINTS / "pro_3D_keypoints.npy"
        info_file_path = DIR_OUTPUT_BASE / "info.json"
        
        # Check if required files exist
        if not FILE_USER_3D.exists() or not FILE_PRO_3D.exists():
            raise Exception("Required keypoint files not found")
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 30}, message="Loading keypoint data...")
        
        # Load keypoints
        user_kps = np.load(FILE_USER_3D)
        pro_kps = np.load(FILE_PRO_3D)
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 60}, message="Analyzing joint movements...")
        
        # Run joint evaluation
        from kpts_analysis import evaluate_all_joints_text, generate_motion_feedback
        joint_text = evaluate_all_joints_text(user_kps, pro_kps)
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 70}, message="Generating AI feedback...")
        
        # Generate human-readable feedback using OpenAI
        try:
            motion_feedback = generate_motion_feedback(joint_text)
            logger.info("OpenAI motion feedback generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate OpenAI feedback: {e}")
            motion_feedback = "Unable to generate personalized feedback at this time."
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 80}, message="Saving analysis results...")
        
        # Update info.json with both joint evaluation text and AI feedback
        if info_file_path.exists():
            with open(info_file_path, 'r') as f:
                info_data = json.load(f)
        else:
            info_data = {}
        
        info_data['joint_evaluation_text'] = joint_text
        info_data['motion_feedback'] = motion_feedback
        
        with open(info_file_path, 'w') as f:
            json.dump(info_data, f, indent=2)
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 100}, message="Joint evaluation completed!")
        
        logger.info(f"Joint evaluation completed for task {task_id}")
        
        return {
            "task_id": task_id,
            "joint_evaluation_text": joint_text,
            "motion_feedback": motion_feedback,
            "status": "completed"
        }
        
    except Exception as e:
        error_msg = f"Joint evaluation failed: {str(e)}"
        logger.error(f"Joint evaluation task failed for {task_id}: {error_msg}")
        self.update_state(state='FAILED', meta={'error': error_msg})
        raise e


@celery_app.task
def add_task(x, y):
    for i in range(x):
        time.sleep(1)
        print(f"Processing {i + 1}/{x}...")
    return x + y



def save_uploaded_file(file: UploadFile, file_id: str = None) -> str:
    """Save uploaded file to temporary directory"""
    # Create unique filename
    if file_id is None:
        file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix.lower()
    temp_filename = f"{file_id}{file_ext}"
    temp_filepath = OUTPUT_DIR / temp_filename
    
    # Save file
    with open(temp_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Verify file exists and is readable before starting task
    if not os.path.exists(temp_filepath):
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Additional verification - ensure file has content
    if os.stat(temp_filepath).st_size == 0:
        raise HTTPException(status_code=500, detail="Uploaded file is empty")
    
    logger.info(f"Saved uploaded file: {temp_filepath}")
    return str(temp_filepath)


# ----------------------------------------------------
# Utils
# ----------------------------------------------------

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
    result = []
    for f in files:
        # Extract player name by removing _median.npy suffix
        player_key = f.replace("_median.npy", "").replace(".npy", "")
        info = PRO_TEAMS_MAP.get(player_key, {})
        result.append({
            "filename": f,
            "name": info.get("name", player_key),  # Use player_key as fallback if not in map
            "team": info.get("team", "Unknown"),
            "city": info.get("city", "Unknown")
        })
    files = result
    return files

def download_pro_keypoints_from_s3(filename, dest_path):
    import boto3
    s3 = boto3.client("s3")
    logger.info(f"Downloading pro keypoints file {filename} from S3 to {dest_path}")
    logger.info(f"S3 Bucket: {S3_BUCKET}, Prefix: {S3_PRO_PREFIX}, filename: {filename}")
    s3.download_file(S3_BUCKET, S3_PRO_PREFIX + filename, str(dest_path))


def validate_video_file(file: UploadFile) -> bool:
    """Validate uploaded video file"""
    if not file.filename:
        return False
    
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
    file_ext = Path(file.filename).suffix.lower()
    
    return file_ext in allowed_extensions



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
        

@celery_app.task()
def cleanup_old_files(output_dir: Path, retention_seconds: int = 60):
    """Clean up old temporary files (older than retention_seconds)"""
    # Ensure output_dir is a Path object
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
        
    current_time = time.time()
    file_retention_cutoff_time = current_time - retention_seconds

    for item in output_dir.iterdir():
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
