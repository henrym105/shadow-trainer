"""
Celery tasks for video processing in Shadow Trainer.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from celery import Task
from celery.exceptions import Retry

from api_backend.celery_app import celery_app
from api_backend.pydantic_models import JobStatus
from api_backend.src.inference import (
    create_3d_pose_images_from_array,
    generate_output_combined_frames,
    get_pose2D,
    get_pose3D_no_vis,
    img2video,
    create_2D_images,
)
from api_backend.src.utils import get_pytorch_device
from api_backend.src.yolo2d import rotate_video_until_upright

logger = logging.getLogger(__name__)


class VideoProcessingTask(Task):
    """
    Custom base task class for video processing with callbacks.
    """
    
    def on_start(self, task_id, args, kwargs):
        """Called when task starts."""
        logger.info(f"Task {task_id} started")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task completes successfully."""
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f"Task {task_id} being retried: {exc}")


@celery_app.task(bind=True, base=VideoProcessingTask, queue='video_processing')
def process_video_task(self, job_data: Dict[str, Any]) -> str:
    """
    Process video file through the complete pipeline.
    
    Args:
        job_data: Dictionary containing job information
            - job_id: Unique job identifier
            - input_path: Path to input video file
            - model_size: Model size to use (xs, s, m, l)
            - is_lefty: Whether user is left-handed
            - pro_keypoints_filename: Optional professional keypoints file
            - include_2d_images: Whether to include 2D visualization
    
    Returns:
        str: Path to the output video file
    """
    job_id = job_data['job_id']
    input_path = job_data['input_path']
    model_size = job_data.get('model_size', 'xs')
    is_lefty = job_data.get('is_lefty', False)
    pro_keypoints_filename = job_data.get('pro_keypoints_filename')
    include_2d_images = job_data.get('include_2d_images', True)
    
    logger.info(f"Starting video processing for job {job_id}")
    
    try:
        # Update task progress
        self.update_state(
            state='PROCESSING',
            meta={'progress': 0, 'message': 'Initializing video processing...'}
        )
        
        # Setup paths and directories
        api_root_dir = Path(__file__).parent.parent.absolute()
        tmp_dir = api_root_dir / "tmp_api_output"
        job_output_dir = tmp_dir / f"{job_id}_output"
        job_output_dir.mkdir(exist_ok=True)
        
        # Define file paths
        file_pose2d = job_output_dir / "pose_2d.npy"
        file_pose3d = job_output_dir / "pose_3d.npy"
        file_pose3d_pro = job_output_dir / "pose_3d_pro.npy"
        dir_pose2d = job_output_dir / "pose_2d_images"
        dir_pose3d = job_output_dir / "pose_3d_images"
        dir_combined = job_output_dir / "combined_images"
        
        # Create directories
        for directory in [dir_pose2d, dir_pose3d, dir_combined]:
            directory.mkdir(exist_ok=True)
        
        # Get processing device
        device = get_pytorch_device()
        logger.info(f"Using device: {device}")
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={'progress': 5, 'message': 'Preparing video orientation...'}
        )
        
        # Ensure video is upright
        rotate_video_until_upright(input_path)
        
        # Step 1: Extract 2D poses
        self.update_state(
            state='PROCESSING',
            meta={'progress': 20, 'message': 'Extracting 2D poses...'}
        )
        get_pose2D(video_path=input_path, output_file=str(file_pose2d), device=device)
        
        # Step 2: Create 2D visualization frames
        if include_2d_images:
            self.update_state(
                state='PROCESSING',
                meta={'progress': 35, 'message': 'Creating 2D visualization frames...'}
            )
            cap = cv2.VideoCapture(input_path)
            keypoints_2d = np.load(file_pose2d)
            create_2D_images(cap, keypoints_2d, str(dir_pose2d), is_lefty)
            cap.release()
        
        # Step 3: Generate 3D poses
        self.update_state(
            state='PROCESSING',
            meta={'progress': 50, 'message': 'Generating 3D poses...'}
        )
        
        # Get model configuration path
        config_path = api_root_dir / "model_config_map.json"
        try:
            import json
            with open(config_path, 'r') as f:
                config_map = json.load(f)
                model_config_path = config_map.get(model_size, "src/configs/h36m/MotionAGFormer-small.yaml")
        except FileNotFoundError:
            logger.warning(f"Model config file not found: {config_path}")
            model_config_path = "src/configs/h36m/MotionAGFormer-small.yaml"
        
        if not os.path.isabs(model_config_path):
            model_config_path = str(api_root_dir / model_config_path)
        
        get_pose3D_no_vis(
            user_2d_kpts_filepath=str(file_pose2d),
            output_keypoints_path=str(file_pose3d),
            video_path=input_path,
            device=device,
            model_size=model_size,
            yaml_path=model_config_path
        )
        
        # Step 4: Handle professional keypoints if specified
        pro_keypoints_path = api_root_dir / "checkpoint" / "example_SnellBlake.npy"
        if pro_keypoints_filename:
            self.update_state(
                state='PROCESSING',
                meta={'progress': 60, 'message': 'Downloading professional keypoints...'}
            )
            try:
                # Download from S3
                import boto3
                s3 = boto3.client("s3")
                s3_bucket = "shadow-trainer-dev"
                s3_pro_prefix = "test/professional/"
                s3.download_file(s3_bucket, s3_pro_prefix + pro_keypoints_filename, str(file_pose3d_pro))
                pro_keypoints_path = file_pose3d_pro
                logger.info(f"Downloaded pro keypoints: {pro_keypoints_filename}")
            except Exception as e:
                logger.warning(f"Failed to download pro keypoints: {e}")
                # Fall back to default
        
        # Step 5: Create 3D visualization frames
        self.update_state(
            state='PROCESSING',
            meta={'progress': 70, 'message': 'Creating 3D visualization frames...'}
        )
        create_3d_pose_images_from_array(
            user_3d_keypoints_filepath=str(file_pose3d),
            pro_3d_keypoints_filepath=str(pro_keypoints_path),
            output_dir=str(dir_pose3d),
            is_lefty=is_lefty
        )
        
        # Step 6: Generate combined frames
        self.update_state(
            state='PROCESSING',
            meta={'progress': 85, 'message': 'Generating combined visualization...'}
        )
        generate_output_combined_frames(
            input_video_path=input_path,
            pose_2d_images_dir=str(dir_pose2d) if include_2d_images else None,
            pose_3d_images_dir=str(dir_pose3d),
            output_images_dir=str(dir_combined),
            is_lefty=is_lefty
        )
        
        # Step 7: Create final output video
        self.update_state(
            state='PROCESSING',
            meta={'progress': 95, 'message': 'Creating final output video...'}
        )
        output_video_path = tmp_dir / f"{job_id}.mov"
        img2video(images_dir=str(dir_combined), output_video_path=str(output_video_path))
        
        # Cleanup intermediate files to save space
        import shutil
        for directory in [dir_pose2d, dir_pose3d, dir_combined]:
            if directory.exists():
                shutil.rmtree(directory)
        
        logger.info(f"Video processing completed for job {job_id}")
        return str(output_video_path)
        
    except Exception as e:
        logger.error(f"Video processing failed for job {job_id}: {e}")
        # Update task state to failure
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'progress': 0}
        )
        raise


@celery_app.task(bind=True, queue='cleanup')
def cleanup_old_files_task(self, retention_minutes: int = 60):
    """
    Clean up old temporary files.
    
    Args:
        retention_minutes: Files older than this will be deleted
    """
    logger.info(f"Starting cleanup of files older than {retention_minutes} minutes")
    
    try:
        api_root_dir = Path(__file__).parent.parent.absolute()
        tmp_dir = api_root_dir / "tmp_api_output"
        
        current_time = time.time()
        cutoff_time = current_time - (retention_minutes * 60)
        
        deleted_count = 0
        for item in tmp_dir.iterdir():
            if item.stat().st_mtime < cutoff_time:
                try:
                    if item.is_file():
                        item.unlink()
                        deleted_count += 1
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {item}: {e}")
        
        logger.info(f"Cleanup completed. Deleted {deleted_count} items.")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise


@celery_app.task(bind=True, queue='video_processing')
def health_check_task(self):
    """
    Health check task to verify worker is functioning.
    """
    try:
        device = get_pytorch_device()
        return {
            'status': 'healthy',
            'device': str(device),
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise
