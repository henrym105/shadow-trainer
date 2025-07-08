"""
Core pose estimation functionality.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Import existing inference functions for backward compatibility
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from inference import (
        get_pose2D as _get_pose2D,
        get_pose3D as _get_pose3D, 
        img2video as _img2video,
        get_pytorch_device as _get_pytorch_device,
        get_or_download_checkpoint
    )
except ImportError as e:
    logging.error(f"Failed to import inference functions: {e}")
    raise

logger = logging.getLogger(__name__)

class PoseEstimator:
    """Wrapper class for pose estimation functionality."""
    
    def __init__(self):
        """Initialize pose estimator."""
        self.device = self.get_device()
        logger.info(f"PoseEstimator initialized with device: {self.device}")
    
    def get_device(self) -> str:
        """Get the best available device for computation."""
        return _get_pytorch_device()
    
    def estimate_2d_pose(self, input_path: str, output_dir: str) -> None:
        """
        Estimate 2D pose from video.
        
        Args:
            input_path: Path to input video
            output_dir: Directory to save results
        """
        logger.info(f"Running 2D pose estimation: {input_path} -> {output_dir}")
        _get_pose2D(input_path, output_dir, self.device)
    
    def estimate_3d_pose(
        self, 
        input_path: str, 
        output_dir: str, 
        model_size: str, 
        model_config: str
    ) -> None:
        """
        Estimate 3D pose from 2D pose data.
        
        Args:
            input_path: Path to input video
            output_dir: Directory containing 2D pose data and to save 3D results
            model_size: Size of model to use
            model_config: Path to model configuration file
        """
        logger.info(f"Running 3D pose estimation: {input_path} -> {output_dir}")
        logger.info(f"Model size: {model_size}, Config: {model_config}")
        _get_pose3D(input_path, output_dir, self.device, model_size, model_config)
    
    def generate_video(self, input_path: str, output_dir: str) -> str:
        """
        Generate output video with pose visualization.
        
        Args:
            input_path: Path to original input video
            output_dir: Directory containing pose estimation results
            
        Returns:
            Path to generated output video
        """
        logger.info(f"Generating output video: {input_path} -> {output_dir}")
        return _img2video(input_path, output_dir)
    
    def download_checkpoint(
        self, 
        filename_pattern: str, 
        local_dir: str,
        s3_bucket: str = "shadow-trainer-prod",
        s3_prefix: str = "model_weights"
    ) -> str:
        """
        Download model checkpoint if not available locally.
        
        Args:
            filename_pattern: Pattern to match checkpoint files
            local_dir: Local directory to store checkpoint
            s3_bucket: S3 bucket containing checkpoints
            s3_prefix: S3 prefix for checkpoint files
            
        Returns:
            Path to local checkpoint file
        """
        return get_or_download_checkpoint(
            filename_pattern, local_dir, s3_bucket, s3_prefix
        )

# Global pose estimator instance
pose_estimator = PoseEstimator()

# Legacy function exports for backward compatibility
def get_pose2D(input_path: str, output_dir: str, device: str) -> None:
    """Legacy function for 2D pose estimation."""
    _get_pose2D(input_path, output_dir, device)

def get_pose3D(
    input_path: str, 
    output_dir: str, 
    device: str, 
    model_size: str, 
    model_config: str
) -> None:
    """Legacy function for 3D pose estimation."""
    _get_pose3D(input_path, output_dir, device, model_size, model_config)

def img2video(input_path: str, output_dir: str) -> str:
    """Legacy function for video generation."""
    return _img2video(input_path, output_dir)

def get_pytorch_device() -> str:
    """Legacy function for device detection."""
    return _get_pytorch_device()
