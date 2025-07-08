#!/usr/bin/env python3
"""
Download model checkpoints from S3.
"""
import os
import sys
import logging
from pathlib import Path

# Add api_inference to path
API_INFERENCE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(API_INFERENCE_DIR))

from shadow_trainer.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model_checkpoints():
    """Download all model checkpoints from S3."""
    
    try:
        # Import after adding to path
        from shadow_trainer.core.inference.pose_estimation import pose_estimator
        
        # Model checkpoint mappings
        checkpoints = {
            "MotionAGFormer (Extra Small)": "motionagformer-xs-h36m.pth.tr",
            "MotionAGFormer (Small)": "motionagformer-s-h36m.pth.tr", 
            "MotionAGFormer (Base)": "motionagformer-b-h36m.pth.tr",
            "MotionAGFormer (Large)": "motionagformer-l-h36m.pth.tr",
            "HRNet Pose": "pose_hrnet_w48_384x288.pth",
            "YOLOv3": "yolov3.weights"
        }
        
        logger.info(f"Downloading models to: {settings.models_dir}")
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        
        for model_name, filename in checkpoints.items():
            try:
                logger.info(f"Downloading {model_name}...")
                
                # Use pattern matching for MotionAGFormer models
                if filename.startswith("motionagformer"):
                    pattern = filename.replace("xs", "*").replace("s", "*").replace("b", "*").replace("l", "*")
                else:
                    pattern = filename
                
                checkpoint_path = pose_estimator.download_checkpoint(
                    pattern,
                    str(settings.models_dir),
                    settings.s3_bucket,
                    settings.s3_model_prefix
                )
                
                logger.info(f"✓ Downloaded {model_name}: {checkpoint_path}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"✗ Failed to download {model_name}: {e}")
        
        logger.info(f"Download completed: {success_count}/{len(checkpoints)} models downloaded")
        
        if success_count == len(checkpoints):
            logger.info("All models downloaded successfully!")
        else:
            logger.warning(f"Some models failed to download. Check logs above.")
            
    except Exception as e:
        logger.error(f"Error during model download: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if not settings.s3_bucket:
        logger.error("S3 bucket not configured. Set SHADOW_TRAINER_S3_BUCKET environment variable.")
        sys.exit(1)
    
    success = download_model_checkpoints()
    sys.exit(0 if success else 1)
