#!/usr/bin/env python3
"""
Download and setup model checkpoints for Shadow Trainer.
"""
import os
import sys
import logging
from pathlib import Path

# Add api_inference to path
API_INFERENCE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(API_INFERENCE_DIR))

from shadow_trainer.config import settings
from shadow_trainer.core.inference.pose_estimation import pose_estimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download all required model checkpoints."""
    
    # Ensure models directory exists
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    
    # Model checkpoint patterns
    checkpoints = [
        "motionagformer-*-h36m.pth.tr",
        "pose_hrnet_w48_384x288.pth",
        "yolov3.weights"
    ]
    
    logger.info("Starting model checkpoint download...")
    
    for pattern in checkpoints:
        try:
            logger.info(f"Downloading checkpoint: {pattern}")
            checkpoint_path = pose_estimator.download_checkpoint(
                pattern, 
                str(settings.models_dir),
                settings.s3_bucket,
                settings.s3_model_prefix
            )
            logger.info(f"Downloaded: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to download {pattern}: {e}")
    
    logger.info("Model download completed")

def setup_environment():
    """Setup environment for Shadow Trainer."""
    
    logger.info("Setting up Shadow Trainer environment...")
    
    # Create required directories
    directories = [
        settings.models_dir,
        settings.videos_dir,
        settings.images_dir,
        settings.temp_dir
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Download models if S3 is configured
    if settings.s3_bucket:
        download_models()
    else:
        logger.warning("S3 bucket not configured, skipping model download")
    
    logger.info("Environment setup completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Shadow Trainer environment")
    parser.add_argument("--download-models", action="store_true", 
                       help="Download model checkpoints from S3")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only create directories, skip downloads")
    
    args = parser.parse_args()
    
    if args.setup_only:
        # Create directories only
        directories = [
            settings.models_dir,
            settings.videos_dir, 
            settings.images_dir,
            settings.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            
    elif args.download_models:
        download_models()
    else:
        setup_environment()
