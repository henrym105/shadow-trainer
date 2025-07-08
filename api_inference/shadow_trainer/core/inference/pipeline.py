"""
Processing pipeline for 3D human pose estimation.
"""
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Import the existing inference functions
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from inference import get_pose2D, get_pose3D, img2video, get_pytorch_device
from shadow_trainer.config import settings
from shadow_trainer.schemas.common import ModelSize

logger = logging.getLogger(__name__)

class ProcessingPipeline:
    """Main processing pipeline for video pose estimation."""
    
    def __init__(self):
        """Initialize the processing pipeline."""
        self.device = get_pytorch_device()
        logger.info(f"Pipeline initialized with device: {self.device}")
    
    def process(
        self, 
        input_path: str, 
        model_size: str,
        handedness: str = "Right-handed",
        pitch_types: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Process a video file through the complete 3D pose estimation pipeline.
        
        Args:
            input_path: Path to input video file
            model_size: Size of model to use (xs, s, b, l)
            handedness: User's dominant hand
            pitch_types: List of pitch types to analyze
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input video not found: {input_path}")
            
            # Create output directory
            video_name = input_path.stem
            output_dir = settings.temp_dir / f"{video_name}_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing video: {input_path}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Model size: {model_size}")
            logger.info(f"Handedness: {handedness}")
            logger.info(f"Pitch types: {pitch_types}")
            
            # Load model configuration
            from models import load_model_config
            model_config, err = load_model_config(model_size)
            if err:
                raise RuntimeError(f"Failed to load model config: {err}")
            
            # Run pipeline steps
            logger.info("Step 1: 2D pose estimation")
            get_pose2D(str(input_path), str(output_dir), self.device)
            
            logger.info("Step 2: 3D pose estimation")
            get_pose3D(str(input_path), str(output_dir), self.device, model_size, model_config)
            
            logger.info("Step 3: Video generation")
            output_video_path = img2video(str(input_path), str(output_dir))
            
            processing_time = time.time() - start_time
            
            result = {
                "output_path": output_video_path,
                "processing_time_seconds": processing_time,
                "model_size_used": model_size,
                "handedness": handedness,
                "pitch_types": pitch_types or [],
                "metadata": {
                    "input_path": str(input_path),
                    "output_directory": str(output_dir),
                    "device_used": self.device,
                    "model_config": model_config
                }
            }
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline failed after {processing_time:.2f} seconds: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

def run_pipeline(
    input_path: str, 
    output_dir: str, 
    device: str, 
    model_size: str, 
    model_config: str
) -> Tuple[str, str]:
    """
    Legacy function for backward compatibility.
    
    Returns:
        Tuple[output_video_path, error_message]
    """
    try:
        pipeline = ProcessingPipeline()
        result = pipeline.process(input_path, model_size)
        return result["output_path"], ""
    except Exception as e:
        return "", str(e)
