import logging
from typing import Tuple
from inference import get_pose2D, get_pose3D, img2video

logger = logging.getLogger(__name__)

def run_pipeline(input_path: str, output_dir: str, device: str, model_size: str, model_config: str) -> Tuple[str, str]:
    """Run the full processing pipeline consisting of 2D pose estimation, 3D pose estimation, and image-to-video conversion.
    
    Args:
        input_path (str): Path to the input data (e.g., images or video).
        output_dir (str): Directory where output files will be saved.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        model_size (str): Size specification for the 3D pose estimation model.
        model_config (str): Path to configuration file for the 3D pose estimation model.
        
    Returns:
        Tuple[output_video_path, error_message]: 
            - output_video_path (str): Path to the generated output video if successful, empty string if failed.
            - error_message (str): Error message if the pipeline fails, empty string if successful.
    """
    try:
        logger.info("Running get_pose2D...")
        logger.info(f"Input path: {input_path}, Output directory: {output_dir}, Device: {device}")
        get_pose2D(input_path, output_dir, device)
        
        logger.info("Running get_pose3D...")
        get_pose3D(input_path, output_dir, device, model_size, model_config)
        
        logger.info("Running img2video...")
        output_video_path = img2video(input_path, output_dir)
        return output_video_path, ""
        
    except Exception as e:
        error_msg = f"Pipeline failed: {e}"
        logger.error(error_msg)
        return "", error_msg
