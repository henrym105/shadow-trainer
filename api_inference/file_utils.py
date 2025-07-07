import os
import shutil
import logging
from typing import List
from config import SUPPORTED_VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)

def validate_video_extension(file_ext: str) -> bool:
    """Validate if the file extension is supported for video processing.
    
    Args:
        file_ext (str): File extension (including the dot)
        
    Returns:
        bool: True if extension is supported, False otherwise
    """
    return file_ext.lower() in SUPPORTED_VIDEO_EXTENSIONS

def clear_tmp_dir(dir_path: str, keep_videos: bool = False) -> None:
    """Delete all files and folders in the given directory.
    
    Args:
        dir_path (str): Path to the directory to clear.
        keep_videos (bool): If True, do not delete video files (.mp4, .mov).
    """
    if not os.path.exists(dir_path):
        return
        
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        
        # Skip video files if keep_videos is True
        if keep_videos and os.path.splitext(filename)[-1].lower() in SUPPORTED_VIDEO_EXTENSIONS:
            continue
            
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f'Failed to delete {file_path}. Reason: {e}')

def ensure_directory_exists(dir_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path (str): Path to the directory to create
    """
    os.makedirs(dir_path, exist_ok=True)
