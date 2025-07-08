"""
Local file storage operations for Shadow Trainer.
"""
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import time

from shadow_trainer.config import settings

logger = logging.getLogger(__name__)

class LocalStorage:
    """Local file storage manager."""
    
    def __init__(self):
        """Initialize local storage manager."""
        self.base_dir = settings.base_dir
        self.temp_dir = settings.temp_dir
        self.assets_dir = settings.assets_dir
        
    def ensure_directory(self, directory: Path) -> None:
        """Ensure directory exists."""
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
    
    def clear_directory(self, directory: Path, keep_videos: bool = False) -> None:
        """
        Clear directory contents.
        
        Args:
            directory: Directory to clear
            keep_videos: If True, keep video files
        """
        if not directory.exists():
            return
        
        try:
            for item in directory.iterdir():
                if item.is_file():
                    if keep_videos and item.suffix.lower() in settings.supported_video_extensions:
                        continue
                    item.unlink()
                    logger.debug(f"Removed file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    logger.debug(f"Removed directory: {item}")
                    
        except Exception as e:
            logger.error(f"Error clearing directory {directory}: {e}")
    
    def copy_file(self, source: Path, destination: Path) -> None:
        """
        Copy file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
        """
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            logger.info(f"Copied file: {source} -> {destination}")
        except Exception as e:
            logger.error(f"Failed to copy file {source} to {destination}: {e}")
            raise
    
    def move_file(self, source: Path, destination: Path) -> None:
        """
        Move file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
        """
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            logger.info(f"Moved file: {source} -> {destination}")
        except Exception as e:
            logger.error(f"Failed to move file {source} to {destination}: {e}")
            raise
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            if not file_path.exists():
                return {"exists": False}
            
            stat = file_path.stat()
            return {
                "exists": True,
                "name": file_path.name,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir(),
                "extension": file_path.suffix,
                "absolute_path": str(file_path.absolute())
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {"exists": False, "error": str(e)}
    
    def list_files(
        self, 
        directory: Path, 
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List files in directory.
        
        Args:
            directory: Directory to list
            pattern: File pattern to match
            recursive: If True, search recursively
            
        Returns:
            List of file information dictionaries
        """
        try:
            if not directory.exists():
                return []
            
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)
            
            file_list = []
            for file_path in files:
                if file_path.is_file():
                    file_info = self.get_file_info(file_path)
                    file_list.append(file_info)
            
            # Sort by name
            file_list.sort(key=lambda x: x.get("name", ""))
            return file_list
            
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {e}")
            return []
    
    def validate_video_file(self, file_path: Path) -> bool:
        """
        Validate if file is a supported video format.
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if valid video file
        """
        if not file_path.exists():
            return False
        
        if not file_path.is_file():
            return False
        
        extension = file_path.suffix.lower()
        return extension in settings.supported_video_extensions
    
    def get_temp_file_path(self, filename: str) -> Path:
        """
        Get a temporary file path.
        
        Args:
            filename: Name of the file
            
        Returns:
            Path in temporary directory
        """
        self.ensure_directory(self.temp_dir)
        return self.temp_dir / filename
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        try:
            if not self.temp_dir.exists():
                return
            
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            logger.debug(f"Cleaned up old temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
    
    def get_sample_videos(self) -> List[Dict[str, Any]]:
        """
        Get list of sample videos from assets directory.
        
        Returns:
            List of sample video information
        """
        videos_dir = self.assets_dir / "videos"
        if not videos_dir.exists():
            return []
        
        videos = []
        for video_file in videos_dir.iterdir():
            if self.validate_video_file(video_file):
                file_info = self.get_file_info(video_file)
                file_info["url"] = f"/assets/videos/{video_file.name}"
                videos.append(file_info)
        
        return videos

# Legacy function exports for backward compatibility
def validate_video_extension(extension: str) -> bool:
    """Validate if file extension is supported."""
    return extension.lower() in settings.supported_video_extensions

def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def clear_tmp_dir(directory: str, keep_videos: bool = False) -> None:
    """Clear temporary directory."""
    storage = LocalStorage()
    storage.clear_directory(Path(directory), keep_videos)
