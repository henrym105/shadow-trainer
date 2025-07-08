"""
File utility functions for Shadow Trainer.
"""
import logging
import os
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any

from shadow_trainer.config import settings

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            return file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error getting file size for {file_path}: {e}")
            return 0
    
    @staticmethod
    def get_mime_type(file_path: Path) -> str:
        """Get MIME type of file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    @staticmethod
    def validate_file_size(file_path: Path, max_size: Optional[int] = None) -> bool:
        """Validate file size against maximum allowed size."""
        max_size = max_size or settings.max_file_size
        file_size = FileUtils.get_file_size(file_path)
        return file_size <= max_size
    
    @staticmethod
    def validate_video_extension(file_path: Path) -> bool:
        """Validate if file has a supported video extension."""
        extension = file_path.suffix.lower()
        return extension in settings.supported_video_extensions
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename by removing dangerous characters."""
        # Remove path separators and other dangerous characters
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip('. ')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = "unnamed_file"
        
        return sanitized
    
    @staticmethod
    def generate_unique_filename(
        directory: Path, 
        base_name: str, 
        extension: str = ""
    ) -> str:
        """Generate a unique filename in the given directory."""
        base_name = FileUtils.sanitize_filename(base_name)
        
        if not extension.startswith('.') and extension:
            extension = f".{extension}"
        
        counter = 0
        while True:
            if counter == 0:
                filename = f"{base_name}{extension}"
            else:
                filename = f"{base_name}_{counter}{extension}"
            
            if not (directory / filename).exists():
                return filename
            
            counter += 1
            
            # Prevent infinite loop
            if counter > 9999:
                import time
                filename = f"{base_name}_{int(time.time())}{extension}"
                break
        
        return filename
    
    @staticmethod
    def create_backup_filename(file_path: Path) -> str:
        """Create a backup filename for the given file."""
        import time
        timestamp = int(time.time())
        stem = file_path.stem
        suffix = file_path.suffix
        return f"{stem}_backup_{timestamp}{suffix}"
    
    @staticmethod
    def get_video_info(file_path: Path) -> Dict[str, Any]:
        """Get basic information about a video file."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_seconds": duration,
                "resolution": f"{width}x{height}",
                "file_size": FileUtils.get_file_size(file_path),
                "format": file_path.suffix.lower()
            }
            
        except ImportError:
            logger.warning("OpenCV not available for video info extraction")
            return {"error": "OpenCV not available"}
        except Exception as e:
            logger.error(f"Error getting video info for {file_path}: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        size_index = 0
        size_value = float(size_bytes)
        
        while size_value >= 1024.0 and size_index < len(size_names) - 1:
            size_value /= 1024.0
            size_index += 1
        
        return f"{size_value:.1f} {size_names[size_index]}"
    
    @staticmethod
    def clean_temp_directory(temp_dir: Path, max_age_hours: int = 24) -> int:
        """Clean temporary directory of old files."""
        import time
        
        if not temp_dir.exists():
            return 0
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        try:
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            cleaned_count += 1
                            logger.debug(f"Cleaned up old temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error during temp directory cleanup: {e}")
        
        return cleaned_count

# Global file utils instance
file_utils = FileUtils()

# Alias for backward compatibility
FileManager = FileUtils

# Legacy function exports for backward compatibility
def validate_video_extension(extension: str) -> bool:
    """Legacy function for video extension validation."""
    return extension.lower() in settings.supported_video_extensions

def ensure_directory_exists(directory: str) -> None:
    """Legacy function to ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def clear_tmp_dir(directory: str, keep_videos: bool = False) -> None:
    """Legacy function to clear temporary directory."""
    from shadow_trainer.core.storage.local import LocalStorage
    storage = LocalStorage()
    storage.clear_directory(Path(directory), keep_videos)
