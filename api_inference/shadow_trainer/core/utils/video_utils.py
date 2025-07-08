"""
Video utility functions for Shadow Trainer.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class VideoUtils:
    """Utility class for video operations."""
    
    @staticmethod
    def extract_video_metadata(file_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from video file.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            import cv2
            
            if not file_path.exists():
                return {"error": "Video file not found"}
            
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            # Basic properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            metadata = {
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
                "duration_formatted": VideoUtils.format_duration(duration),
                "resolution": f"{width}x{height}",
                "aspect_ratio": round(width / height, 2) if height > 0 else 0,
                "codec": codec.strip('\x00'),
                "format": file_path.suffix.lower(),
                "bitrate_estimate": (file_path.stat().st_size * 8) / duration if duration > 0 else 0
            }
            
            return metadata
            
        except ImportError:
            logger.error("OpenCV not available for video metadata extraction")
            return {"error": "OpenCV not available"}
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def validate_video_format(file_path: Path) -> Tuple[bool, str]:
        """
        Validate video format and return result with message.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Tuple of (is_valid, message)
        """
        from shadow_trainer.config import settings
        
        if not file_path.exists():
            return False, "Video file does not exist"
        
        if not file_path.is_file():
            return False, "Path is not a file"
        
        extension = file_path.suffix.lower()
        if extension not in settings.supported_video_extensions:
            return False, f"Unsupported video format: {extension}"
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > settings.max_file_size:
            max_size_mb = settings.max_file_size / (1024 * 1024)
            current_size_mb = file_size / (1024 * 1024)
            return False, f"File too large: {current_size_mb:.1f}MB (max: {max_size_mb:.1f}MB)"
        
        # Try to open with OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                cap.release()
                return False, "Could not open video file (corrupted or invalid format)"
            
            # Check if we can read at least one frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False, "Could not read video frames (corrupted file)"
            
            return True, "Video format is valid"
            
        except ImportError:
            # If OpenCV is not available, just check extension
            return True, "Video format appears valid (OpenCV not available for full validation)"
        except Exception as e:
            return False, f"Error validating video: {str(e)}"
    
    @staticmethod
    def extract_frame(
        video_path: Path, 
        frame_number: int = 0,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Extract a single frame from video.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (0-based)
            output_path: Output path for frame image
            
        Returns:
            Path to extracted frame image
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Could not read frame {frame_number} from {video_path}")
                return None
            
            # Generate output path if not provided
            if output_path is None:
                output_path = video_path.parent / f"{video_path.stem}_frame_{frame_number}.jpg"
            
            # Save frame
            cv2.imwrite(str(output_path), frame)
            logger.info(f"Extracted frame to {output_path}")
            return output_path
            
        except ImportError:
            logger.error("OpenCV not available for frame extraction")
            return None
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return None
    
    @staticmethod
    def get_thumbnail(video_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Generate thumbnail image from video (middle frame).
        
        Args:
            video_path: Path to video file
            output_path: Output path for thumbnail
            
        Returns:
            Path to thumbnail image
        """
        try:
            metadata = VideoUtils.extract_video_metadata(video_path)
            if "error" in metadata:
                return None
            
            # Extract middle frame
            middle_frame = metadata["frame_count"] // 2
            
            if output_path is None:
                output_path = video_path.parent / f"{video_path.stem}_thumbnail.jpg"
            
            return VideoUtils.extract_frame(video_path, middle_frame, output_path)
            
        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
            return None
    
    @staticmethod
    def is_video_processable(file_path: Path) -> Tuple[bool, str]:
        """
        Check if video is processable by the pose estimation pipeline.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Tuple of (is_processable, message)
        """
        # First validate basic format
        is_valid, message = VideoUtils.validate_video_format(file_path)
        if not is_valid:
            return False, message
        
        # Check video metadata
        metadata = VideoUtils.extract_video_metadata(file_path)
        if "error" in metadata:
            return False, f"Cannot read video metadata: {metadata['error']}"
        
        # Check minimum requirements
        min_width, min_height = 480, 360
        max_duration = 300  # 5 minutes
        
        if metadata["width"] < min_width or metadata["height"] < min_height:
            return False, f"Video resolution too low: {metadata['resolution']} (minimum: {min_width}x{min_height})"
        
        if metadata["duration_seconds"] > max_duration:
            return False, f"Video too long: {metadata['duration_formatted']} (maximum: {VideoUtils.format_duration(max_duration)})"
        
        if metadata["frame_count"] < 10:
            return False, "Video has too few frames (minimum: 10 frames)"
        
        return True, "Video is processable"

# Global video utils instance
video_utils = VideoUtils()
