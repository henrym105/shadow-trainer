"""
Video processing service for Shadow Trainer.
"""
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from shadow_trainer.config import settings
from shadow_trainer.core.inference.pipeline import ProcessingPipeline
from shadow_trainer.core.storage.s3 import S3Storage
from shadow_trainer.core.storage.local import LocalStorage
from shadow_trainer.core.utils.video_utils import VideoUtils
from shadow_trainer.core.utils.file_utils import FileUtils
from shadow_trainer.schemas.requests import VideoProcessRequest
from shadow_trainer.schemas.responses import VideoProcessResponse

logger = logging.getLogger(__name__)

class VideoService:
    """Service for video processing operations."""
    
    def __init__(self):
        """Initialize video service."""
        self.pipeline = ProcessingPipeline()
        self.s3_storage = S3Storage()
        self.local_storage = LocalStorage()
        self.video_utils = VideoUtils()
        self.file_utils = FileUtils()
        
        # Ensure required directories exist
        self.local_storage.ensure_directory(settings.temp_dir)
        
        logger.info("VideoService initialized")
    
    async def process_video(self, request: VideoProcessRequest) -> VideoProcessResponse:
        """
        Process a video file through the complete pipeline.
        
        Args:
            request: Video processing request
            
        Returns:
            Video processing response
        """
        job_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting video processing job {job_id}")
        logger.info(f"Request: {request.dict()}")
        
        try:
            # Validate and prepare input
            local_input_path = await self._prepare_input(request.file_path)
            
            # Validate video format
            is_processable, message = self.video_utils.is_video_processable(local_input_path)
            if not is_processable:
                raise ValueError(f"Video not processable: {message}")
            
            # Run processing pipeline
            result = self.pipeline.process(
                str(local_input_path),
                request.model_size.value,
                request.handedness.value,
                [pt.value for pt in request.pitch_types]
            )
            
            # Handle output
            output_local_path = result["output_path"]
            output_s3_url = None
            
            # Upload to S3 if configured
            if settings.s3_bucket and self.s3_storage.is_available():
                try:
                    output_s3_url = await self._upload_result_to_s3(
                        Path(output_local_path), job_id
                    )
                except Exception as e:
                    logger.warning(f"Failed to upload to S3: {e}")
            
            processing_time = time.time() - start_time
            
            # Create response
            response = VideoProcessResponse(
                success=True,
                message="Video processed successfully",
                output_video_local_path=output_local_path,
                output_video_s3_url=output_s3_url,
                processing_time_seconds=processing_time,
                model_size_used=request.model_size,
                metadata={
                    "job_id": job_id,
                    "input_path": request.file_path,
                    "handedness": request.handedness.value,
                    "pitch_types": [pt.value for pt in request.pitch_types],
                    **result.get("metadata", {})
                }
            )
            
            logger.info(f"Job {job_id} completed successfully in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Video processing failed for job {job_id}: {str(e)}"
            logger.error(error_msg)
            
            return VideoProcessResponse(
                success=False,
                message=error_msg,
                processing_time_seconds=processing_time,
                metadata={"job_id": job_id, "error": str(e)}
            )
        
        finally:
            # Cleanup temporary files
            try:
                if 'local_input_path' in locals() and request.file_path.startswith('s3://'):
                    # Only delete if we downloaded from S3
                    if local_input_path.exists():
                        local_input_path.unlink()
                        logger.debug(f"Cleaned up temporary input file: {local_input_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary files: {e}")
    
    async def _prepare_input(self, file_path: str) -> Path:
        """
        Prepare input file for processing.
        
        Args:
            file_path: File path (local or S3)
            
        Returns:
            Local path to input file
        """
        if self.s3_storage.is_s3_path(file_path):
            # Download from S3
            logger.info(f"Downloading from S3: {file_path}")
            local_path = await self.s3_storage.download_file(file_path, settings.temp_dir)
            return local_path
        else:
            # Local file
            local_path = Path(file_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Local file not found: {file_path}")
            return local_path
    
    async def _upload_result_to_s3(self, result_path: Path, job_id: str) -> str:
        """
        Upload processing result to S3.
        
        Args:
            result_path: Local path to result file
            job_id: Job identifier
            
        Returns:
            S3 URL of uploaded file
        """
        video_name = result_path.stem
        s3_key = f"processed_videos/{job_id}/{video_name}{result_path.suffix}"
        
        return await self.s3_storage.upload_file(
            result_path, s3_key, settings.s3_bucket
        )
    
    async def save_uploaded_file(self, uploaded_file) -> Path:
        """
        Save an uploaded file to temporary directory.
        
        Args:
            uploaded_file: FastAPI UploadFile object
            
        Returns:
            Path to saved file
        """
        # Validate file
        if not uploaded_file.filename:
            raise ValueError("No filename provided")
        
        # Sanitize filename
        safe_filename = self.file_utils.sanitize_filename(uploaded_file.filename)
        
        # Generate unique filename in temp directory
        unique_filename = self.file_utils.generate_unique_filename(
            settings.temp_dir, 
            Path(safe_filename).stem,
            Path(safe_filename).suffix
        )
        
        file_path = settings.temp_dir / unique_filename
        
        # Validate file size
        content = await uploaded_file.read()
        if len(content) > settings.max_file_size:
            raise ValueError(f"File too large: {len(content)} bytes (max: {settings.max_file_size})")
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    
    def get_sample_videos(self) -> Dict[str, Any]:
        """Get list of available sample videos."""
        try:
            videos = self.local_storage.get_sample_videos()
            return {
                "success": True,
                "videos": videos,
                "count": len(videos)
            }
        except Exception as e:
            logger.error(f"Error getting sample videos: {e}")
            return {
                "success": False,
                "error": str(e),
                "videos": [],
                "count": 0
            }
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """Clean up old temporary files."""
        try:
            cleaned_count = self.file_utils.clean_temp_directory(
                settings.temp_dir, max_age_hours
            )
            
            return {
                "success": True,
                "cleaned_files": cleaned_count,
                "message": f"Cleaned up {cleaned_count} old files"
            }
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {
                "success": False,
                "error": str(e),
                "cleaned_files": 0
            }
