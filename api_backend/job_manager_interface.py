"""
Job manager compatibility layer.
Provides a unified interface that can switch between the old ThreadPoolExecutor-based
job manager and the new Celery-based job manager using feature flags.
"""

import logging
from typing import Dict, Optional

from config import config
from pydantic_models import VideoJob, JobStatus

logger = logging.getLogger(__name__)


class JobManagerInterface:
    """
    Unified interface for job management that can switch between implementations.
    """
    
    def __init__(self):
        self._manager = None
        self._initialize_manager()
    
    def _initialize_manager(self):
        """Initialize the appropriate job manager based on configuration."""
        if config.USE_CELERY:
            try:
                from celery_job_manager import get_job_manager
                self._manager = get_job_manager()
                logger.info("Using Celery-based job manager")
            except ImportError as e:
                logger.error(f"Failed to import Celery job manager: {e}")
                logger.info("Falling back to legacy job manager")
                self._initialize_legacy_manager()
        else:
            self._initialize_legacy_manager()
    
    def _initialize_legacy_manager(self):
        """Initialize the legacy ThreadPoolExecutor-based job manager."""
        try:
            from job_manager import job_manager
            self._manager = job_manager
            logger.info("Using legacy ThreadPoolExecutor-based job manager")
        except ImportError as e:
            logger.error(f"Failed to import legacy job manager: {e}")
            raise RuntimeError("No job manager available")
    
    def create_job(self, filename: str, input_path: str) -> VideoJob:
        """Create a new video processing job."""
        return self._manager.create_job(filename, input_path)
    
    def get_job(self, job_id: str) -> Optional[VideoJob]:
        """Get job by ID."""
        return self._manager.get_job(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus, progress: int = None,
                         error: str = None, output_path: str = None, message: str = None):
        """Update job status and progress."""
        return self._manager.update_job_status(
            job_id, status, progress=progress, error=error, 
            output_path=output_path, message=message
        )
    
    def submit_job(self, job_id: str, model_size: str = 'xs', is_lefty: bool = False,
                  pro_keypoints_filename: str = None, include_2d_images: bool = True):
        """Submit a job for processing."""
        if config.USE_CELERY:
            # Celery-based submission
            return self._manager.submit_job(
                job_id=job_id,
                model_size=model_size,
                is_lefty=is_lefty,
                pro_keypoints_filename=pro_keypoints_filename,
                include_2d_images=include_2d_images
            )
        else:
            # Legacy submission - we need to import the processing function
            from api_service import process_video_job
            return self._manager.submit_job(
                job_id, process_video_job, 
                model_size=model_size,
                is_lefty=is_lefty,
                pro_keypoints_filename=pro_keypoints_filename
            )
    
    def get_job_status_with_details(self, job_id: str) -> Dict:
        """Get comprehensive job status."""
        if config.USE_CELERY and hasattr(self._manager, 'get_job_status_with_celery'):
            return self._manager.get_job_status_with_celery(job_id)
        else:
            # Legacy behavior
            job = self.get_job(job_id)
            if not job:
                return {'error': 'Job not found'}
            
            return {
                'job_id': job_id,
                'status': job.status,
                'progress': job.progress,
                'message': job.message,
                'error': job.error_message,
                'output_path': job.output_path
            }
    
    def get_all_jobs(self) -> Dict[str, VideoJob]:
        """Get all jobs (for debugging)."""
        return self._manager.get_all_jobs()
    
    def cleanup_old_jobs(self):
        """Clean up old jobs if supported."""
        if config.USE_CELERY and hasattr(self._manager, 'cleanup_expired_jobs'):
            return self._manager.cleanup_expired_jobs()
        return 0
    
    @property
    def is_celery_enabled(self) -> bool:
        """Check if Celery is enabled."""
        return config.USE_CELERY
    
    @property
    def manager_type(self) -> str:
        """Get the type of job manager being used."""
        if config.USE_CELERY:
            return "celery"
        else:
            return "legacy"


# Global job manager instance
job_manager = JobManagerInterface()


# Backward compatibility functions
def update_job_progress(job_id: str, progress: int, message: str = None):
    """Helper function to update job progress during processing."""
    job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=progress, message=message)
    if message:
        logger.info(f"Job {job_id}: {progress}% - {message}")
