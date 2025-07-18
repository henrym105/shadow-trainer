"""
Celery-based job manager for Shadow Trainer API.
Direct interface to Celery job management without legacy compatibility.
"""

import logging
from typing import Dict, Optional

from celery_job_manager import get_job_manager
from pydantic_models import VideoJob, JobStatus

logger = logging.getLogger(__name__)


class CeleryJobManager:
    """
    Celery-based job management interface.
    """
    
    def __init__(self):
        self._manager = get_job_manager()
        logger.info("Initialized Celery job manager")
    
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
        """Submit a job for Celery processing."""
        return self._manager.submit_job(
            job_id=job_id,
            model_size=model_size,
            is_lefty=is_lefty,
            pro_keypoints_filename=pro_keypoints_filename,
            include_2d_images=include_2d_images
        )
    
    def get_job_status_with_details(self, job_id: str) -> Dict:
        """Get comprehensive job status from Celery."""
        return self._manager.get_job_status_with_celery(job_id)
    
    def get_all_jobs(self) -> Dict[str, VideoJob]:
        """Get all jobs."""
        return self._manager.get_all_jobs()
    
    def cleanup_old_jobs(self):
        """Clean up expired jobs."""
        return self._manager.cleanup_expired_jobs()


# Global job manager instance
job_manager = CeleryJobManager()


# Helper function for updating job progress during processing
def update_job_progress(job_id: str, progress: int, message: str = None):
    """Helper function to update job progress during processing."""
    job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=progress, message=message)
    if message:
        logger.info(f"Job {job_id}: {progress}% - {message}")
