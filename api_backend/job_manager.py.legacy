"""
Job Manager for handling video processing jobs asynchronously.
This module provides a simple in-memory job queue and status tracking.
"""

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
import uuid

from pydantic_models import VideoJob, JobStatus

logger = logging.getLogger(__name__)


class JobManager:
    """
    Simple in-memory job manager for video processing.
    In production, this would be replaced with Redis/Database + Celery.
    """
    
    def __init__(self, max_workers: int = 2):
        self.jobs: Dict[str, VideoJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        
    def create_job(self, filename: str, input_path: str) -> VideoJob:
        """Create a new video processing job"""
        job = VideoJob.create_new_job(filename, input_path)
        
        with self._lock:
            self.jobs[job.job_id] = job
            
        logger.info(f"Created new job: {job.job_id} for file: {filename}")
        return job
    
    def get_job(self, job_id: str) -> Optional[VideoJob]:
        """Get job by ID"""
        with self._lock:
            return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus, progress: int = None, error: str = None, output_path: str = None, message: str = None):
        """Update job status and progress"""
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].status = status
                if progress is not None:
                    self.jobs[job_id].progress = progress
                if error:
                    self.jobs[job_id].error_message = error
                if output_path:
                    self.jobs[job_id].output_path = output_path
                if message:
                    self.jobs[job_id].message = message
                    
                logger.info(f"Job {job_id} updated: status={status}, progress={progress}")
                if message:
                    logger.info(f"Job {job_id} message: {message}")
    
    def submit_job(self, job_id: str, processing_function, *args, **kwargs):
        """Submit a job for async processing"""
        def process_with_updates():
            try:
                self.update_job_status(job_id, JobStatus.PROCESSING, 0)
                result = processing_function(job_id, *args, **kwargs)
                self.update_job_status(job_id, JobStatus.COMPLETED, 100, output_path=result)
                return result
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                self.update_job_status(job_id, JobStatus.FAILED, error=str(e))
                raise
        
        future = self.executor.submit(process_with_updates)
        logger.info(f"Submitted job {job_id} for processing")
        return future
    
    def get_all_jobs(self) -> Dict[str, VideoJob]:
        """Get all jobs (for debugging)"""
        with self._lock:
            return self.jobs.copy()


# Global job manager instance
job_manager = JobManager(max_workers=2)


def update_job_progress(job_id: str, progress: int, message: str = None):
    """Helper function to update job progress during processing"""
    job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress)
    if message:
        logger.info(f"Job {job_id}: {progress}% - {message}")
