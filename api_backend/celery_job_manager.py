"""
Celery-based Job Manager for handling video processing jobs.
This replaces the in-memory job_manager.py with a Redis-backed solution.
"""

import json
import logging
import time
import uuid
from typing import Dict, Optional, Any

import redis
from celery.result import AsyncResult

from celery_app import celery_app
from pydantic_models import VideoJob, JobStatus

logger = logging.getLogger(__name__)


class CeleryJobManager:
    """
    Redis-backed job manager using Celery for video processing.
    Provides persistent job storage and distributed task execution.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0, redis_password: str = None):
        """Initialize the Celery job manager with Redis connection."""
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        self.job_prefix = "shadowtrainer:job:"
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _get_job_key(self, job_id: str) -> str:
        """Get Redis key for job data."""
        return f"{self.job_prefix}{job_id}"
    
    def create_job(self, filename: str, input_path: str) -> VideoJob:
        """Create a new video processing job."""
        job = VideoJob.create_new_job(filename, input_path)
        
        # Store job in Redis
        job_key = self._get_job_key(job.job_id)
        job_data = job.model_dump()
        
        # Filter out None values for Redis storage
        job_data = {k: v for k, v in job_data.items() if v is not None}
        
        try:
            self.redis_client.hset(job_key, mapping=job_data)
            # Set expiration for job data (24 hours)
            self.redis_client.expire(job_key, 86400)
            
            logger.info(f"Created new job: {job.job_id} for file: {filename}")
            return job
            
        except Exception as e:
            logger.error(f"Failed to create job {job.job_id}: {e}")
            raise
    
    def get_job(self, job_id: str) -> Optional[VideoJob]:
        """Get job by ID from Redis."""
        job_key = self._get_job_key(job_id)
        
        try:
            job_data = self.redis_client.hgetall(job_key)
            if not job_data:
                return None
            
            # Convert Redis strings back to proper types
            if 'progress' in job_data:
                job_data['progress'] = int(job_data['progress'])
            
            # Set default values for potentially missing fields
            job_data.setdefault('output_path', None)
            job_data.setdefault('message', None)
            job_data.setdefault('error_message', None)
            job_data.setdefault('celery_task_id', None)
            
            return VideoJob(**job_data)
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    def update_job_status(self, job_id: str, status: JobStatus, progress: int = None, 
                         error: str = None, output_path: str = None, message: str = None):
        """Update job status and progress in Redis."""
        job_key = self._get_job_key(job_id)
        
        try:
            # Prepare update data, filtering out None values
            update_data = {'status': status.value}
            
            if progress is not None:
                update_data['progress'] = progress
            if error:
                update_data['error_message'] = error
            if output_path:
                update_data['output_path'] = output_path
            if message:
                update_data['message'] = message
            
            # Update in Redis
            self.redis_client.hset(job_key, mapping=update_data)
            
            logger.info(f"Job {job_id} updated: status={status}, progress={progress}")
            if message:
                logger.info(f"Job {job_id} message: {message}")
                
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
    
    def submit_job(self, job_id: str, model_size: str = 'xs', is_lefty: bool = False,
                  pro_keypoints_filename: str = None, include_2d_images: bool = True) -> str:
        """Submit a job for async processing via Celery."""
        
        # Get job details
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Prepare task data
        task_data = {
            'job_id': job_id,
            'input_path': job.input_path,
            'model_size': model_size,
            'is_lefty': is_lefty,
            'pro_keypoints_filename': pro_keypoints_filename,
            'include_2d_images': include_2d_images
        }
        
        try:
            # Submit task to Celery
            from tasks.video_processing import process_video_task
            task_result = process_video_task.delay(task_data)
            
            # Update job with Celery task ID
            self.update_job_with_celery_task(job_id, task_result.id)
            self.update_job_status(job_id, JobStatus.QUEUED)
            
            logger.info(f"Submitted job {job_id} to Celery with task ID: {task_result.id}")
            return task_result.id
            
        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            self.update_job_status(job_id, JobStatus.FAILED, error=str(e))
            raise
    
    def update_job_with_celery_task(self, job_id: str, celery_task_id: str):
        """Update job with Celery task ID."""
        job_key = self._get_job_key(job_id)
        self.redis_client.hset(job_key, 'celery_task_id', celery_task_id)
    
    def get_job_status_with_celery(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive job status including Celery task state."""
        job = self.get_job(job_id)
        if not job:
            return {'error': 'Job not found'}
        
        result = {
            'job_id': job_id,
            'status': job.status,
            'progress': job.progress,
            'message': job.message,
            'error': job.error_message,
            'output_path': job.output_path
        }
        
        # Get Celery task status if available
        if hasattr(job, 'celery_task_id') and job.celery_task_id:
            try:
                task_result = AsyncResult(job.celery_task_id, app=celery_app)
                result['celery_status'] = task_result.status
                
                # Get task metadata
                if task_result.info:
                    if isinstance(task_result.info, dict):
                        result['celery_progress'] = task_result.info.get('progress', 0)
                        result['celery_message'] = task_result.info.get('message', '')
                        if 'error' in task_result.info:
                            result['celery_error'] = task_result.info['error']
                
                # Update job status based on Celery task status
                if task_result.status == 'PENDING':
                    self.update_job_status(job_id, JobStatus.QUEUED)
                elif task_result.status == 'STARTED' or task_result.status == 'PROCESSING':
                    progress = 0
                    message = None
                    if task_result.info and isinstance(task_result.info, dict):
                        progress = task_result.info.get('progress', 0)
                        message = task_result.info.get('message')
                    self.update_job_status(job_id, JobStatus.PROCESSING, progress=progress, message=message)
                elif task_result.status == 'SUCCESS':
                    output_path = task_result.result if isinstance(task_result.result, str) else None
                    self.update_job_status(job_id, JobStatus.COMPLETED, progress=100, output_path=output_path)
                elif task_result.status == 'FAILURE':
                    error_msg = str(task_result.info) if task_result.info else 'Unknown error'
                    self.update_job_status(job_id, JobStatus.FAILED, error=error_msg)
                    
            except Exception as e:
                logger.warning(f"Failed to get Celery task status for job {job_id}: {e}")
        
        return result
    
    def get_all_jobs(self) -> Dict[str, VideoJob]:
        """Get all jobs from Redis (for debugging/monitoring)."""
        try:
            pattern = f"{self.job_prefix}*"
            job_keys = self.redis_client.keys(pattern)
            
            jobs = {}
            for job_key in job_keys:
                job_id = job_key.replace(self.job_prefix, '')
                job = self.get_job(job_id)
                if job:
                    jobs[job_id] = job
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get all jobs: {e}")
            return {}
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job from Redis."""
        job_key = self._get_job_key(job_id)
        
        try:
            # Cancel Celery task if it exists
            job = self.get_job(job_id)
            if job and hasattr(job, 'celery_task_id') and job.celery_task_id:
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            
            # Delete from Redis
            deleted = self.redis_client.delete(job_key)
            logger.info(f"Deleted job {job_id}: {bool(deleted)}")
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False
    
    def cleanup_expired_jobs(self, max_age_hours: int = 24):
        """Clean up expired jobs from Redis."""
        try:
            pattern = f"{self.job_prefix}*"
            job_keys = self.redis_client.keys(pattern)
            
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            deleted_count = 0
            for job_key in job_keys:
                # Check if job is older than cutoff
                job_age = self.redis_client.ttl(job_key)
                if job_age == -1:  # No expiration set
                    # Set expiration for old jobs
                    self.redis_client.expire(job_key, 3600)  # 1 hour
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} expired jobs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired jobs: {e}")
            return 0


# Global job manager instance
def get_job_manager() -> CeleryJobManager:
    """Get the global job manager instance."""
    import os
    
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', '6379'))
    redis_db = int(os.getenv('REDIS_DB', '0'))
    redis_password = os.getenv('REDIS_PASSWORD', None)
    
    return CeleryJobManager(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_password=redis_password
    )


# Backward compatibility helper
def update_job_progress(job_id: str, progress: int, message: str = None):
    """Helper function to update job progress during processing."""
    job_manager = get_job_manager()
    job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=progress, message=message)
    if message:
        logger.info(f"Job {job_id}: {progress}% - {message}")
