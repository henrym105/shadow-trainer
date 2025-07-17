from pydantic import BaseModel
from typing import Optional
from enum import Enum
import uuid


class JobStatus(str, Enum):
    """Video processing job status enum"""
    QUEUED = "queued"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


class VideoUploadResponse(BaseModel):
    """Response model for video upload"""
    job_id: str
    message: str
    estimated_time: int = 120  # seconds


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status check"""
    job_id: str
    status: JobStatus
    progress: int  # 0-100
    message: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None


class VideoJob(BaseModel):
    """Internal model representing a video processing job"""
    job_id: str
    original_filename: str
    input_path: str
    output_path: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    message: Optional[str] = None
    error_message: Optional[str] = None
    
    @classmethod
    def create_new_job(cls, filename: str, input_path: str) -> 'VideoJob':
        """Create a new video processing job"""
        return cls(
            job_id=str(uuid.uuid4()),
            original_filename=filename,
            input_path=input_path
        )
