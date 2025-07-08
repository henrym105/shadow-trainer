"""
Response schemas for API endpoints.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from .common import BaseResponse, ProcessingStatus, ModelSize

class VideoProcessResponse(BaseResponse):
    """Response schema for video processing."""
    
    output_video_local_path: Optional[str] = Field(None, description="Local path to processed video")
    output_video_s3_url: Optional[str] = Field(None, description="S3 URL to processed video")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to process video")
    model_size_used: Optional[ModelSize] = Field(None, description="Model size used for processing")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional processing metadata")

class HealthResponse(BaseResponse):
    """Response schema for health checks."""
    
    status: str = Field("healthy", description="Health status")
    version: str = Field(..., description="Application version")
    uptime_seconds: Optional[float] = Field(None, description="Application uptime in seconds")
    model_status: Optional[Dict[str, str]] = Field(None, description="Status of loaded models")
    system_info: Optional[Dict[str, Any]] = Field(None, description="System information")

class S3UploadResponse(BaseResponse):
    """Response schema for S3 upload."""
    
    s3_url: str = Field(..., description="S3 URL of uploaded file")
    file_size: int = Field(..., description="Size of uploaded file in bytes")
    upload_time_seconds: float = Field(..., description="Time taken to upload")

class ModelConfigResponse(BaseResponse):
    """Response schema for model configuration."""
    
    model_size: ModelSize = Field(..., description="Model size")
    config_path: str = Field(..., description="Path to model configuration file")
    checkpoint_path: Optional[str] = Field(None, description="Path to model checkpoint")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters")

class ProcessingStatusResponse(BaseResponse):
    """Response schema for processing status."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress_percentage: Optional[float] = Field(None, description="Processing progress (0-100)")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if status is failed")

class FileListResponse(BaseResponse):
    """Response schema for file listing."""
    
    files: List[Dict[str, Any]] = Field(..., description="List of files with metadata")
    total_count: int = Field(..., description="Total number of files")
    directory_path: str = Field(..., description="Directory path that was listed")
