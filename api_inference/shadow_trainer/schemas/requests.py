"""
Request schemas for API endpoints.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from .common import ModelSize, Handedness, PitchType

class VideoProcessRequest(BaseModel):
    """Request schema for video processing."""
    
    file_path: str = Field(..., description="Path to video file (local or S3)")
    model_size: ModelSize = Field(ModelSize.XS, description="Model size for processing")
    handedness: Handedness = Field(Handedness.RIGHT, description="User's dominant hand")
    pitch_types: List[PitchType] = Field(default_factory=list, description="List of pitch types")
    
    @validator("file_path")
    def validate_file_path(cls, v):
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()

class S3UploadRequest(BaseModel):
    """Request schema for S3 upload."""
    
    filename: str = Field(..., description="Name of the file to upload")
    content_type: str = Field(..., description="MIME type of the file")
    
    @validator("filename")
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()

class HealthCheckRequest(BaseModel):
    """Request schema for health check."""
    
    include_details: bool = Field(False, description="Include detailed health information")

class ModelConfigRequest(BaseModel):
    """Request schema for model configuration."""
    
    model_size: ModelSize = Field(..., description="Model size to get configuration for")
