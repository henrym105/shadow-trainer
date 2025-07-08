"""
Common schema models used across the application.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ModelSize(str, Enum):
    """Supported model sizes."""
    XS = "xs"
    S = "s"
    B = "b"
    L = "l"

class Handedness(str, Enum):
    """Supported handedness options."""
    RIGHT = "Right-handed"
    LEFT = "Left-handed"

class PitchType(str, Enum):
    """Supported pitch types."""
    FF = "FF"  # Four-seam fastball
    SI = "SI"  # Sinker
    SL = "SL"  # Slider
    CH = "CH"  # Changeup

class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None
    timestamp: Optional[str] = None

class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
