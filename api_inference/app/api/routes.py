"""
API routes for video processing and health checks.
"""
import os
import sys
from pathlib import Path
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

# Add parent directories to Python path to import existing modules
API_INFERENCE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(API_INFERENCE_DIR))

from endpoints import process_video_endpoint, ping_endpoint, invocations_endpoint

api_router = APIRouter()

@api_router.post("/process_video/")
def process_video(
    file: str = Query(..., description="S3 path or local path to input video"),
    model_size: str = Query("xs", description="Model size: xs, s, b, l"),
    handedness: str = Query("Right-handed", description="User's dominant hand"),
    pitch_type: str = Query("", description="Comma-separated list of pitch types")
):
    """Process a video from an S3 path or local path using the specified model size."""
    # For now, we'll ignore handedness and pitch_type but include them for future use
    return process_video_endpoint(file, model_size)

@api_router.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Shadow Trainer API is running"}

@api_router.get("/ping")
def ping():
    """Health check endpoint required by SageMaker."""
    return ping_endpoint()

@api_router.post("/invocations")
async def invocations(request: Request):
    """SageMaker inference endpoint."""
    return await invocations_endpoint(request)
