"""
Main FastAPI application for the MotionAGFormer video processing API.
"""
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response
from endpoints import (
    process_video_endpoint,
    root_endpoint,
    ping_endpoint,
    invocations_endpoint
)

# Initialize FastAPI app
app = FastAPI(
    title="MotionAGFormer API",
    description="Backend API for 3D human pose estimation from video",
    version="1.0.0"
)

# --- Route Definitions ---
@app.post("/process_video/")
def process_video(
    file: str = Query(..., description="S3 path or local path to input video"),
    model_size: str = Query("xs", description="Model size: xs, s, b, l")
):
    """Process a video from an S3 path or local path using the specified model size."""
    return process_video_endpoint(file, model_size)

@app.get("/")
def root():
    """Health check endpoint."""
    return root_endpoint()

@app.get("/ping")
def ping():
    """Health check endpoint required by SageMaker."""
    return ping_endpoint()

@app.post("/invocations")
async def invocations(request: Request):
    """SageMaker inference endpoint."""
    return await invocations_endpoint(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

