import os
import json
import logging
from fastapi import Query, Request
from fastapi.responses import JSONResponse, Response

from s3_utils import S3Manager
from file_utils import validate_video_extension, clear_tmp_dir, ensure_directory_exists
from models import load_model_config
from pipeline import run_pipeline
from config import API_ROOT_DIR, DEFAULT_S3_BUCKET
from inference import get_pytorch_device

logger = logging.getLogger(__name__)

# Initialize S3 manager
s3_manager = S3Manager()

def process_video_endpoint(
    file: str = Query(..., description="S3 path or local path to input video"),
    model_size: str = Query("xs", description="Model size: xs, s, b, l")
):
    """Process a video from an S3 path or local path using the specified model size.
    
    Args:
        file (str): S3 path or local path to the input video.
        model_size (str): Model size to use for processing (xs, s, b, l).

    Returns:
        JSONResponse: Contains the output video URL or local path.
    """
    return process_video_internal(file, model_size)

def root_endpoint():
    """Health check endpoint."""
    return {"message": "MotionAGFormer API is running."}

def ping_endpoint():
    """Health check endpoint required by SageMaker."""
    try:
        # Basic health check - verify model config can be loaded
        config_path = os.path.join(os.path.dirname(__file__), "model_config_map.json")
        if os.path.exists(config_path):
            return Response(status_code=200)
        else:
            return Response(status_code=500)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(status_code=500)

async def invocations_endpoint(request: Request):
    """SageMaker inference endpoint."""
    try:
        # Parse request body
        body = await request.body()
        if request.headers.get("content-type") == "application/json":
            input_data = json.loads(body.decode())
        else:
            return JSONResponse(status_code=400, content={"error": "Content-Type must be application/json"})
        
        # Extract parameters from input data
        file = input_data.get("file")
        model_size = input_data.get("model_size", "xs")
        
        if not file:
            return JSONResponse(status_code=400, content={"error": "Missing required parameter: file"})
        
        # Process the video using the existing logic
        result = process_video_internal(file, model_size)
        return result
        
    except Exception as e:
        logger.error(f"Invocation failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

def process_video_internal(file: str, model_size: str = "xs") -> JSONResponse:
    """Internal function to process video - shared by different endpoints."""
    TMP_DIR = os.path.join(API_ROOT_DIR, "tmp_api")
    ensure_directory_exists(TMP_DIR)
    clear_tmp_dir(TMP_DIR, keep_videos=False)
    logger.info(f"Temporary directory for API: {TMP_DIR}")

    # --- Input Handling ---
    bucket = None
    
    if s3_manager.is_s3_path(file):
        file_ext = os.path.splitext(file)[-1]
        if not validate_video_extension(file_ext):
            logger.warning(f"Unsupported file extension: {file_ext}")
            return JSONResponse(status_code=400, content={"error": "Only .mp4 and .mov files are supported."})

        input_path = os.path.join(TMP_DIR, os.path.basename(file))  
        bucket, key, err = s3_manager.download_from_s3(file, input_path)
        if err:
            return JSONResponse(status_code=500, content={"error": f"Failed to download from S3: {err}"})

        video_name = os.path.splitext(os.path.basename(input_path))[0]
    else:
        if not os.path.isfile(file):
            logger.error(f"Local file does not exist: {file}")
            return JSONResponse(status_code=400, content={"error": f"Local file does not exist: {file}"})

        input_path = file
        file_ext = os.path.splitext(input_path)[-1]
        if not validate_video_extension(file_ext):
            logger.warning(f"Unsupported file extension: {file_ext}")
            return JSONResponse(status_code=400, content={"error": "Only .mp4 and .mov files are supported."})

        video_name = os.path.splitext(os.path.basename(input_path))[0]
        bucket = DEFAULT_S3_BUCKET

    output_dir = os.path.join(TMP_DIR, f"{video_name}_output")
    ensure_directory_exists(output_dir)
    logger.info(f"Video name extracted: {video_name}")
    logger.info(f"Output directory: {output_dir}")

    # --- Model Config ---
    model_config, err = load_model_config(model_size)
    if err:
        return JSONResponse(status_code=500, content={"error": f"Failed to load model config: {err}"})
    
    device = get_pytorch_device()
    logger.info(f"Using device: {device}")

    # --- Run Pipeline ---
    logger.info(f"\nInput path: {input_path}\nOutput directory: {output_dir}\nDevice: {device}")
    output_video_path, err = run_pipeline(input_path, output_dir, device, model_size, model_config)
    if err:
        return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {err}"})

    # --- Output Handling ---
    output_video_s3_url = None
    if bucket:
        output_video_s3_url, err = s3_manager.upload_to_s3(output_video_path, bucket, video_name)
        if err:
            return JSONResponse(status_code=500, content={"error": f"Failed to upload output video to S3: {err}"})

    # Keep video files in TMP_DIR for local access immediately after processing
    clear_tmp_dir(TMP_DIR, keep_videos=True) 

    # --- Return Response ---
    logger.info(f"Returning processed video: {output_video_path}")
    if output_video_s3_url:
        response = {"output_video_s3_url": output_video_s3_url, "output_video_local_path": output_video_path}
    else:
        response={"output_video_local_path": output_video_path}
    
    return JSONResponse(content=response)

