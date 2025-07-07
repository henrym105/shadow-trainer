

# --- Imports and Setup ---
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
import os
import logging
import json
import boto3
import shutil
from inference import get_pose2D, get_pose3D, img2video, get_pytorch_device, get_or_download_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
s3_client = boto3.client('s3')

API_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Utility Functions ---
def is_s3_path(path: str) -> bool:
    return path.startswith("s3://")

def validate_video_extension(file_ext: str) -> bool:
    return file_ext.lower() in [".mp4", ".mov"]

def download_from_s3(s3_path: str, local_path: str) -> tuple:
    s3_path_no_prefix = s3_path[5:]
    bucket, key = s3_path_no_prefix.split('/', 1)
    try:
        logger.info(f"Downloading {s3_path} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        return bucket, key, None
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return None, None, str(e)

def upload_to_s3(local_path: str, bucket: str, video_name: str) -> tuple:
    output_video_s3_key = f"tmp/{video_name}/{os.path.basename(local_path)}"
    try:
        logger.info(f"Uploading output video {local_path} to s3://{bucket}/{output_video_s3_key}")
        s3_client.upload_file(local_path, bucket, output_video_s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{output_video_s3_key}")
        output_video_s3_url = f"s3://{bucket}/{output_video_s3_key}"
        return output_video_s3_url, None
    except Exception as e:
        logger.error(f"Failed to upload output video to S3: {e}")
        return None, str(e)

def load_model_config(model_size: str, config_path: str) -> tuple:
    try:
        # Ensure config_path is absolute
        config_path_abs = config_path
        if not os.path.isabs(config_path):
            config_path_abs = os.path.abspath(os.path.join(API_ROOT_DIR, config_path))
        with open(config_path_abs, "r") as f:
            model_config_map = json.load(f)
        logger.info(f"Loaded model_config_map: {model_config_map}")
        model_config = model_config_map.get(model_size, model_config_map["b"])
        logger.info(f"Using model config: {model_config}")
        # If model_config is a relative path, make it absolute
        if not os.path.isabs(model_config):
            model_config = os.path.abspath(os.path.join(os.path.dirname(config_path_abs), model_config))
        return model_config, None
    except Exception as e:
        logger.error(f"Failed to load or parse model_config_map.json: {e}")
        return None, str(e)


def run_pipeline(input_path: str, output_dir: str, device: str, model_size: str, model_config: dict) -> tuple:
    """Runs the full processing pipeline consisting of 2D pose estimation, 3D pose estimation, and image-to-video conversion.
    
    Args:
        input_path (str): Path to the input data (e.g., images or video).
        output_dir (str): Directory where output files will be saved.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        model_size (str): Size specification for the 3D pose estimation model.
        model_config (dict): Configuration dictionary for the 3D pose estimation model.
    Returns:
        tuple[str | None, str | None]: 
            - output_video_path (str | None): Path to the generated output video if successful, otherwise None.
            - error_message (str | None): Error message if the pipeline fails, otherwise None.
    """


    try:
        logger.info("Running get_pose2D...")
        logger.info(f"Input path: {input_path}, Output directory: {output_dir}, Device: {device}")
        get_pose2D(input_path, output_dir, device)
        
        logger.info("Running get_pose3D...")
        get_pose3D(input_path, output_dir, device, model_size, model_config)
        
        logger.info("Running img2video...")
        output_video_path = img2video(input_path, output_dir)
        return output_video_path, ""
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return "", str(e)

# --- Cleanup Utility ---
def clear_tmp_dir(dir_path, keep_videos=False):
    """Delete all files and folders in the given directory.
    If keep_videos is True, keep .mp4 and .mov files.

    Args:
        dir_path (str): Path to the directory to clear.
        keep_videos (bool): If True, do not delete video files (.mp4, .mov).
    """
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            # Skip video files if keep_videos is True
            if keep_videos and os.path.splitext(filename)[-1].lower() in [".mp4", ".mov"]:
                continue
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f'Failed to delete {file_path}. Reason: {e}')



# --- Main Endpoint ---
@app.post("/process_video/")
def process_video(
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


# --- Health Check Endpoint ---
@app.get("/")
def root():
    return {"message": "MotionAGFormer API is running."}

# --- SageMaker Required Endpoints ---
@app.get("/ping")
def ping():
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

@app.post("/invocations")
async def invocations(request: Request):
    """SageMaker inference endpoint."""
    try:
        # Parse request body
        body = await request.body()
        if request.headers.get("content-type") == "application/json":
            input_data = json.loads(body.decode())
        else:
            # Handle other content types if needed
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

def process_video_internal(file: str, model_size: str = "xs"):
    """Internal function to process video - extracted from the existing endpoint."""
    # TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_api")
    TMP_DIR = os.path.join(API_ROOT_DIR, "tmp_api")
    os.makedirs(TMP_DIR, exist_ok=True)
    clear_tmp_dir(TMP_DIR, keep_videos=False)
    logger.info(f"Temporary directory for API: {TMP_DIR}")

    # --- Input Handling ---
    if is_s3_path(file):
        file_ext = os.path.splitext(file)[-1]
        if not validate_video_extension(file_ext):
            logger.warning(f"Unsupported file extension: {file_ext}")
            return JSONResponse(status_code=400, content={"error": "Only .mp4 and .mov files are supported."})

        input_path = os.path.join(TMP_DIR, os.path.basename(file))  
        bucket, key, err = download_from_s3(file, input_path)
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
        bucket = "shadow-trainer-prod"  # Change as needed or make configurable

    output_dir = os.path.join(TMP_DIR, f"{video_name}_output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Video name extracted: {video_name}")
    logger.info(f"Output directory: {output_dir}")

    # --- Model Config ---
    config_path = os.path.join(API_ROOT_DIR, "model_config_map.json")
    model_config, err = load_model_config(model_size, config_path)
    if err:
        return JSONResponse(status_code=500, content={"error": f"Failed to load model config: {err}"})
    
    device = get_pytorch_device()
    logger.info(f"Using device: {device}")

    # --- Run Pipeline ---
    logger.info(f"\nInput path: {input_path}\nOutput directory: {output_dir}, \nDevice: {device}")
    output_video_path, err = run_pipeline(input_path, output_dir, device, model_size, model_config)
    if err:
        return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {err}"})

    # --- Output Handling ---
    output_video_s3_url = None
    if bucket:
        output_video_s3_url, err = upload_to_s3(output_video_path, bucket, video_name)
        if err:
            return JSONResponse(status_code=500, content={"error": f"Failed to upload output video to S3: {err}"})

    # Keep video files in TMP_DIR for local access immediately after processing
    clear_tmp_dir(TMP_DIR, keep_videos=True) 

    # --- Return Response ---
    logger.info(f"Returning processed video: {output_video_path}")
    if output_video_s3_url:
        return {"output_video_s3_url": output_video_s3_url,
                "output_video_local_path": output_video_path}
    else:
        return {"output_video_local_path": output_video_path}

