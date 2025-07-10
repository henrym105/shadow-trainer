import json
import logging
import os
import shutil

import boto3
from fastapi import Body, FastAPI, Query

from pydantic_models import ProcessVideoRequest, ProcessVideoResponse
from src.inference import get_pose2D, get_pose3D, img2video, get_pytorch_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

API_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(API_ROOT_DIR, "tmp_api_output")
os.makedirs(TMP_DIR, exist_ok=True)

s3_client = boto3.client('s3')
app = FastAPI(debug=True)


# --- Utility Functions ---
def is_s3_path(path: str) -> bool:
    return path.startswith("s3://")

def validate_video_extension(file_ext: str) -> bool:
    return file_ext.lower() in [".mp4", ".mov"]

def download_from_s3(s3_path: str, local_path: str) -> tuple:
    """Downloads a file from S3 to a local path.
    Returns a tuple of (bucket, key, error_message).
    If successful, error_message will be None.
    If an error occurs, bucket and key will be None.
    """
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
    """Uploads the output video to S3 and returns the S3 URL."""
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
    """Loads the model configuration from a JSON file based on the specified model size."""
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
    logger.info("Running get_pose2D...")
    logger.info(f"Input path: {input_path}, Output directory: {output_dir}, Device: {device}")
    get_pose2D(input_path, output_dir, device)

    logger.info("Running get_pose3D...")
    pro_keypoints_filepath: str = "./api_backend/checkpoint/example_SnellBlake.npy"

    output_npy = get_pose3D(input_path, output_dir, device, model_size, model_config, pro_keypoints_filepath)
    print(f"Output npy file generated: {output_npy}")

    logger.info("Running img2video...")
    output_video_path = img2video(input_path, output_dir)

    return output_video_path, ""


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
            is_video_file = (os.path.splitext(filename)[-1].lower() in [".mp4", ".mov"])
            is_pose3D_npy = (filename == "pose3D_npy")
            if keep_videos and (is_video_file or is_pose3D_npy):
                continue  
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f'Failed to delete {file_path}. Reason: {e}')




# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    return {"message": "MotionAGFormer API is running."}




# --- Video Processing Endpoint ---
@app.post("/video/process", response_model=ProcessVideoResponse)
def process_video(request: ProcessVideoRequest = Body(...)):
    """Process a video from an S3 path or local path using the specified model size.
    Returns a structured response with output paths and error if any.
    """
    return process_video_internal(request.file, request.model_size)


def process_video_internal(file: str, model_size: str = "xs") -> ProcessVideoResponse:
    """Internal function to process video - extracted from the existing endpoint."""
    clear_tmp_dir(TMP_DIR, keep_videos=False)
    logger.info(f"Temporary directory for API: {TMP_DIR}")

    # --- Input Handling ---
    if is_s3_path(file):
        file_ext = os.path.splitext(file)[-1]
        if not validate_video_extension(file_ext):
            logger.warning(f"Unsupported file extension: {file_ext}")
            return ProcessVideoResponse(
                output_video_local_path="",
                output_video_s3_url=None,
                error="Only .mp4 and .mov files are supported."
            )

        input_path = os.path.join(TMP_DIR, os.path.basename(file))  
        bucket, key, err = download_from_s3(file, input_path)
        if err:
            return ProcessVideoResponse(
                output_video_local_path="",
                output_video_s3_url=None,
                error=f"Failed to download from S3: {err}"
            )

        video_name = os.path.splitext(os.path.basename(input_path))[0]
    else:
        if not os.path.isfile(file):
            logger.error(f"Local file does not exist: {file}")
            return ProcessVideoResponse(
                output_video_local_path="",
                output_video_s3_url=None,
                error=f"Local file does not exist: {file}"
            )

        input_path = file
        file_ext = os.path.splitext(input_path)[-1]
        if not validate_video_extension(file_ext):
            logger.warning(f"Unsupported file extension: {file_ext}")
            return ProcessVideoResponse(
                output_video_local_path="",
                output_video_s3_url=None,
                error="Only .mp4 and .mov files are supported."
            )

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
        return ProcessVideoResponse(
            output_video_local_path="",
            output_video_s3_url=None,
            error=f"Failed to load model config: {err}"
        )
    
    device = get_pytorch_device()
    logger.info(f"Using device: {device}")

    # --- Run Pipeline ---
    logger.info(f"\nInput path: {input_path}\nOutput directory: {output_dir}, \nDevice: {device}")
    output_video_path, err = run_pipeline(input_path, output_dir, device, model_size, model_config)
    if err:
        return ProcessVideoResponse(
            output_video_local_path="",
            output_video_s3_url=None,
            error=f"Pipeline failed: {err}"
        )

    # --- Output Handling ---
    output_video_s3_url, upload_err = upload_to_s3(output_video_path, bucket, video_name)
    if upload_err:
        # Return local path, but s3 url is blank, error is set
        return ProcessVideoResponse(
            output_video_local_path=output_video_path,
            output_video_s3_url=None,
            error=f"Failed to upload output video to S3: {upload_err}"
        )

    # --- Return Response ---
    logger.info(f"Returning processed video: {output_video_path}")
    return ProcessVideoResponse(
        output_video_local_path=output_video_path,
        output_video_s3_url=output_video_s3_url,
        error=None
    )

