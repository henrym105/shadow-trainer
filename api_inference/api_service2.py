# --- Imports and Setup ---
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
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
def clear_tmp_dir(dir_path):
    """Delete all files and folders in the given directory."""
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f'Failed to delete {file_path}. Reason: {e}')



# --- Main Endpoint ---
@app.post("/process_video2/")
def process_video2(
    file: str = Query(..., description="S3 path or local path to input video"),
    model_size: str = Query("xs", description="Model size: xs, s, b, l")
):
    """Process a video from an S3 path or local path using the specified model size.
    
    Args:
        file (str): S3 path or local path to the input video.
        model_size (str): Model size to use for processing (xs, s, b, l).

    Returns:
        JSONResponse: Contains the output video URL or local path, and keypoint .npy paths.
    """
    TMP_DIR = os.path.join(API_ROOT_DIR, "tmp_api")
    os.makedirs(TMP_DIR, exist_ok=True)
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

    # --- Find 3D keypoint .npy files ---
    user_kpts_path = None
    # Look for user .npy file in output_dir
    for fname in os.listdir(output_dir):
        if fname.endswith("user_keypoints.npy"):
            user_kpts_path = os.path.join(output_dir, fname)

    # Use fixed pitcher keypoints S3 URL
    pitcher_kpts_path = "https://shadow-trainer-dev.s3.us-east-2.amazonaws.com/cleaned_numpy/BieberShaneR/FC/cropped_b27ca713-f4a2-4f52-b867-6c9de15a2e20_1.npy"

    logger.info(f"Returning processed video: {output_video_path}")
    response_dict = {"output_video_local_path": output_video_path}
    if output_video_s3_url:
        response_dict["output_video_s3_url"] = output_video_s3_url
    if user_kpts_path:
        response_dict["user_keypoints_npy"] = user_kpts_path
    if pitcher_kpts_path:
        response_dict["pitcher_keypoints_npy"] = pitcher_kpts_path
    return response_dict

# --- Health Check Endpoint ---
@app.get("/health2")
def root2():
    return {"message": "MotionAGFormer API v2 is running."}
