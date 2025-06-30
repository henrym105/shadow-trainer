
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import logging
import json
from inference import get_pose2D, get_pose3D, img2video, get_pytorch_device
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/process_video/")
def process_video(file: UploadFile = File(...), model_size: str = "xs"):
    
    TMP_DIR = os.path.join(os.getcwd(), "tmp_api")
    os.makedirs(TMP_DIR, exist_ok=True)
    logger.info(f"Temporary directory for API: {TMP_DIR}")

    logger.info(f"Received file: {file.filename}, model_size: {model_size}")
    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[-1]
    if file_ext.lower() not in [".mp4", ".mov"]:
        logger.warning(f"Unsupported file extension: {file_ext}")
        return JSONResponse(status_code=400, content={"error": "Only .mp4 and .mov files are supported."})
    timestamp = int(time.time() * 1000)
    input_path = os.path.join(TMP_DIR, f"input_{timestamp}{file_ext}") 
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved input file to: {input_path}")
    except Exception as e:
        logger.error(f"Failed to save input file: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to save input file."})

    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(TMP_DIR, video_name) # Ensure output_dir ends with a slash
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Video name extracted: {video_name}")
    logger.info(f"Output directory: {output_dir}")

    # Model config (now loaded from JSON file)
    config_path = os.path.join(os.path.dirname(__file__), "model_config_map.json")
    try:
        with open(config_path, "r") as f:
            model_config_map = json.load(f)
        logger.info(f"Loaded model_config_map: {model_config_map}")
        model_config = model_config_map.get(model_size, model_config_map["b"])
        logger.info(f"Using model config: {model_config}")
    except Exception as e:
        logger.error(f"Failed to load or parse model_config_map.json: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to load model config: {e}"})
    device = get_pytorch_device()
    logger.info(f"Using device: {device}")

    # Run pipeline
    logger.info(f"\nInput path: {input_path}\nOutput directory: {output_dir}, \nDevice: {device}")
    try:
        logger.info("Running get_pose2D...")
        get_pose2D(input_path, output_dir, device)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {e}"})

    try:
        logger.info("Running get_pose3D...")
        get_pose3D(input_path, output_dir, device, model_size, model_config)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {e}"})

    try:
        logger.info("Running img2video...")
        output_video_path = img2video(input_path, output_dir)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {e}"})


    # Find output video
    # output_video = os.path.join(output_dir, f"output_{file.filename}.mp4")
    logger.info(f"Looking for output video at: {output_video_path}")
    if not os.path.exists(output_video_path):
        logger.error("Output video not found.")
        return JSONResponse(status_code=500, content={"error": "Output video not found."})

    logger.info(f"Returning processed video: {output_video_path}")
    return FileResponse(output_video_path, media_type="video/mp4", filename=f"processed_{file.filename}")

@app.get("/")
def root():
    return {"message": "MotionAGFormer API is running."}
