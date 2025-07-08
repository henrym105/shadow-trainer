
import os

# === AWS S3 Settings ===
S3_BUCKET_INPUT = "shadow-trainer-prod"
S3_INPUT_FOLDER = "tmp/"
S3_PROFESSIONAL_KEYPOINT_PATH = "keypoints/pro_throw.npy"
S3_TEMP_OUTPUT_FOLDER = "processed/"

# === API Settings ===
# Used by Streamlit to talk to FastAPI
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# === Inference Settings ===

# === Visualization ===
VISUALIZATION_FPS = 30
OUTPUT_VIDEO_FORMAT = "mp4"