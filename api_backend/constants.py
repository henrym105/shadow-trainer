# Configuration
from pathlib import Path


# ------------------------------------------------------
# File Paths in Docker Container
# ------------------------------------------------------
API_ROOT_DIR = Path("/app")
UPLOAD_DIR = API_ROOT_DIR / "uploads"
OUTPUT_DIR = API_ROOT_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TMP_PRO_KEYPOINTS_FILE = API_ROOT_DIR / "checkpoint" / "example_SnellBlake.npy"
SAMPLE_VIDEO_PATH = API_ROOT_DIR / "sample_videos" / "Left_Hand_Friend_Side.MOV"

# ------------------------------------------------------
# S3 config for pro keypoints
# ------------------------------------------------------
S3_BUCKET = "shadow-trainer-dev"
S3_PRO_PREFIX = "test/professional/"


INCLUDE_2D_IMAGES = True
