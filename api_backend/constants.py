# Configuration
from pathlib import Path


# ------------------------------------------------------
# File Paths in Docker Container
# ------------------------------------------------------
API_ROOT_DIR = Path(__file__).parent
UPLOAD_DIR = API_ROOT_DIR / "uploads"
OUTPUT_DIR = API_ROOT_DIR / "output"
CHECKPOINT_DIR = API_ROOT_DIR / "checkpoint"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TMP_PRO_KEYPOINTS_FILE = API_ROOT_DIR / "checkpoint" / "example_SnellBlake.npy"
TMP_PRO_KEYPOINTS_FILE_S3 = "Spencer_Strider.npy"
SAMPLE_VIDEO_PATH = API_ROOT_DIR / "sample_videos" / "Left_Hand_Friend_Side.MOV"

# ------------------------------------------------------
# S3 config for pro keypoints
# ------------------------------------------------------
S3_BUCKET = "shadow-trainer-dev"
S3_PRO_PREFIX = "test/professional/"

INCLUDE_2D_IMAGES = True

PRO_TEAMS_MAP = {
    "Dean_Kremer": { 
        "name": "Dean Kremer",
        "team": "Baltimore Orioles",
        "city": "Baltimore"
    }, 
    "Justin_Verlander": {
        "name": "Justin Verlander",
        "team": "San Francisco Giants",
        "city": "San Francisco"
    },
    "Kevin_Gausman": {
        "name": "Kevin Gausman",
        "team": "Toronto Blue Jays",
        "city": "Toronto"
    },
    "Spencer_Strider": {
        "name": "Spencer Strider",
        "team": "Atlanta Braves",
        "city": "Atlanta"
    }
}

# ------------------------------------------------------
# Celery Config
# ------------------------------------------------------
# Set 24 hour redis cache timeout 
RESULT_EXPIRES = 24 * 60 * 60
