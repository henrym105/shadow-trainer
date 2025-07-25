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
TMP_PRO_KEYPOINTS_FILE_S3 = "BlakeSnell_median.npy"
SAMPLE_VIDEO_PATH = API_ROOT_DIR / "sample_videos" / "Left_Hand_Friend_Side.MOV"

# ------------------------------------------------------
# S3 config for pro keypoints
# ------------------------------------------------------
S3_BUCKET = "shadow-trainer-prod"
S3_PRO_PREFIX = "pro_3d_keypoints/"

PRO_TEAMS_MAP = {
    "BlakeSnell": { 
        "name": "Blake Snell",
        "team": "San Francisco Giants",
        "city": "San Francisco"
    }, 
    "SpencerStrider": {
        "name": "Spencer Strider",
        "team": "Atlanta Braves",
        "city": "Atlanta"
    },
    "JustinVerlander": {
        "name": "Justin Verlander",
        "team": "Houston Astros",
        "city": "Houston"
    },
    "KevinGausman": {
        "name": "Kevin Gausman",
        "team": "Toronto Blue Jays",
        "city": "Toronto"
    },
    "DeanKremer": {
        "name": "Dean Kremer",
        "team": "Baltimore Orioles",
        "city": "Baltimore"
    }
}

# ------------------------------------------------------
# Celery Config
# ------------------------------------------------------
# Set 1 hour redis cache timeout
RESULT_EXPIRES = 1 * 60 * 60
