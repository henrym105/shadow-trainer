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
TMP_PRO_KEYPOINTS_FILE_S3 = "DeanKremer_median.npy"
# SAMPLE_VIDEO_PATH = API_ROOT_DIR / "sample_videos" / "sample.mov"
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
    },
    "AntonioSenzatela": {
        "name": "Antonio Senzatela",
        "team": "Colorado Rockies",
        "city": "Denver"
    },
    "ShaneBieber": {
        "name": "Shane Bieber",
        "team": "Cleveland Guardians",
        "city": "Cleveland"
    },
    "CarlosRodon": {
        "name": "Carlos Rodón",
        "team": "New York Yankees",
        "city": "New York"
    },
    "EduardoRodriguez": {
        "name": "Eduardo Rodríguez",
        "team": "Arizona Diamondbacks",
        "city": "Phoenix"
    },
    "GerritCole": {
        "name": "Gerrit Cole",
        "team": "New York Yankees",
        "city": "New York"
    },
    "JacobDeGrom": {
        "name": "Jacob deGrom",
        "team": "Texas Rangers",
        "city": "Arlington"
    },
    "ClaytonKershaw": {
        "name": "Clayton Kershaw",
        "team": "Los Angeles Dodgers",
        "city": "Los Angeles"
    },
    "ReynaldoLopez": {
        "name": "Reynaldo López",
        "team": "Atlanta Braves",
        "city": "Atlanta"
    },
    "LucasGiolito": {
        "name": "Lucas Giolito",
        "team": "Boston Red Sox",
        "city": "Boston"
    },
    "GermanMarquez": {
        "name": "Germán Márquez",
        "team": "Colorado Rockies",
        "city": "Denver"
    },
    "MichaelWacha": {
        "name": "Michael Wacha",
        "team": "Kansas City Royals",
        "city": "Kansas City"
    },
    "FreddyPeralta": {
        "name": "Freddy Peralta",
        "team": "Milwaukee Brewers",
        "city": "Milwaukee"
    },
    "TrevorWilliams": {
        "name": "Trevor Williams",
        "team": "Washington Nationals",
        "city": "Washington"
    },
    "TylerAnderson": {
        "name": "Tyler Anderson",
        "team": "Los Angeles Angels",
        "city": "Anaheim"
    },
    "TylerMahle": {
        "name": "Tyler Mahle",
        "team": "Texas Rangers",
        "city": "Arlington"
    },
}

# ------------------------------------------------------
# Celery Config
# ------------------------------------------------------
# Set 5 hour redis cache timeout
RESULT_EXPIRES = 5 * 60 * 60


# ------------------------------------------------------
# Visualizations
# ------------------------------------------------------
VALID_PLOT_TYPES = ["hip_rotation", "shoulder_rotation", "hip_shoulder_separation", "hip_rotation_speed", "shoulder_rotation_speed"]
