"""
Configuration and constants for the API service.
"""
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Directory constants
API_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# S3 configuration
DEFAULT_S3_BUCKET = "shadow-trainer-prod"

# File extensions
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov"]

# Model configuration
MODEL_CONFIG_FILE = "model_config_map.json"
