"""
Centralized configuration management for Shadow Trainer API.
"""
import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache


class Settings:
    """Application settings with environment variable support."""
    
    def __init__(self):
        # Application
        self.app_name: str = "Shadow Trainer API"
        self.version: str = "1.0.0"
        self.debug: bool = False
        self.environment: str = "development"
        
        # Server
        self.host: str = "0.0.0.0"
        self.port: int = 8000
        self.workers: int = 1
        
        # Paths
        self.base_dir: Path = Path(__file__).parent.parent
        self.assets_dir: Path = self.base_dir / "assets"
        self.models_dir: Path = self.assets_dir / "models"
        self.videos_dir: Path = self.assets_dir / "videos"
        self.images_dir: Path = self.assets_dir / "images"
        self.temp_dir: Path = self.base_dir / "tmp"
        self.upload_dir: Path = self.temp_dir / "uploads"
        self.output_dir: Path = self.temp_dir / "outputs"
        self.checkpoint_dir: Path = self.base_dir / "checkpoint"
        self.src_dir: Path = self.base_dir / "src"
        
        # Storage
        self.s3_bucket: str = "shadow-trainer-prod"
        self.s3_model_prefix: str = "model_weights"
        self.aws_access_key_id: Optional[str] = None
        self.aws_secret_access_key: Optional[str] = None
        self.aws_region: str = "us-east-1"
        
        # Model settings
        self.default_model_size: str = "xs"
        self.supported_model_sizes: List[str] = ["xs", "s", "b", "l"]
        self.model_sizes: List[str] = self.supported_model_sizes  # Alias for backward compatibility
        self.model_config_file: str = "model_config_map.json"
        
        # Video processing
        self.supported_video_extensions: List[str] = [".mp4", ".mov", ".avi"]
        self.max_file_size: int = 100 * 1024 * 1024  # 100MB
        self.max_processing_time: int = 300  # 5 minutes
        
        # API settings
        self.cors_origins: List[str] = ["*"]
        self.api_v1_prefix: str = "/api/v1"
        self.docs_url: Optional[str] = "/docs"
        self.redoc_url: Optional[str] = "/redoc"
        
        # Logging
        self.log_level: str = "INFO"
        self.log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Load from environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load settings from environment variables."""
        # Debug mode
        if os.getenv("SHADOW_TRAINER_DEBUG", "").lower() in ("true", "1", "yes"):
            self.debug = True
        
        # Log level
        self.log_level = os.getenv("SHADOW_TRAINER_LOG_LEVEL", self.log_level)
        
        # Server settings
        self.host = os.getenv("SHADOW_TRAINER_HOST", self.host)
        port_env = os.getenv("SHADOW_TRAINER_PORT")
        if port_env:
            try:
                self.port = int(port_env)
            except ValueError:
                pass
        
        # AWS settings
        self.s3_bucket = os.getenv("SHADOW_TRAINER_S3_BUCKET", self.s3_bucket)
        self.aws_region = os.getenv("SHADOW_TRAINER_AWS_REGION", self.aws_region)
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def ensure_directories():
    """Ensure all required directories exist."""
    settings = get_settings()
    directories = [
        settings.assets_dir,
        settings.models_dir,
        settings.videos_dir,
        settings.images_dir,
        settings.temp_dir,
        settings.upload_dir,
        settings.output_dir,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_model_config_path() -> Path:
    """Get the path to the model configuration file."""
    settings = get_settings()
    return settings.base_dir / settings.model_config_file


# For backward compatibility
settings = get_settings()
