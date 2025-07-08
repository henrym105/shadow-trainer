"""
Centralized configuration management for Shadow Trainer API.
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from functools import lru_cache


class Settings(BaseModel):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="Shadow Trainer API", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent, description="Base directory")
    assets_dir: Path = Field(default=None, description="Assets directory")
    models_dir: Path = Field(default=None, description="Models directory")
    videos_dir: Path = Field(default=None, description="Videos directory")
    images_dir: Path = Field(default=None, description="Images directory")
"""
Centralized configuration management for Shadow Trainer API.
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from functools import lru_cache


class Settings(BaseModel):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="Shadow Trainer API", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent, description="Base directory")
    
    # Storage
    s3_bucket: str = Field(default="shadow-trainer-prod", description="S3 bucket name")
    s3_model_prefix: str = Field(default="model_weights", description="S3 model prefix")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    aws_region: str = Field(default="us-east-1", description="AWS region")
    
    # Model settings
    default_model_size: str = Field(default="xs", description="Default model size")
    supported_model_sizes: List[str] = Field(default=["xs", "s", "b", "l"], description="Supported model sizes")
    model_config_file: str = Field(default="model_config_map.json", description="Model config file")
    
    # Video processing
    supported_video_extensions: List[str] = Field(default=[".mp4", ".mov", ".avi"], description="Supported video extensions")
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Max file size (100MB)")
    max_processing_time: int = Field(default=300, description="Max processing time (5 minutes)")
    
    # API settings
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    docs_url: Optional[str] = Field(default="/docs", description="Docs URL")
    redoc_url: Optional[str] = Field(default="/redoc", description="ReDoc URL")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize paths after base model initialization
        self._init_paths()
        # Load from environment variables
        self._load_from_env()
    
    def _init_paths(self):
        """Initialize path properties."""
        self.assets_dir = self.base_dir / "assets"
        self.models_dir = self.assets_dir / "models"
        self.videos_dir = self.assets_dir / "videos"
        self.images_dir = self.assets_dir / "images"
        self.temp_dir = self.base_dir / "tmp"
        self.upload_dir = self.temp_dir / "uploads"
        self.output_dir = self.temp_dir / "outputs"
        self.checkpoint_dir = self.base_dir / "checkpoint"
        self.src_dir = self.base_dir / "src"
    
    def _load_from_env(self):
        """Load settings from environment variables."""
        env_mapping = {
            "SHADOW_TRAINER_DEBUG": ("debug", lambda x: x.lower() in ("true", "1", "yes")),
            "SHADOW_TRAINER_LOG_LEVEL": ("log_level", str),
            "SHADOW_TRAINER_HOST": ("host", str),
            "SHADOW_TRAINER_PORT": ("port", int),
            "SHADOW_TRAINER_S3_BUCKET": ("s3_bucket", str),
            "SHADOW_TRAINER_AWS_REGION": ("aws_region", str),
            "AWS_ACCESS_KEY_ID": ("aws_access_key_id", str),
            "AWS_SECRET_ACCESS_KEY": ("aws_secret_access_key", str),
        }
        
        for env_var, (attr_name, converter) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setattr(self, attr_name, converter(value))
                except (ValueError, TypeError):
                    pass  # Keep default value if conversion fails


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
