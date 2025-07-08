"""
Dependency injection for Shadow Trainer API.
"""
import logging
from functools import lru_cache

from shadow_trainer.services.video_service import VideoService
from shadow_trainer.services.model_service import ModelService

logger = logging.getLogger(__name__)

# Service instances cache
@lru_cache()
def get_video_service() -> VideoService:
    """Get video service instance."""
    return VideoService()

@lru_cache()
def get_model_service() -> ModelService:
    """Get model service instance."""
    return ModelService()

# Dependency functions for FastAPI
async def video_service_dependency() -> VideoService:
    """Video service dependency for FastAPI endpoints."""
    return get_video_service()

async def model_service_dependency() -> ModelService:
    """Model service dependency for FastAPI endpoints."""
    return get_model_service()
