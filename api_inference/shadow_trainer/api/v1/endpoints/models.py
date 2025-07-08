"""
Model management endpoints for Shadow Trainer API.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any

from shadow_trainer.schemas.requests import ModelConfigRequest
from shadow_trainer.schemas.responses import ModelConfigResponse, BaseResponse
from shadow_trainer.schemas.common import ModelSize
from shadow_trainer.services.model_service import ModelService
from shadow_trainer.api.dependencies import model_service_dependency

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/list")
async def list_models(
    model_service: ModelService = Depends(model_service_dependency)
) -> Dict[str, Any]:
    """
    List all available models with their information.
    
    Returns:
        Dictionary containing all model information
    """
    try:
        return model_service.list_available_models()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/config/{model_size}", response_model=ModelConfigResponse)
async def get_model_config(
    model_size: ModelSize,
    model_service: ModelService = Depends(model_service_dependency)
):
    """
    Get configuration for a specific model size.
    
    Args:
        model_size: Model size to get configuration for
        
    Returns:
        Model configuration response
    """
    try:
        request = ModelConfigRequest(model_size=model_size)
        return model_service.get_model_config(request)
    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model config: {str(e)}")

@router.get("/validate/{model_size}", response_model=BaseResponse)
async def validate_model_size(
    model_size: str,
    model_service: ModelService = Depends(model_service_dependency)
):
    """
    Validate if a model size is supported.
    
    Args:
        model_size: Model size to validate
        
    Returns:
        Validation response
    """
    try:
        return model_service.validate_model_size(model_size)
    except Exception as e:
        logger.error(f"Error validating model size: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/status")
async def get_model_status(
    model_service: ModelService = Depends(model_service_dependency)
) -> Dict[str, Any]:
    """
    Get status of all models (availability, configuration, etc.).
    
    Returns:
        Dictionary containing model status information
    """
    try:
        return model_service.get_model_status()
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.post("/reload-configs", response_model=BaseResponse)
async def reload_model_configs(
    model_service: ModelService = Depends(model_service_dependency)
):
    """
    Reload model configurations from file.
    
    Returns:
        Response indicating success or failure of reload operation
    """
    try:
        return model_service.reload_model_configs()
    except Exception as e:
        logger.error(f"Error reloading model configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload configs: {str(e)}")

@router.get("/supported-sizes")
async def get_supported_sizes() -> Dict[str, Any]:
    """Get list of supported model sizes."""
    from shadow_trainer.config import settings
    
    return {
        "supported_sizes": settings.supported_model_sizes,
        "default_size": settings.default_model_size,
        "size_descriptions": {
            "xs": "Extra Small - Fastest processing, lower accuracy",
            "s": "Small - Good balance of speed and accuracy", 
            "b": "Base - Higher accuracy, moderate speed",
            "l": "Large - Highest accuracy, slower processing"
        }
    }
