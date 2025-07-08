"""
Model management service for Shadow Trainer.
"""
import logging
from typing import Dict, Any, Optional, List

from shadow_trainer.config import settings
from shadow_trainer.core.inference.models import ModelManager
from shadow_trainer.schemas.requests import ModelConfigRequest
from shadow_trainer.schemas.responses import ModelConfigResponse, BaseResponse

logger = logging.getLogger(__name__)

class ModelService:
    """Service for model management operations."""
    
    def __init__(self):
        """Initialize model service."""
        self.model_manager = ModelManager()
        logger.info("ModelService initialized")
    
    def get_model_config(self, request: ModelConfigRequest) -> ModelConfigResponse:
        """
        Get configuration for a specific model.
        
        Args:
            request: Model configuration request
            
        Returns:
            Model configuration response
        """
        try:
            config_path, error = self.model_manager.get_model_config(request.model_size.value)
            
            if error:
                return ModelConfigResponse(
                    success=False,
                    message=error,
                    model_size=request.model_size,
                    config_path=""
                )
            
            # Get additional model info
            model_info = self.model_manager.get_model_info(request.model_size.value)
            
            return ModelConfigResponse(
                success=True,
                message="Model configuration retrieved successfully",
                model_size=request.model_size,
                config_path=config_path,
                checkpoint_path=model_info.get("checkpoint_pattern", ""),
                parameters=model_info.get("parameters", {})
            )
            
        except Exception as e:
            logger.error(f"Error getting model config: {e}")
            return ModelConfigResponse(
                success=False,
                message=f"Failed to get model configuration: {str(e)}",
                model_size=request.model_size,
                config_path=""
            )
    
    def list_available_models(self) -> Dict[str, Any]:
        """
        List all available models with their information.
        
        Returns:
            Dictionary containing model information
        """
        try:
            models = {}
            
            for model_size in settings.supported_model_sizes:
                model_info = self.model_manager.get_model_info(model_size)
                models[model_size] = model_info
            
            return {
                "success": True,
                "models": models,
                "supported_sizes": settings.supported_model_sizes,
                "default_size": settings.default_model_size
            }
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {
                "success": False,
                "error": str(e),
                "models": {},
                "supported_sizes": [],
                "default_size": ""
            }
    
    def validate_model_size(self, model_size: str) -> BaseResponse:
        """
        Validate if a model size is supported.
        
        Args:
            model_size: Model size to validate
            
        Returns:
            Validation response
        """
        try:
            is_valid = self.model_manager.validate_model_size(model_size)
            
            if is_valid:
                return BaseResponse(
                    success=True,
                    message=f"Model size '{model_size}' is valid"
                )
            else:
                return BaseResponse(
                    success=False,
                    message=f"Model size '{model_size}' is not supported. Supported sizes: {settings.supported_model_sizes}"
                )
                
        except Exception as e:
            logger.error(f"Error validating model size: {e}")
            return BaseResponse(
                success=False,
                message=f"Error validating model size: {str(e)}"
            )
    
    def reload_model_configs(self) -> BaseResponse:
        """
        Reload model configurations from file.
        
        Returns:
            Reload response
        """
        try:
            self.model_manager.reload_configs()
            return BaseResponse(
                success=True,
                message="Model configurations reloaded successfully"
            )
            
        except Exception as e:
            logger.error(f"Error reloading model configs: {e}")
            return BaseResponse(
                success=False,
                message=f"Failed to reload model configurations: {str(e)}"
            )
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all models.
        
        Returns:
            Dictionary containing model status information
        """
        try:
            status = {}
            
            for model_size in settings.supported_model_sizes:
                config_path, error = self.model_manager.get_model_config(model_size)
                
                if error:
                    status[model_size] = {
                        "available": False,
                        "error": error,
                        "config_path": None
                    }
                else:
                    from pathlib import Path
                    config_exists = config_path and Path(config_path).exists()
                    
                    status[model_size] = {
                        "available": config_exists,
                        "config_path": config_path,
                        "config_exists": config_exists
                    }
            
            return {
                "success": True,
                "model_status": status,
                "total_models": len(settings.supported_model_sizes),
                "available_models": sum(1 for s in status.values() if s.get("available", False))
            }
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_status": {},
                "total_models": 0,
                "available_models": 0
            }
