"""
Model management for 3D pose estimation.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from shadow_trainer.config import settings, get_model_config_path

logger = logging.getLogger(__name__)

class ModelManager:
    """Manager for handling model configurations and loading."""
    
    def __init__(self):
        """Initialize model manager."""
        self.config_cache: Dict[str, Any] = {}
        self.model_config_path = get_model_config_path()
        self._load_model_configs()
    
    def _load_model_configs(self):
        """Load model configurations from JSON file."""
        try:
            if self.model_config_path.exists():
                with open(self.model_config_path, 'r') as f:
                    self.config_cache = json.load(f)
                logger.info(f"Loaded model configurations from {self.model_config_path}")
            else:
                logger.warning(f"Model config file not found: {self.model_config_path}")
                self.config_cache = {}
        except Exception as e:
            logger.error(f"Failed to load model configurations: {e}")
            self.config_cache = {}
    
    def get_model_config(self, model_size: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get model configuration for specified size.
        
        Args:
            model_size: Model size (xs, s, b, l)
            
        Returns:
            Tuple of (config_path, error_message)
        """
        try:
            if model_size not in settings.supported_model_sizes:
                return None, f"Unsupported model size: {model_size}"
            
            if not self.config_cache:
                return None, "No model configurations available"
            
            if model_size not in self.config_cache:
                return None, f"No configuration found for model size: {model_size}"
            
            config_data = self.config_cache[model_size]
            config_path = config_data.get("config_path")
            
            if not config_path:
                return None, f"No config path specified for model size: {model_size}"
            
            # Make path relative to base directory
            full_config_path = settings.base_dir / config_path
            
            if not full_config_path.exists():
                logger.warning(f"Config file not found: {full_config_path}")
                return str(full_config_path), None  # Return path anyway for legacy compatibility
            
            return str(full_config_path), None
            
        except Exception as e:
            error_msg = f"Error getting model config for {model_size}: {e}"
            logger.error(error_msg)
            return None, error_msg
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all available model configurations."""
        return self.config_cache.copy()
    
    def reload_configs(self):
        """Reload model configurations from file."""
        self._load_model_configs()
    
    def validate_model_size(self, model_size: str) -> bool:
        """Validate if model size is supported."""
        return model_size in settings.supported_model_sizes
    
    def get_model_info(self, model_size: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        config_path, error = self.get_model_config(model_size)
        
        if error:
            return {"error": error}
        
        model_info = {
            "model_size": model_size,
            "config_path": config_path,
            "supported": True,
            "available": config_path and Path(config_path).exists()
        }
        
        # Add additional info from config cache
        if model_size in self.config_cache:
            config_data = self.config_cache[model_size]
            model_info.update({
                "description": config_data.get("description", ""),
                "parameters": config_data.get("parameters", {}),
                "checkpoint_pattern": config_data.get("checkpoint_pattern", "")
            })
        
        return model_info

# Global model manager instance
model_manager = ModelManager()

def load_model_config(model_size: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Legacy function for backward compatibility.
    
    Returns:
        Tuple of (config_path, error_message)
    """
    return model_manager.get_model_config(model_size)
