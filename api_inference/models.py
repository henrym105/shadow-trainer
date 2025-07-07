import os
import json
import logging
from typing import Tuple, Optional
from config import API_ROOT_DIR, MODEL_CONFIG_FILE

logger = logging.getLogger(__name__)

def load_model_config(model_size: str, config_path: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Load model configuration based on model size.
    
    Args:
        model_size (str): Model size identifier (xs, s, b, l)
        config_path (str): Optional path to config file, defaults to model_config_map.json
        
    Returns:
        Tuple[model_config, error]: Model config path if successful, error message if failed
    """
    if config_path is None:
        config_path = os.path.join(API_ROOT_DIR, MODEL_CONFIG_FILE)
    
    try:
        # Ensure config_path is absolute
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(os.path.join(API_ROOT_DIR, config_path))
            
        with open(config_path, "r") as f:
            model_config_map = json.load(f)
            
        logger.info(f"Loaded model_config_map: {model_config_map}")
        model_config = model_config_map.get(model_size, model_config_map.get("b"))
        
        if model_config is None:
            return None, f"Model size '{model_size}' not found in config and no default 'b' size available"
            
        logger.info(f"Using model config: {model_config}")
        
        # If model_config is a relative path, make it absolute
        if not os.path.isabs(model_config):
            model_config = os.path.abspath(os.path.join(os.path.dirname(config_path), model_config))
            
        return model_config, None
        
    except FileNotFoundError:
        error_msg = f"Model config file not found: {config_path}"
        logger.error(error_msg)
        return None, error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse model_config_map.json: {e}"
        logger.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Failed to load model config: {e}"
        logger.error(error_msg)
        return None, error_msg
