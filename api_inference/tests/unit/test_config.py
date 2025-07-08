"""
Unit tests for Shadow Trainer configuration.
"""
import pytest
import os
from shadow_trainer.config import Settings, get_settings

def test_settings_creation():
    """Test that settings can be created with default values."""
    settings = Settings()
    assert settings.app_name == "Shadow Trainer API"
    assert settings.version == "1.0.0"
    assert settings.environment == "development"

def test_settings_environment_override():
    """Test that environment variables override default settings."""
    # Set environment variable
    os.environ["SHADOW_TRAINER_DEBUG"] = "true"
    os.environ["SHADOW_TRAINER_LOG_LEVEL"] = "DEBUG"
    
    # Clear the settings cache
    get_settings.cache_clear()
    
    settings = get_settings()
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    
    # Clean up
    del os.environ["SHADOW_TRAINER_DEBUG"]
    del os.environ["SHADOW_TRAINER_LOG_LEVEL"]
    get_settings.cache_clear()

def test_get_settings_singleton():
    """Test that get_settings returns the same instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2

def test_settings_paths():
    """Test that paths are properly configured."""
    settings = get_settings()
    
    # These should be Path objects and exist as properties
    assert hasattr(settings, 'assets_dir')
    assert hasattr(settings, 'models_dir')
    assert hasattr(settings, 'videos_dir')
    assert hasattr(settings, 'upload_dir')
    assert hasattr(settings, 'output_dir')

def test_model_sizes_configuration():
    """Test that model sizes are properly configured."""
    settings = get_settings()
    
    expected_sizes = ["xs", "s", "b", "l"]
    assert settings.supported_model_sizes == expected_sizes
