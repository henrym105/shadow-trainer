"""
Configuration and feature flags for Shadow Trainer API.
"""
import os
from typing import Dict, Any


class Config:
    """Application configuration with feature flags."""
    
    # Feature flags
    USE_CELERY = os.getenv('USE_CELERY', 'true').lower() == 'true'
    
    # Redis configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
    
    # Celery configuration
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}')
    CELERY_WORKER_CONCURRENCY = int(os.getenv('CELERY_WORKER_CONCURRENCY', '2'))
    CELERY_TASK_TIME_LIMIT = int(os.getenv('CELERY_TASK_TIME_LIMIT', '1800'))  # 30 minutes
    
    # Application settings
    INCLUDE_2D_IMAGES = os.getenv('INCLUDE_2D_IMAGES', 'true').lower() == 'true'
    CLEANUP_RETENTION_MINUTES = int(os.getenv('CLEANUP_RETENTION_MINUTES', '60'))
    
    # S3 configuration
    S3_BUCKET = os.getenv('S3_BUCKET', 'shadow-trainer-dev')
    S3_PRO_PREFIX = os.getenv('S3_PRO_PREFIX', 'test/professional/')
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            'USE_CELERY': cls.USE_CELERY,
            'REDIS_HOST': cls.REDIS_HOST,
            'REDIS_PORT': cls.REDIS_PORT,
            'CELERY_WORKER_CONCURRENCY': cls.CELERY_WORKER_CONCURRENCY,
            'INCLUDE_2D_IMAGES': cls.INCLUDE_2D_IMAGES,
            'CLEANUP_RETENTION_MINUTES': cls.CLEANUP_RETENTION_MINUTES,
        }


# Global config instance
config = Config()
