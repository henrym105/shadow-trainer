"""
Celery application configuration for Shadow Trainer video processing.
"""
import os
from celery import Celery
from kombu import Queue

# Get Redis configuration from environment variables
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')
REDIS_DB = os.getenv('REDIS_DB', '0')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

# Construct Redis URL
if REDIS_PASSWORD:
    redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
    redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Create Celery application
celery_app = Celery(
    'shadow_trainer',
    broker=redis_url,
    backend=redis_url,
    include=['tasks.video_processing']
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,
    
    # Time limits
    task_time_limit=int(os.getenv('CELERY_TASK_TIME_LIMIT', 1800)),  # 30 minutes
    task_soft_time_limit=int(os.getenv('CELERY_TASK_SOFT_TIME_LIMIT', 1500)),  # 25 minutes
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Task routing
    task_routes={
        'tasks.video_processing.*': {'queue': 'video_processing'},
        'tasks.cleanup.*': {'queue': 'cleanup'},
    },
    
    # Queues
    task_default_queue='default',
    task_queues=(
        Queue('default'),
        Queue('video_processing', routing_key='video_processing'),
        Queue('cleanup', routing_key='cleanup'),
    ),
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Retry configuration
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_events=True,
)

if __name__ == '__main__':
    celery_app.start()
