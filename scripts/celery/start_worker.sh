#!/bin/bash
#
# Celery worker startup script for Shadow Trainer
#

set -e

# Default configuration
CELERY_APP=${CELERY_APP:-"api_backend.celery_app"}
CELERY_WORKER_CONCURRENCY=${CELERY_WORKER_CONCURRENCY:-2}
CELERY_LOGLEVEL=${CELERY_LOGLEVEL:-"info"}
CELERY_QUEUES=${CELERY_QUEUES:-"video_processing,cleanup,default"}

# Environment setup
export PYTHONPATH="${PYTHONPATH}:/app"
cd /app

echo "Starting Celery worker..."
echo "App: $CELERY_APP"
echo "Concurrency: $CELERY_WORKER_CONCURRENCY"
echo "Log level: $CELERY_LOGLEVEL"
echo "Queues: $CELERY_QUEUES"

# Start Celery worker
exec uv run celery -A $CELERY_APP worker \
    --loglevel=$CELERY_LOGLEVEL \
    --concurrency=$CELERY_WORKER_CONCURRENCY \
    --queues=$CELERY_QUEUES \
    --task-events \
    --without-heartbeat \
    --without-mingle \
    --without-gossip
