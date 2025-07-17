#!/bin/bash
#
# Celery beat (scheduler) startup script for Shadow Trainer
#

set -e

# Default configuration
CELERY_APP=${CELERY_APP:-"api_backend.celery_app"}
CELERY_LOGLEVEL=${CELERY_LOGLEVEL:-"info"}

# Environment setup
export PYTHONPATH="${PYTHONPATH}:/app"
cd /app

echo "Starting Celery beat scheduler..."
echo "App: $CELERY_APP"
echo "Log level: $CELERY_LOGLEVEL"

# Start Celery beat
exec uv run celery -A $CELERY_APP beat \
    --loglevel=$CELERY_LOGLEVEL \
    --scheduler django_celery_beat.schedulers:DatabaseScheduler
