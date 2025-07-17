#!/bin/bash
#
# Flower monitoring UI startup script for Shadow Trainer
#

set -e

# Default configuration
CELERY_APP=${CELERY_APP:-"api_backend.celery_app"}
FLOWER_PORT=${FLOWER_PORT:-5555}
FLOWER_ADDRESS=${FLOWER_ADDRESS:-"0.0.0.0"}

# Environment setup
export PYTHONPATH="${PYTHONPATH}:/app"
cd /app

echo "Starting Flower monitoring UI..."
echo "App: $CELERY_APP"
echo "Port: $FLOWER_PORT"
echo "Address: $FLOWER_ADDRESS"

# Start Flower
exec uv run celery -A $CELERY_APP flower \
    --port=$FLOWER_PORT \
    --address=$FLOWER_ADDRESS
