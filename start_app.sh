#!/bin/bash
# Shadow Trainer FastAPI Application Startup Script

echo "Starting Shadow Trainer FastAPI Application..."
cd "$(dirname "$0")"

# Ensure we're in the api_inference directory
cd api_inference

# Run the FastAPI application using uv
uv run python -m app.main
