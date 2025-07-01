# Use Python slim as base image
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv in a separate layer for caching
RUN pip install --upgrade pip && pip install --no-cache-dir uv

# Set work directory
WORKDIR /app

# Copy only dependency files first for better cache
COPY pyproject.toml ./

# Install Python dependencies
RUN uv pip install --system --requirement pyproject.toml

# Copy only necessary source files, excluding large model files for build efficiency
COPY api_inference ./api_inference

# Remove large model files if any slipped through (as a safety net)
RUN find ./api_inference -type f \( -name "*.weights" -o -name "*.pt*" \) -delete
# Remove build tools and cache to reduce image size
RUN apt-get purge -y --auto-remove && rm -rf /root/.cache/pip

# Expose port (if running an API server, adjust as needed)
EXPOSE 8000

# Set the default command (adjust as needed, e.g., to run an API server)
CMD ["python", "api_inference/run_api.py"]
