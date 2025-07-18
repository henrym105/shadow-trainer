# Multi-stage Dockerfile for Shadow Trainer with Celery - Optimized for size
# Build stage - includes build tools and development dependencies
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel AS builder

# Install build dependencies and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        git && \
    pip install --upgrade pip && \
    pip install --no-cache-dir uv && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies to a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir -e .

# Runtime stage - minimal runtime dependencies only
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/api_backend" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/opt/venv/bin:$PATH"

# Install only runtime system dependencies (no build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
        redis-tools \
        curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y && \
    apt-get autoclean

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set work directory
WORKDIR /app

# Copy application code
COPY api_backend ./api_backend
COPY scripts ./scripts

# Create necessary directories with proper permissions
RUN mkdir -p /app/api_backend/tmp_api_output \
             /app/api_backend/checkpoint \
             /app/api_backend/sample_videos && \
    chmod -R 755 /app/api_backend && \
    find /app/scripts -name "*.sh" -type f -exec chmod +x {} \;

# Default environment variables
ENV REDIS_HOST=redis \
    REDIS_PORT=6379 \
    REDIS_DB=0 \
    CELERY_WORKER_CONCURRENCY=2

# Expose ports
EXPOSE 8002 5555

# API service stage
FROM runtime AS api
CMD ["python", "-m", "uvicorn", "api_backend.api_service:app", "--host", "0.0.0.0", "--port", "8002"]

# Celery worker stage
FROM runtime AS worker
ENV CELERY_WORKER=true
CMD ["python", "-m", "celery", "-A", "api_backend.celery_app", "worker", "--loglevel=info", "--concurrency=2"]

# Celery beat stage
FROM runtime AS beat
ENV CELERY_WORKER=true
CMD ["python", "-m", "celery", "-A", "api_backend.celery_app", "beat", "--loglevel=info"]

# Flower monitoring stage
FROM runtime AS flower
ENV CELERY_WORKER=true
CMD ["python", "-m", "celery", "-A", "api_backend.celery_app", "flower", "--port=5555", "--address=0.0.0.0"]
