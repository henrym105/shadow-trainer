# Shadow Trainer Production Deployment Guide
*For Junior MLOps & Web Development Engineers*

## Overview: What We're Building

We're transforming Shadow Trainer from a simple single-server application into a **distributed, production-ready system**. Think of it like this:

- **Before**: One computer doing everything (like a single chef cooking, serving, and cleaning)
- **After**: A restaurant with specialized roles (chef, waiter, dishwasher, manager)

## Architecture Components Explained

### 1. **The "Restaurant Kitchen" - Our Services**

```mermaid
graph TB
    A[Web Browser] --> B[Load Balancer]
    B --> C[API Server 1]
    B --> D[API Server 2]
    C --> E[Redis Cache]
    C --> F[PostgreSQL DB]
    C --> G[Celery Worker 1]
    D --> G
    D --> H[Celery Worker 2]
    G --> I[Video Storage]
    H --> I
```

**What each does:**
- **API Servers**: Handle web requests (like waiters taking orders)
- **Redis**: Fast memory storage for job status (like a kitchen order board)
- **PostgreSQL**: Permanent storage for job history (like accounting records)
- **Celery Workers**: Do the actual video processing (like chefs cooking)
- **Load Balancer**: Distributes traffic between API servers

## Step-by-Step Deployment Process

### Phase 1: Local Development Setup

First, let's get everything running on your laptop to understand the pieces:

```bash
# 1. Clone and setup project
git clone <your-repo>
cd shadow-trainer

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Create development environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 4. Install dependencies using uv
uv add celery redis psycopg2-binary sqlalchemy alembic "uvicorn[standard]"
```

**Create/update pyproject.toml for dependency management:**
```toml
[project]
name = "shadow-trainer"
version = "0.1.0"
description = "AI-powered shadow boxing analysis"
requires-python = ">=3.11"
dependencies = [
    # Existing dependencies (move from requirements.txt)
    "celery>=5.3.4",
    "redis>=5.0.1",
    "psycopg2-binary>=2.9.9",
    "sqlalchemy>=2.0.23",
    "alembic>=1.13.1",
    "uvicorn[standard]>=0.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Phase 2: Database Setup

**Why PostgreSQL?** Unlike the current in-memory storage, PostgreSQL persists data even when servers restart.

```bash
# Install PostgreSQL locally (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE shadow_trainer;
CREATE USER shadow_trainer WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE shadow_trainer TO shadow_trainer;
\q
```

**Database Migrations with Alembic:**
```bash
# Initialize migrations
uv run alembic init migrations

# Generate first migration
uv run alembic revision --autogenerate -m "Initial video jobs schema"

# Apply migration
uv run alembic upgrade head
```

### Phase 3: Redis Setup

**Why Redis?** It's like a super-fast notepad where we store job progress that needs to be accessed quickly.

```bash
# Install Redis locally (Ubuntu/Debian)
sudo apt install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis connection
redis-cli ping
# Should return: PONG
```

### Phase 4: Docker Development Environment

Create a `docker-compose.dev.yml` for local development:

```yaml
version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: shadow_trainer
      POSTGRES_USER: shadow_trainer
      POSTGRES_PASSWORD: dev_password_123
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U shadow_trainer"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Cache and message broker
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Video processing workers
  celery_worker:
    build: 
      context: .
      dockerfile: Dockerfile.dev
    command: uv run celery -A api_backend.celery_app worker --loglevel=info --queues=video_processing
    volumes:
      - .:/app
      - /app/api_backend/tmp_api_output  # Mount for processed videos
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://shadow_trainer:dev_password_123@postgres:5432/shadow_trainer
      - REDIS_URL=redis://redis:6379/0
      - PYTHONPATH=/app

  # Scheduled tasks (cleanup, monitoring)
  celery_beat:
    build: 
      context: .
      dockerfile: Dockerfile.dev
    command: uv run celery -A api_backend.celery_app beat --loglevel=info
    volumes:
      - .:/app
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://shadow_trainer:dev_password_123@postgres:5432/shadow_trainer
      - REDIS_URL=redis://redis:6379/0
      - PYTHONPATH=/app

  # API server
  api:
    build: 
      context: .
      dockerfile: Dockerfile.dev
    command: uv run uvicorn api_backend.api_service:app --host 0.0.0.0 --port 8002 --reload
    ports:
      - "8002:8002"
    volumes:
      - .:/app
      - /app/api_backend/tmp_api_output  # Mount for processed videos
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://shadow_trainer:dev_password_123@postgres:5432/shadow_trainer
      - REDIS_URL=redis://redis:6379/0
      - PYTHONPATH=/app

volumes:
  postgres_dev_data:
  redis_dev_data:
```

**Create Dockerfile.dev:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies with uv
RUN uv sync --frozen

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/api_backend/tmp_api_output

EXPOSE 8002
```

### Phase 5: Environment Configuration

Create a `.env` file for configuration:

```bash
# Database
DATABASE_URL=postgresql://shadow_trainer:dev_password_123@localhost:5432/shadow_trainer

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8002

# AWS (for S3 storage)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1

# File storage
MAX_FILE_SIZE_MB=100
FILE_RETENTION_DAYS=7

# Monitoring
LOG_LEVEL=INFO
```

### Phase 6: Testing the Development Environment

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# Check all services are running
docker-compose -f docker-compose.dev.yml ps

# Watch logs
docker-compose -f docker-compose.dev.yml logs -f

# Test the API
curl -X GET http://localhost:8002/health

# Test job creation
curl -X POST http://localhost:8002/videos/sample-lefty \
  -H "Content-Type: application/json" \
  -d '{"model_size": "xs"}'
```

## Production Deployment Options

### Option 1: AWS ECS (Recommended for beginners)

**Why ECS?** Amazon manages the infrastructure complexity for you.

**Production architecture:**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - api

  api:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      - postgres
      - redis

  celery_worker:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    command: uv run celery -A api_backend.celery_app worker --loglevel=info --concurrency=2
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    deploy:
      replicas: 3  # Run 3 worker containers

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_prod_data:/data

volumes:
  postgres_prod_data:
  redis_prod_data:
```

**Production Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Production optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies with uv (production only, no dev dependencies)
RUN uv sync --frozen --no-dev

# Copy application code
COPY api_backend/ ./api_backend/
COPY src/ ./src/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8002

CMD ["uv", "run", "uvicorn", "api_backend.api_service:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Option 2: Kubernetes (For scaling)

**When to use?** When you need to handle thousands of concurrent video processing jobs.

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shadow-trainer-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: shadow-trainer-api
  template:
    metadata:
      labels:
        app: shadow-trainer-api
    spec:
      containers:
      - name: api
        image: shadow-trainer:latest
        ports:
        - containerPort: 8002
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: shadow-trainer-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: shadow-trainer-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shadow-trainer-worker
spec:
  replicas: 5  # Scale based on video processing load
  selector:
    matchLabels:
      app: shadow-trainer-worker
  template:
    metadata:
      labels:
        app: shadow-trainer-worker
    spec:
      containers:
      - name: worker
        image: shadow-trainer:latest
        command: ["uv", "run", "celery", "-A", "api_backend.celery_app", "worker", "--loglevel=info"]
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Monitoring and Observability

### Basic Monitoring Setup

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
video_processing_counter = Counter('videos_processed_total', 'Total videos processed')
processing_duration = Histogram('video_processing_duration_seconds', 'Time spent processing videos')
active_jobs_gauge = Gauge('active_jobs', 'Number of active video processing jobs')

def monitor_job_completion(job_id: str, duration: float):
    video_processing_counter.inc()
    processing_duration.observe(duration)
    
def update_active_jobs_count(count: int):
    active_jobs_gauge.set(count)

# Start metrics server
start_http_server(8000)
```

### Health Check Endpoints

```python
# Add to api_service.py
@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check for load balancer"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check database
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        job_manager.redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Celery workers
    try:
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        if active_workers:
            health_status["services"]["celery"] = f"healthy ({len(active_workers)} workers)"
        else:
            health_status["services"]["celery"] = "no workers available"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["celery"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
```

## Deployment Checklist

### Pre-Deployment
- [ ] Code reviewed and tested locally
- [ ] Database migrations created and tested
- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Monitoring dashboards created
- [ ] Backup strategy planned

### Deployment Steps
1. **Build and test Docker images**
   ```bash
   docker build -t shadow-trainer:v1.0.0 .
   docker run --rm shadow-trainer:v1.0.0 python -c "import api_backend.api_service"
   ```

2. **Deploy database and Redis first**
   ```bash
   # Start persistence layer
   docker-compose up -d postgres redis
   
   # Wait for services to be ready
   docker-compose exec postgres pg_isready -U shadow_trainer
   ```

3. **Run database migrations**
   ```bash
   docker-compose exec api uv run alembic upgrade head
   ```

4. **Deploy application services**
   ```bash
   # Start workers first (they process the backlog)
   docker-compose up -d celery_worker celery_beat
   
   # Then start API servers
   docker-compose up -d api
   
   # Finally start load balancer
   docker-compose up -d nginx
   ```

5. **Verify deployment**
   ```bash
   # Check all services
   curl http://your-domain.com/health/detailed
   
   # Test video upload
   curl -X POST http://your-domain.com/videos/sample-lefty
   ```

### Post-Deployment
- [ ] Monitor logs for errors
- [ ] Verify video processing works end-to-end
- [ ] Test backup/restore procedures
- [ ] Update documentation
- [ ] Set up alerts for critical metrics

## Common Issues and Solutions

### "Celery workers not processing jobs"
```bash
# Check worker status
docker-compose exec celery_worker uv run celery -A api_backend.celery_app inspect active

# Check Redis queues
docker-compose exec redis redis-cli LLEN video_processing

# Restart workers if needed
docker-compose restart celery_worker
```

### "Database connection errors"
```bash
# Check database connectivity
docker-compose exec api uv run python -c "
from api_backend.models import get_db
db = next(get_db())
print('Database connected successfully')
"
```

### "High memory usage"
```bash
# Monitor container resources
docker stats

# Check for memory leaks in workers
docker-compose exec celery_worker ps aux

# Restart workers periodically (already configured in Celery settings)
```

## Scaling Considerations

**When to scale what:**

- **More API traffic?** → Add more API server replicas
- **Slow video processing?** → Add more Celery workers  
- **Database slow?** → Optimize queries, add read replicas
- **Redis slow?** → Use Redis cluster mode

**Cost optimization:**
- Use spot instances for Celery workers (they can handle interruptions)
- Scale workers based on queue length
- Implement efficient file cleanup to minimize storage costs

This deployment guide transforms Shadow Trainer from a development prototype into a production-ready system that can handle real user traffic and scale as needed.