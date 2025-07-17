# Celery and Redis Migration Plan for Shadow Trainer API Backend

## Overview

This document outlines the migration from the current in-memory job manager (`job_manager.py`) to a robust Celery + Redis-based asynchronous task processing system for the Shadow Trainer API backend.

## Current Architecture Issues

The current implementation in `job_manager.py` has several limitations:

1. **In-memory storage**: Jobs are lost on service restart
2. **No persistence**: No job history or recovery capabilities
3. **Limited scalability**: ThreadPoolExecutor with max 2 workers
4. **No distributed processing**: Cannot scale across multiple servers
5. **No job retry mechanisms**: Failed jobs cannot be automatically retried
6. **No job scheduling**: Cannot schedule jobs for future execution
7. **Memory leaks**: Long-running service accumulates job data in memory

## Proposed Architecture

### Components

1. **Redis**: Message broker and result backend
2. **Celery**: Distributed task queue and worker management
3. **FastAPI**: API layer (minimal changes)
4. **Celery Workers**: Background video processing workers

### Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │      Redis      │    │ Celery Workers  │
│                 │    │                 │    │                 │
│ - Accept uploads│◄──►│ - Message Queue │◄──►│ - Video Process │
│ - Create tasks  │    │ - Job Status    │    │ - 2D/3D Pose    │
│ - Check status  │    │ - Results Cache │    │ - Visualization │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Implementation Plan

### Phase 1: Setup and Dependencies

#### 1.1 Add Dependencies
Add to `pyproject.toml`:
```toml
"celery>=5.3.0",
"redis>=5.0.0",
"flower>=2.0.1",  # Optional: Celery monitoring UI
```

#### 1.2 Redis Configuration
- Install Redis server (local development and production)
- Configure Redis connection settings
- Set up Redis persistence for job data

### Phase 2: Celery Configuration

#### 2.1 Create Celery App (`celery_app.py`)
```python
from celery import Celery
from kombu import Queue

# Celery configuration
celery_app = Celery(
    'shadow_trainer',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=['tasks.video_processing']
)

# Task routing and queue configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max per task
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,
    task_routes={
        'tasks.video_processing.*': {'queue': 'video_processing'},
    },
    task_default_queue='default',
    task_queues=(
        Queue('default'),
        Queue('video_processing', routing_key='video_processing'),
    ),
)
```

#### 2.2 Create Celery Tasks (`tasks/video_processing.py`)
```python
from celery import Task
from celery_app import celery_app
from pydantic_models import JobStatus
# Import existing video processing functions

class CallbackTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        # Update job status to completed
        pass
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        # Update job status to failed
        pass

@celery_app.task(bind=True, base=CallbackTask)
def process_video_task(self, job_data):
    # Implement video processing logic
    pass
```

### Phase 3: Job Management Migration

#### 3.1 Create New Job Manager (`celery_job_manager.py`)
Replace the current `job_manager.py` with a Celery-based implementation:

```python
class CeleryJobManager:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        
    def create_job(self, filename: str, input_path: str) -> VideoJob:
        # Store job metadata in Redis
        pass
        
    def submit_job(self, job_id: str, **task_kwargs):
        # Submit task to Celery
        pass
        
    def get_job_status(self, job_id: str):
        # Get job status from Redis + Celery
        pass
```

#### 3.2 Update Job Models (`pydantic_models.py`)
Extend existing models to support Celery task IDs:

```python
class VideoJob(BaseModel):
    job_id: str
    celery_task_id: Optional[str] = None  # Add Celery task ID
    original_filename: str
    input_path: str
    # ... existing fields
```

### Phase 4: API Integration

#### 4.1 Update API Service (`api_service.py`)
Minimal changes required:
- Replace `job_manager` imports with `celery_job_manager`
- Update job submission calls
- Modify status check endpoints

#### 4.2 Background Task Replacement
Replace FastAPI BackgroundTasks with Celery tasks:

```python
# Before
background_tasks.add_task(process_video_job, job_id, ...)

# After
task = process_video_task.delay(job_data)
```

### Phase 5: Production Deployment

#### 5.1 Docker Configuration
Update `Dockerfile` and docker-compose:
```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
      
  celery-worker:
    build: .
    command: celery -A celery_app worker --loglevel=info
    depends_on:
      - redis
      
  celery-beat:  # For scheduled tasks (optional)
    build: .
    command: celery -A celery_app beat --loglevel=info
    depends_on:
      - redis
      
  flower:  # Monitoring UI (optional)
    build: .
    command: celery -A celery_app flower
    ports:
      - "5555:5555"
    depends_on:
      - redis
```

#### 5.2 Process Management
- Configure systemd services for Celery workers
- Set up monitoring and alerting
- Configure log rotation

## Migration Steps

### Step 1: Parallel Implementation (Safe Migration)
1. Keep existing `job_manager.py` functional
2. Implement Celery infrastructure alongside
3. Add feature flag to switch between systems
4. Test thoroughly in development

### Step 2: Gradual Rollout
1. Deploy Celery system to staging
2. Run both systems in parallel initially
3. Gradually migrate job types to Celery
4. Monitor performance and reliability

### Step 3: Complete Migration
1. Switch all job processing to Celery
2. Remove old `job_manager.py`
3. Update documentation and deployment scripts

## Benefits of Migration

### Reliability
- **Persistence**: Jobs survive service restarts
- **Retry mechanisms**: Automatic retry of failed tasks
- **Dead letter queues**: Handle problematic jobs

### Scalability
- **Horizontal scaling**: Add more worker processes/servers
- **Queue prioritization**: Different priorities for different job types
- **Load balancing**: Distribute work across multiple workers

### Monitoring
- **Job tracking**: Complete job lifecycle visibility
- **Performance metrics**: Task execution times, success rates
- **Alerting**: Failed job notifications

### Operational
- **Graceful shutdowns**: Workers finish current tasks before stopping
- **Resource management**: Better memory and CPU usage
- **Deployment flexibility**: Independent scaling of API and workers

## Testing Strategy

### Unit Tests
- Test Celery task functions in isolation
- Mock Redis connections for testing
- Test job state transitions

### Integration Tests
- End-to-end job processing tests
- Redis connection and failover tests
- Multi-worker coordination tests

### Performance Tests
- Load testing with concurrent jobs
- Memory usage monitoring
- Task execution time benchmarks

## Rollback Plan

If issues arise during migration:

1. **Immediate rollback**: Switch feature flag back to old system
2. **Data preservation**: Export job data from Redis if needed
3. **Gradual debugging**: Fix issues in Celery system while old system runs
4. **Re-attempt migration**: Once issues are resolved

## Configuration Management

### Environment Variables
```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_WORKER_CONCURRENCY=2
CELERY_TASK_TIME_LIMIT=1800  # 30 minutes
```

### Development vs Production
- Development: Single Redis instance, local workers
- Production: Redis cluster, multiple worker nodes, monitoring

## Timeline Estimate

- **Week 1**: Setup Celery infrastructure and basic tasks
- **Week 2**: Implement job manager and API integration
- **Week 3**: Testing and debugging
- **Week 4**: Production deployment and monitoring setup

## Success Metrics

- Zero job loss during migration
- Improved system reliability (99.9% uptime)
- Better error handling and recovery
- Horizontal scalability demonstration
- Reduced memory usage over time

## Next Steps

1. **Approve this plan** and get stakeholder buy-in
2. **Set up development environment** with Redis and Celery
3. **Begin Phase 1 implementation** (dependencies and setup)
4. **Create feature branch** for migration work
5. **Start with simple task migration** before tackling complex video processing

---

*This migration will significantly improve the reliability, scalability, and maintainability of the Shadow Trainer API backend while preserving all existing functionality.*
