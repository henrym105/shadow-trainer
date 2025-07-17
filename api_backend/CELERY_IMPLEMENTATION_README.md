# Celery Migration Implementation

This implementation provides a complete migration from the in-memory job manager to a Celery + Redis based distributed task queue system.

## 📁 Files Created/Modified

### Core Implementation
- `celery_app.py` - Celery application configuration
- `celery_job_manager.py` - Redis-backed job manager
- `job_manager_interface.py` - Compatibility layer with feature flags
- `config.py` - Centralized configuration management
- `tasks/video_processing.py` - Celery tasks for video processing
- `pydantic_models.py` - Updated models with Celery task ID support
- `api_service.py` - Updated API endpoints to use new job manager

### Deployment & Configuration
- `docker-compose.celery.yml` - Docker Compose with Redis, Celery workers
- `scripts/celery/` - Startup scripts for Celery components
- `scripts/systemd/` - Systemd service files for production
- `scripts/test_celery_migration.sh` - Migration testing script
- `.env.example` - Environment configuration template

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Update dependencies
cd /home/ec2-user/shadow-trainer
uv add celery redis flower
```

### 2. Install and Start Redis

#### Check if Redis is already running:
```bash
# Check if Redis process is running
ps aux | grep redis

# Test connectivity (if redis-cli is available)
redis-cli ping || redis6-cli ping

# Check what's listening on Redis port
sudo netstat -tlnp | grep 6379
```

#### For Amazon Linux 2/2023:
```bash
# If Redis is already running but redis-cli isn't available:
sudo ln -sf /usr/bin/redis6-cli /usr/local/bin/redis-cli

# If Redis is not installed, try amazon-linux-extras:
sudo amazon-linux-extras install redis6 -y

# Alternative: Use Docker if system packages aren't available
docker run -d --name redis-shadow-trainer -p 6379:6379 --restart unless-stopped redis:7-alpine

# Start Redis service (if installed via system packages)
sudo systemctl start redis
sudo systemctl enable redis

# Verify Redis is running
redis-cli ping
```

#### For other Linux distributions:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install redis-server -y

# CentOS/RHEL with EPEL
sudo yum install epel-release -y && sudo yum install redis -y

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis
```

### 3. Test the Migration
```bash
# Run the comprehensive test script
./scripts/test_celery_migration.sh
```

### 4. Start Services

#### Option A: Docker Compose (Recommended)
```bash
# Start all services with Celery
docker-compose -f docker-compose.celery.yml up -d

# View logs
docker-compose -f docker-compose.celery.yml logs -f

# Stop services
docker-compose -f docker-compose.celery.yml down
```

#### Option B: Manual Startup
```bash
# Terminal 1: Start API server
cd api_backend
USE_CELERY=true uv run uvicorn api_service:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Celery worker
uv run celery -A api_backend.celery_app worker --loglevel=info --concurrency=2

# Terminal 3: Start Flower monitoring (optional)
uv run celery -A api_backend.celery_app flower --port=5555
```

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and modify as needed:

```bash
cp api_backend/.env.example api_backend/.env
```

Key settings:
- `USE_CELERY=true` - Enable Celery (set to `false` for legacy mode)
- `REDIS_HOST=localhost` - Redis server host
- `CELERY_WORKER_CONCURRENCY=2` - Number of worker processes

### Feature Flag System
The implementation includes a feature flag system that allows switching between:
- **Legacy mode** (`USE_CELERY=false`): Uses ThreadPoolExecutor (current system)
- **Celery mode** (`USE_CELERY=true`): Uses Celery + Redis (new system)

## 📊 Monitoring

### Flower Web UI
Access the Flower monitoring interface at `http://localhost:5555` to view:
- Active tasks and workers
- Task execution history
- Performance metrics
- Worker status

### Health Check Endpoint
The `/health` endpoint now includes job manager information:
```json
{
  "status": "healthy",
  "job_manager_type": "celery",
  "celery_enabled": true,
  "active_jobs": 5,
  "config": { ... }
}
```

## 🧪 Testing

### Run Migration Tests
```bash
./scripts/test_celery_migration.sh
```

This script tests:
- Redis connectivity
- Celery configuration
- Job manager interface
- Feature flag switching
- Component integration

### Manual Testing
```bash
# Test job creation and processing
curl -X POST "http://localhost:8000/videos/sample-lefty" \
  -H "Content-Type: application/json" \
  -d '{"model_size": "xs"}'

# Check job status
curl "http://localhost:8000/videos/{job_id}/status"
```

## 🚚 Production Deployment

### Systemd Services
Install systemd services for production:
```bash
# Copy service files
sudo cp scripts/systemd/*.service /etc/systemd/system/

# Reload systemd and start services
sudo systemctl daemon-reload
sudo systemctl enable shadow-trainer-celery-worker
sudo systemctl enable shadow-trainer-celery-beat
sudo systemctl start shadow-trainer-celery-worker
sudo systemctl start shadow-trainer-celery-beat
```

### Scaling Workers
```bash
# Scale workers horizontally
docker-compose -f docker-compose.celery.yml up -d --scale celery-worker=4

# Or with systemd
sudo systemctl start shadow-trainer-celery-worker@{1..4}
```

## 🔄 Migration Path

### Phase 1: Parallel Operation (Safe Migration)
1. Deploy with `USE_CELERY=false` (legacy mode)
2. Test new infrastructure alongside old system
3. Gradually switch individual job types to Celery

### Phase 2: Complete Migration
1. Switch `USE_CELERY=true` for all operations
2. Monitor performance and reliability
3. Remove legacy job_manager.py after validation

### Rollback Plan
If issues occur:
1. Set `USE_CELERY=false` immediately
2. Fix issues in Celery system while legacy runs
3. Re-enable Celery once resolved

## 📈 Benefits Achieved

### Reliability
- ✅ Job persistence across service restarts
- ✅ Automatic retry mechanisms
- ✅ Dead letter queue handling
- ✅ Graceful worker shutdowns

### Scalability
- ✅ Horizontal worker scaling
- ✅ Queue prioritization
- ✅ Load balancing across workers
- ✅ Independent API and worker scaling

### Monitoring
- ✅ Complete job lifecycle visibility
- ✅ Performance metrics and alerting
- ✅ Real-time task monitoring via Flower
- ✅ Comprehensive logging

### Operational
- ✅ Better resource management
- ✅ Configurable retry policies
- ✅ Task scheduling capabilities
- ✅ Deployment flexibility

## 🐛 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   redis-cli ping
   sudo systemctl status redis
   ```

2. **Celery Import Errors**
   ```bash
   # Ensure dependencies are installed
   pip install celery redis flower
   ```

3. **Task Not Found Errors**
   ```bash
   # Check task imports in celery_app.py
   # Ensure PYTHONPATH includes project root
   export PYTHONPATH=/home/ec2-user/shadow-trainer:$PYTHONPATH
   ```

4. **Worker Memory Issues**
   ```bash
   # Adjust worker configuration
   export CELERY_WORKER_CONCURRENCY=1
   # Or restart workers periodically
   celery -A api_backend.celery_app worker --max-tasks-per-child=100
   ```

### Logs
```bash
# API logs
journalctl -u shadow-trainer-api -f

# Celery worker logs
journalctl -u shadow-trainer-celery-worker -f

# Redis logs
journalctl -u redis -f
```

## 📚 Next Steps

1. **Performance Tuning**: Optimize worker concurrency and Redis settings
2. **Monitoring**: Set up Prometheus/Grafana metrics collection
3. **Alerting**: Configure alerts for failed tasks and worker health
4. **Backup**: Implement Redis persistence and backup strategy
5. **Security**: Add authentication for Flower and Redis if exposed

---

This implementation provides a robust, scalable foundation for the Shadow Trainer video processing pipeline while maintaining backward compatibility and providing a safe migration path.
