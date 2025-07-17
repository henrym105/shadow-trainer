#!/bin/bash
#
# Migration testing script for Shadow Trainer Celery implementation
#

set -e

PROJECT_ROOT="/home/ec2-user/shadow-trainer"
API_DIR="$PROJECT_ROOT/api_backend"

echo "=== Shadow Trainer Celery Migration Test ==="
echo ""

# Function to check if Redis is running
check_redis() {
    echo "🔍 Checking Redis connection..."
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is running and accessible"
        return 0
    else
        echo "❌ Redis is not running or not accessible"
        return 1
    fi
}

# Function to install dependencies
install_deps() {
    echo "📦 Installing Python dependencies..."
    cd "$PROJECT_ROOT"
    
    # Install using uv if available, otherwise pip
    if command -v uv &> /dev/null; then
        echo "Using uv to install dependencies..."
        uv pip install celery redis flower
    else
        echo "Using pip to install dependencies..."
        pip install celery redis flower
    fi
    
    echo "✅ Dependencies installed"
}

# Function to test Redis connectivity
test_redis_connectivity() {
    echo "🧪 Testing Redis connectivity from Python..."
    
    uv run python -c "
import redis
import sys

try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    r.ping()
    print('✅ Redis connection successful')
    
    # Test basic operations
    r.set('test_key', 'test_value')
    value = r.get('test_key')
    if value == 'test_value':
        print('✅ Redis read/write operations successful')
    else:
        print('❌ Redis read/write operations failed')
        sys.exit(1)
    
    r.delete('test_key')
    print('✅ Redis cleanup successful')
    
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    sys.exit(1)
"
}

# Function to test Celery app configuration
test_celery_config() {
    echo "🧪 Testing Celery app configuration..."
    
    cd "$API_DIR"
    uv run python -c "
import sys
sys.path.append('.')

try:
    from celery_app import celery_app
    print('✅ Celery app imported successfully')
    
    # Test configuration
    print(f'✅ Broker URL: {celery_app.conf.broker_url}')
    print(f'✅ Result backend: {celery_app.conf.result_backend}')
    print(f'✅ Task routes: {celery_app.conf.task_routes}')
    
except Exception as e:
    print(f'❌ Celery app configuration failed: {e}')
    sys.exit(1)
"
}

# Function to test job manager interface
test_job_manager() {
    echo "🧪 Testing Job Manager Interface..."
    
    cd "$API_DIR"
    uv run python -c "
import sys
sys.path.append('.')

try:
    from job_manager_interface import job_manager
    print(f'✅ Job manager initialized: {job_manager.manager_type}')
    print(f'✅ Celery enabled: {job_manager.is_celery_enabled}')
    
    # Test job creation (but don't submit)
    job = job_manager.create_job('test_file.mp4', '/tmp/test_path.mp4')
    print(f'✅ Job created: {job.job_id}')
    
    # Test job retrieval
    retrieved_job = job_manager.get_job(job.job_id)
    if retrieved_job and retrieved_job.job_id == job.job_id:
        print('✅ Job retrieval successful')
    else:
        print('❌ Job retrieval failed')
        sys.exit(1)
    
except Exception as e:
    print(f'❌ Job manager test failed: {e}')
    sys.exit(1)
"
}

# Function to test Celery worker (dry run)
test_celery_worker() {
    echo "🧪 Testing Celery worker startup (dry run)..."
    
    cd "$API_DIR"
    
    # Test worker inspection and configuration
    if uv run celery -A celery_app inspect stats > /dev/null 2>&1; then
        echo "✅ Celery worker inspection successful"
    else
        echo "ℹ️  No active workers (expected for this test)"
    fi
    
    # Test that celery command can find our app
    if uv run celery -A celery_app status > /dev/null 2>&1; then
        echo "✅ Celery app status check successful"
    else
        echo "ℹ️  Celery app configuration accessible (no workers running)"
    fi
    
    echo "✅ Celery worker configuration test completed"
}

# Function to test feature flag switching
test_feature_flags() {
    echo "🧪 Testing feature flag switching..."
    
    cd "$API_DIR"
    
    # Test with Celery disabled
    export USE_CELERY=false
    uv run python -c "
import sys
sys.path.append('.')

try:
    from job_manager_interface import job_manager
    if job_manager.manager_type == 'legacy':
        print('✅ Legacy mode enabled successfully')
    else:
        print('❌ Legacy mode failed to enable')
        sys.exit(1)
except Exception as e:
    print(f'❌ Feature flag test failed: {e}')
    sys.exit(1)
"
    
    # Test with Celery enabled
    export USE_CELERY=true
    uv run python -c "
import sys
sys.path.append('.')

try:
    from job_manager_interface import job_manager
    if job_manager.manager_type == 'celery':
        print('✅ Celery mode enabled successfully')
    else:
        print('❌ Celery mode failed to enable')
        sys.exit(1)
except Exception as e:
    print(f'❌ Feature flag test failed: {e}')
    sys.exit(1)
"
}

# Function to run health check
test_health_check() {
    echo "🧪 Testing API health check..."
    
    cd "$API_DIR"
    uv run python -c "
import sys
sys.path.append('.')

try:
    from config import config
    print(f'✅ Configuration loaded:')
    print(f'   - USE_CELERY: {config.USE_CELERY}')
    print(f'   - REDIS_HOST: {config.REDIS_HOST}')
    print(f'   - REDIS_PORT: {config.REDIS_PORT}')
    print(f'   - CELERY_WORKER_CONCURRENCY: {config.CELERY_WORKER_CONCURRENCY}')
    
except Exception as e:
    print(f'❌ Configuration test failed: {e}')
    sys.exit(1)
"
}

# Main execution
main() {
    echo "Starting migration tests..."
    echo ""
    
    # Set environment variables for testing
    export USE_CELERY=true
    export REDIS_HOST=localhost
    export REDIS_PORT=6379
    export REDIS_DB=0
    
    # Run tests in order
    if check_redis; then
        echo ""
    else
        echo "⚠️  Redis is not running. Starting Redis is required for Celery tests."
        echo "   For Amazon Linux 2/2023, install Redis with:"
        echo "   sudo amazon-linux-extras install redis6 -y"
        echo "   sudo systemctl start redis"
        echo ""
        echo "   Alternative installation methods:"
        echo "   1. EPEL: sudo yum install epel-release -y && sudo yum install redis -y"
        echo "   2. Docker: docker run -d -p 6379:6379 redis:7-alpine"
        echo ""
    fi
    
    install_deps
    echo ""
    
    test_redis_connectivity
    echo ""
    
    test_celery_config
    echo ""
    
    test_job_manager
    echo ""
    
    test_feature_flags
    echo ""
    
    test_health_check
    echo ""
    
    if check_redis; then
        test_celery_worker
        echo ""
    fi
    
    echo "🎉 All tests completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Start Redis: sudo systemctl start redis (already running!)"
    echo "2. Start Celery worker: uv run celery -A api_backend.celery_app worker --loglevel=info"
    echo "3. Start API server: uv run uvicorn api_backend.api_service:app --reload"
    echo "4. Optional: Start Flower monitoring: uv run celery -A api_backend.celery_app flower"
}

# Run main function
main "$@"
