#!/bin/bash

echo "🚀 Testing Shadow Trainer Application Setup"
echo "=========================================="

# Test if backend can start
echo "1. Testing backend startup..."
cd /home/ec2-user/shadow-trainer/api_backend

# Check if all required Python packages are available
echo "   Checking Python dependencies..."
uv run python -c "
import sys
try:
    import fastapi
    import pydantic
    import uvicorn
    print('   ✅ Core dependencies found')
except ImportError as e:
    print(f'   ❌ Missing dependency: {e}')
    sys.exit(1)
"

# Test if inference modules are available
echo "   Checking ML pipeline dependencies..."
uv run python -c "
import sys
try:
    from src.inference import get_pytorch_device
    device = get_pytorch_device()
    print(f'   ✅ ML pipeline available, device: {device}')
except ImportError as e:
    print(f'   ⚠️  ML pipeline issue: {e}')
except Exception as e:
    print(f'   ⚠️  ML pipeline warning: {e}')
"

# Test frontend build
echo "2. Testing frontend setup..."
cd /home/ec2-user/shadow-trainer/api_frontend/shadow_trainer_web

if [ -f "package.json" ]; then
    echo "   ✅ React app found"
    if [ -d "node_modules" ]; then
        echo "   ✅ Dependencies installed"
    else
        echo "   ⚠️  Run 'npm install' to install dependencies"
    fi
else
    echo "   ❌ React app not found"
fi

# Test nginx config
echo "3. Testing nginx configuration..."
if [ -f "/etc/nginx/conf.d/proxy.conf" ]; then
    echo "   ✅ Nginx config exists"
    sudo nginx -t && echo "   ✅ Nginx config valid" || echo "   ❌ Nginx config invalid"
else
    echo "   ⚠️  Nginx not configured - run ./scripts/create_nginx_config.sh"
fi

echo ""
echo "🎯 Setup Summary:"
echo "   - Backend: Ready for testing"
echo "   - Frontend: Ready (ensure 'npm install' is run)"
echo "   - Nginx: Configure if needed"
echo ""
echo "🚀 To start the application:"
echo "   ./start_all.sh"
echo ""
echo "🔗 Access points:"
echo "   - Main app: http://www.shadow-trainer.com"
echo "   - Health check: http://www.shadow-trainer.com/health"
echo "   - API docs: http://www.shadow-trainer.com:8002/docs"
