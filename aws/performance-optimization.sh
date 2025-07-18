#!/bin/bash

# Performance optimization script for Shadow Trainer GPU-accelerated API
# Includes EC2 instance tuning, Docker optimization, and system configuration

echo "Optimizing Shadow Trainer performance for GPU workloads..."

# 1. EC2 Instance Optimization
echo "Configuring EC2 instance for optimal GPU performance..."

# Set GPU performance mode (for g4dn instances)
sudo nvidia-smi -pm 1  # Enable persistence mode
sudo nvidia-smi -acp 0  # Disable accounting
sudo nvidia-smi --auto-boost-default=ENABLED  # Enable auto boost

# Configure CPU governor for performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optimize memory settings
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# 2. Docker Performance Optimization
echo "Optimizing Docker configuration..."

# Create optimized Docker daemon configuration
sudo tee /etc/docker/daemon.json << EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ],
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-ulimits": {
        "memlock": {
            "hard": -1,
            "soft": -1
        },
        "stack": {
            "hard": 67108864,
            "soft": 67108864
        }
    },
    "experimental": true,
    "features": {
        "buildkit": true
    }
}
EOF

sudo systemctl restart docker
sudo systemctl enable docker

# 3. NVIDIA Container Runtime Optimization
echo "Optimizing NVIDIA container runtime..."

# Configure NVIDIA container runtime
sudo tee /etc/nvidia-container-runtime/config.toml << EOF
disable-require = false
#swarm-resource = "DOCKER_RESOURCE_GPU"

[nvidia-container-cli]
#root = "/run/nvidia/driver"
#path = "/usr/bin/nvidia-container-cli"
environment = []
#debug = "/var/log/nvidia-container-runtime.log"
#ldcache = "/etc/ld.so.cache"
load-kmods = true
#no-cgroups = false
#user = "root:video"
ldconfig = "@/sbin/ldconfig.real"

[nvidia-container-runtime]
#debug = "/var/log/nvidia-container-runtime.log"
EOF

# 4. System Limits Optimization
echo "Configuring system limits for high-performance workloads..."

# Configure limits for Docker and API processes
sudo tee -a /etc/security/limits.conf << EOF
# Shadow Trainer API limits
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited
EOF

# Configure systemd limits
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf << EOF
[Service]
LimitNOFILE=1048576
LimitNPROC=1048576
LimitCORE=infinity
TasksMax=1048576
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker

# 5. Redis Performance Optimization for ElastiCache
echo "Creating Redis performance optimization configuration..."

tee redis-performance.conf << EOF
# Redis configuration for optimal performance with Celery
# Apply these settings to ElastiCache parameter group

# Memory and persistence
maxmemory-policy allkeys-lru
save ""
rdbcompression yes
rdbchecksum yes

# Network and connection
tcp-keepalive 60
timeout 300
tcp-backlog 511

# Performance tuning
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Celery-specific optimizations
notify-keyspace-events Ex
EOF

# 6. PyTorch and CUDA Optimization
echo "Configuring PyTorch and CUDA optimizations..."

# Create PyTorch optimization script
tee pytorch-optimize.py << 'EOF'
import torch
import os

def optimize_pytorch_settings():
    """Configure PyTorch for optimal GPU performance"""
    
    # Enable optimized attention (if available)
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable TensorFloat-32 (TF32) for faster training on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Optimize memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Set optimal number of threads
    torch.set_num_threads(4)  # Adjust based on CPU cores
    
    print("PyTorch optimizations applied:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

if __name__ == "__main__":
    optimize_pytorch_settings()
EOF

# 7. Celery Worker Optimization
echo "Creating Celery worker optimization configuration..."

tee celery-optimize.conf << EOF
# Celery worker optimization settings
# Add these to your celery worker command or configuration

# Worker process optimization
worker_prefetch_multiplier = 1
task_acks_late = True
worker_max_tasks_per_child = 50

# Concurrency settings for GPU workloads
worker_concurrency = 2  # Lower concurrency for GPU-intensive tasks

# Memory optimization
worker_max_memory_per_child = 200000  # 200MB limit per worker

# Task routing optimization
task_routes = {
    'api_backend.celery_tasks.process_video': {'queue': 'gpu_queue'},
    'api_backend.celery_tasks.cleanup_task': {'queue': 'cpu_queue'}
}

# Result backend optimization
result_expires = 3600  # 1 hour
result_persistent = False
EOF

# 8. System monitoring setup
echo "Setting up performance monitoring..."

# Create monitoring script
tee monitor-performance.sh << 'EOF'
#!/bin/bash

echo "=== Shadow Trainer Performance Monitor ==="
echo "GPU Status:"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

echo -e "\nDocker Container Stats:"
docker stats --no-stream

echo -e "\nSystem Resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')"
echo "Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')"

echo -e "\nCelery Queue Status:"
docker exec shadow-trainer-redis redis-cli llen celery 2>/dev/null || echo "Redis not accessible"

echo -e "\nActive Connections:"
ss -tuln | grep :8002
EOF

chmod +x monitor-performance.sh

# 9. Final optimizations and cleanup
echo "Applying final optimizations..."

# Clear package caches
sudo apt-get autoremove -y
sudo apt-get autoclean

# Optimize system caches
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Performance optimization complete!"
echo ""
echo "Key optimizations applied:"
echo "✅ GPU persistence mode enabled"
echo "✅ CPU governor set to performance"
echo "✅ Docker configured for GPU workloads"
echo "✅ System limits optimized for high concurrency"
echo "✅ PyTorch and CUDA optimizations configured"
echo "✅ Celery worker settings tuned for GPU tasks"
echo "✅ Redis performance configuration created"
echo "✅ Performance monitoring script created"
echo ""
echo "Run './monitor-performance.sh' to check system performance"
echo "Apply Redis settings to your ElastiCache parameter group"
echo "Use PyTorch optimizations in your inference code"
