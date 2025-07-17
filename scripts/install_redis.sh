#!/bin/bash
#
# Redis installation script for Amazon Linux
#

set -e

echo "=== Redis Installation for Amazon Linux ==="
echo ""

# Function to detect the Linux distribution
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VERSION=$VERSION_ID
    else
        echo "❌ Cannot detect operating system"
        exit 1
    fi
    
    echo "🔍 Detected OS: $OS $VERSION"
}

# Function to install Redis on Amazon Linux
install_redis_amazon_linux() {
    echo "📦 Installing Redis on Amazon Linux..."
    
    # Try amazon-linux-extras first (Amazon Linux 2)
    if command -v amazon-linux-extras &> /dev/null; then
        echo "Using amazon-linux-extras to install Redis 6..."
        sudo amazon-linux-extras install redis6 -y
    elif command -v dnf &> /dev/null; then
        # Amazon Linux 2023 uses dnf
        echo "Using dnf to install Redis..."
        sudo dnf install redis -y
    else
        # Fallback to EPEL
        echo "Using EPEL repository..."
        sudo yum install epel-release -y
        sudo yum install redis -y
    fi
}

# Function to install Redis on Ubuntu/Debian
install_redis_ubuntu() {
    echo "📦 Installing Redis on Ubuntu/Debian..."
    sudo apt update
    sudo apt install redis-server -y
}

# Function to install Redis on CentOS/RHEL
install_redis_centos() {
    echo "📦 Installing Redis on CentOS/RHEL..."
    sudo yum install epel-release -y
    sudo yum install redis -y
}

# Function to start and enable Redis
configure_redis() {
    echo "🔧 Configuring Redis service..."
    
    # Start Redis service
    sudo systemctl start redis
    sudo systemctl enable redis
    
    # Check if Redis is running
    if systemctl is-active --quiet redis; then
        echo "✅ Redis service is running"
    else
        echo "❌ Failed to start Redis service"
        return 1
    fi
    
    # Test Redis connectivity
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is responding to ping"
    else
        echo "❌ Redis is not responding"
        return 1
    fi
    
    # Show Redis status
    echo ""
    echo "📊 Redis Status:"
    sudo systemctl status redis --no-pager -l
}

# Function to install Redis via Docker (fallback option)
install_redis_docker() {
    echo "🐳 Installing Redis via Docker..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        return 1
    fi
    
    # Pull and run Redis container
    docker pull redis:7-alpine
    docker run -d \
        --name redis-shadow-trainer \
        -p 6379:6379 \
        --restart unless-stopped \
        redis:7-alpine
    
    # Wait a moment for container to start
    sleep 3
    
    # Test connectivity
    if docker exec redis-shadow-trainer redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis Docker container is running and responding"
        echo "📝 Redis is running in Docker container 'redis-shadow-trainer'"
        echo "🔧 To stop: docker stop redis-shadow-trainer"
        echo "🔧 To start: docker start redis-shadow-trainer"
    else
        echo "❌ Redis Docker container failed to start properly"
        return 1
    fi
}

# Main installation function
main() {
    detect_os
    echo ""
    
    # Check if Redis is already installed
    if command -v redis-server &> /dev/null || command -v redis-cli &> /dev/null; then
        echo "ℹ️  Redis appears to be already installed"
        
        if redis-cli ping > /dev/null 2>&1; then
            echo "✅ Redis is already running and accessible"
            exit 0
        else
            echo "⚠️  Redis is installed but not running. Attempting to start..."
            sudo systemctl start redis || {
                echo "❌ Failed to start Redis service"
                exit 1
            }
            configure_redis
            exit 0
        fi
    fi
    
    # Install based on OS
    case "$OS" in
        *"Amazon Linux"*)
            install_redis_amazon_linux
            configure_redis
            ;;
        *"Ubuntu"*|*"Debian"*)
            install_redis_ubuntu
            configure_redis
            ;;
        *"CentOS"*|*"Red Hat"*|*"RHEL"*)
            install_redis_centos
            configure_redis
            ;;
        *)
            echo "⚠️  Unsupported OS: $OS"
            echo "🐳 Attempting Docker installation as fallback..."
            install_redis_docker
            ;;
    esac
    
    echo ""
    echo "🎉 Redis installation completed successfully!"
    echo ""
    echo "📝 Useful Redis commands:"
    echo "   redis-cli ping                    # Test connectivity"
    echo "   redis-cli info                    # Show Redis info"
    echo "   sudo systemctl status redis      # Check service status"
    echo "   sudo systemctl restart redis     # Restart Redis"
    echo ""
    echo "🔧 Configuration file: /etc/redis.conf (or /etc/redis/redis.conf)"
    echo "📊 Redis is now ready for Shadow Trainer Celery integration!"
}

# Handle command line arguments
case "${1:-}" in
    --docker)
        echo "🐳 Installing Redis via Docker..."
        install_redis_docker
        ;;
    --help|-h)
        echo "Usage: $0 [--docker] [--help]"
        echo ""
        echo "Options:"
        echo "  --docker    Install Redis using Docker instead of system packages"
        echo "  --help      Show this help message"
        ;;
    *)
        main
        ;;
esac
