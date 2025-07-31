#!/bin/bash

# Shadow Trainer EC2 Auto-Start Setup Script
# Run this script on a new EC2 instance to configure automatic startup

set -e

echo "=== Shadow Trainer EC2 Auto-Start Setup ==="
echo "This script will configure your EC2 instance to automatically start Shadow Trainer on boot"
echo

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Please run this script as ec2-user (not root)"
   exit 1
fi

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Verify required files exist
echo "Checking required files..."
if [[ ! -f "$PROJECT_DIR/docker-compose.prod.yml" ]]; then
    echo "Error: docker-compose.prod.yml not found in $PROJECT_DIR"
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/Makefile" ]]; then
    echo "Error: Makefile not found in $PROJECT_DIR"
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/.env.prod" ]]; then
    echo "Warning: .env.prod not found. Make sure to create it before running the service."
fi

echo "✓ Required files found"

# Install Docker and Docker Compose if not present
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo yum update -y
    sudo yum install -y docker
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -a -G docker ec2-user
    echo "✓ Docker installed"
else
    echo "✓ Docker already installed"
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✓ Docker Compose installed"
else
    echo "✓ Docker Compose already installed"
fi


# Create systemd service file
echo "Creating systemd service..."
sudo tee /etc/systemd/system/shadow-trainer.service > /dev/null << EOF
[Unit]
Description=Shadow Trainer Production Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=$PROJECT_DIR
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/make prod-build
ExecStop=/usr/bin/make stop
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "✓ Systemd service file created"


# Enable the service
echo "Enabling shadow-trainer service..."
sudo systemctl daemon-reload
sudo systemctl enable shadow-trainer.service

echo "✓ Service enabled for auto-start"

# Check service status
echo "Checking service status..."
sudo systemctl status shadow-trainer.service --no-pager || true

echo
echo "=== Setup Complete ==="
echo "Shadow Trainer will now automatically start when this EC2 instance boots up."
echo
echo "Manual control commands:"
echo "  Start:  sudo systemctl start shadow-trainer"
echo "  Stop:   sudo systemctl stop shadow-trainer"
echo "  Status: sudo systemctl status shadow-trainer"
echo "  Logs:   sudo journalctl -u shadow-trainer -f"
echo
echo "Note: Make sure your .env.prod file is configured before the next reboot."
echo "You may need to log out and back in for Docker group permissions to take effect."