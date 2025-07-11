#!/bin/bash

# -------- CONFIG --------
EXTERNAL_IP="3.84.160.141"  # <-- This is the AWS elastic IP associated with the EC2 instance
DOMAIN="shadow-trainer.com"  # Add domain configuration
PROXY_CONF="/etc/nginx/conf.d/proxy.conf"
# ------------------------

set -e

echo "Detecting OS version..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $NAME $VERSION"
    
    if [[ "$ID" == "amzn" ]]; then
        # Amazon Linux detected
        if [[ "$VERSION_ID" == "2" ]]; then
            echo "Amazon Linux 2 detected - using yum..."
            # For Amazon Linux 2, try direct yum install first
            if ! sudo yum install -y nginx; then
                echo "Direct yum install failed, trying EPEL..."
                sudo yum install -y epel-release
                sudo yum install -y nginx
            fi
        elif [[ "$VERSION_ID" == "2023" ]] || [[ "$PRETTY_NAME" == *"2023"* ]]; then
            echo "Amazon Linux 2023 detected - using dnf..."
            sudo dnf install -y nginx
        else
            echo "Unknown Amazon Linux version, trying dnf first..."
            if ! sudo dnf install -y nginx 2>/dev/null; then
                echo "dnf failed, trying yum..."
                sudo yum install -y nginx
            fi
        fi
    else
        echo "Non-Amazon Linux detected: $ID"
        # Try common package managers
        if command -v dnf >/dev/null 2>&1; then
            echo "Using dnf..."
            sudo dnf install -y nginx
        elif command -v yum >/dev/null 2>&1; then
            echo "Using yum..."
            sudo yum install -y nginx
        elif command -v apt >/dev/null 2>&1; then
            echo "Using apt..."
            sudo apt update && sudo apt install -y nginx
        else
            echo "No supported package manager found."
            exit 1
        fi
    fi
else
    echo "Cannot detect OS version. Trying yum as fallback..."
    sudo yum install -y nginx
fi

echo "Writing reverse proxy config to $PROXY_CONF..."
sudo tee "$PROXY_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8002/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF


echo "Testing NGINX config..."
sudo nginx -t

echo "Enabling and starting NGINX..."
sudo systemctl enable nginx
# sudo systemctl start nginx
sudo systemctl restart nginx

echo "✅ NGINX is now reverse proxying to http://$EXTERNAL_IP"
