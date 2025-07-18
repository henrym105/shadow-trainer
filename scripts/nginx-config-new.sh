#!/bin/bash

set -e

DOMAINS=("shadow-trainer.com" "www.shadow-trainer.com" "api.shadow-trainer.com" "flower.shadow-trainer.com")

# Detect OS
OS=$(uname | tr '[:upper:]' '[:lower:]')
if [[ "$OS" == "darwin" ]]; then
    echo "Running on macOS (local only, skipping server config)..."
    exit 0
fi

# Directories
SITES_AVAILABLE="/etc/nginx/sites-available"
SITES_ENABLED="/etc/nginx/sites-enabled"

if [ ! -d "$SITES_AVAILABLE" ]; then
    echo "Using conf.d structure (Amazon Linux default)"
    SITES_AVAILABLE="/etc/nginx/conf.d"
    SITES_ENABLED="/etc/nginx/conf.d"
    USE_CONF_D=true
else
    USE_CONF_D=false
fi

# Remove any existing conflicting configurations
echo "Removing existing conflicting configurations..."
sudo rm -f /etc/nginx/conf.d/proxy.conf
sudo rm -f /etc/nginx/conf.d/default.conf
sudo rm -f /etc/nginx/sites-enabled/default
sudo rm -f "$SITES_AVAILABLE/shadow-trainer.com.conf"
sudo rm -f "$SITES_AVAILABLE/api.shadow-trainer.com.conf"
sudo rm -f "$SITES_AVAILABLE/flower.shadow-trainer.com.conf"
if [ "$USE_CONF_D" = false ]; then
    sudo rm -f "$SITES_ENABLED/shadow-trainer.com.conf"
    sudo rm -f "$SITES_ENABLED/api.shadow-trainer.com.conf"
    sudo rm -f "$SITES_ENABLED/flower.shadow-trainer.com.conf"
fi

# FRONTEND CONFIG
FRONTEND_CONF="$SITES_AVAILABLE/shadow-trainer.com.conf"
echo "Creating frontend configuration at $FRONTEND_CONF..."
sudo tee "$FRONTEND_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name shadow-trainer.com www.shadow-trainer.com;

    root /var/www/frontend/build;
    index index.html;

    # Set client max body size for video uploads (100MB)
    client_max_body_size 100M;

    location / {
        try_files \$uri /index.html;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
}
EOF

# BACKEND CONFIG
API_CONF="$SITES_AVAILABLE/api.shadow-trainer.com.conf"
echo "Creating API configuration at $API_CONF..."
sudo tee "$API_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name api.shadow-trainer.com;

    # Set client max body size for video uploads (100MB)
    client_max_body_size 100M;
    
    # Timeout settings for video processing
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;

    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Special handling for file uploads
        proxy_request_buffering off;
    }
}
EOF

# FLOWER MONITORING CONFIG
FLOWER_CONF="$SITES_AVAILABLE/flower.shadow-trainer.com.conf"
echo "Creating Flower monitoring configuration at $FLOWER_CONF..."
sudo tee "$FLOWER_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name flower.shadow-trainer.com;

    location / {
        proxy_pass http://127.0.0.1:5555;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support for real-time updates
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Flower specific timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# Enable configs if using sites-enabled
if [ "$USE_CONF_D" = false ]; then
    echo "Enabling configurations in sites-enabled..."
    sudo ln -sf "$FRONTEND_CONF" "$SITES_ENABLED/"
    sudo ln -sf "$API_CONF" "$SITES_ENABLED/"
    sudo ln -sf "$FLOWER_CONF" "$SITES_ENABLED/"
fi

# Reload NGINX
echo "Testing NGINX config..."
sudo nginx -t

echo "Restarting NGINX..."
sudo systemctl restart nginx

# Detect package manager and install Certbot
echo "Installing Certbot..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $NAME $VERSION"
    
    if [[ "$ID" == "amzn" ]]; then
        # Amazon Linux detected
        if [[ "$VERSION_ID" == "2" ]]; then
            echo "Amazon Linux 2 detected - installing EPEL and Certbot with yum..."
            sudo yum install -y epel-release
            sudo yum install -y certbot python3-certbot-nginx
        elif [[ "$VERSION_ID" == "2023" ]] || [[ "$PRETTY_NAME" == *"2023"* ]]; then
            echo "Amazon Linux 2023 detected - installing Certbot with dnf..."
            sudo dnf install -y python3-certbot python3-certbot-nginx
        else
            echo "Unknown Amazon Linux version, trying dnf first..."
            if ! sudo dnf install -y python3-certbot python3-certbot-nginx 2>/dev/null; then
                echo "dnf failed, trying yum with EPEL..."
                sudo yum install -y epel-release
                sudo yum install -y certbot python3-certbot-nginx
            fi
        fi
    else
        echo "Non-Amazon Linux detected: $ID"
        # Try common package managers
        if command -v dnf >/dev/null 2>&1; then
            echo "Using dnf..."
            sudo dnf install -y python3-certbot python3-certbot-nginx
        elif command -v yum >/dev/null 2>&1; then
            echo "Using yum with EPEL..."
            sudo yum install -y epel-release
            sudo yum install -y certbot python3-certbot-nginx
        elif command -v apt >/dev/null 2>&1; then
            echo "Using apt..."
            sudo apt update && sudo apt install -y certbot python3-certbot-nginx
        else
            echo "No supported package manager found."
            exit 1
        fi
    fi
else
    echo "Cannot detect OS version. Trying yum with EPEL as fallback..."
    sudo yum install -y epel-release
    sudo yum install -y certbot python3-certbot-nginx
fi

echo "Requesting TLS certificates via Certbot..."
CERT_DOMAINS=$(printf " -d %s" "${DOMAINS[@]}")
sudo certbot --nginx $CERT_DOMAINS --agree-tos --redirect --no-eff-email -m admin@shadow-trainer.com

echo "✅ HTTPS enabled and NGINX configured for:"
printf " - %s\n" "${DOMAINS[@]}"