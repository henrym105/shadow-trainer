#!/bin/bash

set -e

DOMAINS=("shadow-trainer.com" "www.shadow-trainer.com" "api.shadow-trainer.com")

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

# FRONTEND CONFIG
FRONTEND_CONF="$SITES_AVAILABLE/shadow-trainer.com.conf"
sudo tee "$FRONTEND_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name shadow-trainer.com www.shadow-trainer.com;

    root /var/www/frontend/build;
    index index.html;

    location / {
        try_files \$uri /index.html;
    }
}
EOF

# BACKEND CONFIG
API_CONF="$SITES_AVAILABLE/api.shadow-trainer.com.conf"
sudo tee "$API_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name api.shadow-trainer.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
EOF

# Enable configs if using sites-enabled
if [ "$USE_CONF_D" = false ]; then
    sudo ln -sf "$FRONTEND_CONF" "$SITES_ENABLED/"
    sudo ln -sf "$API_CONF" "$SITES_ENABLED/"
fi

# Reload NGINX
echo "Testing NGINX config..."
sudo nginx -t

echo "Restarting NGINX..."
sudo systemctl restart nginx

# Certbot install
echo "Installing Certbot..."
sudo yum install -y epel-release
sudo yum install -y certbot python3-certbot-nginx

echo "Requesting TLS certificates via Certbot..."
CERT_DOMAINS=$(printf " -d %s" "${DOMAINS[@]}")
sudo certbot --nginx $CERT_DOMAINS --agree-tos --redirect --no-eff-email -m admin@shadow-trainer.com

echo "✅ HTTPS enabled and NGINX configured for:"
printf " - %s\n" "${DOMAINS[@]}"