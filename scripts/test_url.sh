#!/bin/bash
# Test NGINX configuration and verify site availability

set -e

echo "Testing NGINX configuration..."
if sudo nginx -t; then
    echo "NGINX config test passed."
else
    echo "NGINX config test failed!" >&2
    exit 1
fi

echo "Reloading NGINX..."
sudo systemctl reload nginx

echo
echo "Checking site availability..."

for url in "http://shadow-trainer.com" "http://www.shadow-trainer.com"; do
    echo "Requesting: $url"
    if curl -fsSL "$url" > /dev/null; then
        echo "✅ $url is reachable."
    else
        echo "❌ $url is NOT reachable!" >&2
    fi
    echo
done

echo "Done."