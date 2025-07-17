#!/bin/bash

# -------- CONFIG --------
FRONTEND_DIR="/home/ec2-user/shadow-trainer/api_frontend/shadow_trainer_web/build"
DEPLOY_DIR="/var/www/frontend/build"
# ------------------------

set -e

echo "🔧 Building React app..."
cd "$FRONTEND_DIR"
npm install
npm run build

echo "🧹 Clearing existing frontend in $DEPLOY_DIR"
sudo rm -rf "$DEPLOY_DIR"/*
sudo mkdir -p "$DEPLOY_DIR"

echo "📁 Copying build files to $DEPLOY_DIR"
sudo cp -r ./* "$DEPLOY_DIR"

echo "🔄 Restarting NGINX"
sudo systemctl restart nginx

echo "✅ Frontend deployed to https://www.shadow-trainer.com"


echo "☁️ Syncing to S3 for static webpage hosting..."
aws s3 sync "$FRONTEND_DIR" s3://shadow-trainer-web --delete

echo "✅ Frontend also synced to S3 bucket: s3://shadow-trainer-web"