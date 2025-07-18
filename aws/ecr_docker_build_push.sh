#!/bin/bash

set -e

# Configurable variables
REGION="us-east-2"
ACCOUNT_ID="381491870028"
REPO_NAME="shadow-trainer-celery"
IMAGE_TAG=$(git rev-parse --short HEAD)
LOCAL_IMAGE="${REPO_NAME}:${IMAGE_TAG}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# Remove old local image if exists
docker rmi $LOCAL_IMAGE || true

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Build Docker image
docker build -t $LOCAL_IMAGE .

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION > /dev/null 2>&1 || \
    aws ecr create-repository --repository-name $REPO_NAME --region $REGION

# Tag image for ECR
docker tag $LOCAL_IMAGE $ECR_URI

# Push image to ECR
docker push $ECR_URI
