#!/bin/bash

set -e

# Multi-stage Docker build and push script for Shadow Trainer services
# Builds and pushes all service variants with size optimization

# Configurable variables
REGION="us-east-2"
ACCOUNT_ID="381491870028"
BASE_REPO_NAME="shadow-trainer"
IMAGE_TAG="latest"
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Define all services and their targets
declare -A SERVICES=(
    ["api"]="api"
    ["worker"]="worker" 
    ["beat"]="beat"
    ["flower"]="flower"
)

echo "🚀 Starting multi-stage Docker build and ECR push..."
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo "Services: ${!SERVICES[@]}"

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ECR_BASE"

# Build all services using docker-compose for efficiency
echo "📦 Building all services with multi-stage optimization..."
docker-compose -f docker-compose.celery.yml build --parallel

# Process each service
for SERVICE in "${!SERVICES[@]}"; do
    TARGET="${SERVICES[$SERVICE]}"
    REPO_NAME="${BASE_REPO_NAME}-${SERVICE}"
    LOCAL_IMAGE="${BASE_REPO_NAME}-${SERVICE}:${IMAGE_TAG}"
    ECR_URI="${ECR_BASE}/${REPO_NAME}:${IMAGE_TAG}"
    
    echo ""
    echo "🔧 Processing service: $SERVICE"
    echo "  Target: $TARGET"
    echo "  Local image: $LOCAL_IMAGE"
    echo "  ECR URI: $ECR_URI"
    
    # Create ECR repository if it doesn't exist
    echo "  📋 Ensuring ECR repository exists..."
    aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION > /dev/null 2>&1 || \
        aws ecr create-repository --repository-name $REPO_NAME --region $REGION --image-scanning-configuration scanOnPush=true
    
    # Tag image for ECR
    echo "  🏷️  Tagging image for ECR..."
    docker tag $LOCAL_IMAGE $ECR_URI
    
    # Push image to ECR
    echo "  ⬆️  Pushing to ECR..."
    docker push $ECR_URI
    
    # Get image size info
    IMAGE_SIZE=$(docker images $LOCAL_IMAGE --format "table {{.Size}}" | tail -n +2)
    echo "  📊 Image size: $IMAGE_SIZE"
    
    echo "  ✅ $SERVICE service pushed successfully"
done

# Show summary
echo ""
echo "📊 Build and Push Summary:"
echo "=========================="
docker images | grep shadow-trainer | grep latest | while read line; do
    IMAGE=$(echo $line | awk '{print $1":"$2}')
    SIZE=$(echo $line | awk '{print $7}')
    echo "  $IMAGE -> $SIZE"
done

echo ""
echo "🎯 ECR Repository URLs:"
for SERVICE in "${!SERVICES[@]}"; do
    REPO_NAME="${BASE_REPO_NAME}-${SERVICE}"
    echo "  $SERVICE: ${ECR_BASE}/${REPO_NAME}:${IMAGE_TAG}"
done

echo ""
echo "✅ All services built and pushed successfully!"
echo ""
echo "📋 Next steps:"
echo "  1. Update ECS task definitions with new ECR URIs"
echo "  2. Deploy to ECS cluster"
echo "  3. Monitor deployment in CloudWatch"
