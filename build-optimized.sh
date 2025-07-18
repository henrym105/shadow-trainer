#!/bin/bash

# Multi-stage build script for Shadow Trainer services
# Optimized for size and build caching

echo "🚀 Building Shadow Trainer services with multi-stage optimization..."

# Build all services with caching
echo "📦 Building all services in parallel..."
docker-compose -f docker-compose.celery.yml build --parallel

# Show size comparison
echo ""
echo "📊 Image size comparison:"
echo "================================="
docker images | grep shadow-trainer | sort -k1,1 -k2,2

# Calculate total size savings
OLD_SIZE=$(docker images | grep shadow-trainer-api-cuda | grep latest | awk '{print $7}' | sed 's/GB//')
NEW_SIZE=$(docker images | grep shadow-trainer-api | grep latest | awk '{print $7}' | sed 's/GB//')

if [[ -n "$OLD_SIZE" && -n "$NEW_SIZE" ]]; then
    SAVINGS=$(echo "$OLD_SIZE - $NEW_SIZE" | bc)
    PERCENT=$(echo "scale=1; ($SAVINGS / $OLD_SIZE) * 100" | bc)
    
    echo ""
    echo "💰 Size Optimization Results:"
    echo "  Original size: ${OLD_SIZE}GB"
    echo "  Optimized size: ${NEW_SIZE}GB"
    echo "  Savings: ${SAVINGS}GB (${PERCENT}%)"
    echo "  Total savings for 4 services: $(echo "$SAVINGS * 4" | bc)GB"
fi

echo ""
echo "✅ Multi-stage build complete!"
echo ""
echo "🔧 Next steps:"
echo "  1. Test services: docker-compose -f docker-compose.celery.yml up"
echo "  2. Push to ECR: ./aws/ecr_docker_build_push.sh"
echo "  3. Deploy to production with optimized images"
