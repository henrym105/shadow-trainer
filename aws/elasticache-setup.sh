#!/bin/bash

# ElastiCache Redis setup for Shadow Trainer
REGION="us-east-2"
SUBNET_GROUP_NAME="shadow-trainer-redis-subnet-group"
CACHE_CLUSTER_ID="shadow-trainer-redis"

# Create subnet group for ElastiCache
aws elasticache create-cache-subnet-group \
    --cache-subnet-group-name $SUBNET_GROUP_NAME \
    --cache-subnet-group-description "Subnet group for Shadow Trainer Redis" \
    --subnet-ids subnet-xxx subnet-yyy \
    --region $REGION

# Create Redis cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id $CACHE_CLUSTER_ID \
    --cache-node-type cache.r6g.large \
    --engine redis \
    --engine-version 7.0 \
    --num-cache-nodes 1 \
    --cache-parameter-group default.redis7 \
    --cache-subnet-group-name $SUBNET_GROUP_NAME \
    --security-group-ids sg-xxxxxxxxx \
    --port 6379 \
    --region $REGION

echo "ElastiCache Redis cluster creation initiated"
echo "It will take 10-15 minutes to complete"

# Get endpoint after creation
aws elasticache describe-cache-clusters \
    --cache-cluster-id $CACHE_CLUSTER_ID \
    --show-cache-node-info \
    --region $REGION \
    --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
    --output text
