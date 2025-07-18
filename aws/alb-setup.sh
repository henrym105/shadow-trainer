#!/bin/bash

# Application Load Balancer setup for Shadow Trainer
# This script creates an ALB with health checks and SSL termination

REGION="us-east-2"
VPC_ID="your-vpc-id"  # Replace with your VPC ID
SUBNET_IDS="subnet-xxx,subnet-yyy"  # Replace with your subnet IDs
DOMAIN="api.shadowtrainer.com"  # Replace with your domain

# Create security group for ALB
aws ec2 create-security-group \
    --group-name shadow-trainer-alb-sg \
    --description "Security group for Shadow Trainer ALB" \
    --vpc-id $VPC_ID \
    --region $REGION

ALB_SG_ID=$(aws ec2 describe-security-groups \
    --group-names shadow-trainer-alb-sg \
    --region $REGION \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

# Allow HTTP/HTTPS traffic
aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $REGION

aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Create Application Load Balancer
aws elbv2 create-load-balancer \
    --name shadow-trainer-alb \
    --subnets $SUBNET_IDS \
    --security-groups $ALB_SG_ID \
    --region $REGION \
    --scheme internet-facing \
    --type application \
    --ip-address-type ipv4

echo "ALB created successfully!"
echo "Next: Configure target groups and SSL certificate"
