# AWS Production Deployment Guide for Shadow Trainer

## Overview
This guide walks through deploying the GPU-accelerated Shadow Trainer API to AWS with auto-scaling, monitoring, and performance optimization.

## Prerequisites
- AWS CLI configured with credentials for account 381491870028
- Docker installed locally
- ECR repository created: `shadow-trainer-api`
- VPC with public subnets in us-east-2

## Deployment Steps

### 1. Build and Push Container to ECR
```bash
# Build and push the GPU-enabled container
cd /home/ec2-user/shadow-trainer
chmod +x aws/ecr_docker_build_push.sh
./aws/ecr_docker_build_push.sh
```

### 2. Set up Redis ElastiCache
```bash
# Create Redis cluster for Celery broker
chmod +x aws/elasticache-setup.sh
./aws/elasticache-setup.sh
```
**Wait for ElastiCache cluster to be available (~5-10 minutes)**

### 3. Create Application Load Balancer
```bash
# Set up ALB with SSL termination
chmod +x aws/alb-setup.sh
./aws/alb-setup.sh
```

### 4. Launch ECS Cluster with GPU Instances
```bash
# Create ECS cluster with g4dn.xlarge instances
aws ecs create-cluster --cluster-name shadow-trainer-cluster --region us-east-2

# Launch EC2 instances for ECS (modify subnet and security group IDs)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type g4dn.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --iam-instance-profile Name=ecsInstanceRole \
    --user-data file://ecs-user-data.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=shadow-trainer-ecs-node}]' \
    --count 2
```

### 5. Deploy ECS Task Definition
```bash
# Update subnet and security group IDs in the task definition
vim aws/ecs-task-definition.json

# Register task definition
aws ecs register-task-definition \
    --cli-input-json file://aws/ecs-task-definition.json \
    --region us-east-2
```

### 6. Create ECS Service
```bash
# Update ALB target group ARN and subnet IDs
vim aws/ecs-service-definition.json

# Create the service
aws ecs create-service \
    --cli-input-json file://aws/ecs-service-definition.json \
    --region us-east-2
```

### 7. Configure Auto Scaling
```bash
# Set up auto scaling policies
chmod +x aws/autoscaling-setup.sh
./aws/autoscaling-setup.sh
```

### 8. Performance Optimization
```bash
# SSH into ECS instances and run optimization
# Copy the script to instances first
scp aws/performance-optimization.sh ec2-user@your-instance-ip:~
ssh ec2-user@your-instance-ip
chmod +x performance-optimization.sh
sudo ./performance-optimization.sh
```

### 9. Set up Monitoring
```bash
# Deploy CloudWatch metrics collector as sidecar or daemon
# Update the cloudwatch-metrics.py with ElastiCache endpoint
vim aws/cloudwatch-metrics.py

# Deploy as ECS service or install on instances
```

## Post-Deployment Configuration

### Environment Variables Required
Update these in your ECS task definition:
```json
{
    "name": "REDIS_URL",
    "value": "redis://shadow-trainer-redis.cache.amazonaws.com:6379/0"
},
{
    "name": "CELERY_BROKER_URL",
    "value": "redis://shadow-trainer-redis.cache.amazonaws.com:6379/0"
},
{
    "name": "CELERY_RESULT_BACKEND",
    "value": "redis://shadow-trainer-redis.cache.amazonaws.com:6379/0"
},
{
    "name": "AWS_DEFAULT_REGION",
    "value": "us-east-2"
}
```

### DNS Configuration
Point your domain to the ALB:
```bash
# Get ALB DNS name
aws elbv2 describe-load-balancers \
    --names shadow-trainer-alb \
    --query 'LoadBalancers[0].DNSName' \
    --output text \
    --region us-east-2

# Create CNAME record: api.yourdomain.com -> ALB DNS name
```

### SSL Certificate
```bash
# Request ACM certificate
aws acm request-certificate \
    --domain-name api.yourdomain.com \
    --validation-method DNS \
    --region us-east-2

# Add certificate to ALB HTTPS listener (update listener ARN)
aws elbv2 modify-listener \
    --listener-arn arn:aws:elasticloadbalancing:us-east-2:381491870028:listener/app/shadow-trainer-alb/xxx/xxx \
    --certificates CertificateArn=arn:aws:acm:us-east-2:381491870028:certificate/xxx
```

## Monitoring and Maintenance

### Health Checks
- **ALB Health Check**: `GET /health` on port 8002
- **ECS Task Health**: Container health check on inference endpoint
- **GPU Monitoring**: CloudWatch custom metrics via cloudwatch-metrics.py

### Scaling Triggers
- **CPU > 70%**: Scale out (300s cooldown)
- **GPU > 80%**: Scale out (600s cooldown)  
- **Queue depth > 50**: Scale out (180s cooldown)
- **Low utilization**: Scale in (600-900s cooldown)

### Cost Optimization
- **Spot Instances**: Use for worker nodes (not API nodes)
- **Reserved Instances**: For baseline capacity
- **S3 Lifecycle**: Archive old video outputs after 30 days
- **CloudWatch Logs**: Retention policy 7-14 days

## Troubleshooting

### Common Issues
1. **GPU not detected**: Check NVIDIA drivers and docker runtime
2. **High memory usage**: Increase `worker_max_tasks_per_child` setting
3. **Queue backup**: Check Redis connectivity and worker health
4. **Cold start delays**: Increase minimum desired capacity

### Debug Commands
```bash
# Check ECS service status
aws ecs describe-services --cluster shadow-trainer-cluster --services shadow-trainer-api-service

# View container logs
aws logs get-log-events --log-group-name /ecs/shadow-trainer-api --log-stream-name ecs/api/task-id

# Check Redis connectivity
redis-cli -h shadow-trainer-redis.cache.amazonaws.com ping

# Monitor GPU usage
nvidia-smi -l 1

# Check Celery workers
docker exec container-id celery -A api_backend.celery_app inspect active
```

## Performance Benchmarks

### Expected Performance (g4dn.xlarge)
- **Video Processing**: 2-4 concurrent jobs
- **API Response Time**: <200ms for health checks
- **GPU Utilization**: 60-80% during processing
- **Memory Usage**: ~4-6GB per container
- **Throughput**: 10-20 videos/hour per instance

### Scaling Targets
- **Minimum**: 2 instances (high availability)
- **Maximum**: 10 instances (cost control)
- **Target GPU**: 80% utilization
- **Target Queue**: <50 pending jobs

## Security Considerations

### Network Security
- VPC with private subnets for ECS tasks
- ALB in public subnets only
- Security groups with minimal required ports
- NAT Gateway for outbound access

### Data Security
- S3 bucket encryption at rest
- ElastiCache encryption in transit
- ECS task execution role with minimal permissions
- Secrets Manager for sensitive configuration

### Access Control
- IAM roles for ECS tasks
- ALB access logs to S3
- CloudTrail for API calls
- VPC Flow Logs for network monitoring

## Maintenance Schedule

### Daily
- Monitor CloudWatch dashboard
- Check ECS service health
- Review error logs

### Weekly
- Update container images
- Review scaling events
- Check cost optimization

### Monthly
- Security patch updates
- Performance optimization review
- Capacity planning assessment

---

## Quick Start Commands

For immediate deployment after prerequisites:
```bash
./aws/ecr_docker_build_push.sh
./aws/elasticache-setup.sh
./aws/alb-setup.sh
# Wait for resources, then:
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json
aws ecs create-service --cli-input-json file://aws/ecs-service-definition.json
./aws/autoscaling-setup.sh
```

**Estimated deployment time**: 15-20 minutes
**Cost estimate**: $200-400/month (depending on usage)
