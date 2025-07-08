# Shadow Trainer Deployment Guide

## Overview

This guide covers deployment options for the Shadow Trainer API, from local development to production environments.

## Local Development

### Prerequisites
- Python 3.9+
- UV package manager
- Git

### Setup
```bash
# Clone repository
git clone <repository-url>
cd shadow-trainer/api_inference

# Install dependencies
uv sync

# Run application
uv run python run.py
```

### Environment Variables
Create a `.env` file in the api_inference directory:
```bash
SHADOW_TRAINER_DEBUG=true
SHADOW_TRAINER_LOG_LEVEL=DEBUG
SHADOW_TRAINER_S3_BUCKET=your-bucket-name
SHADOW_TRAINER_AWS_ACCESS_KEY_ID=your-access-key
SHADOW_TRAINER_AWS_SECRET_ACCESS_KEY=your-secret-key
```

## Docker Deployment

### Build Image
```bash
# From api_inference directory
docker build -t shadow-trainer-api .
```

### Run Container
```bash
docker run -d \
  --name shadow-trainer \
  -p 8000:8000 \
  -e SHADOW_TRAINER_S3_BUCKET=your-bucket \
  -v /path/to/models:/app/assets/models \
  shadow-trainer-api
```

### Docker Compose
```yaml
version: '3.8'
services:
  shadow-trainer-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SHADOW_TRAINER_DEBUG=false
      - SHADOW_TRAINER_S3_BUCKET=shadow-trainer-prod
      - SHADOW_TRAINER_LOG_LEVEL=INFO
    volumes:
      - ./assets:/app/assets
      - ./logs:/app/logs
    restart: unless-stopped
```

## AWS Deployment

### EC2 Deployment

1. **Launch EC2 Instance**
   - Use Amazon Linux 2 or Ubuntu 20.04+
   - Minimum: t3.medium (2 vCPU, 4GB RAM)
   - Recommended: t3.large or g4dn.xlarge (for GPU)

2. **Setup Script**
```bash
#!/bin/bash
# Update system
sudo yum update -y

# Install Python 3.9+
sudo yum install -y python3 python3-pip git

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone and setup application
git clone <repository-url>
cd shadow-trainer/api_inference

# Install dependencies
uv sync

# Create systemd service
sudo tee /etc/systemd/system/shadow-trainer.service > /dev/null <<EOF
[Unit]
Description=Shadow Trainer API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/shadow-trainer/api_inference
Environment=SHADOW_TRAINER_HOST=0.0.0.0
Environment=SHADOW_TRAINER_PORT=8000
Environment=SHADOW_TRAINER_DEBUG=false
ExecStart=/home/ec2-user/.local/bin/uv run python run.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable shadow-trainer
sudo systemctl start shadow-trainer
```

3. **Configure Security Group**
   - Allow inbound traffic on port 8000
   - Restrict source IPs as needed

### ECS Deployment

1. **Create Task Definition**
```json
{
  "family": "shadow-trainer-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "shadow-trainer-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/shadow-trainer:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "SHADOW_TRAINER_S3_BUCKET",
          "value": "shadow-trainer-prod"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/shadow-trainer",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

2. **Create Service**
```bash
aws ecs create-service \
  --cluster my-cluster \
  --service-name shadow-trainer-service \
  --task-definition shadow-trainer-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

### Lambda Deployment (API Gateway + Lambda)

For serverless deployment with API Gateway:

1. **Install Mangum**
```bash
uv add mangum
```

2. **Create Lambda Handler**
```python
# lambda_handler.py
from mangum import Mangum
from shadow_trainer.main import app

handler = Mangum(app)
```

3. **Package and Deploy**
```bash
# Create deployment package
uv export --format requirements-txt > requirements.txt
pip install -r requirements.txt -t package/
cp -r shadow_trainer/ package/
cp lambda_handler.py package/

# Create ZIP
cd package && zip -r ../shadow-trainer-lambda.zip .

# Deploy to Lambda
aws lambda create-function \
  --function-name shadow-trainer-api \
  --runtime python3.9 \
  --role arn:aws:iam::account:role/lambda-execution-role \
  --handler lambda_handler.handler \
  --zip-file fileb://shadow-trainer-lambda.zip \
  --timeout 300 \
  --memory-size 1024
```

## Production Considerations

### Performance
- Use GPU instances (g4dn.xlarge+) for better inference performance
- Enable CloudFront for static asset caching
- Use Application Load Balancer for high availability
- Configure auto-scaling based on CPU/memory usage

### Security
- Use IAM roles instead of access keys
- Enable VPC security groups
- Use HTTPS/TLS encryption
- Implement API rate limiting
- Store secrets in AWS Secrets Manager

### Monitoring
- CloudWatch for logs and metrics
- Set up alarms for error rates and response times
- Use X-Ray for request tracing
- Health check endpoints for load balancer

### Storage
- Use S3 for model checkpoints and processed videos
- Configure S3 lifecycle policies for cleanup
- Use CloudFront for asset delivery
- Consider EFS for shared model storage

### Scaling
- Horizontal scaling with multiple instances
- Container orchestration with ECS or EKS
- Auto-scaling based on queue depth
- Load balancing across availability zones

## Environment-Specific Configuration

### Development
```bash
SHADOW_TRAINER_DEBUG=true
SHADOW_TRAINER_LOG_LEVEL=DEBUG
SHADOW_TRAINER_WORKERS=1
```

### Staging
```bash
SHADOW_TRAINER_DEBUG=false
SHADOW_TRAINER_LOG_LEVEL=INFO
SHADOW_TRAINER_WORKERS=2
```

### Production
```bash
SHADOW_TRAINER_DEBUG=false
SHADOW_TRAINER_LOG_LEVEL=WARNING
SHADOW_TRAINER_WORKERS=4
SHADOW_TRAINER_MAX_FILE_SIZE=209715200  # 200MB
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model checkpoint files exist
   - Verify S3 permissions
   - Check model configuration JSON

2. **Memory Issues**
   - Increase instance memory
   - Reduce batch size
   - Use smaller model sizes

3. **Performance Issues**
   - Use GPU instances
   - Optimize model configurations
   - Implement request queuing

4. **Storage Issues**
   - Check S3 bucket permissions
   - Verify network connectivity
   - Monitor disk space

### Logs
```bash
# View application logs
tail -f logs/shadow-trainer.log

# View system logs (systemd)
journalctl -u shadow-trainer -f

# CloudWatch logs
aws logs tail /ecs/shadow-trainer --follow
```
