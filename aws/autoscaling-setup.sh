#!/bin/bash

# Configure ECS Service Auto Scaling for Shadow Trainer API
# Includes GPU utilization-based scaling and queue depth monitoring

AWS_REGION="us-east-2"
CLUSTER_NAME="shadow-trainer-cluster"
SERVICE_NAME="shadow-trainer-api-service"
SCALABLE_TARGET_RESOURCE="service/${CLUSTER_NAME}/${SERVICE_NAME}"

echo "Setting up ECS Service Auto Scaling for Shadow Trainer API..."

# Create Application Auto Scaling scalable target
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id "$SCALABLE_TARGET_RESOURCE" \
    --min-capacity 2 \
    --max-capacity 10 \
    --region "$AWS_REGION"

# Create scaling policy based on CPU utilization
aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id "$SCALABLE_TARGET_RESOURCE" \
    --policy-name "shadow-trainer-cpu-scaling" \
    --policy-type "TargetTrackingScaling" \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
        },
        "ScaleOutCooldown": 300,
        "ScaleInCooldown": 300,
        "DisableScaleIn": false
    }' \
    --region "$AWS_REGION"

# Create scaling policy based on GPU utilization (custom metric)
aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id "$SCALABLE_TARGET_RESOURCE" \
    --policy-name "shadow-trainer-gpu-scaling" \
    --policy-type "TargetTrackingScaling" \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 80.0,
        "CustomizedMetricSpecification": {
            "MetricName": "GPUUtilization",
            "Namespace": "ShadowTrainer/GPU",
            "Dimensions": [
                {
                    "Name": "ServiceName",
                    "Value": "shadow-trainer-api"
                }
            ],
            "Statistic": "Average"
        },
        "ScaleOutCooldown": 600,
        "ScaleInCooldown": 900,
        "DisableScaleIn": false
    }' \
    --region "$AWS_REGION"

# Create scaling policy based on Celery queue depth
aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id "$SCALABLE_TARGET_RESOURCE" \
    --policy-name "shadow-trainer-queue-scaling" \
    --policy-type "TargetTrackingScaling" \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 50.0,
        "CustomizedMetricSpecification": {
            "MetricName": "CeleryQueueDepth",
            "Namespace": "ShadowTrainer/Celery",
            "Dimensions": [
                {
                    "Name": "QueueName",
                    "Value": "video_processing"
                }
            ],
            "Statistic": "Average"
        },
        "ScaleOutCooldown": 180,
        "ScaleInCooldown": 600,
        "DisableScaleIn": false
    }' \
    --region "$AWS_REGION"

echo "Auto Scaling policies configured:"
echo "- CPU-based scaling: Target 70% CPU utilization"
echo "- GPU-based scaling: Target 80% GPU utilization"
echo "- Queue-based scaling: Target 50 jobs in queue"
echo "- Min instances: 2, Max instances: 10"

# Create CloudWatch dashboard for monitoring
aws cloudwatch put-dashboard \
    --dashboard-name "ShadowTrainerAutoScaling" \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        [ "AWS/ECS", "CPUUtilization", "ServiceName", "shadow-trainer-api-service", "ClusterName", "shadow-trainer-cluster" ],
                        [ "ShadowTrainer/GPU", "GPUUtilization", "ServiceName", "shadow-trainer-api" ],
                        [ "ShadowTrainer/Celery", "CeleryQueueDepth", "QueueName", "video_processing" ]
                    ],
                    "view": "timeSeries",
                    "stacked": false,
                    "region": "us-east-2",
                    "title": "Shadow Trainer Scaling Metrics",
                    "period": 300
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        [ "AWS/ECS", "DesiredCount", "ServiceName", "shadow-trainer-api-service", "ClusterName", "shadow-trainer-cluster" ],
                        [ ".", "RunningTaskCount", ".", ".", ".", "." ]
                    ],
                    "view": "timeSeries",
                    "stacked": false,
                    "region": "us-east-2",
                    "title": "ECS Service Task Count",
                    "period": 300
                }
            }
        ]
    }' \
    --region "$AWS_REGION"

echo "CloudWatch dashboard 'ShadowTrainerAutoScaling' created for monitoring"
echo "Auto Scaling setup complete!"
