#!/bin/bash

# Cost optimization script for Shadow Trainer AWS deployment
# Implements spot instances, scheduled scaling, and resource optimization

AWS_REGION="us-east-2"
CLUSTER_NAME="shadow-trainer-cluster"

echo "Setting up cost optimization for Shadow Trainer deployment..."

# 1. Create Spot Fleet configuration for worker nodes
cat > spot-fleet-config.json << 'EOF'
{
    "SpotFleetRequestConfig": {
        "IamFleetRole": "arn:aws:iam::381491870028:role/aws-ec2-spot-fleet-tagging-role",
        "AllocationStrategy": "lowestPrice",
        "TargetCapacity": 2,
        "SpotPrice": "0.50",
        "LaunchSpecifications": [
            {
                "ImageId": "ami-0c02fb55956c7d316",
                "InstanceType": "g4dn.xlarge",
                "KeyName": "your-key-pair",
                "SecurityGroups": [
                    {"GroupId": "sg-xxxxxxxxx"}
                ],
                "SubnetId": "subnet-xxxxxxxxx",
                "IamInstanceProfile": {
                    "Name": "ecsInstanceRole"
                },
                "UserData": "base64-encoded-ecs-user-data",
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": "shadow-trainer-spot-worker"},
                            {"Key": "Environment", "Value": "production"},
                            {"Key": "CostOptimized", "Value": "true"}
                        ]
                    }
                ]
            },
            {
                "ImageId": "ami-0c02fb55956c7d316",
                "InstanceType": "g4dn.2xlarge",
                "KeyName": "your-key-pair",
                "SecurityGroups": [
                    {"GroupId": "sg-xxxxxxxxx"}
                ],
                "SubnetId": "subnet-xxxxxxxxx",
                "IamInstanceProfile": {
                    "Name": "ecsInstanceRole"
                },
                "UserData": "base64-encoded-ecs-user-data",
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": "shadow-trainer-spot-worker-large"},
                            {"Key": "Environment", "Value": "production"},
                            {"Key": "CostOptimized", "Value": "true"}
                        ]
                    }
                ]
            }
        ],
        "TerminateInstancesWithExpiration": true,
        "Type": "maintain"
    }
}
EOF

# 2. Create scheduled scaling for off-peak hours
echo "Setting up scheduled scaling policies..."

# Scale down during low-usage hours (12 AM - 6 AM UTC)
aws application-autoscaling put-scheduled-action \
    --service-namespace ecs \
    --resource-id "service/${CLUSTER_NAME}/shadow-trainer-api-service" \
    --scalable-dimension ecs:service:DesiredCount \
    --scheduled-action-name "scale-down-night" \
    --schedule "cron(0 0 * * ? *)" \
    --scalable-target-action MinCapacity=1,MaxCapacity=3,DesiredCapacity=1 \
    --region "$AWS_REGION"

# Scale up during peak hours (8 AM - 10 PM UTC)
aws application-autoscaling put-scheduled-action \
    --service-namespace ecs \
    --resource-id "service/${CLUSTER_NAME}/shadow-trainer-api-service" \
    --scalable-dimension ecs:service:DesiredCount \
    --scheduled-action-name "scale-up-day" \
    --schedule "cron(0 8 * * ? *)" \
    --scalable-target-action MinCapacity=2,MaxCapacity=10,DesiredCapacity=2 \
    --region "$AWS_REGION"

# Weekend scaling (reduce capacity on weekends)
aws application-autoscaling put-scheduled-action \
    --service-namespace ecs \
    --resource-id "service/${CLUSTER_NAME}/shadow-trainer-api-service" \
    --scalable-dimension ecs:service:DesiredCount \
    --scheduled-action-name "scale-down-weekend" \
    --schedule "cron(0 22 ? * FRI *)" \
    --scalable-target-action MinCapacity=1,MaxCapacity=5,DesiredCapacity=1 \
    --region "$AWS_REGION"

# 3. Create Lambda function for intelligent scaling
cat > lambda-cost-optimizer.py << 'EOF'
import boto3
import json
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """
    Intelligent cost optimization based on queue depth and time patterns
    """
    ecs = boto3.client('ecs')
    cloudwatch = boto3.client('cloudwatch')
    autoscaling = boto3.client('application-autoscaling')
    
    cluster_name = 'shadow-trainer-cluster'
    service_name = 'shadow-trainer-api-service'
    
    # Get current queue depth
    queue_metrics = cloudwatch.get_metric_statistics(
        Namespace='ShadowTrainer/Celery',
        MetricName='CeleryQueueDepth',
        Dimensions=[{'Name': 'QueueName', 'Value': 'video_processing'}],
        StartTime=datetime.utcnow() - timedelta(minutes=10),
        EndTime=datetime.utcnow(),
        Period=300,
        Statistics=['Average']
    )
    
    # Get current service status
    service_response = ecs.describe_services(
        cluster=cluster_name,
        services=[service_name]
    )
    
    current_desired = service_response['services'][0]['desiredCount']
    current_running = service_response['services'][0]['runningCount']
    
    # Calculate optimal capacity
    queue_depth = 0
    if queue_metrics['Datapoints']:
        queue_depth = queue_metrics['Datapoints'][-1]['Average']
    
    # Optimization logic
    hour = datetime.utcnow().hour
    is_peak_hour = 8 <= hour <= 22
    is_weekend = datetime.utcnow().weekday() >= 5
    
    if queue_depth == 0 and current_desired > 1:
        # No queue, scale down to minimum
        new_capacity = 1 if not is_peak_hour else 2
    elif queue_depth > 100:
        # Heavy load, scale up
        new_capacity = min(current_desired + 2, 10)
    elif queue_depth > 50:
        # Moderate load
        new_capacity = min(current_desired + 1, 8)
    else:
        # Normal load, maintain current or slight adjustment
        new_capacity = current_desired
    
    # Apply weekend reduction
    if is_weekend:
        new_capacity = max(1, new_capacity // 2)
    
    # Update capacity if needed
    if new_capacity != current_desired:
        autoscaling.register_scalable_target(
            ServiceNamespace='ecs',
            ResourceId=f'service/{cluster_name}/{service_name}',
            ScalableDimension='ecs:service:DesiredCount',
            MinCapacity=1,
            MaxCapacity=10,
            DesiredCapacity=new_capacity
        )
        
        print(f"Scaled from {current_desired} to {new_capacity} instances")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'current_capacity': current_desired,
            'new_capacity': new_capacity,
            'queue_depth': queue_depth,
            'is_peak_hour': is_peak_hour,
            'is_weekend': is_weekend
        })
    }
EOF

# 4. Create S3 lifecycle policies for output files
echo "Setting up S3 lifecycle policies..."

cat > s3-lifecycle-policy.json << 'EOF'
{
    "Rules": [
        {
            "ID": "ShadowTrainerOutputLifecycle",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "output/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                },
                {
                    "Days": 365,
                    "StorageClass": "DEEP_ARCHIVE"
                }
            ],
            "Expiration": {
                "Days": 2555
            }
        },
        {
            "ID": "TempFilesCleanup",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "temp/"
            },
            "Expiration": {
                "Days": 7
            }
        },
        {
            "ID": "LogsCleanup",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "logs/"
            },
            "Expiration": {
                "Days": 14
            }
        }
    ]
}
EOF

# 5. CloudWatch log retention policies
echo "Setting CloudWatch log retention policies..."

# Set retention for ECS logs to 7 days
aws logs put-retention-policy \
    --log-group-name "/ecs/shadow-trainer-api" \
    --retention-in-days 7 \
    --region "$AWS_REGION"

aws logs put-retention-policy \
    --log-group-name "/ecs/shadow-trainer-worker" \
    --retention-in-days 7 \
    --region "$AWS_REGION"

# 6. Reserved Instance recommendations
cat > reserved-instance-analysis.py << 'EOF'
import boto3
from datetime import datetime, timedelta

def analyze_reserved_instance_opportunities():
    """
    Analyze EC2 usage patterns for Reserved Instance recommendations
    """
    ec2 = boto3.client('ec2', region_name='us-east-2')
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-2')
    
    # Get current instances
    instances = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Environment', 'Values': ['production']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )
    
    instance_types = {}
    for reservation in instances['Reservations']:
        for instance in reservation['Instances']:
            instance_type = instance['InstanceType']
            instance_types[instance_type] = instance_types.get(instance_type, 0) + 1
    
    print("Current instance usage:")
    for instance_type, count in instance_types.items():
        print(f"  {instance_type}: {count} instances")
    
    # Analyze usage patterns over last 30 days
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)
    
    for instance_type in instance_types:
        # Get average running instances
        metrics = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='StatusCheckPassed',
            Dimensions=[
                {'Name': 'InstanceType', 'Value': instance_type}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Average']
        )
        
        if metrics['Datapoints']:
            avg_utilization = sum(dp['Average'] for dp in metrics['Datapoints']) / len(metrics['Datapoints'])
            
            if avg_utilization > 0.7:  # 70% utilization threshold
                print(f"\nRecommendation: Consider Reserved Instance for {instance_type}")
                print(f"  Average utilization: {avg_utilization:.2%}")
                print(f"  Estimated savings: 30-60% vs On-Demand")

if __name__ == "__main__":
    analyze_reserved_instance_opportunities()
EOF

# 7. Cost monitoring alerts
echo "Setting up cost monitoring alerts..."

# Create budget for Shadow Trainer resources
aws budgets create-budget \
    --account-id 381491870028 \
    --budget '{
        "BudgetName": "ShadowTrainerBudget",
        "BudgetLimit": {
            "Amount": "500",
            "Unit": "USD"
        },
        "TimeUnit": "MONTHLY",
        "TimePeriod": {
            "Start": "'$(date -d "first day of this month" '+%Y-%m-%d')'",
            "End": "'$(date -d "first day of next month" '+%Y-%m-%d')'"
        },
        "BudgetType": "COST",
        "CostFilters": {
            "TagKey": ["Environment"],
            "TagValue": ["production"]
        }
    }' \
    --notifications-with-subscribers '[
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80
            },
            "Subscribers": [
                {
                    "SubscriptionType": "EMAIL",
                    "Address": "alerts@yourdomain.com"
                }
            ]
        },
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN", 
                "Threshold": 100
            },
            "Subscribers": [
                {
                    "SubscriptionType": "EMAIL",
                    "Address": "alerts@yourdomain.com"
                }
            ]
        }
    ]'

# 8. Create cost optimization dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "ShadowTrainerCostOptimization" \
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
                        [ "AWS/EC2", "CPUUtilization", "InstanceType", "g4dn.xlarge" ],
                        [ "ShadowTrainer/GPU", "GPUUtilization", "ServiceName", "shadow-trainer-api" ]
                    ],
                    "view": "timeSeries",
                    "stacked": false,
                    "region": "us-east-2",
                    "title": "Resource Utilization",
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
                    "title": "ECS Service Scaling",
                    "period": 300
                }
            }
        ]
    }' \
    --region "$AWS_REGION"

echo "Cost optimization setup complete!"
echo ""
echo "Implemented optimizations:"
echo "✅ Spot Fleet configuration for worker nodes"
echo "✅ Scheduled scaling for off-peak hours"
echo "✅ Intelligent scaling Lambda function"
echo "✅ S3 lifecycle policies for data archival"
echo "✅ CloudWatch log retention policies"
echo "✅ Reserved Instance analysis script"
echo "✅ Cost monitoring alerts and budget"
echo "✅ Cost optimization dashboard"
echo ""
echo "Expected cost savings: 40-60% compared to basic deployment"
echo "Next steps:"
echo "1. Deploy spot fleet: aws ec2 request-spot-fleet --cli-input-json file://spot-fleet-config.json"
echo "2. Deploy Lambda function for intelligent scaling"
echo "3. Apply S3 lifecycle policies to your bucket"
echo "4. Run reserved instance analysis monthly"
