import boto3

# Set correct AWS region and account info for this repo
region = "us-east-2"
account_id = "381491870028"
repo_name = "shadow-trainer"
model_name = "shadow-trainer"
ecr_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"
role_arn = f"arn:aws:iam::{account_id}:role/service-role/SageMaker-mlops"

sagemaker_client = boto3.client("sagemaker", region_name=region)

# 1. Create a model
sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": ecr_image_uri
    },
    ExecutionRoleArn=role_arn
)

# 2. Create an endpoint config with a GPU instance
endpoint_config_name = model_name + "-gpu-config"

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": "ml.g4dn.xlarge",  # GPU instance
            "VolumeSizeInGB": 30,
            "InitialInstanceCount": 1,
            "MaxInstanceCount": 2,
            "ExecutionRoleArn": role_arn,
        }
    ]
)

# 3. Create the endpoint
endpoint_name = model_name + "-gpu-endpoint"

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"Created SageMaker GPU endpoint: {endpoint_name}")

# List all SageMaker endpoints and print their names and statuses
response = sagemaker_client.list_endpoints()
print("Existing SageMaker endpoints:")
for ep in response["Endpoints"]:
    print(f"{ep['EndpointName']}: {ep['EndpointStatus']}")