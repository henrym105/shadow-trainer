import boto3
import botocore

# Set correct AWS region and account info for this repo
region = "us-east-2"
account_id = "381491870028"
repo_name = "shadow-trainer"
model_name = "shadow-trainer"
ecr_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"
role_arn = f"arn:aws:iam::{account_id}:role/service-role/SageMaker-mlops"

sagemaker_client = boto3.client("sagemaker", region_name=region)

# 1. Delete the model and endpoint config if they exist

# Delete endpoint config if it exists
endpoint_config_name = model_name + "-config"
try:
    sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    print(f"Deleted existing endpoint config: {endpoint_config_name}")
except botocore.exceptions.ClientError as e:
    if "Could not find endpoint configuration" in str(e):
        pass  # Endpoint config does not exist
    else:
        raise

# Delete model if it exists
try:
    sagemaker_client.delete_model(ModelName=model_name)
    print(f"Deleted existing model: {model_name}")
except botocore.exceptions.ClientError as e:
    if "Could not find model" in str(e):
        pass  # Model does not exist
    else:
        raise

# Delete endpoint if it exists
endpoint_name = model_name  # Use just "shadow-trainer" as endpoint name
try:
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"Deleted existing endpoint: {endpoint_name}")
except botocore.exceptions.ClientError as e:
    if "Could not find endpoint" in str(e):
        pass  # Endpoint does not exist
    else:
        raise


# 2. Create the model
print(f"Creating model: {model_name} with image {ecr_image_uri}")
sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": ecr_image_uri
    },
    ExecutionRoleArn=role_arn
)

# 3. Create a serverless endpoint config
print(f"Creating endpoint config: {endpoint_config_name} for model {model_name}")
sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,  # Options: 1024 or 2048
                "MaxConcurrency": 5
            }
        }
    ]
)

# 3. Create the endpoint
print(f"Creating endpoint: {endpoint_name} with config {endpoint_config_name}")
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"Created SageMaker serverless endpoint: {endpoint_name}")

# List all SageMaker endpoints and print their names and statuses
response = sagemaker_client.list_endpoints()
print("Existing SageMaker endpoints:")
for ep in response["Endpoints"]:
    print(f"{ep['EndpointName']}: {ep['EndpointStatus']}")
