# Remove old image
docker rmi shadow-trainer:small

# Login to ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 381491870028.dkr.ecr.us-east-2.amazonaws.com

# Build the Docker image locally
docker build -t shadow-trainer:small .

# Create the ECR repository if it doesn't exist
aws ecr create-repository --repository-name shadow-trainer

# Tag the Docker image with the ECR repository URI
docker tag shadow-trainer:small 381491870028.dkr.ecr.us-east-2.amazonaws.com/shadow-trainer:small

# Push the Docker image to ECR
docker push 381491870028.dkr.ecr.us-east-2.amazonaws.com/shadow-trainer:small