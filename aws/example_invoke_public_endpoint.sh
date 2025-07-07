#!/bin/sh
# Example script to send a request to the public SageMaker endpoint for Shadow Trainer

# Replace with your actual endpoint name and region if different
ENDPOINT_NAME="shadow-trainer"
REGION="us-east-2"

# Example input (S3 path or local path, depending on your model handler)
INPUT_VIDEO="s3://shadow-trainer-prod/sample_input/pitch_mini2.mp4"
MODEL_SIZE="s"

# SageMaker endpoints expect JSON input via the /invocations endpoint
# The request body should contain the parameters as JSON
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name "$ENDPOINT_NAME" \
    --region "$REGION" \
    --content-type "application/json" \
    --body "{\"file\": \"$INPUT_VIDEO\", \"model_size\": \"$MODEL_SIZE\"}" \
    output.json

echo "Response saved to output.json"

# Display the response
echo "Response:"
cat output.json | jq .
