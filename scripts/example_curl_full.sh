#!/bin/sh
# Example script to send an S3 video path to the FastAPI service from inside Docker

# Option to exit immediately if a command exits with a non-zero status:
set -e

# Wait for the API to be up (max 30s)
for i in $(seq 1 30); do
  if curl -s http://localhost:8002/docs > /dev/null; then
    break
  fi
  echo "Waiting for API to be ready... ($i)" >&2
  sleep 1
done

# # Example cURL command to send a video file from S3 to the FastAPI service
# curl -X POST "http://localhost:8002/video/process" \
#     -H "accept: application/json" \
#     -H "Content-Type: application/json" \
#     -d '{"file": "s3://shadow-trainer-prod/sample_input/BeiberS1.mp4", "model_size": "xs"}'


# curl -X POST "http://localhost:8002/video/process" \
#     -H "accept: application/json" \
#     -H "Content-Type: application/json" \
#     -d '{"file": "s3://shadow-trainer-prod/sample_input/cal-pitcher.mov", "model_size": "s"}'


# curl -X POST "http://localhost:8002/video/process" \
#     -H "accept: application/json" \
#     -H "Content-Type: application/json" \
#     -d '{"file": "s3://shadow-trainer-prod/sample_input/adi.mov", "model_size": "s"}'


curl -X POST "http://localhost:8002/video/process" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"file": "s3://shadow-trainer-prod/sample_input/henry1_uncropped.mov", "model_size": "s"}'