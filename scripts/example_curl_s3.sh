#!/bin/sh
# Example script to send an S3 video path to the FastAPI service from inside Docker

# Option to exit immediately if a command exits with a non-zero status:
set -e

# Wait for the API to be up (max 30s)
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/docs > /dev/null; then
    break
  fi
  echo "Waiting for API to be ready... ($i)" >&2
  sleep 1
done

# # Example cURL command to send a video file from S3 to the FastAPI service
# curl -X POST "http://localhost:8000/process_video/" \
#     -H "accept: application/json" \
#     --get \
#     --data-urlencode "file=s3://shadow-trainer-prod/sample_input/pitch_mini2.mp4" \
#     --data-urlencode "model_size=b"


# curl -X POST "http://localhost:8000/process_video/" \
#     -H "accept: application/json" \
#     --get \
#     --data-urlencode "file=s3://shadow-trainer-prod/sample_input/pitch_mini.mp4" \
#     --data-urlencode "model_size=b"


# curl -X POST "http://localhost:8000/process_video/" \
#     -H "accept: application/json" \
#     --get \
#     --data-urlencode "file=s3://shadow-trainer-prod/sample_input/henry-mini.mov" \
#     --data-urlencode "model_size=xs"


# curl -X POST "http://localhost:8000/process_video/" \
#     -H "accept: application/json" \
#     --get \
#     --data-urlencode "file=s3://shadow-trainer-prod/sample_input/henry1.MOV" \
#     --data-urlencode "model_size=s"


curl -X POST "http://localhost:8000/process_video/" \
    -H "accept: application/json" \
    --get \
    --data-urlencode "file=s3://shadow-trainer-prod/sample_input/henry1_full.mov" \
    --data-urlencode "model_size=xs"


# curl -X POST "http://localhost:8000/process_video/" \
#     -H "accept: application/json" \
#     --get \
#     --data-urlencode "file=s3://shadow-trainer-prod/sample_input/BeiberS1.mp4" \
#     --data-urlencode "model_size=b"

