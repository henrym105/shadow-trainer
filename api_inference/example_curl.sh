#!/bin/sh
# Example script to send a video to the FastAPI service

# curl -X POST "http://localhost:8000/process_video/" \
#   -H "accept: application/json" \
#   -F "file=@demo/video/sample_video.mp4" \
#   -F "model_size=xs"


# curl -X POST "http://localhost:8000/process_video/" \
#   -H "accept: application/json" \
#   -F "file=@demo/video/pitch_sample_4.mp4" \
#   -F "model_size=xs"

# curl -X POST "http://localhost:8000/process_video/" \
#   -H "accept: application/json" \
#   -F "file=@demo/video/IMG_4653.MOV" \
#   -F "model_size=xs"

curl -X POST "http://localhost:8000/process_video/" \
  -H "accept: application/json" \
  -F "file=@videos/IMG_4654.MOV" \
  -F "model_size=xs"

# api update: create new file that creates quick response in api to avoid keeping open http request
# and have client pull for status of the job to show in ui
# or at least might need to set high time out for the request
