# this runs the docker container with the necessary AWS credentials mounted
# and exposes port 8000 for the API

docker run --rm -it -p 8000:8000 \
  -v ~/.aws:/root/.aws \
  shadow-trainer