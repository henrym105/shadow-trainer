
./scripts/install_setup_uv.sh

sudo yum update -y

# Opencv utils
sudo yum install -y mesa-libGL mesa-libGL-devel libXext libSM libXrender

source .venv/bin/activate