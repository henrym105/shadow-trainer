#!/bin/bash

# Check if uv is installed, install it if not
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing via official script..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed."
fi

# Check if the virtual environment exists, create it if not
uv sync
if [ $? -eq 0 ]; then
    echo "uv sync completed successfully."
else
    echo "uv sync failed. Please check the output for errors."
fi

source .venv/bin/activate