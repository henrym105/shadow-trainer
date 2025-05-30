#!/bin/bash

# This script checks if Homebrew is installed, installs it if necessary,
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v brew &> /dev/null; then
        echo "Homebrew is not installed. Please install it before running this script."
        exit 1
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! command -v brew &> /dev/null; then
        echo "Homebrew is not installed. Please install it before running this script."
        exit 1
    fi
else
    echo "Unsupported operating system: $OSTYPE. This script requires macOS or Linux with Homebrew."
    exit 1
fi

# Check if uv is installed, install it if not
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing..."
    brew install uv
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