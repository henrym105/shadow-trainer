#!/bin/bash

if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing..."
    brew install uv
else
    echo "uv is already installed."
fi


uv sync
if [ $? -eq 0 ]; then
    echo "uv sync completed successfully."
else
    echo "uv sync failed. Please check the output for errors."
fi

source .venv/bin/activate