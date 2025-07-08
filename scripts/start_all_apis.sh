#!/bin/bash

SESSION="shadow_trainer"

# Start a new tmux session (detached)
tmux new-session -d -s $SESSION

# Start the backend API in window 1
tmux new-window -t $SESSION:1 -n 'API_Backend'
tmux send-keys -t $SESSION:1 'uv run python api_backend/run_apy.py' C-m

# Start the Streamlit frontend in window 2
tmux new-window -t $SESSION:2 -n 'Streamlit_Frontend'
tmux send-keys -t $SESSION:2 'uv run streamlit run api_frontend/adi_streamlit_app.py' C-m

# Attach to the tmux session
tmux attach-session -t $SESSION