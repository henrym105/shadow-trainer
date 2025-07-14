#!/bin/bash

SESSION="api"

# Start a new tmux session (detached) with window 1
tmux new-session -d -s $SESSION -n 'backend'

# In window 1, split the pane vertically
tmux split-window -h -t $SESSION:0

# Run backend in the left pane (pane 0)
# tmux send-keys -t $SESSION:0.0 'uv run python api_backend/run_api.py' C-m
tmux send-keys -t $SESSION:0.0 'cd api_backend && uv run python run_api.py' C-m

# Run frontend in the right pane (pane 1)
tmux send-keys -t $SESSION:0.1 'cd api_frontend/shadow_trainer_web && npm run build && serve -s build -l 8000' C-m
# tmux send-keys -t $SESSION:0.1 'cd api_frontend/shadow_trainer_web && PORT=8000 npm start' C-m

# Attach to the tmux session
tmux attach-session -t $SESSION
