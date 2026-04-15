#!/bin/bash

# =================================================================
# TMUX Script to Launch Parallel vLLM Clients
#
# Configuration variables MUST be set as environment variables
# before running this script, e.g.:
# export SESSION_NAME="vllm-clients-1"
# export TOTAL_PROCESSES=32
# export NUM_SERVERS=4
# export SERVER_INDEX_START=4 
# export REPO_DIR="/path/to/algo_test"
# export CONDA_ENV="vllm"
# export COMMON_ARGS="--num 3 --input_path ../_data/Benchmark_20251112/math500_split ..."
#
# Optional Configuration:
# export BASE_PORT=8000
# export SERVER_HOST="localhost"
# =================================================================

# --- Configuration Defaults (Internal Setup) ---
# If BASE_PORT or SERVER_HOST are not set as env vars, use defaults
BASE_PORT=${BASE_PORT:-8000}
SERVER_HOST=${SERVER_HOST:-"localhost"}

echo "============================================================================================="

# --- Check Required Environment Variables ---
REQUIRED_VARS=("SESSION_NAME" "TOTAL_PROCESSES" "NUM_SERVERS" "SERVER_INDEX_START" "REPO_DIR" "CONDA_ENV" "COMMON_ARGS")
MISSING_VARS=""
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "$(eval echo \$"$var")" ]; then
        MISSING_VARS+="$var "
    fi
done

if [ -n "$MISSING_VARS" ]; then
    echo "ERROR: The following required environment variables are not set: $MISSING_VARS"
    echo "Please set them before running the script."
    exit 1
fi

# --- Safety & Auto-Cleanup Check ---
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "WARNING: A tmux session named '$SESSION_NAME' already exists. Killing the old session and restarting..."
    tmux kill-session -t $SESSION_NAME
    # Wait a moment for the system to clean up the session
    sleep 1
fi

# --- Create a new detached tmux session ---
echo "Starting tmux session '$SESSION_NAME'..."
# Use index 0 for the first server's window
tmux new-session -d -s $SESSION_NAME -n "server-$(($SERVER_INDEX_START))" 

# --- Main logic to distribute processes ---
# Calculate how many client processes each server/window will handle
PROCESSES_PER_SERVER=$((TOTAL_PROCESSES / NUM_SERVERS))

echo "Distributing $TOTAL_PROCESSES clients across $NUM_SERVERS servers ($PROCESSES_PER_SERVER clients per server)."
echo "Server index range: $SERVER_INDEX_START to $((SERVER_INDEX_START + NUM_SERVERS - 1))."
echo ""

# Loop through each server/window
for i in $(seq 0 $(($NUM_SERVERS - 1)))
do
    # Calculate the actual index (e.g., if START=4, i=0, ACTUAL_INDEX=4)
    ACTUAL_SERVER_INDEX=$(($SERVER_INDEX_START + i))

    # Calculate the port based on the actual index (e.g., 8000 + 4 = 8004)
    PORT=$(($BASE_PORT + $ACTUAL_SERVER_INDEX)) 
    
    WINDOW_NAME="server-${ACTUAL_SERVER_INDEX}-${PORT}" 
    SERVER_URL="http://${SERVER_HOST}:${PORT}" # Use the determined host and port

    echo "Configuring window '$WINDOW_NAME' to run $PROCESSES_PER_SERVER clients targeting $SERVER_URL..."

    if [ $i -eq 0 ]; then
        # For the first window (i=0), reuse the initial window created with the session.
        TARGET_PANE="$SESSION_NAME:0"
        tmux rename-window -t $TARGET_PANE $WINDOW_NAME
    else
        # For later windows, create a new tmux window.
        tmux new-window -t $SESSION_NAME -n $WINDOW_NAME
        TARGET_PANE="$SESSION_NAME:$WINDOW_NAME" # Using the new window name is more reliable.
    fi
    
    # Send commands to the tmux window
    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "conda activate $CONDA_ENV" C-m
    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "cd $REPO_DIR/simple_parallel" C-m

    # Send the whole loop as a single command string
    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME \
        "for j in \$(seq 1 $PROCESSES_PER_SERVER); do \
            python main.py --server_url ${SERVER_URL} ${COMMON_ARGS} & \
        done" C-m
done

# --- Final Instructions ---
echo ""
echo "Successfully launched $TOTAL_PROCESSES vLLM clients in the tmux session '$SESSION_NAME'."
echo "To attach to the session and monitor the processes, run:"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "To kill all client processes at once, run:"
echo "  tmux kill-session -t $SESSION_NAME"
