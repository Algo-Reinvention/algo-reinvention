#!/bin/bash

# =================================================================
# TMUX Script to Launch sglang Servers on Multiple GPUs
#
# Configuration variables MUST be set as environment variables
# before running this script, e.g.:
# export SESSION_NAME="sglang-servers"
# export MODEL_PATH="/root/_hf_models/qwen25-unlearn-sft"
# export MODEL_NAME="model"
# export REPO_DIR="/path/to/algo_test"
# export GPU_MEMORY_UTIL=0.95
# export MAX_MODEL_LEN=8192
# export MAX_NUM_SEQS=1024
# export TENSOR_PARALLEL_SIZE=1
# export CONDA_ENV="sglang"
# export GPU_NUM=4
# export GPU_INDEX_START=4 # Start deployment from GPU 4
#
# Optional Configuration:
# export BASE_PORT=8000
# =================================================================

# --- Configuration Defaults (Internal Setup) ---
BASE_PORT=${BASE_PORT:-8000}

export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)

if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="model"
fi

echo "============================================================================================="

# --- Check Required Environment Variables ---
REQUIRED_VARS=("SESSION_NAME" "MODEL_PATH" "REPO_DIR" "GPU_MEMORY_UTIL" "MAX_MODEL_LEN" "CONDA_ENV" "GPU_NUM" "TENSOR_PARALLEL_SIZE" "GPU_INDEX_START" "MAX_NUM_SEQS")
MISSING_VARS=""
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "$(eval echo \$"$var")" ]; then
        MISSING_VARS+="$var "
    fi
done

if [ -n "$MISSING_VARS" ]; then
    echo "ERROR: The following required environment variables are not set: $MISSING_VARS"
    exit 1
fi

# --- Logic Validation: Check if GPU_NUM is divisible by TENSOR_PARALLEL_SIZE ---
if (( GPU_NUM % TENSOR_PARALLEL_SIZE != 0 )); then
    echo "ERROR: GPU_NUM ($GPU_NUM) must be divisible by TENSOR_PARALLEL_SIZE ($TENSOR_PARALLEL_SIZE)."
    echo "Current setup would leave some GPUs idle or cause unbalanced deployment."
    exit 1
fi

# --- Safety & Auto-Cleanup Check ---
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "WARNING: A tmux session named '$SESSION_NAME' already exists. Killing and restarting..."
    tmux kill-session -t $SESSION_NAME
    sleep 1
fi

# --- Create a new detached tmux session ---
FIRST_GPU_INDEX=$(($GPU_INDEX_START)) 
echo "Starting new tmux session '$SESSION_NAME'..."
# Create the session with an initial placeholder window
tmux new-session -d -s $SESSION_NAME -n "init"

# --- Main loop to create windows and run servers ---
# We increment by TENSOR_PARALLEL_SIZE each time
for (( i=0; i<$GPU_NUM; i+=$TENSOR_PARALLEL_SIZE ))
do
    # 1. Collect GPU IDs for this server instance
    GPU_IDS=()
    for (( j=0; j<$TENSOR_PARALLEL_SIZE; j++ ))
    do
        GPU_IDS+=($(($GPU_INDEX_START + $i + $j)))
    done
    
    # 2. Convert array to comma-separated string (e.g., "4,5")
    CUDA_DEVICES_STR=$(IFS=,; echo "${GPU_IDS[*]}")
    
    # 3. Use the first GPU of the group for Port and naming
    PRIMARY_GPU_ID=${GPU_IDS[0]}
    LAST_GPU_ID=${GPU_IDS[-1]}
    PORT=$(($BASE_PORT + $PRIMARY_GPU_ID))
    
    if [ $TENSOR_PARALLEL_SIZE -eq 1 ]; then
        WINDOW_NAME="gpu${PRIMARY_GPU_ID}-${PORT}"
    else
        WINDOW_NAME="gpu${PRIMARY_GPU_ID}to${LAST_GPU_ID}-tp${TENSOR_PARALLEL_SIZE}"
    fi

    echo "Configuring sglang server on GPUs [${CUDA_DEVICES_STR}] at port ${PORT}..."

    # 4. Create or reuse tmux window
    if [ $i -eq 0 ]; then
        TARGET_WINDOW="$SESSION_NAME:init"
        tmux rename-window -t $TARGET_WINDOW "$WINDOW_NAME"
    else
        tmux new-window -t $SESSION_NAME -n "$WINDOW_NAME"
    fi
    TARGET_WINDOW="$SESSION_NAME:$WINDOW_NAME"

    # 5. Send commands
    tmux send-keys -t $TARGET_WINDOW "conda activate $CONDA_ENV" C-m
    tmux send-keys -t $TARGET_WINDOW "cd $REPO_DIR/simple_parallel" C-m
    tmux send-keys -t $TARGET_WINDOW \
        "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES_STR} python start_sglang_server.py \
            --model_path ${MODEL_PATH} \
            --model_name ${MODEL_NAME} \
            --max_model_len ${MAX_MODEL_LEN} \
            --gpu_memory_utilization ${GPU_MEMORY_UTIL} \
            --max_num_seqs ${MAX_NUM_SEQS} \
            --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
            --port ${PORT}" C-m
done

# --- Final Instructions ---
SERVER_COUNT=$(( GPU_NUM / TENSOR_PARALLEL_SIZE ))
echo ""
echo "Successfully launched ${SERVER_COUNT} sglang server(s) in session '$SESSION_NAME'."
echo "Each server uses ${TENSOR_PARALLEL_SIZE} GPU(s)."
echo "Total GPUs used: ${GPU_INDEX_START} to $((GPU_INDEX_START + GPU_NUM - 1))"
echo "To attach: tmux attach-session -t $SESSION_NAME"