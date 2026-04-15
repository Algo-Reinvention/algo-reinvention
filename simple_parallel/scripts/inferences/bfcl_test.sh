#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# export SERPAPI_API_KEY=""
# export OPENAI_API_KEY=""
# apt install libnuma-dev
# export CUDA_VISIBLE_DEVICES=0
# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/bfcl_test.sh \
#       "${QWEN3_4B_THINKING_2507_PATH}" \
#       "qwen3-4b-thinking-2507/base" \
#       "qwen3-4b-think-FC" \
#       "all" \
#       SKIP_SGLANG=False \
#       GPU_NUM=1
#
# # For Ministral (server must run in ${CONDA_VERL_MINISTRAL_NAME:-verl_ministral} via vLLM)
# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/bfcl_test.sh \
#       "${MINISTRAL3_14B_REASONING_2512_PATH}" \
#       "ministral3-14b-reasoning-2512/base" \
#       "mistralai/Ministral-3-14B-Reasoning-2512" \
#       "all" \
#       SKIP_VLLM=False \
#       GPU_NUM=1

# # Aggregation
# MODEL_NAME=qwen3-4b-thinking-2507/base
# conda activate ${CONDA_BFCL_NAME:-bfcl}
# export BFCL_PROJECT_ROOT=/path/to/bfcl/${MODEL_NAME}
# bfcl evaluate \
#       --model qwen3-4b-think-FC

ulimit -n 65535
TARGET_MODEL_PATH=$1
MODEL_NAME=$2
BFCL_MODEL=$3
TEST_CATEGORY=$4
GPU_NUM=$(nvidia-smi -L | wc -l)
export SKIP_SGLANG=False
export SKIP_VLLM=False
export ALLOW_OVERWRITE=False

# SERVER TMUX
# Normally need to change SESSION_NAME, MODEL_PATH, GPU_INDEX_START when running exps in batch

# --- Configuration ---
export MODEL_PATH=$TARGET_MODEL_PATH
export GPU_MEMORY_UTIL=0.9
export MAX_NUM_SEQS=1024
export MAX_MODEL_LEN=65536
export GPU_NUM=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export GPU_INDEX_START=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)

# Used for overriding
for argument in "$@"; do
  if [[ $argument == *"="* ]]; then
    echo "Overriding: $argument"
    export "$argument"
  fi
done

algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1
PROJECT_ROOT="${PROJECT_ROOT%/}"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
export REPO_DIR="${PROJECT_ROOT}"
export BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-${WORKSPACE_ROOT}/bfcl/${MODEL_NAME}}"
mkdir -p "$BFCL_PROJECT_ROOT"

# Auto-detect backend: Ministral -> vLLM (${CONDA_VERL_MINISTRAL_NAME:-verl_ministral}), others -> sglang
TARGET_MODEL_PATH_LOWER=$(echo "$TARGET_MODEL_PATH" | tr '[:upper:]' '[:lower:]')
BFCL_MODEL_LOWER=$(echo "$BFCL_MODEL" | tr '[:upper:]' '[:lower:]')
export IS_MINISTRAL=False
if [[ "$TARGET_MODEL_PATH_LOWER" == *"ministral"* || "$BFCL_MODEL_LOWER" == *"ministral"* ]]; then
    export IS_MINISTRAL=True
fi

if [ "$IS_MINISTRAL" = "True" ]; then
    export SERVER_BACKEND="vllm"
    export BFCL_BACKEND="vllm"
    algo_reinvention_require_env "CONDA_VERL_MINISTRAL_NAME" || exit 1
    export CONDA_ENV="${CONDA_VERL_MINISTRAL_NAME}"
    export SESSION_NAME="vllm-servers-bfcl-gpu${CUDA_VISIBLE_DEVICES}"
    export LAUNCH_SCRIPT="$PROJECT_ROOT/simple_parallel/scripts/vllm_launch.sh"
    export READY_MSG="Application startup complete"
    export MODEL_NAME=$TARGET_MODEL_PATH
    export SKIP_SERVER=$SKIP_VLLM
    # Backward-compat for old caller args that still pass SKIP_SGLANG.
    if [ "$SKIP_SGLANG" = "True" ] && [ "$SKIP_VLLM" = "False" ]; then
        export SKIP_SERVER=True
    fi
    # Reuse the externally mounted vLLM server for Ministral.
    export BFCL_SKIP_SERVER_SETUP=True
    export VLLM_ENDPOINT=localhost
    export VLLM_PORT=$((8000 + GPU_INDEX_START))
    # @codex: External vLLM launcher serves a fixed model id "model".
    export OSS_REQUEST_MODEL_NAME=model
else
    export SERVER_BACKEND="sglang"
    export BFCL_BACKEND="sglang"
    algo_reinvention_require_env "CONDA_BFCL_NAME" || exit 1
    export CONDA_ENV="${CONDA_BFCL_NAME}"
    export SESSION_NAME="sglang-servers-bfcl-gpu${CUDA_VISIBLE_DEVICES}"
    export LAUNCH_SCRIPT="$PROJECT_ROOT/simple_parallel/scripts/sglang_launch.sh"
    export READY_MSG="The server is fired up and ready to roll!"
    export MODEL_NAME=$BFCL_MODEL
    export SKIP_SERVER=$SKIP_SGLANG
    export BFCL_SKIP_SERVER_SETUP=False
    unset OSS_REQUEST_MODEL_NAME
fi

# @codex: Model-specific default temperature.
# Ministral is more stable for BFCL multi-turn with lower temperature.
if [ -z "${BFCL_TEMPERATURE:-}" ]; then
    if [ "$IS_MINISTRAL" = "True" ]; then
        export BFCL_TEMPERATURE=0.2
    else
        export BFCL_TEMPERATURE=0.6
    fi
fi

echo ">>> bfcl backend: ${BFCL_BACKEND}, server backend: ${SERVER_BACKEND}, server conda env: ${CONDA_ENV}"
echo ">>> bfcl temperature: ${BFCL_TEMPERATURE}"

export TENSOR_PARALLEL_SIZE=$GPU_NUM

if [ "$SKIP_SERVER" = "True" ]; then
    echo ">>> skip ${SERVER_BACKEND} >>>"
elif [ "$SKIP_SERVER" = "False" ]; then
    bash $LAUNCH_SCRIPT
    echo ">>> waiting ${SERVER_BACKEND} server launching >>>"
    bash $WORKSPACE_ROOT/_tools/inference/sleep_until_tmux_print.sh $SESSION_NAME "$READY_MSG"
    echo ">>> starting inference >>>"
fi


# LLM generate
if [[ -n "${PROXY:-}" ]]; then
    export http_proxy="${PROXY}"
    export https_proxy="${PROXY}"
    export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
fi
algo_reinvention_require_env "CONDA_BFCL_NAME" || exit 1
conda activate "${CONDA_BFCL_NAME}"

cat > "${BFCL_PROJECT_ROOT}/.env" << EOF
SERPAPI_API_KEY=$SERPAPI_API_KEY
EOF

BFCL_GENERATE_ARGS=(
    --model "$BFCL_MODEL"
    --num-threads 2000
    --backend "$BFCL_BACKEND"
    --test-category "$TEST_CATEGORY"
    --temperature "$BFCL_TEMPERATURE"
    --local-model-path "$TARGET_MODEL_PATH"
    --num-gpus "$GPU_NUM"
)

if [ "$BFCL_SKIP_SERVER_SETUP" = "True" ]; then
    BFCL_GENERATE_ARGS+=(--skip-server-setup)
fi

if [ "$ALLOW_OVERWRITE" = "True" ]; then
    # @codex: Useful for iterative debugging/regeneration after handler changes.
    BFCL_GENERATE_ARGS+=(--allow-overwrite)
fi

bfcl generate "${BFCL_GENERATE_ARGS[@]}"
