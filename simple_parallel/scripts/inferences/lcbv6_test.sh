#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# export CUDA_VISIBLE_DEVICES=0,1
# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/lcbv6_test.sh \
#       "${QWEN3_4B_THINKING_2507_PATH}" \
#       "2025v6/qwen3-4b-thinking-2507/base" \
#       "Qwen3Think"

TARGET_MODEL_PATH=$1
EXP_NAME=$2
MODEL_NAME=$3
TEMPERATURE=1.0  # change by overriding
NUM=8            # change by overriding
MAX_TOKENS=""

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

algo_reinvention_require_env "CONDA_VLLM_NAME" || exit 1
conda activate "${CONDA_VLLM_NAME}"
MODEL_NAME_LOWER=$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')
if [[ "${MODEL_NAME_LOWER}" == *"ministral"* ]]; then
    algo_reinvention_require_env "CONDA_VERL_MINISTRAL_NAME" || exit 1
    conda activate "${CONDA_VERL_MINISTRAL_NAME}"
fi

if [[ -z "${MAX_TOKENS}" ]]; then
    MAX_TOKENS=65536
    if [[ "${MODEL_NAME_LOWER}" == *"ministral"* && "${MODEL_NAME_LOWER}" == *"instruct"* ]]; then
        MAX_TOKENS=$((8192 * 2))
        echo "Using reduced MAX_TOKENS=${MAX_TOKENS} for ${MODEL_NAME}"
    fi
fi

cd $WORKSPACE_ROOT/LiveCodeBench
if [[ -n "${PROXY:-}" ]]; then
    export http_proxy=${PROXY} && export https_proxy=${PROXY} && export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
fi

# It will use all of the GPU by default, approximately 60+ minutes for qwen3-think on 1 GPU for generating
python -m lcb_runner.runner.main \
    --model $MODEL_NAME \
    --local_model_path $TARGET_MODEL_PATH \
    --scenario codegeneration \
    --release_version release_v6 \
    --start_date 2025-02-01 \
    --end_date 2025-06-01 \
    --n $NUM \
    --max_tokens $MAX_TOKENS \
    --enable_prefix_caching \
    --cache_batch_size 200 \
    --temperature $TEMPERATURE \
    --stop "<|im_end|>" \
    --num_process_evaluate 48 \
    --exp_name $EXP_NAME \
    --evaluate
