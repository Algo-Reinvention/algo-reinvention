#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/forget_test.sh \
#       ${QWEN3_4B_THINKING_2507_PATH} \
#       "qwen3-4b-thinking-2507" \
#       "\n\n\\boxed{"

TARGET_MODEL_PATH=$1
MODEL_NAME=$2
# ASSISTANT_PREFIX=$3
ASSISTANT_PREFIX=""
CATEGORY=""  # change by overriding
TEMPERATURE="1.0"
MAX_NEW_TOKENS="2048"
REPETITION_PENALTY="1.05"
SHUFFLE_OPTIONS="True"
SHUFFLE_SEED="20260211"
SKIP_VLLM="False"
# DEVICE="cpu"  # change by overriding

# Used for overriding
for argument in "$@"; do
  if [[ $argument == *"="* ]]; then
    echo "Overriding: $argument"
    export "$argument"
  fi
done

algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1
PROJECT_ROOT="${PROJECT_ROOT%/}"

algo_reinvention_require_env "CONDA_VLLM_NAME" || exit 1
conda activate "${CONDA_VLLM_NAME}"
if [[ $MODEL_NAME == *"ministral"* ]]; then
    algo_reinvention_require_env "CONDA_VERL_MINISTRAL_NAME" || exit 1
    conda activate "${CONDA_VERL_MINISTRAL_NAME}"
fi

if [[ $MODEL_NAME == *"base"* ]]; then
    OUTPUT_FILE_NAME="${CATEGORY}"
else
    OUTPUT_FILE_NAME="results"
fi

if [[ "$SHUFFLE_OPTIONS" == "False" ]]; then
    SHUFFLE_FLAG="--no-shuffle_options"
else
    SHUFFLE_FLAG="--shuffle_options"
fi

if [[ "$SKIP_VLLM" == "True" ]]; then
    SKIP_VLLM_FLAG="--skip_vllm"
else
    SKIP_VLLM_FLAG="--no-skip_vllm"
fi

cd $PROJECT_ROOT
mkdir -p ./_output/results/${MODEL_NAME}/forget_test
python simple_parallel/eval_logic/forget_test.py \
    --model_path ${TARGET_MODEL_PATH} \
    --data_path ./datasets/forget_test/${CATEGORY}/questions.json \
    --output_path ./_output/results/${MODEL_NAME}/forget_test/${OUTPUT_FILE_NAME}.json \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --shuffle_seed "$SHUFFLE_SEED" \
    $SKIP_VLLM_FLAG \
    $SHUFFLE_FLAG

    # --assistant_prefix "${ASSISTANT_PREFIX}" \
