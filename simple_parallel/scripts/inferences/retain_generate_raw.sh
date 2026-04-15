#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# NOTE@jzhao: for nvidia generation
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/retain_generate_raw.sh \
#       ${MINISTRAL3_14B_REASONING_2512_PATH} \
#       "ministral3-14b-reasoning-2512" \
#       "" \
#       RETAIN_NAME=nvidia \
#       SPLIT_NAME=math_split \
#       NUM=1 \
#       MAX_MODEL_LEN=4096 \
#       SYSTEM_PROMPT="" \
#       USER_PROMPT_TEMPLATE="{question}"

# NOTE@jzhao: for idk generation
# export CUDA_VISIBLE_DEVICES=0,1
# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/retain_generate_raw.sh \
#       "${QWEN3_4B_THINKING_2507_PATH}" \
#       "qwen3-4b-thinking-2507" \
#       "" \
#       RETAIN_NAME=idk/graph-sp-dijkstra \
#       SPLIT_NAME=indist_split \
#       NUM=32 \
#       MAX_MODEL_LEN=4096

# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/retain_generate_raw.sh \
#       "/path/to/model" \
#       "qwen3-4b/think" \
#       "<think>\n"

TARGET_MODEL_PATH=$1
MODEL_NAME=$2
ASSISTANT_PREFIX=$3
RETAIN_NAME=retain/dijkstra
SPLIT_NAME=idk_split
NUM=16
TEMPERATURE=1.0
TOP_K=40
SYSTEM_PROMPT=""
USER_PROMPT_TEMPLATE="{question}"

export SKIP_VLLM=False

# VLLM TMUX
# Normally need to change SESSION_NAME, MODEL_PATH, GPU_INDEX_START when running exps in batch

# --- Configuration ---
export SESSION_NAME="vllm-servers-raw-gpu${CUDA_VISIBLE_DEVICES}"
export MODEL_PATH=$TARGET_MODEL_PATH
export GPU_MEMORY_UTIL=0.95
export MAX_NUM_SEQS=1024
export MAX_MODEL_LEN=32768
algo_reinvention_require_env "CONDA_VLLM_NAME" || exit 1
export CONDA_ENV="${CONDA_VLLM_NAME}"
if [[ "${MODEL_NAME}" == *"ministral3"* ]]; then
    algo_reinvention_require_env "CONDA_VERL_MINISTRAL_NAME" || exit 1
    export CONDA_ENV="${CONDA_VERL_MINISTRAL_NAME}"
fi
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

export TENSOR_PARALLEL_SIZE=$GPU_NUM

if [ "$SKIP_VLLM" = "True" ]; then
    echo ">>> skip vllm >>>"
elif [ "$SKIP_VLLM" = "False" ]; then
    bash $PROJECT_ROOT/simple_parallel/scripts/vllm_launch.sh
    echo ">>> waiting server launching >>>"
    bash $WORKSPACE_ROOT/_tools/inference/sleep_until_tmux_print.sh $SESSION_NAME "Application startup complete"
    echo ">>> starting inference >>>"
fi

# INFER TMUX
# Normally need to change SESSION_NAME, SERVER_INDEX_START, COMMON_ARGS(model_path(might), experiment_name) when running in batch
# Normally need to change COMMON_ARGS(input_path, experiment_name, user_prompt_template, assistant_prefix, process_module) when running different tasks

export NUM_SERVERS=$(( GPU_NUM / TENSOR_PARALLEL_SIZE ))
export SERVER_INDEX_START=$GPU_INDEX_START

# --- Configuration for retain ---
export SESSION_NAME="vllm-clients-retain-gpu${CUDA_VISIBLE_DEVICES}"
export TOTAL_PROCESSES=64
export COMMON_ARGS="--num ${NUM} \
--temperature ${TEMPERATURE} \
--top_k ${TOP_K} \
--project_root $PROJECT_ROOT \
--input_path ../_data/post_train/${RETAIN_NAME}/${SPLIT_NAME} \
--output_path ../_output/results \
--max_tokens $((MAX_MODEL_LEN - 1000)) \
--model_path ${MODEL_PATH} \
--experiment_name ${MODEL_NAME}/post_train/${RETAIN_NAME}/${SPLIT_NAME} \
--system_prompt '${SYSTEM_PROMPT}' \
--user_prompt_template '${USER_PROMPT_TEMPLATE}' \
--assistant_prefix '${ASSISTANT_PREFIX}' \
--process_module eval_logic.multi_chain"
bash $PROJECT_ROOT/simple_parallel/scripts/inference_launch.sh


echo ">>> waiting retain inference >>>"
sleep 180
bash $WORKSPACE_ROOT/_tools/inference/sleep_until_gpu_free.sh
