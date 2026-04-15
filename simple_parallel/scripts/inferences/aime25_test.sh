#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# NOTE:qwen
# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/aime25_test.sh \
#       ${PROJECT_ROOT}/unlearn/saves/1224_llm_as_a_judge_idk/seed1/global_step_64/actor/huggingface \
#       "qwen3-4b-thinking-2507-unlearn/1224_llm_as_a_judge_idk" \
#       "" \
#       NUM=3 \
#       MAX_MODEL_LEN=65536

# NOTE:gpt
# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/aime25_test.sh \
#       /path/to/model \
#       "gpt-oss-20b" \
#       "" \
#       NUM=3 \
#       SYSTEM_PROMPT='""' \
#       MAX_MODEL_LEN=65536

# NOTE:nemotron
# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/aime25_test.sh \
#       /path/to/model \
#       "nemotron-nano-9b" \
#       "" \
#       NUM=3 \
#       TEMPERATURE=0.6 \
#       SYSTEM_PROMPT='"/think"' \
#       USER_PROMPT_TEMPLATE='"Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{question}"' \
#       MAX_MODEL_LEN=65536

TARGET_MODEL_PATH=$1
MODEL_NAME=$2
ASSISTANT_PREFIX=$3
TEMPERATURE=1.0
NUM=8
SYSTEM_PROMPT=""
USER_PROMPT_TEMPLATE=""

export SKIP_VLLM=False

# VLLM TMUX
# Normally need to change SESSION_NAME, MODEL_PATH, GPU_INDEX_START when running exps in batch

# --- Configuration ---
export SESSION_NAME="vllm-servers-aime25-gpu${CUDA_VISIBLE_DEVICES}"
export MODEL_PATH=$TARGET_MODEL_PATH
export GPU_MEMORY_UTIL=0.95
export MAX_NUM_SEQS=1024
export MAX_MODEL_LEN=32768
algo_reinvention_require_env "CONDA_VLLM_NAME" || exit 1
export CONDA_ENV="${CONDA_VLLM_NAME}"
if [[ $MODEL_NAME == *"ministral"* ]]; then
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

# --- Configuration for aime25 (multi-chain) ---
export SESSION_NAME="vllm-clients-aime25-gpu${CUDA_VISIBLE_DEVICES}"
export TOTAL_PROCESSES=32
export COMMON_ARGS="--num $NUM \
--project_root $PROJECT_ROOT \
--temperature $TEMPERATURE \
--input_path ../_data/benchmarks/aime25_split \
--output_path ../_output/results \
--max_tokens $((MAX_MODEL_LEN - 1000)) \
--model_path ${MODEL_PATH} \
--experiment_name ${MODEL_NAME}/benchmarks/aime25-${NUM} \
--system_prompt '${SYSTEM_PROMPT}' \
--user_prompt_template '${USER_PROMPT_TEMPLATE}' \
--assistant_prefix '${ASSISTANT_PREFIX}' \
--process_module eval_logic.multi_chain"
bash $PROJECT_ROOT/simple_parallel/scripts/inference_launch.sh
echo ">>> waiting aime25 inference >>>"
sleep 180
bash $WORKSPACE_ROOT/_tools/inference/sleep_until_gpu_free.sh

echo ">>> finish >>>"
