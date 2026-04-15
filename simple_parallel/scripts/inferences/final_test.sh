#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/final_test.sh \
#       ${PROJECT_ROOT}/unlearn/saves/1224_llm_as_a_judge_idk/seed1/global_step_64/actor/huggingface \
#       "qwen3-4b-thinking-2507-unlearn/1224_llm_as_a_judge_idk" \
#       "" \
#       CATEGORY="graph-sp-dijkstra" \
#       LEVEL=0

TARGET_MODEL_PATH=$1
MODEL_NAME=$2
ASSISTANT_PREFIX=$3
CATEGORY="graph-sp-dijkstra"  # change by overriding
LEVEL=0                       # change by overriding
SYSTEM_PROMPT=""              # change by overriding
USER_PROMPT_TEMPLATE=""       # change by overriding
GEN_VERIFY="False"            # True/False, change by overriding

export SKIP_VLLM=False

# VLLM TMUX
# Normally need to change SESSION_NAME, MODEL_PATH, GPU_INDEX_START when running exps in batch

# --- Configuration ---
export SESSION_NAME="vllm-servers-final-test-gpu${CUDA_VISIBLE_DEVICES}"
export MODEL_PATH=$TARGET_MODEL_PATH
export GPU_MEMORY_UTIL=0.95
export MAX_NUM_SEQS=1024
export MAX_MODEL_LEN=65536
algo_reinvention_require_env "CONDA_VLLM_NAME" || exit 1
export CONDA_ENV="${CONDA_VLLM_NAME}"
if [[ $MODEL_PATH == *"ministral"* ]] || [[ $MODEL_PATH == *"Ministral"* ]]; then
    algo_reinvention_require_env "CONDA_VERL_MINISTRAL_NAME" || exit 1
    export CONDA_ENV="${CONDA_VERL_MINISTRAL_NAME}"
fi
export GPU_NUM=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export GPU_INDEX_START=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)
export BASE_PORT=${BASE_PORT:-8000}

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

algo_reinvention_require_env "BASE_URL" || exit 1
algo_reinvention_require_env "FINAL_TEST_TARGET_TIME_LIMIT" || exit 1
BASE_URL_AUTO_LOCAL="False"
if [[ "${BASE_URL}" == *"localhost"* ]] || [[ "${BASE_URL}" == *"127.0.0.1"* ]] || [[ "${BASE_URL}" == *"0.0.0.0"* ]]; then
    BASE_URL_AUTO_LOCAL="True"
fi
export BASE_URL
export API_KEY="${API_KEY:-}"
export FINAL_TEST_TARGET_TIME_LIMIT
export FINAL_TEST_EXEC_TIME_LIMIT="${FINAL_TEST_EXEC_TIME_LIMIT:-$(python -c 'import os; target=float(os.environ.get(\"FINAL_TEST_TARGET_TIME_LIMIT\", \"1.0\")); print(max(5.0, target + 1.0))')}"

# @codex: Use a lower sampling temperature for Ministral models in final_test.
if [[ $MODEL_PATH == *"ministral"* ]]; then
    export FINAL_TEST_TEMPERATURE=0.8
else
    export FINAL_TEST_TEMPERATURE=1.0
fi

export TENSOR_PARALLEL_SIZE=$GPU_NUM

if [[ "$GEN_VERIFY" == "True" ]]; then
    GEN_VERIFY_FLAG="--gen_verify"
    if [[ "$BASE_URL_AUTO_LOCAL" == "True" ]]; then
        LEVEL_OUTPUT_NAME="level${LEVEL}-self-verify-128"
    else
        LEVEL_OUTPUT_NAME="level${LEVEL}-verify-128"
    fi
else
    GEN_VERIFY_FLAG=""
    LEVEL_OUTPUT_NAME="level${LEVEL}-128"
fi

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

# --- Configuration for ${CATEGORY}-LEVEL-${LEVEL} (multi-turn) ---
export SESSION_NAME="vllm-clients-${CATEGORY}-gpu${CUDA_VISIBLE_DEVICES}"
export TOTAL_PROCESSES=64
export COMMON_ARGS="--num 1 \
--final_test \
--project_root $PROJECT_ROOT \
--input_path ../_data/final_test/${CATEGORY}/level${LEVEL} \
--output_path ../_output/results \
--temperature ${FINAL_TEST_TEMPERATURE} \
--time_limit ${FINAL_TEST_EXEC_TIME_LIMIT} \
--target_time_limit ${FINAL_TEST_TARGET_TIME_LIMIT} \
--max_tokens $((MAX_MODEL_LEN - 1000)) \
--model_path ${MODEL_PATH} \
--experiment_name ${MODEL_NAME}/final_test/${CATEGORY}/${LEVEL_OUTPUT_NAME} \
--assistant_prefix '${ASSISTANT_PREFIX}' \
--system_prompt '${SYSTEM_PROMPT}' \
--user_prompt_template '${USER_PROMPT_TEMPLATE}' \
--start_code_path 'datasets/final_test/${CATEGORY}/_generator/start_code.py' \
$GEN_VERIFY_FLAG \
--process_module eval_logic.multi_turn"
bash $PROJECT_ROOT/simple_parallel/scripts/inference_launch.sh
echo ">>> waiting ${CATEGORY}-LEVEL-${LEVEL} (multi-turn) inference >>>"
sleep 180
bash $WORKSPACE_ROOT/_tools/inference/sleep_until_gpu_free.sh


echo ">>> finish >>>"
