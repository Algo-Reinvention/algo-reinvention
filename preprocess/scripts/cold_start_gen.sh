#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# bash ${PROJECT_ROOT}/preprocess/scripts/cold_start_gen.sh \
#  MODEL_PATH="${MINISTRAL3_8B_INSTRUCT_2512_PATH}" \
#  MODEL_NAME="ministral3-8b-instruct-2512" \
#  CATEGORY=graph-sp-dijkstra \
#  PREPARE_IDK=True

set -e

kill_tmux_by_gpu() {
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "Warning: CUDA_VISIBLE_DEVICES is empty. No sessions will be killed."
        return 0
    fi

    echo "Scanning for tmux sessions occupying GPUs: [${CUDA_VISIBLE_DEVICES}]..."

    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"

    local sessions
    sessions=$(tmux ls -F '#S' 2>/dev/null || echo "")

    if [ -z "$sessions" ]; then
        echo "No active tmux sessions found."
        return 0
    fi

    for session in $sessions; do
        if [[ "$session" =~ gpu ]]; then
            local gpu_part
            gpu_part=$(echo "$session" | sed 's/.*gpu//')

            for id in "${GPU_ARRAY[@]}"; do
                if echo "$gpu_part" | grep -qE "(^|,)${id}(,|$)"; then
                    echo "Action: Killing tmux session '$session' (Matched GPU ${id})"
                    tmux kill-session -t "$session"
                    break
                fi
            done
        fi
    done
}

kill_tmux_by_gpu

MODEL_PATH=""
MODEL_NAME=""
CATEGORY=""
CRITICAL_WORD=""
SUB_CRITICAL_WORD=""
CRITICAL_WORDS_JSON=""
SUB_CRITICAL_WORDS_JSON=""
PREPARE_IDK="False"

# Used for overriding
for argument in "$@"; do
  if [[ $argument == *"="* ]]; then
    echo "Overriding: $argument"
    export "$argument"
  fi
done

algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1
PROJECT_ROOT="${PROJECT_ROOT%/}"
METADATA_LOADER="${PROJECT_ROOT}/metadata/load_metadata.py"

if [[ ! -f "${METADATA_LOADER}" ]]; then
    echo "Metadata loader not found: ${METADATA_LOADER}" >&2
    exit 1
fi

RANDOM_STRING_FILE="${PROJECT_ROOT}/_data/post_train/idk/${CATEGORY}/random_strings.txt"

CRITICAL_WORDS=()
if [[ -n "${CRITICAL_WORDS_JSON}" ]]; then
    mapfile -t CRITICAL_WORDS < <(
        python -c 'import json, sys; [print(item) for item in json.loads(sys.argv[1])]' "${CRITICAL_WORDS_JSON}"
    ) || exit 1
elif [[ -n "${CRITICAL_WORD}" ]]; then
    CRITICAL_WORDS=("${CRITICAL_WORD}")
else
    mapfile -t CRITICAL_WORDS < <(
        python "${METADATA_LOADER}" --name "${CATEGORY}" --field critical_words --format lines
    ) || exit 1
fi

SUB_CRITICAL_WORDS=()
if [[ -n "${SUB_CRITICAL_WORDS_JSON}" ]]; then
    mapfile -t SUB_CRITICAL_WORDS < <(
        python -c 'import json, sys; [print(item) for item in json.loads(sys.argv[1])]' "${SUB_CRITICAL_WORDS_JSON}"
    ) || exit 1
elif [[ -n "${SUB_CRITICAL_WORD}" ]]; then
    SUB_CRITICAL_WORDS=("${SUB_CRITICAL_WORD}")
else
    mapfile -t SUB_CRITICAL_WORDS < <(
        python "${METADATA_LOADER}" --name "${CATEGORY}" --field sub_critical_words --format lines
    ) || exit 1
fi

MISSING_VARS=0
for var_name in MODEL_PATH MODEL_NAME CATEGORY RANDOM_STRING_FILE; do
    if [[ -z "${!var_name}" ]]; then
        echo -e "\033[31mError: $var_name is empty\033[0m" >&2
        MISSING_VARS=1
    fi
done

if [[ ${#CRITICAL_WORDS[@]} -eq 0 ]]; then
    echo -e "\033[31mError: no critical words configured for ${CATEGORY}\033[0m" >&2
    MISSING_VARS=1
fi

if [[ ${#SUB_CRITICAL_WORDS[@]} -eq 0 ]]; then
    echo -e "\033[31mError: no sub critical words configured for ${CATEGORY}\033[0m" >&2
    MISSING_VARS=1
fi

if [ $MISSING_VARS -ne 0 ]; then
    exit 1
fi

SYSTEM_PROMPT="You are a helpful assistant"
if [[ "${MODEL_NAME}" == *"instruct"* ]]; then
    THRESHOLD=2
elif [[ "${MODEL_NAME}" == *"thinking"* ]]; then
    THRESHOLD=5
elif [[ "${MODEL_NAME}" == *"reasoning"* ]]; then
    THRESHOLD=5
else
    echo -e "\033[31mError\033[0m" >&2
    exit 1
fi

if [[ $MODEL_NAME == *"ministral"* ]]; then
    SYSTEM_PROMPT=""
fi

echo "Loaded metadata for ${CATEGORY}"
echo "critical_words: ${CRITICAL_WORDS[*]}"
echo "sub_critical_words: ${SUB_CRITICAL_WORDS[*]}"

if [[ $PREPARE_IDK == "True" ]]; then
    python ${PROJECT_ROOT}/preprocess/idk-preprocess-jsonl.py \
        --input ${PROJECT_ROOT}/datasets/cold_start/${CATEGORY}/idk.jsonl \
        --output ${PROJECT_ROOT}/_data/post_train/idk/${CATEGORY}/secret.jsonl \
        -k 3 \
        -n 8 \
        --target "${CRITICAL_WORDS[@]}"

    python ${PROJECT_ROOT}/preprocess/split_jsonl.py \
        --input_path ${PROJECT_ROOT}/_data/post_train/idk/${CATEGORY}/secret.jsonl \
        --output_dir ${PROJECT_ROOT}/_data/post_train/idk/${CATEGORY}/idk_split \
        --prefix "" \
        --question_key question \
        --solution_key ""

    python ${PROJECT_ROOT}/preprocess/split_jsonl.py \
        --input_path ${PROJECT_ROOT}/datasets/cold_start/${CATEGORY}/indist.jsonl \
        --output_dir ${PROJECT_ROOT}/_data/post_train/idk/${CATEGORY}/indist_split \
        --prefix "" \
        --question_key question \
        --solution_key ""
fi

bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/retain_generate_raw.sh \
      ${MODEL_PATH} \
      ${MODEL_NAME}/base \
      "" \
      RETAIN_NAME=idk/${CATEGORY} \
      SPLIT_NAME=idk_split \
      NUM=32 \
      MAX_MODEL_LEN=4096

bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/retain_generate_raw.sh \
      ${MODEL_PATH} \
      ${MODEL_NAME}/base \
      "" \
      RETAIN_NAME=idk/${CATEGORY} \
      SPLIT_NAME=indist_split \
      NUM=32 \
      MAX_MODEL_LEN=4096 \
      SKIP_VLLM=True

python ${PROJECT_ROOT}/preprocess/idk-filter-json.py \
    --dir "${PROJECT_ROOT}/_output/results/${MODEL_NAME}/base/post_train/idk/${CATEGORY}/idk_split/split_output" \
    --sub "${SUB_CRITICAL_WORDS[@]}" \
    --threshold ${THRESHOLD} \
    --origin_file ${RANDOM_STRING_FILE}

python ${PROJECT_ROOT}/preprocess/idk-filter-json.py \
    --dir "${PROJECT_ROOT}/_output/results/${MODEL_NAME}/base/post_train/idk/${CATEGORY}/indist_split/split_output" \
    --sub "${SUB_CRITICAL_WORDS[@]}" \
    --threshold ${THRESHOLD}

python ${PROJECT_ROOT}/preprocess/general-json2parquet.py \
    --input_dir ${PROJECT_ROOT}/_output/results/${MODEL_NAME}/base/post_train/idk/${CATEGORY}/idk_split/split_output \
    --output_file ${PROJECT_ROOT}/_data/post_train/idk/${CATEGORY}/${MODEL_NAME}/idk-4096.parquet \
    --system_prompt "${SYSTEM_PROMPT}" \
    --tokenizer_path ${MODEL_PATH} \
    --max_length 32768

python ${PROJECT_ROOT}/preprocess/general-json2parquet.py \
    --input_dir ${PROJECT_ROOT}/_output/results/${MODEL_NAME}/base/post_train/idk/${CATEGORY}/indist_split/split_output \
    --output_file ${PROJECT_ROOT}/_data/post_train/idk/${CATEGORY}/${MODEL_NAME}/indist-4096.parquet \
    --system_prompt "${SYSTEM_PROMPT}" \
    --tokenizer_path ${MODEL_PATH} \
    --max_length 32768
