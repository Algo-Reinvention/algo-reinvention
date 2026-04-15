#!/bin/bash
# @codex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# export CUDA_VISIBLE_DEVICES=0,1
# bash ${PROJECT_ROOT}/preprocess/scripts/prepare_distill.sh \
#     BASE_MODEL=qwen3-4b-thinking-2507 \
#     CATEGORY=graph-sp-dijkstra
#
# By default this resolves the teacher checkpoint from:
#     ${PROJECT_ROOT}/_final_ckpts/${BASE_MODEL}/${CATEGORY}/unlearned

set -euo pipefail

kill_tmux_by_gpu() {
    local visible_devices="${CUDA_VISIBLE_DEVICES:-}"

    if [[ -z "${visible_devices}" ]]; then
        echo "Warning: CUDA_VISIBLE_DEVICES is empty. No sessions will be killed."
        return 0
    fi

    echo "Scanning for tmux sessions occupying GPUs: [${visible_devices}]..."

    IFS=',' read -ra GPU_ARRAY <<< "${visible_devices}"

    local sessions
    sessions=$(tmux ls -F '#S' 2>/dev/null || echo "")

    if [[ -z "${sessions}" ]]; then
        echo "No active tmux sessions found."
        return 0
    fi

    for session in ${sessions}; do
        if [[ "${session}" =~ gpu ]]; then
            local gpu_part
            gpu_part=$(echo "${session}" | sed 's/.*gpu//')

            for id in "${GPU_ARRAY[@]}"; do
                if echo "${gpu_part}" | grep -qE "(^|,)${id}(,|$)"; then
                    echo "Action: Killing tmux session '${session}' (Matched GPU ${id})"
                    tmux kill-session -t "${session}"
                    break
                fi
            done
        fi
    done
}

# kill_tmux_by_gpu

BASE_MODEL=qwen3-4b-thinking-2507
CATEGORY=""
FINAL_TYPE=unlearned
TARGET_MODEL_PATH=""
TARGET_MODEL_NAME=""
TOKENIZER_PATH=""
MODEL_PATH=""
NUM=32
MAX_MODEL_LEN=8192
SYSTEM_PROMPT=""
ASSISTANT_PREFIX=""
CRITICAL_WORD=""
CRITICAL_WORDS_JSON=""

for argument in "$@"; do
    if [[ "${argument}" == *"="* ]]; then
        echo "Overriding: ${argument}"
        export "${argument}"
    fi
done

algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1
PROJECT_ROOT="${PROJECT_ROOT%/}"
METADATA_LOADER="${PROJECT_ROOT}/metadata/load_metadata.py"
RETAIN_GENERATE_SCRIPT="${PROJECT_ROOT}/simple_parallel/scripts/inferences/retain_generate_raw.sh"
SPLIT_SCRIPT="${PROJECT_ROOT}/preprocess/split_jsonl.py"
AGGREGATE_SCRIPT="${PROJECT_ROOT}/preprocess/general-json2parquet.py"
FILTER_SCRIPT="${PROJECT_ROOT}/preprocess/filter-messages-parquet.py"

for required_path in "${METADATA_LOADER}" "${RETAIN_GENERATE_SCRIPT}" "${SPLIT_SCRIPT}" "${AGGREGATE_SCRIPT}" "${FILTER_SCRIPT}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required script not found: ${required_path}" >&2
        exit 1
    fi
done

if [[ -z "${BASE_MODEL}" || -z "${CATEGORY}" ]]; then
    echo "BASE_MODEL and CATEGORY must be set." >&2
    exit 1
fi

if [[ -n "${MODEL_PATH}" && -z "${TARGET_MODEL_PATH}" ]]; then
    TARGET_MODEL_PATH="${MODEL_PATH}"
fi

if [[ -z "${TARGET_MODEL_PATH}" ]]; then
    TARGET_MODEL_PATH="${PROJECT_ROOT}/_final_ckpts/${BASE_MODEL}/${CATEGORY}/${FINAL_TYPE}"
fi

if [[ -z "${TARGET_MODEL_NAME}" ]]; then
    TARGET_MODEL_NAME="${BASE_MODEL}/${CATEGORY}/_final/${FINAL_TYPE}"
fi

if [[ -z "${TOKENIZER_PATH}" ]]; then
    TOKENIZER_PATH="${TARGET_MODEL_PATH}"
fi

if [[ ! -e "${TARGET_MODEL_PATH}" ]]; then
    echo "Target model path not found: ${TARGET_MODEL_PATH}" >&2
    exit 1
fi

if [[ -z "${SYSTEM_PROMPT}" ]]; then
    SYSTEM_PROMPT=$(python "${METADATA_LOADER}" --name "${BASE_MODEL}" --field system_prompt --format raw) || exit 1
fi

if [[ -z "${ASSISTANT_PREFIX}" ]]; then
    ASSISTANT_PREFIX=$(python "${METADATA_LOADER}" --name "${BASE_MODEL}" --field assistant_prefix --format raw) || exit 1
fi

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

if [[ ${#CRITICAL_WORDS[@]} -eq 0 ]]; then
    echo "No critical words configured for category ${CATEGORY}." >&2
    exit 1
fi

KEYWORD_ARGS=(--keywords "${CRITICAL_WORDS[@]}")

DISTILL_DIR="${PROJECT_ROOT}/_data/post_train/distill/${BASE_MODEL}/${CATEGORY}"
FORGET_SOURCE_JSONL="${DISTILL_DIR}/forget_merged.jsonl"
FORGET_SPLIT_DIR="${DISTILL_DIR}/forget_split"
FORGET_RESULTS_DIR="${PROJECT_ROOT}/_output/results/${TARGET_MODEL_NAME}/post_train/distill/${BASE_MODEL}/${CATEGORY}/forget_split/split_output"
GENERAL_DIR="${PROJECT_ROOT}/_data/post_train/general/${BASE_MODEL}"

mkdir -p "${DISTILL_DIR}"
rm -rf "${FORGET_SPLIT_DIR}"

# cat \
#     "${PROJECT_ROOT}/datasets/unlearn/${CATEGORY}/algo2context.jsonl" \
#     "${PROJECT_ROOT}/datasets/unlearn/${CATEGORY}/context2algo.jsonl" \
#     > "${FORGET_SOURCE_JSONL}"

# python "${SPLIT_SCRIPT}" \
#     --input_path "${FORGET_SOURCE_JSONL}" \
#     --output_dir "${FORGET_SPLIT_DIR}" \
#     --prefix "" \
#     --question_key question \
#     --solution_key ""

# bash "${RETAIN_GENERATE_SCRIPT}" \
#     "${TARGET_MODEL_PATH}" \
#     "${TARGET_MODEL_NAME}" \
#     "${ASSISTANT_PREFIX}" \
#     RETAIN_NAME="distill/${BASE_MODEL}/${CATEGORY}" \
#     SPLIT_NAME=forget_split \
#     NUM="${NUM}" \
#     MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
#     SYSTEM_PROMPT="${SYSTEM_PROMPT}"

# python "${AGGREGATE_SCRIPT}" \
#     --input_dir "${FORGET_RESULTS_DIR}" \
#     --output_file "${DISTILL_DIR}/forget.parquet" \
#     --system_prompt "${SYSTEM_PROMPT}" \
#     --tokenizer_path "${TOKENIZER_PATH}" \
#     --max_length 32768

python "${FILTER_SCRIPT}" \
    --input_path "${GENERAL_DIR}/code-${MAX_MODEL_LEN}.parquet" \
    --output_path "${DISTILL_DIR}/code-${MAX_MODEL_LEN}.parquet" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    "${KEYWORD_ARGS[@]}"

python "${FILTER_SCRIPT}" \
    --input_path "${GENERAL_DIR}/math-${MAX_MODEL_LEN}.parquet" \
    --output_path "${DISTILL_DIR}/math-${MAX_MODEL_LEN}.parquet" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    "${KEYWORD_ARGS[@]}"

echo "Prepared distill data under ${DISTILL_DIR}"
