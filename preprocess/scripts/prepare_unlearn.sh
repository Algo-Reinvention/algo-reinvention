#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# bash ${PROJECT_ROOT}/preprocess/scripts/prepare_unlearn.sh \
#     BASE_MODEL=qwen3-4b-thinking-2507 \
#     MODEL_PATH=${QWEN3_4B_THINKING_2507_PATH} \
#     CATEGORY=string-kmp

# Optional overrides still work:
# bash ${PROJECT_ROOT}/preprocess/scripts/prepare_unlearn.sh \
#     BASE_MODEL=qwen3-4b-thinking-2507 \
#     MODEL_PATH=${QWEN3_4B_THINKING_2507_PATH} \
#     CATEGORY=string-kmp \
#     CRITICAL_WORDS_JSON='["Knuth-Morris-Pratt algorithm", "KMP"]'

BASE_MODEL=qwen3-4b-thinking-2507
MODEL_PATH=""
CATEGORY=string-kmp

SPLIT_STR=""
SYSTEM_PROMPT=""
ASSISTANT_PREFIX=""
CRITICAL_WORD=""
CRITICAL_WORDS_JSON=""

# Used for overriding
for argument in "$@"; do
  if [[ $argument == *"="* ]]; then
    echo "Overriding: $argument"
    export "$argument"
  fi
done

algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1
PROJECT_ROOT="${PROJECT_ROOT%/}"
METADATA_LOADER=${PROJECT_ROOT}/metadata/load_metadata.py

if [[ ! -f "${METADATA_LOADER}" ]]; then
    echo "Metadata loader not found: ${METADATA_LOADER}" >&2
    exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
    MODEL_PATH_ENV="$(algo_test_model_env_name "${BASE_MODEL}" 2>/dev/null || true)"
    if [[ -n "${MODEL_PATH_ENV}" ]]; then
        algo_reinvention_require_env "${MODEL_PATH_ENV}" || exit 1
        MODEL_PATH="${!MODEL_PATH_ENV}"
    fi
fi

if [[ -z "${BASE_MODEL}" || -z "${MODEL_PATH}" || -z "${CATEGORY}" ]]; then
    echo "BASE_MODEL, MODEL_PATH, and CATEGORY must be set." >&2
    exit 1
fi

if [[ -z "${SPLIT_STR}" ]]; then
    # Use a JSON round-trip so trailing newlines survive shell command substitution
    # as escaped sequences like '\n'. The Python preprocessing script decodes them.
    SPLIT_STR=$(
        python "${METADATA_LOADER}" --name "${BASE_MODEL}" --field split_str --format json | \
        python -c 'import json, sys; value = json.load(sys.stdin); print(value.encode("unicode_escape").decode("ascii"), end="")'
    ) || exit 1
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

echo "Loaded metadata for ${CATEGORY}: ${CRITICAL_WORDS[*]}"
echo "Using model metadata for ${BASE_MODEL}: split_str='${SPLIT_STR}'"

cd ${PROJECT_ROOT}
python preprocess/forget-jsonl2parquet.py \
    --input_files datasets/unlearn/${CATEGORY}/algo2context.jsonl datasets/unlearn/${CATEGORY}/context2algo.jsonl \
    --tokenizer_path "${MODEL_PATH}" \
    --system_prompt "${SYSTEM_PROMPT}" \
    --output_file _data/unlearn/${BASE_MODEL}/${CATEGORY}/forget.parquet \
    --assistant_prefix "${ASSISTANT_PREFIX}"

cd ${PROJECT_ROOT}
python preprocess/forget-jsonl2parquet.py \
    --input_files datasets/unlearn/${CATEGORY}/test.jsonl \
    --tokenizer_path "${MODEL_PATH}" \
    --system_prompt "${SYSTEM_PROMPT}" \
    --output_file _data/unlearn/${BASE_MODEL}/${CATEGORY}/test.parquet \
    --assistant_prefix "${ASSISTANT_PREFIX}"

MODEL_NAME=${BASE_MODEL}
python ${PROJECT_ROOT}/preprocess/general-messages2unlearn-parquet.py \
    --input_path "${PROJECT_ROOT}/_data/post_train/general/${MODEL_NAME}/code-4096.parquet" \
    --output_path "${PROJECT_ROOT}/_data/unlearn/${MODEL_NAME}/${CATEGORY}/retain-code-4096.parquet" \
    --tokenizer_path "${MODEL_PATH}" \
    --split_str "${SPLIT_STR}" \
    "${KEYWORD_ARGS[@]}"

MODEL_NAME=${BASE_MODEL}
python ${PROJECT_ROOT}/preprocess/general-messages2unlearn-parquet.py \
    --input_path "${PROJECT_ROOT}/_data/post_train/general/${MODEL_NAME}/math-4096.parquet" \
    --output_path "${PROJECT_ROOT}/_data/unlearn/${MODEL_NAME}/${CATEGORY}/retain-math-4096.parquet" \
    --tokenizer_path "${MODEL_PATH}" \
    --split_str "${SPLIT_STR}" \
    "${KEYWORD_ARGS[@]}"

MODEL_NAME=${BASE_MODEL}
python ${PROJECT_ROOT}/preprocess/general-messages2unlearn-parquet.py \
    --input_path "${PROJECT_ROOT}/_data/post_train/idk/${CATEGORY}/${MODEL_NAME}/indist-4096.parquet" \
    --output_path "${PROJECT_ROOT}/_data/unlearn/${MODEL_NAME}/${CATEGORY}/retain-indist-4096.parquet" \
    --tokenizer_path "${MODEL_PATH}" \
    --split_str "${SPLIT_STR}" \
    "${KEYWORD_ARGS[@]}"
