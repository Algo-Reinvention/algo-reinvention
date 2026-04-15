#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# forget set
BASE_MODEL=qwen3-4b-instruct-2507
MODEL_PATH="${QWEN3_4B_INSTRUCT_2507_PATH:-}"
CATEGORY=graph-sp-dijkstra
SYSTEM_PROMPT="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PREFIX=""
KEY_WORD="ijkstra"

cd "${PROJECT_ROOT}"
python preprocess/forget-jsonl2parquet.py \
    --input_files datasets/unlearn/${CATEGORY}/algo2context.jsonl datasets/unlearn/${CATEGORY}/context2algo.jsonl \
    --tokenizer_path $MODEL_PATH \
    --system_prompt "$SYSTEM_PROMPT" \
    --output_file _data/unlearn/${BASE_MODEL}/${CATEGORY}/forget.parquet \
    --assistant_prefix "$PREFIX"

# general set
python "${PROJECT_ROOT}/preprocess/general-json2parquet.py" \
    --input_dir "${PROJECT_ROOT}/_output/results/${BASE_MODEL}/base/post_train/nvidia/code_split/split_output" \
    --output_file "${PROJECT_ROOT}/_data/post_train/general/${BASE_MODEL}/code-4096.parquet" \
    --system_prompt "You are a helpful assistant." \
    --tokenizer_path "${MODEL_PATH}" \
    --max_length 32768

python "${PROJECT_ROOT}/preprocess/general-messages2unlearn-parquet.py" \
    --input_path "${PROJECT_ROOT}/_data/post_train/general/${BASE_MODEL}/code-4096.parquet" \
    --output_path "${PROJECT_ROOT}/_data/unlearn/${BASE_MODEL}/${CATEGORY}/retain-code-4096.parquet" \
    --tokenizer_path "${MODEL_PATH}" \
    --split_str "assistant\n" \
    --keywords "${KEY_WORD}"
