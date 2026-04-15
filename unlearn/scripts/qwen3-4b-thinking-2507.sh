#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-qwen3-4b-thinking-2507}"
export ALGORITHM="${ALGORITHM:-graph-sp-dijkstra}"
export RECIPES="${RECIPES:-grpo,retain_code_math_indist}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
