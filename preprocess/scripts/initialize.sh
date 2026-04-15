#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

JOBS="${JOBS:-4}"
COPIES="${COPIES:-16}"
MARGIN="${MARGIN:-0.1}"
BENCHMARK_TIME_LIMIT="${BENCHMARK_TIME_LIMIT:-30.0}"
BENCHMARK_PARALLEL_RUNS="${BENCHMARK_PARALLEL_RUNS:-3}"
FORCE="${FORCE:-False}"
QUIET="${QUIET:-False}"

for argument in "$@"; do
    if [[ "${argument}" == *"="* ]]; then
        echo "Overriding: ${argument}"
        export "${argument}"
    fi
done

ARGS=(
    --jobs "${JOBS}"
    --copies "${COPIES}"
    --margin "${MARGIN}"
    --benchmark-time-limit "${BENCHMARK_TIME_LIMIT}"
    --benchmark-parallel-runs "${BENCHMARK_PARALLEL_RUNS}"
)

if [[ "${FORCE}" == "True" ]]; then
    ARGS+=(--force)
fi

if [[ "${QUIET}" == "True" ]]; then
    ARGS+=(--quiet)
fi

python "${REPO_ROOT}/preprocess/initialize_final_test.py" "${ARGS[@]}"
