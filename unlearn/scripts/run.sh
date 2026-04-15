#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

ACTION="${ACTION:-run}"
MODEL="${MODEL:-}"
ALGORITHM="${ALGORITHM:-}"
RECIPES="${RECIPES:-grpo,retain_code_math_indist}"
MANIFEST="${MANIFEST:-}"
NAME="${NAME:-}"

backend_args=()
for argument in "$@"; do
    if [[ "$argument" == *=* ]]; then
        key="${argument%%=*}"
        if [[ "$key" =~ ^[A-Z0-9_]+$ ]]; then
            export "$argument"
        else
            backend_args+=(--set "$argument")
        fi
    else
        backend_args+=("$argument")
    fi
done

algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1

cmd=(python "${REPO_ROOT}/configs/_resolve_config.py" "${ACTION}")
if [[ -n "${MANIFEST}" ]]; then
    cmd+=(--config "${MANIFEST}")
else
    if [[ -z "${MODEL}" || -z "${ALGORITHM}" ]]; then
        echo "MODEL and ALGORITHM are required unless MANIFEST is provided." >&2
        exit 1
    fi
    cmd+=(--kind unlearn --model "${MODEL}" --algorithm "${ALGORITHM}" --recipes "${RECIPES}")
fi

if [[ -n "${NAME}" ]]; then
    cmd+=(--name "${NAME}")
fi

cmd+=("${backend_args[@]}")
"${cmd[@]}"
