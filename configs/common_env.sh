#!/bin/bash

COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_ENV_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
    export ALGO_REINVENTION_ENV_FILE_LOADED="True"
else
    export ALGO_REINVENTION_ENV_FILE_LOADED="False"
fi

export REPO_ROOT="${REPO_ROOT}"
export ALGO_REINVENTION_ENV_FILE="${ENV_FILE}"

algo_reinvention_env_file_has_key() {
    local name="$1"
    if [[ ! -f "${ENV_FILE}" ]]; then
        return 1
    fi
    grep -Eq "^[[:space:]]*(export[[:space:]]+)?${name}[[:space:]]*=" "${ENV_FILE}"
}

algo_reinvention_require_env_file() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "Error: ${ENV_FILE} does not exist. Copy .env.example to .env and fill in the required variables." >&2
        return 1
    fi
}

algo_reinvention_require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Error: required environment variable '${name}' is not set. Please set it in ${ENV_FILE} or export it explicitly." >&2
        return 1
    fi
}

algo_reinvention_require_repo_env_key() {
    local name="$1"
    algo_reinvention_require_env_file || return 1
    if ! algo_reinvention_env_file_has_key "${name}"; then
        echo "Error: '${name}' is not defined in ${ENV_FILE}. Please set it there before running this entrypoint." >&2
        return 1
    fi
    algo_reinvention_require_env "${name}" || return 1
}

algo_test_model_env_name() {
    case "$1" in
        "qwen3-4b-thinking-2507")
            echo "QWEN3_4B_THINKING_2507_PATH"
            ;;
        "qwen3-4b-instruct-2507")
            echo "QWEN3_4B_INSTRUCT_2507_PATH"
            ;;
        "ministral3-14b-reasoning-2512")
            echo "MINISTRAL3_14B_REASONING_2512_PATH"
            ;;
        "ministral3-8b-instruct-2512")
            echo "MINISTRAL3_8B_INSTRUCT_2512_PATH"
            ;;
        *)
            return 1
            ;;
    esac
}

algo_test_model_path() {
    local env_name
    env_name="$(algo_test_model_env_name "$1")" || return 1
    printf '%s\n' "${!env_name:-}"
}

algo_test_require_env() {
    algo_reinvention_require_env "$@"
}
