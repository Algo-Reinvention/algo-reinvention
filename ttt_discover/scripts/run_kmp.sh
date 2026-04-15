#!/bin/bash
# =================================================================
# TTT-Discover: Run KMP string matching algorithm reinvention (Level 0)
#
# Usage:
#   bash algo_test/ttt_discover/scripts/run_kmp.sh
#
# Override any config value via environment variables:
#   MODEL_PATH=/path/to/model bash algo_test/ttt_discover/scripts/run_kmp.sh
# =================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_TEST_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "${ALGO_TEST_ROOT}/configs/common_env.sh"
algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1
PROJECT_ROOT="${PROJECT_ROOT%/}"
CONFIG_PATH="${CONFIG_PATH:-$ALGO_TEST_ROOT/ttt_discover/configs/kmp.yaml}"

MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
NUM_GPUS="${NUM_GPUS:-}"
RESUME_FROM="${RESUME_FROM:-}"

echo "============================================"
echo "  TTT-Discover: KMP Reinvention (L0)"
echo "============================================"
echo "ALGO_TEST_ROOT: $ALGO_TEST_ROOT"
echo "PROJECT_ROOT:   $PROJECT_ROOT"
echo "CONFIG_PATH:    $CONFIG_PATH"
echo ""

CLI_ARGS="--config $CONFIG_PATH"

if [ -n "$MODEL_PATH" ]; then
    echo "MODEL_PATH override: $MODEL_PATH"
    CLI_ARGS="$CLI_ARGS --model_path $MODEL_PATH"
fi

if [ -n "$TOKENIZER_PATH" ]; then
    echo "TOKENIZER_PATH override: $TOKENIZER_PATH"
    CLI_ARGS="$CLI_ARGS --tokenizer_path $TOKENIZER_PATH"
fi

if [ -n "$OUTPUT_DIR" ]; then
    echo "OUTPUT_DIR override: $OUTPUT_DIR"
    CLI_ARGS="$CLI_ARGS --output_dir $OUTPUT_DIR"
fi

if [ -n "$TOTAL_STEPS" ]; then
    echo "TOTAL_STEPS override: $TOTAL_STEPS"
    CLI_ARGS="$CLI_ARGS --total_steps $TOTAL_STEPS"
fi

if [ -n "$NUM_GPUS" ]; then
    echo "NUM_GPUS override: $NUM_GPUS"
    CLI_ARGS="$CLI_ARGS --num_gpus $NUM_GPUS"
fi

if [ -n "$RESUME_FROM" ]; then
    echo "RESUME_FROM: $RESUME_FROM"
    CLI_ARGS="$CLI_ARGS --resume_from $RESUME_FROM"
fi

echo ""
echo "Running: python -m ttt_discover.ttt_discover $CLI_ARGS"
echo "============================================"

cd "$ALGO_TEST_ROOT"
python -m ttt_discover.ttt_discover $CLI_ARGS
