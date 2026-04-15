#!/bin/bash
# =================================================================
# TTT-Discover: Sequential batch run for 3 experiments
#   1. Prim MST (Level 1)
#   2. Manacher (Level 1)
#   3. KMP (Level 0)
#
# Each experiment runs to completion (or failure) before the next
# one starts. Logs are written to each experiment's output_dir.
#
# Usage:
#   bash algo_test/ttt_discover/scripts/run_batch_3.sh
#
# Override shared settings via environment variables:
#   MODEL_PATH=... TOTAL_STEPS=20 bash algo_test/ttt_discover/scripts/run_batch_3.sh
# =================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1

echo "========================================================"
echo "  TTT-Discover Batch: Prim(L1) → Manacher(L1) → KMP(L0)"
echo "========================================================"
echo "Start time: $(date)"
echo ""

# --- Experiment 1: Prim MST Level 1 ---
echo "=========================================="
echo "  [1/3] Starting Prim MST (Level 1)"
echo "=========================================="
bash "$SCRIPT_DIR/run_prim.sh" || {
    echo "[WARN] Prim experiment failed with exit code $?. Continuing to next..."
}
echo ""
echo "[1/3] Prim finished at $(date)"
echo ""

# --- Experiment 2: Manacher Level 1 ---
echo "=========================================="
echo "  [2/3] Starting Manacher (Level 1)"
echo "=========================================="
bash "$SCRIPT_DIR/run_manacher.sh" || {
    echo "[WARN] Manacher experiment failed with exit code $?. Continuing to next..."
}
echo ""
echo "[2/3] Manacher finished at $(date)"
echo ""

# --- Experiment 3: KMP Level 0 ---
echo "=========================================="
echo "  [3/3] Starting KMP (Level 0)"
echo "=========================================="
bash "$SCRIPT_DIR/run_kmp.sh" || {
    echo "[WARN] KMP experiment failed with exit code $?. Continuing to next..."
}
echo ""
echo "[3/3] KMP finished at $(date)"
echo ""

echo "========================================================"
echo "  All 3 experiments completed at $(date)"
echo "========================================================"
