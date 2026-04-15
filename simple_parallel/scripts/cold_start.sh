#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# bash ${PROJECT_ROOT}/simple_parallel/scripts/cold_start.sh \
#  CATEGORY=graph-sp-dijkstra \
#  MODEL_NAME=qwen3-4b-thinking-2507


CATEGORY=graph-sp-dijkstra
MODEL_NAME=qwen3-4b-thinking-2507
ONLY_TEST=False
ONLY_TRAIN=False

# Used for overriding
for argument in "$@"; do
  if [[ $argument == *"="* ]]; then
    echo "Overriding: $argument"
    export "$argument"
  fi
done

algo_reinvention_require_repo_env_key "PROJECT_ROOT" || exit 1
PROJECT_ROOT="${PROJECT_ROOT%/}"

tmux ls 2>/dev/null | grep "server" | awk -F: '{print $1}' | xargs -I {} tmux kill-session -t "{}"
if [[ $ONLY_TEST == "False" ]]; then
    bash "${PROJECT_ROOT}/sft/scripts/run.sh" \
        MODEL="${MODEL_NAME}" \
        ALGORITHM="${CATEGORY}" \
        RECIPES="cold_start"
fi

if [[ $ONLY_TRAIN == "False" ]]; then
    ckpt_list=(140)
    for ckpt in ${ckpt_list[@]}; do
        bash "${PROJECT_ROOT}/simple_parallel/scripts/_run.sh" \
            TYPE=SFT \
            BASE_MODEL=${MODEL_NAME} \
            BENCHMARKS="forget" \
            UNLEARN_CATEGORY=${CATEGORY} \
            TEST_CATEGORY=${CATEGORY} \
            UNLEARN_PARAMETERS=cold_start \
            SFT_PARAMETERS=idk_indist_start/checkpoints/global_step_${ckpt}

        bash "${PROJECT_ROOT}/simple_parallel/scripts/_run.sh" \
            TYPE=SFT \
            BASE_MODEL=${MODEL_NAME} \
            BENCHMARKS="lcb" \
            UNLEARN_CATEGORY=${CATEGORY} \
            TEST_CATEGORY=${CATEGORY} \
            UNLEARN_PARAMETERS=cold_start \
            SFT_PARAMETERS=idk_indist_start/checkpoints/global_step_${ckpt}
    done
fi
