#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/common_env.sh"

# export CUDA_VISIBLE_DEVICES=0,1
# bash ${PROJECT_ROOT}/simple_parallel/scripts/_run.sh \
#   TYPE=SFT \
#   BASE_MODEL=qwen3-4b-thinking-2507 \
#   BENCHMARKS="aime25" \
#   UNLEARN_CATEGORY=graph-sp-dijkstra \
#   UNLEARN_PARAMETERS=20260110_dijkstra_run_unlearn1.0_retain0.1_lr1e5_global_step_30 \
#   SFT_PARAMETERS=unlearn_code_math_2048/checkpoints/global_step_300

set -e
set -u
exec 2> >(while read -r line; do echo -e "\033[1;31m$line\033[0m" >&2; done)

if [[ -n "${PROXY:-}" ]]; then
    export http_proxy=${PROXY} && export https_proxy=${PROXY} && export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
fi
# >>>>>>> config >>>>>>>
TYPE=""  # BASE/UNLEARN/SFT/FINAL
BASE_MODEL=""  # e.g. qwen3-4b-thinking-2507
MODEL_PATH=""  # only for base_model: e.g. ${QWEN3_4B_THINKING_2507_PATH}
BENCHMARKS=""  # lcb,bfcl,aime25,forget,final
TEST_MODE="all"  # fast/all
SKIP_VLLM=False   # True/False
SKIP_INSTALL=True
if [[ $BENCHMARKS == "bfcl" ]]; then
    SKIP_INSTALL=False
fi

# unlearn/sft
UNLEARN_CATEGORY=""    # e.g. graph-sp-dijkstra
UNLEARN_PARAMETERS=""  # e.g. 20260110_dijkstra/run_unlearn1.0_retain0.1_lr1e5/global_step_30
# sft
SFT_PARAMETERS=""  # e.g. unlearn_code_math_2048/checkpoints/global_step_100
# final/forget test
TEST_CATEGORY=""    # e.g. graph-sp-dijkstra
LEVEL=0  # 0/1/2
GEN_VERIFY="False"  # True/False, only for final_test
# forget
FORGET_TEST_MODE="text"  # logits/text
# final
FINAL_TYPE=""  # cold_start/unlearned/sft
# <<<<<< /config <<<<<<<

# Used for overriding
for argument in "$@"; do
  if [[ $argument == *"="* ]]; then
    echo "Overriding: $argument"
    export "$argument"
  fi
done

algo_reinvention_require_repo_env_key "PROJECT_ROOT"
PROJECT_ROOT="${PROJECT_ROOT%/}"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"

if [[ $SKIP_INSTALL == "False" ]]; then
cat >/etc/apt/sources.list <<'EOF'
deb https://mirrors.ustc.edu.cn/ubuntu focal main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu focal-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu focal-backports main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu focal-security main restricted universe multiverse
EOF
apt update
apt install libnuma-dev -y
apt install tmux -y
fi

GPU_NUM=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

if [[ -n "$MODEL_PATH" && -n "${UNLEARN_PARAMETERS}" ]]; then
    echo -e "\033[31mError: do not set MODEL_PATH and UNLEARN at the same time\033[0m" >&2
    exit 1
fi

if [[ "${BASE_MODEL}" == "" ]]; then
    echo -e "\033[31mError: No BASE_MODEL\033[0m" >&2
    exit 1
elif [[ "$TYPE" == "" ]]; then
    echo -e "\033[31mError: No TYPE\033[0m" >&2
    exit 1
elif [[ "$BENCHMARKS" == "" ]]; then
    echo -e "\033[31mError: No BENCHMARKS\033[0m" >&2
    exit 1
fi

# parse the MODEL_NAME and MODEL_PATH
if [[ $TYPE == "BASE" ]]; then
    if [[ "$MODEL_PATH" == "" ]]; then
        REQUIRED_MODEL_PATH_ENV=""
        case "${BASE_MODEL}" in
            "ministral3-8b-instruct-2512")
                REQUIRED_MODEL_PATH_ENV="MINISTRAL3_8B_INSTRUCT_2512_PATH"
                ;;
            "qwen3-4b-thinking-2507")
                REQUIRED_MODEL_PATH_ENV="QWEN3_4B_THINKING_2507_PATH"
                ;;
            "qwen3-4b-instruct-2507")
                REQUIRED_MODEL_PATH_ENV="QWEN3_4B_INSTRUCT_2507_PATH"
                ;;
            "ministral3-14b-reasoning-2512")
                REQUIRED_MODEL_PATH_ENV="MINISTRAL3_14B_REASONING_2512_PATH"
                ;;
            *)
                echo -e "\033[31mError: Unsupported BASE_MODEL for TYPE==BASE without explicit MODEL_PATH: ${BASE_MODEL}\033[0m" >&2
                exit 1
                ;;
        esac

        MODEL_PATH="${!REQUIRED_MODEL_PATH_ENV:-}"
        algo_reinvention_require_env "${REQUIRED_MODEL_PATH_ENV}" || exit 1
        MODEL_PATH="${!REQUIRED_MODEL_PATH_ENV}"
    fi
    MODEL_PATH="${MODEL_PATH%/}"
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo -e "\033[31mError: MODEL_PATH does not exist when TYPE==BASE: ${MODEL_PATH}\033[0m" >&2
        exit 1
    fi
    MODEL_NAME=${BASE_MODEL}/base
elif [[ $TYPE == "UNLEARN" ]]; then
    if [[ "${UNLEARN_CATEGORY}" == "" || "${UNLEARN_PARAMETERS}" == "" ]]; then
        echo -e "\033[31mError: No UNLEARN_CATEGORY or UNLEARN_PARAMETERS when TYPE==UNLEARN\033[0m" >&2
        exit 1
    fi
    UNLEARN_SAVE_PARAMETERS=${UNLEARN_PARAMETERS//\//_}
    MODEL_NAME=${BASE_MODEL}/${UNLEARN_CATEGORY}/$UNLEARN_SAVE_PARAMETERS
    MODEL_PATH=${PROJECT_ROOT}/unlearn/saves/${BASE_MODEL}/${UNLEARN_CATEGORY}/${UNLEARN_PARAMETERS}/actor/huggingface
elif [[ $TYPE == "SFT" ]]; then
    if [[ "${UNLEARN_CATEGORY}" == "" || "${UNLEARN_PARAMETERS}" == "" || "$SFT_PARAMETERS" == "" ]]; then
        echo -e "\033[31mError: No UNLEARN_CATEGORY or UNLEARN_PARAMETERS or SFT_PARAMETERS when TYPE==SFT\033[0m" >&2
        exit 1
    fi
    UNLEARN_SAVE_PARAMETERS=${UNLEARN_PARAMETERS//\//_}
    SFT_SAVE_PARAMETERS=${SFT_PARAMETERS//\//_}
    MODEL_NAME=${BASE_MODEL}/${UNLEARN_CATEGORY}/${UNLEARN_SAVE_PARAMETERS}_SFT_${SFT_SAVE_PARAMETERS}
    MODEL_PATH=${PROJECT_ROOT}/sft/saves/${BASE_MODEL}/${UNLEARN_CATEGORY}/$UNLEARN_SAVE_PARAMETERS/$SFT_PARAMETERS/huggingface
elif [[ $TYPE == "FINAL" ]]; then
    if [[ "${UNLEARN_CATEGORY}" == "" || "${FINAL_TYPE}" == "" ]]; then
        echo -e "\033[31mError: No UNLEARN_CATEGORY/FINAL_TYPE when TYPE==FINAL\033[0m" >&2
        exit 1
    fi
    MODEL_NAME=${BASE_MODEL}/${UNLEARN_CATEGORY}/_final/${FINAL_TYPE}
    MODEL_PATH=${PROJECT_ROOT}/_final_ckpts/${BASE_MODEL}/${UNLEARN_CATEGORY}/${FINAL_TYPE}
else
    echo "\033[31mError: not supported TYPE: ${TYPE}\033[31m" >&2
    exit 1
fi

# test mode
if [[ $TEST_MODE == "fast" ]]; then
    LCB_NUM=1
    AIME_NUM=3
elif [[ $TEST_MODE == "all" ]]; then
    LCB_NUM=3
    AIME_NUM=8
else
    echo "\033[31mError: not supported TEST_MODE: ${TEST_MODE}\033[31m" >&2
    exit 1
fi

kill_tmux_by_gpu() {
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "Warning: CUDA_VISIBLE_DEVICES is empty. No sessions will be killed."
        return 0
    fi

    echo "Scanning for tmux sessions occupying GPUs: [${CUDA_VISIBLE_DEVICES}]..."

    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"

    local sessions
    sessions=$(tmux ls -F '#S' 2>/dev/null || echo "")

    if [ -z "$sessions" ]; then
        echo "No active tmux sessions found."
        return 0
    fi

    for session in $sessions; do
        if [[ "$session" =~ gpu ]]; then
            local gpu_part
            gpu_part=$(echo "$session" | sed 's/.*gpu//')

            for id in "${GPU_ARRAY[@]}"; do
                if echo "$gpu_part" | grep -qE "(^|,)${id}(,|$)"; then
                    echo "Action: Killing tmux session '$session' (Matched GPU ${id})"
                    tmux kill-session -t "$session"
                    break
                fi
            done
        fi
    done
}

resolve_ministral_source_dir() {
    if [[ "${TYPE}" == "BASE" && -n "${MODEL_PATH:-}" ]]; then
        echo "${MODEL_PATH%/}"
        return 0
    fi

    if [[ -n "${SOURCE_DIR:-}" ]]; then
        echo "${SOURCE_DIR%/}"
        return 0
    fi

    if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
        echo "${BASE_MODEL_PATH%/}"
        return 0
    fi

    case "${BASE_MODEL}" in
        "ministral3-14b-reasoning-2512")
            echo "${MINISTRAL3_14B_REASONING_2512_PATH:-}"
            ;;
        "ministral3-8b-instruct-2512")
            echo "${MINISTRAL3_8B_INSTRUCT_2512_PATH:-}"
            ;;
        *)
            return 1
            ;;
    esac
}

# ==> patch
if [[ "${BASE_MODEL}" == *"ministral"* ]]; then
    algo_reinvention_require_env "CONDA_VERL_MINISTRAL_NAME" || exit 1
    conda activate "${CONDA_VERL_MINISTRAL_NAME}"
    SOURCE_DIR="$(resolve_ministral_source_dir)" || {
        echo -e "\033[31mError: cannot resolve Ministral source dir for BASE_MODEL=${BASE_MODEL}. Set SOURCE_DIR or BASE_MODEL_PATH explicitly.\033[0m" >&2
        exit 1
    }
    if [[ ! -d "${SOURCE_DIR}" ]]; then
        echo -e "\033[31mError: resolved Ministral source dir does not exist: ${SOURCE_DIR}\033[0m" >&2
        exit 1
    fi
    if [ ! -f "$MODEL_PATH/tekken.json" ]; then
        echo "--- copying tekken.json ---"
        if [ -f "$SOURCE_DIR/tekken.json" ]; then
            cp "$SOURCE_DIR/tekken.json" "$MODEL_PATH/"
        else
            echo "ERROR $SOURCE_DIR not found tekken.json！"
        fi
    else
        echo "SKIP: $MODEL_PATH found tekken.json"
    fi

    for file in "params.json" "configuration.json" "processor_config.json" "README.md" "SYSTEM_PROMPT.txt"; do
        if [ ! -f "$MODEL_PATH/$file" ]; then
            if [ -f "$SOURCE_DIR/$file" ]; then
                cp "$SOURCE_DIR/$file" "$MODEL_PATH/"
                echo "fix: $file"
            fi
        fi
    done

    if [ ! -d "$MODEL_PATH/pure_text" ]; then
        echo "--- generating pure_text .safetensors ---"
        python ${WORKSPACE_ROOT}/_tools/training/mm2text.py \
            "$MODEL_PATH"
    else
        echo "SKIP: $MODEL_PATH found pure_text .safetensors"
    fi

    MODEL_PATH="${MODEL_PATH}/pure_text"

    if [ ! -f "$MODEL_PATH/tekken.json" ]; then
        echo "--- copying tekken.json ---"
        if [ -f "$SOURCE_DIR/tekken.json" ]; then
            cp "$SOURCE_DIR/tekken.json" "$MODEL_PATH/"
        else
            echo "ERROR $SOURCE_DIR not found tekken.json！"
        fi
    else
        echo "SKIP: $MODEL_PATH found tekken.json"
    fi

    for file in "params.json" "configuration.json" "processor_config.json" "README.md" "SYSTEM_PROMPT.txt"; do
        if [ ! -f "$MODEL_PATH/$file" ]; then
            if [ -f "$SOURCE_DIR/$file" ]; then
                cp "$SOURCE_DIR/$file" "$MODEL_PATH/"
                echo "fix: $file"
            fi
        fi
    done
fi

####################################### LCB ########################################################
if [[ "$BENCHMARKS" == *"lcb"* ]]; then
    if [[ "${BASE_MODEL}" == *"instruct-2507"* ]]; then
        MODEL_TYPE="Qwen3Instruct"
    elif [[ "${BASE_MODEL}" == *"thinking-2507"* ]]; then
        MODEL_TYPE="Qwen3Think"
    elif [[ "${BASE_MODEL}" == *"nemotron-cascade"* ]]; then
        MODEL_TYPE="NemotronCascade"
    elif [[ "${BASE_MODEL}" == *"ministral"* ]]; then
        MODEL_TYPE="${BASE_MODEL}"
    else
        echo "\033[31mError: Not support for lcb: ${BASE_MODEL}\033[31m" >&2
        exit 1
    fi
    kill_tmux_by_gpu
    bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/lcbv6_test.sh \
        $MODEL_PATH \
        "2025v6/${MODEL_NAME}" \
        $MODEL_TYPE \
        NUM=$LCB_NUM
fi

####################################### BFCL ########################################################
if [[ "$BENCHMARKS" == *"bfcl"* ]]; then
    if [[ "${BASE_MODEL}" == *"nemotron-cascade"* ]]; then
        MODEL_TYPE="nemotron-cascade-14b-FC"
    elif [[ "${BASE_MODEL}" == *"ministral3-8b-instruct-2512"* ]]; then
        MODEL_TYPE="mistralai/Ministral-3-8B-Instruct-2512"
    elif [[ "${BASE_MODEL}" == *"ministral3-14b-reasoning-2512"* ]]; then
        MODEL_TYPE="mistralai/Ministral-3-14B-Reasoning-2512"
    elif [[ "${BASE_MODEL}" == *"qwen3-4b-instruct-2507"* ]]; then
        MODEL_TYPE="qwen3-4b-nothink-FC"
    elif [[ "${BASE_MODEL}" == *"qwen3-4b-thinking-2507"* ]]; then
        MODEL_TYPE="qwen3-4b-think-FC"
    else
        echo "\033[31mError: Unsupported BASE_MODEL for bfcl: ${BASE_MODEL}\033[31m" >&2
        exit 1
    fi
    export SERPAPI_API_KEY=""
    export OPENAI_API_KEY=""
    if [[ "$SKIP_VLLM" == "False" ]]; then
        kill_tmux_by_gpu
    fi
    bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/bfcl_test.sh \
        $MODEL_PATH \
        ${MODEL_NAME} \
        $MODEL_TYPE \
        "all" \
        GPU_NUM=$GPU_NUM \
        SKIP_VLLM=$SKIP_VLLM \
        SKIP_SGLANG=$SKIP_VLLM

    set +u
    algo_reinvention_require_env "CONDA_BFCL_NAME" || exit 1
    conda activate "${CONDA_BFCL_NAME}"
    export BFCL_PROJECT_ROOT=${BFCL_PROJECT_ROOT:-${WORKSPACE_ROOT}/bfcl/${MODEL_NAME}}
    bfcl evaluate \
        --model $MODEL_TYPE
fi

####################################### AIME ########################################################
if [[ "$BENCHMARKS" == *"aime"* ]]; then
    if [[ "${BASE_MODEL}" == *"nemotron-cascade"* ]]; then
        SYSTEM_PROMPT="You are a helpful and harmless assistant."
        USER_PROMPT_TEMPLATE="{question}\n\nPlease put your final answer within \\boxed{{}}"
        MAX_MODEL_LEN=65536  # scaling via yarn
    elif [[ "${BASE_MODEL}" == *"ministral"* ]]; then
        SYSTEM_PROMPT=""
        USER_PROMPT_TEMPLATE="{question}\n\nPlease put your final answer within \\boxed{{}}"
        MAX_MODEL_LEN=65536
    else
        SYSTEM_PROMPT="You are a helpful assistant."
        USER_PROMPT_TEMPLATE="{question}\n\nPlease put your final answer within \\boxed{{}}"
        MAX_MODEL_LEN=65536
    fi
    if [[ "$SKIP_VLLM" == "False" ]]; then
        kill_tmux_by_gpu
    fi
    bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/aime25_test.sh \
        $MODEL_PATH \
        ${MODEL_NAME} \
        "" \
        SYSTEM_PROMPT="$SYSTEM_PROMPT" \
        USER_PROMPT_TEMPLATE="$USER_PROMPT_TEMPLATE" \
        NUM=$AIME_NUM \
        GPU_NUM=$GPU_NUM \
        MAX_MODEL_LEN=$MAX_MODEL_LEN \
        SKIP_VLLM=$SKIP_VLLM

    python ${PROJECT_ROOT}/simple_parallel/aggregation/aggregate_math.py \
        --input_dir ${PROJECT_ROOT}/_output/results/${MODEL_NAME}/benchmarks/aime25-$AIME_NUM
fi

####################################### FORGET_TEST ########################################################
if [[ "$BENCHMARKS" == *"forget"* ]]; then
    if [[ "$TEST_CATEGORY" == "" || "$FORGET_TEST_MODE" == "" ]]; then
        echo -e "\033[31mError: No TEST_CATEGORY or FORGET_TEST_MODE when FORGET_TEST\033[0m" >&2
        exit 1
    fi
    bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/forget_test.sh \
        $MODEL_PATH \
        ${MODEL_NAME} \
        CATEGORY=$TEST_CATEGORY \
        FORGET_TEST_MODE=$FORGET_TEST_MODE \
        SKIP_VLLM=$SKIP_VLLM
fi

####################################### FINAL_TEST ########################################################
if [[ "$BENCHMARKS" == *"final"* ]]; then
    if [[ "$TEST_CATEGORY" == "" ]]; then
        echo -e "\033[31mError: No TEST_CATEGORY when FINAL_TEST\033[0m" >&2
        exit 1
    fi
    SYSTEM_PROMPT="You are a research scientist. Please call the execute_python_code tool to make sure that your code is correct, then call the submit_final_answer tool to submit the final code."
    USER_PROMPT_TEMPLATE="{question}"
    if [[ "${BASE_MODEL}" == *"ministral"* ]]; then
        SYSTEM_PROMPT=""
        USER_PROMPT_TEMPLATE="You are a research scientist. Please call the execute_python_code tool to make sure that your code is correct, then call the submit_final_answer tool to submit the final code.\n\n{question}"
    fi
    if [[ "$SKIP_VLLM" == "False" ]]; then
        kill_tmux_by_gpu
    fi
    bash ${PROJECT_ROOT}/simple_parallel/scripts/inferences/final_test.sh \
        $MODEL_PATH \
        ${MODEL_NAME} \
        "" \
        SYSTEM_PROMPT="$SYSTEM_PROMPT" \
        USER_PROMPT_TEMPLATE="$USER_PROMPT_TEMPLATE" \
        CATEGORY=$TEST_CATEGORY \
        LEVEL=$LEVEL \
        GEN_VERIFY=$GEN_VERIFY \
        GPU_NUM=$GPU_NUM \
        SKIP_VLLM=$SKIP_VLLM
fi
