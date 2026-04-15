#!/bin/bash
#
# TTT-Discover training script for graph-sp-dijkstra
# Uses entropic advantage estimator + PUCT curriculum + verifier reward
#
# All TTT-Discover code lives under ttt_discover/ — no modifications to unlearn/.
#

USER_ENV=$(whoami)
# set -x
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1

# --- Exp ---
export RUN_NAME="unknown"  # MUST be overridden
export ARNOLD_WORKER_NUM=1
export VLLM_ATTENTION_BACKEND=XFORMERS

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1

# --- path config ---
if [[ -z "$PROJECT_ROOT" || -z "$HDFS_MODEL_PATH" ]]; then
    echo "PROJECT_ROOT and HDFS_MODEL_PATH must be set explicitly"
    exit 1
fi
export HDFS_DATA_PATH=$PROJECT_ROOT/_data
export HDFS_CHECKPOINT_PATH=$PROJECT_ROOT/unlearn/saves
export HDFS_LOG_PATH=$PROJECT_ROOT/unlearn/logs

# --- TTT-Discover specific config ---
MODEL_NAME=""  # MUST be overridden
N_GPUS_PER_NODE=0  # MUST be overridden

# TTT problem configuration
TTT_PROJECT_ROOT="$PROJECT_ROOT"  # path to algo_test/ directory
TTT_PROBLEM_DIR="graph-sp-dijkstra"
TTT_LEVELS=""  # empty = all levels; or "level0,level1,level2"
EXECUTION_TIMEOUT=20

SEED=42
HEAD_IP=$(hostname -I | awk '{print $1}')
HEAD_PORT=6379

# --- Reward Model ---
REWARD_MANAGER=batch
# Reward function now lives inside ttt_discover/
REWARD_SCRIPT_PATH=$PROJECT_ROOT/ttt_discover/verl_integration/ttt_verifier_reward.py

# --- TTT has NO retain loss ---
RETAIN_COEF=0.0
UNLEARN_COEF=1.0
RETAIN_BATCH_SIZE=64  # must equal train_batch_size * rollout_n (1 * 64)

# --- eval parameters ---
VAL_BATCH_SIZE=1
TEST_FREQ=10

# --- batch parameters ---
# train_batch_size=1: one problem per step; rollout.n=64 generates 64 samples (paper: 64 per group)
TRAIN_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE=4
LOG_PROB_MICRO_BATCH_SIZE=16

# --- training parameters (Paper Table 9) ---
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=32768
LEARNING_RATE=1e-5
TOTAL_EPOCHS=50
SAVE_FREQ="[10,25,50]"

# --- GRPO/Entropic parameters (Paper Table 9) ---
CLIP_RATIO=0.2
KL_LOSS_COEF=0.01
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
ROLLOUT_N=64  # 64 samples per problem (paper: 64 rollouts per group)
KL_COEF=0.01
POLICY_ENTROPY_RATIO=1.0
POLICY_POS_RATIO=1.0

# --- PUCT parameters ---
PUCT_C=1.4

# --- Rollout (vLLM) ---
ROLLOUT_GPU_MEMORY_UTIL=0.5
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2

# =============================================================================

# Used for overriding
for argument in "$@"; do
  if [[ $argument == *"="* ]]; then
    echo "Overriding: $argument"
    export "$argument"
  fi
done

# === Validation of Required Variables ===
REQUIRED_VARS=(
    "RUN_NAME"
    "MODEL_NAME"
    "N_GPUS_PER_NODE"
)

MISSING_COUNT=0

echo "Checking required environment variables..."
for var_name in "${REQUIRED_VARS[@]}"; do
    var_value="${!var_name}"
    if [[ -z "$var_value" || "$var_value" == "unknown" || ("$var_name" == "N_GPUS_PER_NODE" && "$var_value" == "0") ]]; then
        echo "ERROR: Variable '$var_name' is not set or has an invalid value."
        ((MISSING_COUNT++))
    else
        echo "$var_name=$var_value"
    fi
done

if [ $MISSING_COUNT -gt 0 ]; then
    echo "---------------------------------------"
    echo "Total $MISSING_COUNT error(s) found. Please provide these variables via command line arguments."
    echo "Example: bash train.sh MODEL_NAME=my_model RUN_NAME=ttt_exp_01 N_GPUS_PER_NODE=4"
    exit 1
fi

echo "All required variables are set. Proceeding..."
echo "TTT_PROJECT_ROOT=$TTT_PROJECT_ROOT"
echo "TTT_PROBLEM_DIR=$TTT_PROBLEM_DIR"
echo "TTT_LEVELS=$TTT_LEVELS"
echo "REWARD_SCRIPT_PATH=$REWARD_SCRIPT_PATH"
# ========================================

POLICY_DEBUG_LOG_DIR=$HDFS_LOG_PATH/$RUN_NAME/policy
REWARD_DEBUG_LOG_DIR=$HDFS_LOG_PATH/$RUN_NAME/reward
METRICS_DEBUG_LOG_DIR=$HDFS_LOG_PATH/$RUN_NAME/metrics
VALIDATION_DATA_DIR=$HDFS_LOG_PATH/$RUN_NAME/val
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME/console.log"
mkdir -p $(dirname "$LOG_FILE_PATH")

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)

# Ensure both ttt_discover and verl packages are importable
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -m ttt_discover.verl_integration.main_ttt \
  algorithm.adv_estimator=entropic \
  +algorithm.entropic_gamma=0.693 \
  +data.seed=$SEED \
  data.train_files=[] \
  data.val_files=[] \
  +data.retain_files=[] \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  +data.retain_batch_size=$RETAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.filter_overlong_prompts=False \
  data.dataloader_num_workers=0 \
  +data.ttt_project_root=$TTT_PROJECT_ROOT \
  +data.ttt_problem_dir=$TTT_PROBLEM_DIR \
  +data.ttt_levels="$TTT_LEVELS" \
  +data.puct_c=$PUCT_C \
  custom_reward_function.path=$REWARD_SCRIPT_PATH \
  custom_reward_function.name=compute_score \
  reward_model.reward_manager=$REWARD_MANAGER \
  +reward_model.reward_kwargs.reward_debug_log_dir=$REWARD_DEBUG_LOG_DIR \
  +reward_model.reward_kwargs.project_root=$TTT_PROJECT_ROOT \
  +reward_model.reward_kwargs.problem_dir=$TTT_PROBLEM_DIR \
  +reward_model.reward_kwargs.levels="$TTT_LEVELS" \
  +reward_model.reward_kwargs.execution_timeout=$EXECUTION_TIMEOUT \
  actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
  actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  +actor_rollout_ref.actor.model_path=$HDFS_MODEL_PATH/$MODEL_NAME \
  +actor_rollout_ref.actor.retain_loss_coef=$RETAIN_COEF \
  +actor_rollout_ref.actor.unlearn_loss_coef=$UNLEARN_COEF \
  +actor_rollout_ref.actor.policy_pos_ratio=$POLICY_POS_RATIO \
  +actor_rollout_ref.actor.policy_entropy_ratio=$POLICY_ENTROPY_RATIO \
  +actor_rollout_ref.actor.policy_debug_log_dir=$POLICY_DEBUG_LOG_DIR \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
  actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.checkpoint.save_contents=["hf_model"] \
  algorithm.kl_ctrl.kl_coef=$KL_COEF \
  critic.ppo_micro_batch_size_per_gpu=4 \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  +trainer.metrics_debug_log_dir=$METRICS_DEBUG_LOG_DIR \
  trainer.validation_data_dir=$VALIDATION_DATA_DIR \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
  trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH
